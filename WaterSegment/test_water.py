from SegNet import SegNet
from Pavements import Pavements
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import json
import csv
import time
from pathlib import Path
from torch.utils.data import DataLoader


def load_model_json():
    """加载模型配置"""
    with open(os.path.join(os.getcwd(), 'model.json')) as f:
        model_json = json.load(f)
    return model_json


def compute_metrics(pred_mask, true_mask, num_classes):
    """计算分割指标：Precision, Recall, F1, IoU（宏平均）"""
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    metrics_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()
        
        tp = (pred_cls * true_cls).sum()
        fp = (pred_cls * (1 - true_cls)).sum()
        fn = ((1 - pred_cls) * true_cls).sum()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)
        
        if true_cls.sum() > 0:
            metrics_per_class.append({
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'iou': iou.item(),
            })
    
    if metrics_per_class:
        return {
            'precision': np.mean([m['precision'] for m in metrics_per_class]),
            'recall': np.mean([m['recall'] for m in metrics_per_class]),
            'f1': np.mean([m['f1'] for m in metrics_per_class]),
            'miou': np.mean([m['iou'] for m in metrics_per_class]),
        }
    return {'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0}


@torch.no_grad()
def test_model(model, dataloader, device, num_classes, loss_fn):
    """测试模型，返回各项指标"""
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    inference_times = []
    
    for data in dataloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        # 测量推理时间
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        output = model(images)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_times.append(time.time() - start)
        
        # 计算 loss
        loss = loss_fn(output, labels)
        total_loss += loss.item()
        num_batches += 1
        
        # 获取预测结果
        preds = output.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
    
    # 计算指标
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets, num_classes)
    metrics['loss'] = total_loss / num_batches
    metrics['inference_time_ms'] = np.mean(inference_times) * 1000
    metrics['fps'] = images.size(0) / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
    metrics['total_images'] = len(all_preds)
    
    return metrics


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        'total_params': total_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def save_results(metrics, model_info, save_path, args, model_json):
    """保存测试结果到 CSV"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建结果字典
    result = {
        'model_path': args.model,
        'test_images': args.images,
        'test_masks': args.masks,
        'total_params': model_info['total_params'],
        'model_size_mb': f"{model_info['model_size_mb']:.2f}",
        'num_classes': model_json['out_chn'],
        'batch_size': model_json['batch_size'],
        'test_loss': f"{metrics['loss']:.6f}",
        'test_precision': f"{metrics['precision']:.6f}",
        'test_recall': f"{metrics['recall']:.6f}",
        'test_f1': f"{metrics['f1']:.6f}",
        'test_miou': f"{metrics['miou']:.6f}",
        'inference_time_ms': f"{metrics['inference_time_ms']:.4f}",
        'fps': f"{metrics['fps']:.2f}",
        'total_images': metrics['total_images'],
    }
    
    # 写入 CSV（追加模式）
    header = list(result.keys())
    file_exists = save_path.exists()
    
    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    
    print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test SegNet on test set')
    
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the trained model .pth.tar file')
    parser.add_argument('--images', '-i', type=str, required=True,
                        help='Path to the test images directory')
    parser.add_argument('--masks', type=str, required=True,
                        help='Path to the test masks directory')
    parser.add_argument('--output', '-o', type=str, default='./logs/tensorboard/test_results.csv',
                        help='Path to save test results CSV')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    # 加载配置
    model_json = load_model_json()
    
    # 加载模型
    print(f'Loading model from {args.model}')
    model = SegNet(
        in_chn=model_json['in_chn'],
        out_chn=model_json['out_chn'],
        BN_momentum=model_json['bn_momentum']
    )
    
    # 加载权重（过滤非模型参数）
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # 移除可能存在的额外键
    for key in ['epoch', 'optimizer', 'miou', 'metrics']:
        checkpoint.pop(key, None)
    
    # 如果 state_dict 被嵌套保存
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # 获取模型信息
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")
    
    # 准备测试数据
    testset = Pavements(args.images, args.masks)
    testloader = DataLoader(
        testset,
        batch_size=model_json['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    print(f'Test set size: {len(testset)} images')
    
    # 损失函数
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(model_json['cross_entropy_loss_weights'])
    ).to(device)
    
    # 测试
    print('Starting evaluation...')
    metrics = test_model(model, testloader, device, model_json['out_chn'], loss_fn)
    
    # 打印结果
    print('=' * 50)
    print('TEST RESULTS')
    print('=' * 50)
    print(f"Loss:           {metrics['loss']:.6f}")
    print(f"Precision:      {metrics['precision']:.6f}")
    print(f"Recall:         {metrics['recall']:.6f}")
    print(f"F1-Score:       {metrics['f1']:.6f}")
    print(f"mIoU:           {metrics['miou']:.6f}")
    print(f"Inference Time: {metrics['inference_time_ms']:.4f} ms")
    print(f"FPS:            {metrics['fps']:.2f}")
    print(f"Total Images:   {metrics['total_images']}")
    print('=' * 50)
    
    # 保存结果
    save_results(metrics, model_info, args.output, args, model_json)
    
    # 同时保存详细文本报告
    report_path = Path(args.output).parent / 'test_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Images: {args.images}\n")
        f.write(f"Test Masks: {args.masks}\n\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Number of Classes: {model_json['out_chn']}\n")
        f.write(f"Batch Size: {model_json['batch_size']}\n\n")
        f.write(f"Test Set Size: {metrics['total_images']} images\n\n")
        f.write(f"Loss:           {metrics['loss']:.6f}\n")
        f.write(f"Precision:      {metrics['precision']:.6f}\n")
        f.write(f"Recall:         {metrics['recall']:.6f}\n")
        f.write(f"F1-Score:       {metrics['f1']:.6f}\n")
        f.write(f"mIoU:           {metrics['miou']:.6f}\n")
        f.write(f"Inference Time: {metrics['inference_time_ms']:.4f} ms\n")
        f.write(f"FPS:            {metrics['fps']:.2f}\n")
    
    print(f"Text report saved to {report_path}")


if __name__ == '__main__':
    main()