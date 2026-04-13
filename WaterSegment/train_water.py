from SegNet import SegNet
from Pavements import Pavements
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import json
import csv
import time
from pathlib import Path
from collections import defaultdict


def save_checkpoint(state, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("Created directory: {}".format(directory))
    torch.save(state, path)
    print("Checkpoint saved at {}".format(path))


def load_model_json():
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
def evaluate_metrics(model, dataloader, device, num_classes, loss_fn):
    """评估模型，返回各项指标和推理时间"""
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
    
    return metrics


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


class CSVLogger:
    """CSV日志记录器，每epoch一行"""
    
    def __init__(self, save_path):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.header = [
            'epoch',
            'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        self.rows = []
        
    def log(self, epoch, train_metrics, val_metrics, lr):
        """记录一轮数据"""
        row = {
            'epoch': epoch,
            'train_loss': f"{train_metrics['loss']:.6f}",
            'train_precision': f"{train_metrics['precision']:.6f}",
            'train_recall': f"{train_metrics['recall']:.6f}",
            'train_f1': f"{train_metrics['f1']:.6f}",
            'train_miou': f"{train_metrics['miou']:.6f}",
            'val_loss': f"{val_metrics['loss']:.6f}",
            'val_precision': f"{val_metrics['precision']:.6f}",
            'val_recall': f"{val_metrics['recall']:.6f}",
            'val_f1': f"{val_metrics['f1']:.6f}",
            'val_miou': f"{val_metrics['miou']:.6f}",
            'inference_time_ms': f"{val_metrics['inference_time_ms']:.4f}",
            'fps': f"{val_metrics['fps']:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        self.rows.append(row)
        
    def save(self):
        """保存CSV"""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Training log saved to {self.save_path}")


def main(args):

    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    
    writer = SummaryWriter(args.tensorboard_logs_dir)
    
    # 创建CSV记录器
    csv_path = Path(args.tensorboard_logs_dir) / 'training_log.csv'
    logger = CSVLogger(csv_path)

    weight_fn = args.weight_fn
    model_json = load_model_json()

    assert len(model_json['cross_entropy_loss_weights']) == model_json['out_chn'], \
        "CrossEntropyLoss class weights must be same as no. of output channels"

    # 训练集
    trainset = Pavements(args.images, args.masks)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=model_json['batch_size'], 
        shuffle=True, 
        num_workers=4
    )

    # 验证集（可选）
    valloader = None
    if args.val_images and args.val_masks:
        valset = Pavements(args.val_images, args.val_masks)
        valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=model_json['batch_size'], 
            shuffle=False, 
            num_workers=4
        )
        print(f"Validation set loaded: {len(valset)} samples")

    model = SegNet(
        in_chn=model_json['in_chn'], 
        out_chn=model_json['out_chn'], 
        BN_momentum=model_json['bn_momentum']
    )
    optimizer = optim.SGD(
        model.parameters(), 
        lr=model_json['learning_rate'], 
        momentum=model_json['sgd_momentum']
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(model_json['cross_entropy_loss_weights'])
    )

    if cuda_available:
        model.cuda()
        loss_fn.cuda()

    # 记录模型信息
    model_info = get_model_info(model)
    info_path = Path(args.tensorboard_logs_dir) / 'model_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Input Channels: {model_json['in_chn']}\n")
        f.write(f"Output Channels: {model_json['out_chn']}\n")
        f.write(f"Batch Size: {model_json['batch_size']}\n")
    print(f"Model info: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

    run_epoch = model_json['epochs']
    start_epoch = 1
    
    # 加载检查点
    if weight_fn is not None:
        if os.path.isfile(weight_fn):
            print("Loading checkpoint '{}'".format(weight_fn))
            checkpoint = torch.load(weight_fn, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (epoch {})".format(weight_fn, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'. Will create new checkpoint.".format(weight_fn))
            weight_fn = os.path.join(
                os.getcwd(), 
                "weights/checkpoint_pavements_{}.pth.tar".format(
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                )
            )
    else:
        print("Starting new checkpoint.")
        weight_fn = os.path.join(
            os.getcwd(), 
            "weights/checkpoint_pavements_{}.pth.tar".format(
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        )

    best_miou = 0.0

    for epoch in range(start_epoch, run_epoch + 1):
        print('\n=== Epoch {} ==='.format(epoch))

        # ===== 训练阶段 =====
        model.train()
        sum_loss = 0.0
        all_preds = []
        all_targets = []

        for j, data in enumerate(trainloader, 1):
            images, labels = data
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            # 收集预测用于计算指标
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

            # TensorBoard 记录
            writer.add_scalar('Train/Loss', loss.item()/trainloader.batch_size, 
                            (epoch-1)*len(trainloader) + j)
            sum_loss += loss.item()

            print('Train Loss @ batch {}: {:.4f}'.format(j, loss.item() / trainloader.batch_size))

        # 计算训练指标
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        train_metrics = compute_metrics(all_preds, all_targets, model_json['out_chn'])
        train_metrics['loss'] = sum_loss / (j * trainloader.batch_size)

        # TensorBoard 记录 epoch 级别指标
        writer.add_scalar('Train/Avg_Loss_per_Epoch', train_metrics['loss'], epoch)
        writer.add_scalar('Train/Precision', train_metrics['precision'], epoch)
        writer.add_scalar('Train/Recall', train_metrics['recall'], epoch)
        writer.add_scalar('Train/F1', train_metrics['f1'], epoch)
        writer.add_scalar('Train/mIoU', train_metrics['miou'], epoch)

        print('>>> Average TRAIN loss: {:.4f}, mIoU: {:.4f}, F1: {:.4f}'.format(
            train_metrics['loss'], train_metrics['miou'], train_metrics['f1']))

        # ===== 验证阶段 =====
        val_metrics = {
            'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0,
            'inference_time_ms': 0, 'fps': 0
        }
        
        if valloader is not None:
            val_metrics = evaluate_metrics(model, valloader, device, 
                                          model_json['out_chn'], loss_fn)
            
            # TensorBoard 记录
            writer.add_scalar('Val/Avg_Loss_per_Epoch', val_metrics['loss'], epoch)
            writer.add_scalar('Val/Precision', val_metrics['precision'], epoch)
            writer.add_scalar('Val/Recall', val_metrics['recall'], epoch)
            writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
            writer.add_scalar('Val/mIoU', val_metrics['miou'], epoch)
            writer.add_scalar('Val/Inference_Time_ms', val_metrics['inference_time_ms'], epoch)
            writer.add_scalar('Val/FPS', val_metrics['fps'], epoch)

            print('>>> Average VAL loss: {:.4f}, mIoU: {:.4f}, F1: {:.4f}, FPS: {:.2f}'.format(
                val_metrics['loss'], val_metrics['miou'], val_metrics['f1'], val_metrics['fps']))

            # 保存最佳模型
            if val_metrics['miou'] > best_miou:
                best_miou = val_metrics['miou']
                best_path = os.path.join(os.path.dirname(weight_fn), 'checkpoint_best.pth.tar')
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'miou': best_miou,
                }, best_path)
                print(f"New best model! mIoU: {best_miou:.4f}")

        # 记录到 CSV
        logger.log(epoch, train_metrics, val_metrics, optimizer.param_groups[0]['lr'])

    # 保存训练日志
    logger.save()
    
    # 保存最终模型
    print("\nTraining complete. Saving checkpoint...")
    save_checkpoint({
        'epoch': run_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, weight_fn)
    
    print(f"Best Val mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", required=True, type=str, 
                        help="Directory: Training raw images")
    parser.add_argument("--masks", required=True, type=str, 
                        help="Directory: Training annotated masks")
    parser.add_argument("--val-images", type=str, default=None,
                        help="Directory: Validation raw images (optional)")
    parser.add_argument("--val-masks", type=str, default=None,
                        help="Directory: Validation annotated masks (optional)")
    parser.add_argument("--logs", dest="tensorboard_logs_dir", required=True, type=str,
                        help="Directory: Logs for tensorboard and CSV")
    parser.add_argument("--weight-fn", type=str, help="Path: Trained weights", default=None)

    args = parser.parse_args()
    main(args)