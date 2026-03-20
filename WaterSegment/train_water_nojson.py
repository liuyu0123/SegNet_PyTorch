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
import csv
import time
from pathlib import Path

def save_checkpoint(state, path):
    """保存检查点"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("Created directory: {}".format(directory))
    torch.save(state, path)
    print("Checkpoint saved at {}".format(path))

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
            'epoch', 'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
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
        """保存CSV (覆盖写入以保留所有历史)"""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Training log saved to {self.save_path}")

def main(args):
    # 设备设置
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")

    # 确保保存目录存在
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # TensorBoard & CSV Logger
    writer = SummaryWriter(args.log_dir)
    csv_path = Path(args.log_dir) / f"{args.log_name}.csv"
    logger = CSVLogger(csv_path)

    # 数据集加载
    trainset = Pavements(args.images, args.masks)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    valloader = None
    if args.val_images and args.val_masks:
        valset = Pavements(args.val_images, args.val_masks)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        print(f"Validation set loaded: {len(valset)} samples")

    # 模型初始化
    model = SegNet(
        in_chn=args.in_channels,
        out_chn=args.out_channels,
        BN_momentum=args.bn_momentum
    ).to(device)

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.sgd_momentum
    )

    # 损失函数
    # 解析权重字符串，例如 "1.0,1.0"
    weights = [float(w) for w in args.loss_weights.split(',')]
    assert len(weights) == args.out_channels, "Loss weights length must match output channels"
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    # 记录模型信息
    model_info = get_model_info(model)
    info_path = Path(args.log_dir) / f"{args.log_name}_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Input Channels: {args.in_channels}\n")
        f.write(f"Output Channels: {args.out_channels}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
    print(f"Model info: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

    start_epoch = 1
    best_miou = 0.0

    # 加载检查点 (用于断点续训)
    if args.load:
        if os.path.isfile(args.load):
            print(f"Loading checkpoint '{args.load}'")
            checkpoint = torch.load(args.load, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'best_miou' in checkpoint:
                best_miou = checkpoint['best_miou']
            print(f"Loaded checkpoint '{args.load}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.load}'. Starting fresh.")

    # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        print('\n=== Epoch {} ==='.format(epoch))
        
        # ===== 训练阶段 =====
        model.train()
        sum_loss = 0.0
        all_preds = []
        all_targets = []
        
        for j, data in enumerate(trainloader, 1):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

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
            writer.add_scalar('Train/Loss', loss.item() / trainloader.batch_size, (epoch-1)*len(trainloader) + j)
            sum_loss += loss.item()
            print('Train Loss @ batch {}: {:.4f}'.format(j, loss.item() / trainloader.batch_size))

        # 计算训练指标
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        train_metrics = compute_metrics(all_preds, all_targets, args.out_channels)
        train_metrics['loss'] = sum_loss / (j * trainloader.batch_size)

        writer.add_scalar('Train/Avg_Loss_per_Epoch', train_metrics['loss'], epoch)
        writer.add_scalar('Train/mIoU', train_metrics['miou'], epoch)
        print('>>> Average TRAIN loss: {:.4f}, mIoU: {:.4f}, F1: {:.4f}'.format(
            train_metrics['loss'], train_metrics['miou'], train_metrics['f1']))

        # ===== 验证阶段 =====
        val_metrics = {
            'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0, 'inference_time_ms': 0, 'fps': 0
        }
        if valloader is not None:
            val_metrics = evaluate_metrics(model, valloader, device, args.out_channels, loss_fn)
            writer.add_scalar('Val/Avg_Loss_per_Epoch', val_metrics['loss'], epoch)
            writer.add_scalar('Val/mIoU', val_metrics['miou'], epoch)
            writer.add_scalar('Val/FPS', val_metrics['fps'], epoch)
            print('>>> Average VAL loss: {:.4f}, mIoU: {:.4f}, F1: {:.4f}, FPS: {:.2f}'.format(
                val_metrics['loss'], val_metrics['miou'], val_metrics['f1'], val_metrics['fps']))

        # ===== 保存逻辑 =====
        
        # 1. 保存最佳模型
        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            best_path = os.path.join(args.model_dir, f"{args.model_name}_best.pth")
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_miou': best_miou,
            }, best_path)
            print(f"New best model! mIoU: {best_miou:.4f}")

        # 2. 分步保存模型
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            interval_path = os.path.join(args.model_dir, f"{args.model_name}_epoch{epoch}.pth")
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, interval_path)

        # 3. 记录 CSV (每个 epoch 立即保存)
        logger.log(epoch, train_metrics, val_metrics, optimizer.param_groups[0]['lr'])
        logger.save()

    # ===== 训练结束 =====
    print("\nTraining complete.")
    # 保存最终模型 (_last.pth)
    last_path = os.path.join(args.model_dir, f"{args.model_name}_last.pth")
    save_checkpoint({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_miou': best_miou,
    }, last_path)

    print(f"Best Val mIoU: {best_miou:.4f}")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SegNet Training Script')
    
    # 数据参数
    parser.add_argument('--images', type=str, required=True, help='Directory: Training raw images')
    parser.add_argument('--masks', type=str, required=True, help='Directory: Training masks')
    parser.add_argument('--val-images', type=str, default=None, help='Directory: Validation raw images')
    parser.add_argument('--val-masks', type=str, default=None, help='Directory: Validation masks')
    
    # 保存与命名参数 (新增)
    parser.add_argument('--model-dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--model-name', type=str, default='segnet', help='Model filename prefix (e.g., exp1)')
    parser.add_argument('--log-name', type=str, default='training_log', help='Log CSV filename')
    parser.add_argument('--save-interval', type=int, default=0, help='Save checkpoint every N epochs (0 to disable)')
    parser.add_argument('--load', type=str, default=None, help='Path to checkpoint to resume training')

    # 原本 model.json 中的参数 (新增默认值)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--sgd-momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--bn-momentum', type=float, default=0.5, help='BatchNorm momentum')
    parser.add_argument('--loss-weights', type=str, default='1.0,1.0', help='CrossEntropy loss weights, comma separated')
    parser.add_argument('--in-channels', type=int, default=3, help='Input channels')
    parser.add_argument('--out-channels', type=int, default=2, help='Output channels')
    
    # 其他参数
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')

    args = parser.parse_args()
    main(args)
