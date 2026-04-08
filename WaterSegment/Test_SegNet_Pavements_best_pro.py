import SegNet
import os
import argparse
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv
from glob import glob

def get_image_paths(path):
    """自适应获取图片列表（支持单文件或文件夹）"""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            paths.extend(glob(os.path.join(path, ext)))
            paths.extend(glob(os.path.join(path, ext.upper())))
        return sorted(list(set(paths)))
    else:
        raise ValueError(f"输入路径无效: {path}")

def find_ground_truth(img_name, gt_dir):
    """智能查找真值文件（支持后缀不匹配）"""
    base_name = os.path.splitext(img_name)[0]
    exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '']
    
    for ext in exts:
        gt_path = os.path.join(gt_dir, base_name + ext)
        if os.path.exists(gt_path):
            return gt_path
    return None

def load_model(weight_fn, model_json, cuda):
    """加载模型权重"""
    model = SegNet.SegNet(
        in_chn=model_json['in_chn'], 
        out_chn=model_json['out_chn'], 
        BN_momentum=model_json.get('bn_momentum', 0.1)
    )
    if cuda:
        model.cuda()
    
    checkpoint = torch.load(weight_fn, map_location='cuda' if cuda else 'cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"✓ 加载权重: {weight_fn} | Epoch: {checkpoint.get('epoch', 'unknown')}\n")
    return model

def save_overlay_result(pil_img, pred_mask, save_path, alpha=0.4):
    """
    保存红色透明叠加图（水为红色，背景保持原图不变暗）
    pred_mask: 二值数组 (H, W)，1表示水，0表示背景
    """
    img_array = np.array(pil_img).astype(np.float32)
    
    # 创建红色叠加层
    red_overlay = np.zeros_like(img_array)
    red_overlay[pred_mask == 1] = [255, 0, 0]  # 水体区域设为红色
    
    # 关键修改：只在mask==1的区域进行alpha混合，背景(img_array)保持不变
    # result = img_array * mask_weight + red_overlay * (1 - mask_weight)
    # 其中 mask_weight = 1 在背景区（不变），mask_weight = (1-alpha) 在水体区（混合）
    mask_3ch = np.stack([pred_mask] * 3, axis=-1)  # (H,W,3)
    
    # 背景权重：mask=0时为1（完全原图），mask=1时为(1-alpha)（混合）
    bg_weight = 1 - (mask_3ch * alpha)
    # 红色权重：mask=0时为0（无红色），mask=1时为alpha（有红色）
    red_weight = mask_3ch * alpha
    
    result = img_array * bg_weight + red_overlay * red_weight
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    Image.fromarray(result).save(save_path)
    print(f"  ✓ 已保存: {os.path.basename(save_path)}")

def compute_metrics(pred_mask, gt_mask):
    """计算分割指标"""
    TP = np.sum((pred_mask == 1) & (gt_mask == 1))
    FP = np.sum((pred_mask == 1) & (gt_mask == 0))
    FN = np.sum((pred_mask == 0) & (gt_mask == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    miou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'miou': float(miou)
    }

def print_metrics_table(metrics_list):
    """终端打印指标表格"""
    print("\n" + "="*80)
    print("分割性能评估结果")
    print("-"*80)
    print(f"{'文件名':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'mIoU':<10}")
    print("-"*80)
    
    for m in metrics_list:
        print(f"{m['image']:<30} {m['precision']:<10.4f} {m['recall']:<10.4f} "
              f"{m['f1']:<10.4f} {m['miou']:<10.4f}")
    
    avg_p = np.mean([m['precision'] for m in metrics_list])
    avg_r = np.mean([m['recall'] for m in metrics_list])
    avg_f1 = np.mean([m['f1'] for m in metrics_list])
    avg_iou = np.mean([m['miou'] for m in metrics_list])
    
    print("-"*80)
    print(f"{'[整体平均]':<30} {avg_p:<10.4f} {avg_r:<10.4f} {avg_f1:<10.4f} {avg_iou:<10.4f}")
    print("="*80)

def save_csv(metrics_list, save_path):
    """保存CSV文件"""
    if not metrics_list:
        return
    
    avg_metrics = {
        'image': 'AVERAGE',
        'precision': np.mean([m['precision'] for m in metrics_list]),
        'recall': np.mean([m['recall'] for m in metrics_list]),
        'f1': np.mean([m['f1'] for m in metrics_list]),
        'miou': np.mean([m['miou'] for m in metrics_list])
    }
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'precision', 'recall', 'f1', 'miou'])
        writer.writeheader()
        writer.writerows(metrics_list + [avg_metrics])
    
    print(f"\n✓ 指标已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='SegNet 水体分割推理工具（全参数版）')
    
    # 所有参数均为 --param_name 形式
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入图片路径或文件夹路径（必需）')
    parser.add_argument('--weights', '-w', type=str, required=True,
                       help='模型权重文件路径 .pth（必需）')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出文件夹路径（保存红色叠加图，可选）')
    parser.add_argument('--ground_truth', '-g', type=str, default=None,
                       help='真值标签文件夹路径（可选，用于计算指标）')
    parser.add_argument('--alpha', '-a', type=float, default=0.4,
                       help='红色蒙版透明度 (0.0-1.0)，默认0.4，建议0.3-0.5')
    parser.add_argument('--use_normalize', action='store_true',
                       help='使用ImageNet Normalize（如果训练时使用了Normalize）')
    
    args = parser.parse_args()
    
    # 获取待处理图片
    try:
        input_paths = get_image_paths(args.input)
        print(f"发现 {len(input_paths)} 张待处理图片\n")
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    # 创建输出目录
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        print(f"输出目录: {args.output}\n")
    
    # 加载配置和模型
    try:
        with open('./model.json') as f:
            model_json = json.load(f)
    except FileNotFoundError:
        print("错误: 未找到 model.json 文件")
        return
    
    cuda = torch.cuda.is_available()
    model = load_model(args.weights, model_json, cuda)
    
    # 预处理流程（与训练时对齐）
    transform_list = [
        transforms.Resize((320, 640)),  # (H, W) - SegNet标准尺寸
        transforms.ToTensor(),
    ]
    
    if args.use_normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        print("使用预处理: Resize(320,640) + ToTensor + Normalize")
    else:
        print("使用预处理: Resize(320,640) + ToTensor")
    
    transform = transforms.Compose(transform_list)
    
    metrics_list = []
    
    # 批量推理
    for img_path in input_paths:
        img_name = os.path.basename(img_path)
        
        try:
            # 保留原始PIL用于保存（未resize版本）
            pil_img_original = Image.open(img_path).convert('RGB')
            # 用于推理的tensor（已resize）
            img_tensor = transform(pil_img_original)
        except Exception as e:
            print(f"跳过 {img_name}: 无法读取 ({e})")
            continue
        
        if cuda:
            img_tensor = img_tensor.cuda()
        
        # 推理
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0))
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (320, 640)
        
        # 将预测mask resize回原始尺寸用于可视化（保持原图比例）
        pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
        pred_pil_resized = pred_pil.resize(pil_img_original.size, Image.NEAREST)
        pred_original_size = (np.array(pred_pil_resized) > 127).astype(np.uint8)
        
        # 保存叠加图（背景不变暗，仅水体变红）
        if args.output:
            save_path = os.path.join(args.output, img_name)
            save_overlay_result(pil_img_original, pred_original_size, save_path, args.alpha)
        
        # 处理真值和指标（在模型输出尺寸320x640上计算）
        if args.ground_truth:
            gt_path = find_ground_truth(img_name, args.ground_truth)
            
            if gt_path and os.path.exists(gt_path):
                try:
                    gt_img = Image.open(gt_path).convert('L')
                    gt_array = np.array(gt_img)
                    gt_mask = (gt_array > 127).astype(np.uint8)
                    
                    # 尺寸对齐到模型输出尺寸
                    if gt_mask.shape != pred.shape:
                        gt_mask_pil = Image.fromarray((gt_mask * 255).astype(np.uint8))
                        gt_mask_resized = gt_mask_pil.resize((pred.shape[1], pred.shape[0]), Image.NEAREST)
                        gt_mask = (np.array(gt_mask_resized) > 127).astype(np.uint8)
                    
                    metrics = compute_metrics(pred, gt_mask)
                    metrics['image'] = img_name
                    metrics_list.append(metrics)
                        
                except Exception as e:
                    print(f"  警告: 处理真值失败: {e}")
            else:
                print(f"  警告: 未找到真值文件 (尝试: {os.path.splitext(img_name)[0]}.*)")
    
    # 输出指标结果
    if args.ground_truth and metrics_list:
        print_metrics_table(metrics_list)
        if args.output:
            save_csv(metrics_list, os.path.join(args.output, 'metrics.csv'))
    elif args.ground_truth:
        print("\n警告: 未找到任何有效真值数据")
    
    print(f"\n✓ 处理完成！共处理 {len(input_paths)} 张图片")

if __name__ == "__main__":
    main()