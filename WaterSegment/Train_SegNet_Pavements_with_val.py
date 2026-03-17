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

def save_checkpoint(state, path):
    # 自动创建父目录（如果不存在）
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("Created directory: {}".format(directory))

    torch.save(state, path)
    print("Checkpoint saved at {}".format(path))


def load_model_json():

    # batch_size: Training batch-size
    # epochs: No. of epochs to run
    # lr: Optimizer learning rate
    # momentum: SGD momentum
    # no_cuda: Disables CUDA training (**To be implemented)
    # seed: Random seed
    # in-chn: Input image channels (3 for RGB, 4 for RGB-A)
    # out-chn: Output channels/semantic classes (2 for Pavements dataset)

    with open(os.path.join(os.getcwd(), 'model.json')) as f:
        model_json = json.load(f)

    return model_json


def main(args):

    cuda_available = torch.cuda.is_available()
    writer = SummaryWriter(args.tensorboard_logs_dir)

    weight_fn = args.weight_fn
    model_json = load_model_json()

    assert len(model_json['cross_entropy_loss_weights']) == model_json['out_chn'], "CrossEntropyLoss class weights must be same as no. of output channels"

    # 训练集
    trainset = Pavements(args.images, args.masks)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=model_json['batch_size'], shuffle=True, num_workers=4)

    # 验证集（新增，可选）
    valloader = None
    if args.val_images and args.val_masks:
        valset = Pavements(args.val_images, args.val_masks)
        valloader = torch.utils.data.DataLoader(valset, batch_size=model_json['batch_size'], shuffle=False, num_workers=4)
        print(f"Validation set loaded: {len(valset)} samples")

    model = SegNet(in_chn=model_json['in_chn'], out_chn=model_json['out_chn'], BN_momentum=model_json['bn_momentum'])
    optimizer = optim.SGD(model.parameters(), lr=model_json['learning_rate'], momentum=model_json['sgd_momentum'])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(model_json['cross_entropy_loss_weights']))

    if cuda_available:
      model.cuda()
      loss_fn.cuda()

    run_epoch = model_json['epochs']
    epoch = None
    if weight_fn is not None:
        if os.path.isfile(weight_fn):
            print("Loading checkpoint '{}'".format(weight_fn))
            checkpoint = torch.load(weight_fn)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (epoch {})".format(weight_fn, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'. Will create new checkpoint.".format(weight_fn))
    else:
        print("Starting new checkpoint.".format(weight_fn))
        weight_fn = os.path.join(os.getcwd(), "weights/checkpoint_pavements_{}.pth.tar".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print('\n=== Epoch {} ==='.format(i))

        # ===== 训练阶段 =====
        model.train()
        sum_loss = 0.0

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

            writer.add_scalar('Train/Loss', loss.item()/trainloader.batch_size, (i-1)*len(trainloader) + j)
            sum_loss += loss.item()

            print('Train Loss @ batch {}: {:.4f}'.format(j, loss.item() / trainloader.batch_size))

        avg_train_loss = sum_loss / (j * trainloader.batch_size)
        print('>>> Average TRAIN loss: {:.4f}'.format(avg_train_loss))
        writer.add_scalar('Train/Avg_Loss_per_Epoch', avg_train_loss, i)

        # ===== 验证阶段（新增）=====
        if valloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for k, val_data in enumerate(valloader, 1):
                    val_images, val_labels = val_data
                    if cuda_available:
                        val_images = val_images.cuda()
                        val_labels = val_labels.cuda()

                    val_output = model(val_images)
                    v_loss = loss_fn(val_output, val_labels)
                    val_loss += v_loss.item()

                    writer.add_scalar('Val/Loss', v_loss.item()/valloader.batch_size, (i-1)*len(valloader) + k)

            avg_val_loss = val_loss / (k * valloader.batch_size)
            print('>>> Average VAL loss: {:.4f}'.format(avg_val_loss))
            writer.add_scalar('Val/Avg_Loss_per_Epoch', avg_val_loss, i)

    print("\nTraining complete. Saving checkpoint...")
    save_checkpoint({'epoch': run_epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, weight_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 训练集参数（改为可选参数形式）
    parser.add_argument("--images", required=True, type=str, 
                        help="Directory: Training raw images")
    parser.add_argument("--masks", required=True, type=str, 
                        help="Directory: Training annotated masks")

    # 验证集参数（可选）
    parser.add_argument("--val-images", type=str, 
                        help="Directory: Validation raw images (optional)", default=None)
    parser.add_argument("--val-masks", type=str, 
                        help="Directory: Validation annotated masks (optional)", default=None)

    # TensorBoard日志目录
    parser.add_argument("--logs", dest="tensorboard_logs_dir", required=True, type=str,
                        help="Directory: Logs for tensorboard")

    # 其他可选参数
    parser.add_argument("--weight-fn", type=str, help="Path: Trained weights", default=None)

    args = parser.parse_args()

    main(args)