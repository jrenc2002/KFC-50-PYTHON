import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# 设置图像、掩码和检查点的目录路径
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model,  # 要训练的模型
        device,  # 训练过程中使用的设备（如 'cuda' 或 'cpu'）
        epochs: int = 5,  # 训练的轮数，默认为5
        batch_size: int = 1,  # 每个批次中包含的样本数，默认为1
        learning_rate: float = 1e-5,  # 学习率，控制模型权重更新的步长，默认为1e-5
        val_percent: float = 0.1,  # 用于验证的数据集比例（百分比），默认为10%
        save_checkpoint: bool = True,  # 是否在每个epoch结束时保存模型检查点，默认为True
        img_scale: float = 0.5,  # 图像缩放比例，用于调整输入图像的尺寸，默认为0.5
        amp: bool = False,  # 是否使用自动混合精度训练，默认为False
        weight_decay: float = 1e-8,  # 权重衰减（L2正则化）的值，默认为1e-8
        momentum: float = 0.999,  # 优化器的动量参数，默认为0.999
        gradient_clipping: float = 1.0,  # 梯度裁剪的阈值，以避免梯度爆炸，默认为1.0
):

    # 1. 创建数据集
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. 将数据集划分为训练集和验证集
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 初始化日志记录
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''开始训练:
        轮数:          {epochs}
        批大小:        {batch_size}
        学习率:        {learning_rate}
        训练集大小:    {n_train}
        验证集大小:    {n_val}
        保存检查点:    {save_checkpoint}
        设备:          {device.type}
        图像缩放比例:  {img_scale}
        混合精度:      {amp}
    ''')

    # 4. 设置优化器、损失函数、学习率调度器和AMP的损失缩放
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # 目标: 最大化Dice得分
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'轮 {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'网络定义了 {model.n_channels} 个输入通道，' \
                    f'但加载的图像有 {images.shape[1]} 个通道。请检查图像是否正确加载。'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    '训练损失': loss.item(),
                    '步数': global_step,
                    '轮数': epoch
                })
                pbar.set_postfix(**{'损失（批次）': loss.item()})

                # 评估轮
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['权重/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['梯度/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('验证Dice得分: {}'.format(val_score))
                        try:
                            experiment.log({
                                '学习率': optimizer.param_groups[0]['lr'],
                                '验证Dice得分': val_score,
                                '图像': wandb.Image(images[0].cpu()),
                                '掩码': {
                                    '真实': wandb.Image(true_masks[0].float().cpu()),
                                    '预测': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                '步数': global_step,
                                '轮数': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'检查点 {epoch} 已保存！')

def get_args():
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description='训练U-Net模型用于图像和目标掩码')

    # 添加训练轮数参数，短选项为-e，长选项为--epochs，默认值为5
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='训练轮数')

    # 添加批大小参数，短选项为-b，长选项为--batch-size，默认值为1
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='批大小')

    # 添加学习率参数，短选项为-l，长选项为--learning-rate，默认值为1e-5
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='学习率', dest='lr')

    # 添加加载模型参数，短选项为-f，长选项为--load，默认值为False，表示是否从.pth文件加载模型
    parser.add_argument('--load', '-f', type=str, default=False, help='从.pth文件加载模型')

    # 添加图像下采样因子参数，短选项为-s，长选项为--scale，默认值为0.5
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='图像下采样因子')

    # 添加用于验证的数据百分比参数，短选项为-v，长选项为--validation，默认值为10.0，表示0到100之间的百分比
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='用于验证的数据百分比 (0-100)')

    # 添加混合精度参数，长选项为--amp，默认值为False，表示是否使用混合精度训练
    parser.add_argument('--amp', action='store_true', default=False, help='使用混合精度')

    # 添加双线性上采样参数，长选项为--bilinear，默认值为False，表示是否使用双线性上采样
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性上采样')

    # 添加类别数参数，短选项为-c，长选项为--classes，默认值为2
    parser.add_argument('--classes', '-c', type=int, default=2, help='类别数')

    # 解析命令行参数并返回结果
    return parser.parse_args()


if __name__ == '__main__':
    # 获取命令行参数
    args = get_args()

    # 配置日志记录，设置日志级别为INFO，日志格式为'日志级别: 日志消息'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 检查是否有可用的CUDA设备，如果有则使用CUDA，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备 {device}')

    # 创建U-Net模型
    # n_channels=3 表示输入图像是RGB图像
    # n_classes 表示输出的类别数（每个像素的类别）
    # bilinear 表示是否使用双线性上采样
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # 设置模型的内存格式为channels_last，以优化内存访问模式
    model = model.to(memory_format=torch.channels_last)

    # 记录模型的结构信息，包括输入通道数、输出通道数和上采样方式
    logging.info(f'网络结构:\n'
                 f'\t{model.n_channels} 个输入通道\n'
                 f'\t{model.n_classes} 个输出通道（类别）\n'
                 f'\t{"双线性" if model.bilinear else "转置卷积"} 上采样')

    # 如果指定了加载模型的路径，则加载模型的权重
    if args.load:
        # 加载模型状态字典
        state_dict = torch.load(args.load, map_location=device)
        # 删除自定义的'mask_values'键，以便与当前模型的状态字典兼容
        del state_dict['mask_values']
        # 将加载的状态字典应用到模型中
        model.load_state_dict(state_dict)
        logging.info(f'模型已从 {args.load} 加载')

    # 将模型移动到指定设备（CPU或GPU）
    model.to(device=device)

    try:
        # 调用训练函数，开始训练模型
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        # 捕获CUDA内存不足错误
        logging.error('检测到内存不足错误！'
                      '启用检查点以减少内存使用，但这会降低训练速度。'
                      '考虑启用AMP (--amp) 以实现快速和高效的内存训练')
        # 清空CUDA缓存以释放内存
        torch.cuda.empty_cache()
        # 启用模型检查点功能，以减少内存使用
        model.use_checkpointing()
        # 再次调用训练函数，尝试再次训练模型
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
