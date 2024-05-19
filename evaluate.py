import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    # 将模型设置为评估模式
    net.eval()
    # 获取验证批次数量
    num_val_batches = len(dataloader)
    # 初始化Dice得分
    dice_score = 0

    # 迭代验证集
    # 使用自动混合精度（如果启用），根据设备类型设置计算环境
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # 将图像和标签移动到正确的设备和类型
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 预测掩码
            mask_pred = net(image)

            if net.n_classes == 1:
                # 对于二分类任务，检查真实掩码的索引范围
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # 使用Sigmoid函数并将预测结果二值化
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # 计算Dice得分
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # 对于多分类任务，检查真实掩码的索引范围
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # 将真实掩码转换为one-hot格式
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                # 将预测掩码转换为one-hot格式
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # 计算Dice得分，忽略背景类
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    # 将模型恢复到训练模式
    net.train()
    # 返回平均Dice得分
    return dice_score / max(num_val_batches, 1)
