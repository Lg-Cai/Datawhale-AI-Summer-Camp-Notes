# Datawhale AI 夏令营 Task3 笔记 - 数据增强

## 一、数据增强基础

1. 目的：通过人工方式增加数据的多样性，从而提高模型的泛化能力，使其能够在未见过的数据上表现得更好。
2. 对于图像而言，数据增强包括修改纹理、修改颜色、边缘增强、边缘提取、翻转旋转等。
3. 数据增强的变换操作与目标任务的实际场景应该相适应，否则可能引入无关的噪声。过度的数据增强，可能导致图像失真，使得模型难以学习到有效的特征。

## 二、常见数据增强方法

Pytorch框架下使用的数据增强方法主要位于 `torchvision.transforms` 和 `torch vision.transforms.v2` 中。

1. 几何变换：
    1. 调整大小：`Resize`
    2. 随机裁剪：`RandomCrop` `RandomResizedCrop`
    3. 中心裁剪：`CenterCrop`
    4. 五裁剪和十裁剪：`FiveCrop` `TenCrop`
    5. 翻转：`RandomHorizontalFlip` `RandomVerticalFlip`
    6. 旋转：`RandomRotation`
    7. 仿射变换：`RandomAffine`
    8. 透视变换：`RandomPerspective`
2. 颜色变换
    1. 颜色抖动：`ColorJitter`
    2. 灰度化：`Grayscale` `RandomGrayscale`
    3. 高斯模糊：`GaussianBlur`
    4. 颜色反转：`RandomInvert`
    5. 减少图像中每个颜色通道的位数：`RandomPosterize`
    6. 反转图像中高于阈值的像素值：`RandomSolarize`
3. 自动增强（可控性差，谨慎使用）：
    1. `AutoAugment` ：可以根据数据集自动学习数据增强策略
    2. `RandAugment` ：可以随机应用一系列数据增强操作
    3. `TrivialAugmentWide` ：提供于数据集无关的数据增强
    4. `AugMix` ：通过混合多个增强操作进行数据增强

pytorch框架下使用常见数据增强方法：

```python
dataset.transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.RandomErasing(p=p, scale=scale, ratio=ratio),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

1. 数据增强：`transforms.Resize()` 等
2. 转化为张量：`transforms.ToTensor()`
3. `transforms.RandomErasing()` 需在转化为张量后进行。
4. 归一化：`transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` ，这里的均值和标准差是根据ImangNet数据集计算出来的，用于将图像像素值标准化，有助于模型的训练稳定性和收敛速度。

## 三、进阶数据增强方法

1. MixUp：通过将两个不同的图像及其标签按照一定的比例混合，从而创建一个新的训练样本。

    1. 混合比例：`alpha` ，一个0到1之间的值。
    2. 实现过程：
        1. 从训练集中随机选择两个图像和它们的标签。
        2. 将图像和标签分别按照 `alpha` 的比例混合，得到一个新的图像和标签。
        3. 实际应用中，更多地使用混合损失的方式，而不是混合标签的方式
    3. 优点：增加数据多样性，减少过拟合，提高泛化能力。

    ```python
    def mixup(inputs, targets, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(len(inputs))
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        targets_a, targets_b = targets, targets[index]
        return mixed_inputs, targets_a, targets_b, lam
        
    def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
        return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)
        
    def train():
    	# prepare the data, model, etc.
    	inputs, targets = inputs.to(device), targets.to(device)
    	inputs, targets_a, targets_b, lam = mixup(inputs, targets)
    	pred = model(inputs)
        loss = mixup_criterion(criterion, pred, targets_a, targets_b, lam)
        # continue training step
    ```

2. Cutmix：通过将一个图像的一部分剪切并粘贴到另一个图像上来创建新的训练样本。

    1. 实现过程：
        1. 从训练集中随机选择两个图像和它们的标签。
        2. 随机选择一个剪切区域的大小和位置。
        3. 将第一个图像的剪切区域粘贴到第二个图像上，得到一个新的图像。
        4. 根据剪切区域的大小，计算两个图像的标签的加权平均值，作为新的标签。

    ```python
    def cutmix(inputs, targets, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(len(inputs))
        _, h, w = inputs.size()
        bbox_area = (w * h) * lam
        cut_rat = np.sqrt(bbox_area / (w * h))
        cut_w = np.round(cut_rat * w).astype(int)
        cut_h = np.round(cut_rat * h).astype(int)
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        inputs[:, y1:y2, x1:x2] = inputs[index, y1:y2, x1:x2]
        targets_a, targets_b = targets, targets[index]
        return inputs, targets_a, targets_b, lam
    # function mixup_criterion is as same as the MixUp augmentation
    ```

    

