from torchvision import transforms
from base import BaseDataLoader
  # 假设你有一个自定义数据集类 MyDataset
import os
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    自定义数据集类，用于加载图像分类数据
    假设数据目录结构为：
    data_dir/
        train/
            class1/
                img1.jpg
                img2.jpg
                ...
            class2/
                img3.jpg
                ...
        val/ 或 test/
            class1/
                ...
            class2/
                ...
    """

    def __init__(self, data_dir, train=True, transform=None):
        """
        参数:
        - data_dir: 数据集根目录
        - train: 是否加载训练集（True加载train目录，False加载val/test目录）
        - transform: 数据预处理变换
        """
        self.transform = transform
        self.data_path = os.path.join(data_dir, 'train' if train else 'val')
        if not os.path.exists(self.data_path):
            self.data_path = os.path.join(data_dir, 'test')  # 后备检查test目录

        # 获取类别列表并创建映射字典
        self.classes = sorted(os.listdir(self.data_path))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 收集所有样本路径和标签
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_path, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 支持常见图片格式
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单个样本"""
        img_path, label = self.samples[idx]

        # 加载图像并转换为RGB格式
        image = Image.open(img_path).convert('RGB')

        # 应用数据增强/预处理
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """获取有序的类别名称列表"""
        return self.classes

class MyDataLoader(BaseDataLoader):
    """
    Custom data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # 定义图像预处理流程
        trsfm = transforms.Compose([
            transforms.Resize((300, 300)),  # 缩放为 300x300
            transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),  # 随机裁剪并缩放到 300x300
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(15),  # 随机旋转 ±15 度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])

        self.data_dir = data_dir
        self.dataset = MyDataset(self.data_dir, train=training, transform=trsfm)  # 使用自定义数据集类
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)