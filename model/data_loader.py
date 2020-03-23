import os

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# 借鉴于 http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# 和 http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# 定义一个训练图像加载器，用于指定图像的变换
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize到（64*64）
    transforms.RandomHorizontalFlip(),  # 水平随机翻转图像
    transforms.ToTensor()])  # 转换成tensor

# 加载验证数据
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize到（64*64）
    transforms.ToTensor()])  # 转换成tensor



def fetch_dataloader(types, args):
    """
    从data_dir中获取各类别的DataLoader对象
    Args:
        types: (list) 根据需要的数据具有一个或多个“ train”，“ val”，“ test”
    Returns:
        data: (dict) 包含类型中每种类型的DataLoader对象
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(args.data_dir, "{}_signs".format(split))

            # 如果训练数据，则使用train_transformer，否则使用eval_transformer而不进行随机翻转
            if split == 'train':
                dl = DataLoader(ImageFolder(path, train_transformer), batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=args.cuda)
            else:
                dl = DataLoader(ImageFolder(path, eval_transformer), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.cuda)

            dataloaders[split] = dl
            # classes = dl.dataset.classes
            # print('classes:',classes)  #查看类别是否正确
    return dataloaders

# class SIGNSDataset(Dataset):
    # """
    # 数据集的标准PyTorch定义，定义了函数__len__和__getitem__。
    # """
    # def __init__(self, data_dir, transform):
        # """
        # 存储要使用的jpg的文件名。 指定要应用于图像的变换。
        # Args:
            # data_dir: (string) 包含数据集的目录
            # transform: (torchvision.transforms) 要应用的转换方法
        # """
        # self.filenames = os.listdir(data_dir)
        # self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        
        # self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        # # print(os.path.split(self.filenames[2]))
        # self.transform = transform

    # def __len__(self):
        # # 返回数据集的大小
        # return len(self.filenames)

    # def __getitem__(self, idx):
        # """
        # 从数据集中获取索引idx图像和标签。 对图像执行变换。

        # Args:
            # idx: (int) size_of_dataset的索引

        # Returns:
            # image: (Tensor) 转换后的图片
            # label: (int) 对应图片的标签
        # """
        # image = Image.open(self.filenames[idx])  # PIL image
        # image = self.transform(image)
        # return image, self.labels[idx]


# def fetch_dataloader(types, args):
    # """
    # 从data_dir中获取每种类型的DataLoader对象

    # Args:
        # types: (list) 根据需要的数据具有一个或多个“ train”，“ val”，“ test”

    # Returns:
        # data: (dict) 包含类型中每种类型的DataLoader对象
    # """
    # dataloaders = {}

    # for split in ['train', 'val', 'test']:
        # if split in types:
            # path = os.path.join(args.data_dir, "{}_signs".format(split))

            # # 如果训练数据，则使用train_transformer，否则使用eval_transformer而不进行随机翻转
            # if split == 'train':
                # dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=args.batch_size, shuffle=True,
                                        # num_workers=args.num_workers,
                                        # pin_memory=args.cuda)
            # else:
                # dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=args.batch_size, shuffle=False,
                                # num_workers=args.num_workers,
                                # pin_memory=args.cuda)

            # dataloaders[split] = dl

    # return dataloaders
