import json
import logging
import os
import shutil

import torch

class RunningAverage():
    """一个简单的类，用来计算平均值
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """训练日志打印相关
    将训练日志打印下来有助于我们分析模型
    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) 日志路径
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 将日志输出到磁盘文件中
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # 将日志输出到 console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)



def save_checkpoint(state, is_best, checkpoint):
    """将最佳和最后一个模型保存在checkpoint文件夹下

    Args:
        state: (dict) 包含模型、epoch等信息的字典
        is_best: (bool) 是否为最好的模型
        checkpoint: (string) 保存路径
    """
    filepath = os.path.join(checkpoint, 'last.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        pass
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best-{}-{:.3f}.pth'.format(state['epoch'], state['acc'])))


def load_checkpoint(checkpoint, model, optimizer=None):
    """加载checkpoint与其他信息

    Args:
        checkpoint: (string) checkpoint名字
        model: (torch.nn.Module) 需要加载参数的网络
        optimizer: (torch.optim) checkpoint中的优化器（可选）
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint