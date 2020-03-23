import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader



def evaluate(model, loss_fn, dataloader, metrics, args):
    """验证模型.

    Args:
        model: (torch.nn.Module) 网络结构
        loss_fn: 损失函数
        dataloader: (DataLoader) 加载验证数据
        metrics: (dict) 指标
    """

    # 将模型设置为验证模式
    model.eval()

    # 当前验证摘要
    summ = []

    # 计算指标
    for data_batch, labels_batch in dataloader:

        # 若GPU可用，则将数据移到PU
        if args.cuda:
            data_batch, labels_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True)
        # 获取batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # 计算验证输出
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # 从torch变量中提取数据，移至cpu，转换为numpy数组
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # 计算指标
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # 计算平均指标
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        验证模型
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data\\64x64_SIGNS',
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir', default='experiments\\base_model',
                        help="Directory containing net")
    parser.add_argument('--restore_file', default='best-5-0.882.pth',
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    parser.add_argument('--cuda', default=0, type=float, help="0:cuda true, 1:cuda false")
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 4')
    parser.add_argument('--save_summary_steps', default=100, help="save summary frequency")

    args = parser.parse_args()
    # 判断GPU是否可用
    args.cuda = torch.cuda.is_available()

    # 设置随机种子来重构实验
    torch.manual_seed(230)
    if args.cuda:
        torch.cuda.manual_seed(230)

    # 创建日志文件
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")

    # 获取数据
    dataloaders = data_loader.fetch_dataloader(['test'], args)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # 定义网络结构
    model = net.Net().cuda() if args.cuda else net.Net()

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # 加载训练模型
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file), model)

    # 开始验证
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, args)
