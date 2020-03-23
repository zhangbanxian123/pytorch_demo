import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate


def train(model, optimizer, loss_fn, dataloader, metrics, args):
    """***训练模型***

    args:
        model: (torch.nn.Module) 定义的网络结构
        optimizer: (torch.optim) 网络参数优化器
        loss_fn: 损失函数
        dataloader: (DataLoader) 用来加载训练数据
        metrics: (dict) 模型指标，例如准确率
    """

    # 设置训练模式
    model.train()

    # 当前迭代和平均损失的摘要
    summ = []
    loss_avg = utils.RunningAverage()

    # 使用tqdm作为进度条
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # 如果GPU可用，则将数据移动到GPU上
            if args.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            # 转换为torch Variable
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # 计算模型输出与损失
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # 请出之前的梯度, 计算所有损失变量的梯度并反向传播
            optimizer.zero_grad()
            loss.backward()

            # 使用计算的梯度更新参数
            optimizer.step()

            # 间隔一段时间评估一下
            if i % args.save_summary_steps == 0:
                # 从torch Variable提取数据, 转移到cpu, 转换格式为numpy类型
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # 计算一个batch的所有指标
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # 更新平均损失
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # 汇总所有指标的平均值
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, args):
    """训练并在每个epoch结束时进行评估

    args:
        model: (torch.nn.Module)定义的网络结构
        train_dataloader: (DataLoader) 用来加载训练数据
        val_dataloader: (DataLoader) 用来加载验证数据
        optimizer: (torch.optim) 网络参数优化器
        loss_fn:  损失函数
        metrics: (dict) 模型指标，例如准确率
    """
    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(args.num_epochs):
        # 计算一个epoch
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))

        # 训练一个epoch
        train(model, optimizer, loss_fn, train_dataloader, metrics, args)

        # 在验证集上评估模型结果
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, args)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # 保存checkpoints
        utils.save_checkpoint({'epoch': epoch + 1,
                                'acc': val_acc,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=args.model_dir)

        # 保存评估最好的模型的指标
        if is_best:
            logging.info("- Found new best accuracy：epoch-{}-{:.3f}".format(epoch, val_acc))
            best_val_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data\SIGNS',
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir', default='checkpoint',
                        help="Directory containing net")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before training")
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
        print('Training on GPU!')
        torch.cuda.manual_seed(230)

    # 创建日志文件
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    # 获取数据
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], args)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # 定义网络结构与优化器
    model = net.Net().cuda() if args.cuda else net.Net()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 定义损失函数与指标
    loss_fn = net.loss_fn
    metrics = net.metrics

    # 训练与评估
    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, args)
