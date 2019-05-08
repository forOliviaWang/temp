"""main.py"""
import argparse    #argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。

import numpy as np
import torch

from solver import Solver
from utils.utils import str2bool

def main(args):

    torch.backends.cudnn.enabled = True  #使用CuDNN运行代码，设置为False时将不再使用CuDNN运行
    torch.backends.cudnn.benchmark = True   #在程序开始时加这条语句可以提升一点训练速度。大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。

    seed = args.seed
    torch.manual_seed(seed)          #为CPU设置种子用于生成随机数，以使随机数固定
    torch.cuda.manual_seed(seed)     #为当前GPU设置随机种子
    np.random.seed(seed)       #利用随机数种子，使后面的随机数按一定顺序生成

    np.set_printoptions(precision=4)  #强制numpy打印全部数据，浮点输出精度为4
    torch.set_printoptions(precision=4)  #设定输出tensor查看时的精度为4

    print()
    print('[ARGUMENTS]')
    print(args)
    print()

    net = Solver(args)

    if args.mode == 'train':
        net.train()
    elif args.mode == 'test':
        net.test()
    elif args.mode == 'generate':
        net.generate(num_sample=args.batch_size,
                     target=args.target,
                     epsilon=args.epsilon,
                     alpha=args.alpha,
                     iteration=args.iteration)
    elif args.mode == 'universal':
        net.universal(args)
    else: return

    print('[*] Finished')


if __name__ == "__main__":
    #处理命令行参数
    parser = argparse.ArgumentParser(description='toynet template')    #创建一个解析对象，向该对象中添加你要关注的命令行参数和选项
    parser.add_argument('--epoch', type=int, default=20, help='epoch size')    #每一个add_argument方法对应一个你要关注的参数或选项，help用来描述这个选项的作用
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--y_dim', type=int, default=10, help='the number of classes')
    parser.add_argument('--target', type=int, default=-1, help='target class for targeted generation')
    parser.add_argument('--eps', type=float, default=1e-9, help='epsilon')
    parser.add_argument('--env_name', type=str, default='main', help='experiment name')
    parser.add_argument('--dataset', type=str, default='FMNIST', help='dataset type')
    parser.add_argument('--dset_dir', type=str, default='datasets', help='dataset directory path')
    parser.add_argument('--summary_dir', type=str, default='summary', help='summary directory path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory path')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
    parser.add_argument('--load_ckpt', type=str, default='', help='')
    parser.add_argument('--cuda', type=str2bool, default=True, help='enable cuda')
    parser.add_argument('--silent', type=str2bool, default=False, help='')
    parser.add_argument('--mode', type=str, default='train', help='train / test / generate / universal')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--iteration', type=int, default=1, help='the number of iteration for FGSM')
    parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon for FGSM and i-FGSM')
    parser.add_argument('--alpha', type=float, default=2/255, help='alpha for i-FGSM')
    parser.add_argument('--tensorboard', type=str2bool, default=False, help='enable tensorboard')
    parser.add_argument('--visdom', type=str2bool, default=False, help='enable visdom')
    parser.add_argument('--visdom_port', type=str, default=55558, help='visdom port')
    args = parser.parse_args()   #最后使用parse_args() 方法进行解析

    main(args)
