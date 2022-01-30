from __future__ import print_function
import argparse


def get_parser():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--batch_size', type=int, default=4)
    parser_.add_argument('--train_shuffle', type=bool, default=True)
    parser_.add_argument('--val_shuffle', type=bool, default=False)
    parser_.add_argument('--test_shuffle', type=bool, default=False)
    parser_.add_argument('--crop_ratio', type=int, default=1)
    parser_.add_argument('--print', type=int, default=10)
    parser_.add_argument('--lr', type=float, default=5e-3)
    parser_.add_argument('--use_cycle', type=bool, default=True)
    parser_.add_argument('--num_classes', type=int, default=19)
    parser_.add_argument('--weight_decay', type=float, default=5e-4)
    parser_.add_argument('--momentum', type=float, default=0.9)
    parser_.add_argument('--num_epoch', type=int, default=200)
    parser_.add_argument('--curr_epoch', type=int, default=1)
    parser_.add_argument('--best_iou', type=float, default=0.0)
    parser_.add_argument('--start_epoch', type=int, default=0)
    parser_.add_argument('--resume', type=str, default=None)
    parser_.add_argument('--pre_trained', type=str, default=None)
    parser_.add_argument('--use_gpu', type=bool, default=True)
    parser_.add_argument('--dataset', type=str, default='Cityscapes')
    parser_.add_argument('--local_path', type=str, default='F:/Cityscapes')
    parser_.add_argument('--save_root', type=str, default='F:/save')
    return parser_


parser = get_parser()
args = parser.parse_args()
print(args)
