import datetime
import math
import os
import random
import numpy as np
import torch
from torch.autograd import Variable
from data.Dataset import train_loader, valid_loader
from utils.Utils import RunningScore, cross_entropy2d, poly_lr_scheduler
from model.DFFNet import M2_semantic
from utils.Config import args
import torch.nn.functional as F

# Setup Model
model = M2_semantic(num_classes=args.num_classes)
use_gpu = torch.cuda.is_available()
if args.use_gpu:
    model = model.cuda()
else:
    model = model.cpu()

# Setup Metrics
running_metrics = RunningScore(args.num_classes)

# Setup Optimizer
if hasattr(model.modules(), 'optimizer'):
    optimizer = model.modules().optimizer
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

# Define Class_weight
class_weight = None
if hasattr(model.modules(), 'loss'):
    print('> Using custom loss')
    loss_fn = model.modules().loss
else:
    class_weight = np.array([3.045384, 12.862123, 4.509889, 38.15694, 35.25279, 31.482613,
                             45.792305, 39.694073, 6.0639296, 32.16484, 17.109228, 31.563286,
                             47.333973, 11.610675, 44.60042, 45.23716, 45.283024, 48.14782,
                             41.924667], dtype=float) / 10.0

    class_weight = torch.from_numpy(class_weight).float().cuda()
    loss_fn = cross_entropy2d

# Resume Model
weight_dir = "{}/".format(args.save_root)
if args.resume is not None:
    full_path = "{}{}".format(weight_dir, args.resume)
    if os.path.isfile(full_path):
        print("> Loading model and optimizer from checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(full_path)
        args.start_epoch = checkpoint['epoch']
        args.best_iou = checkpoint['best_iou']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        del checkpoint
        print("> Loaded checkpoint '{}' (epoch {}, iou {}, lr {},weight_decay {})".format(args['resume'],
                                                                                          args['start_epoch'],
                                                                                          args['best_iou'],
                                                                                          optimizer.param_groups[0][
                                                                                              'lr'],
                                                                                          optimizer.param_groups[0][
                                                                                              'weight_decay']))
    else:
        print("> No checkpoint found at '{}'".format(args['resume']))
else:
    if args.pre_trained is not None:
        print("> Loading weights from pre-trained model '{}'".format(args.pre_trained))
        full_path = "{}{}".format(weight_dir, args.pre_trained)

        pre_weight = torch.load(full_path)
        model.load_state_dict(pre_weight)

# Train and Validate Model
print("> Model Training start...")
num_batches = int(math.ceil(len(train_loader.dataset.files[train_loader.dataset.split]) /
                            float(train_loader.batch_size)))
print("number_batches:", num_batches)
lr_period = 20 * num_batches
swa_weights = model.state_dict()

# Mini-Batch Learning
for epoch in np.arange(args.start_epoch, args.num_epoch):
    print("> Training Epoch [%d/%d]:" % (epoch + 1, args.num_epoch))
    prev_time = datetime.datetime.now()
    model.train()

    last_loss = 0.0

    for train_i, (images, labels) in enumerate(train_loader):

        full_iter = (epoch * num_batches) + train_i + 1

        poly_lr_scheduler(optimizer, init_lr=args.lr, iter_num=full_iter,
                          lr_decay_iter=1, max_iter=args.num_epoch * num_batches, power=0.9)

        # if args.use_cycle: one_cycle = OneCycle(int(len(train_loader) * args.num_epoch), args.up_lr,
        # print = args.print, momentum_val=(0.95, 0.8)) lr, mom = one_cycle.calc() update_lr(optimizer,
        # lr) update_mom(optimizer, mom)

        if args.use_gpu:
            images = Variable(images.cuda(), requires_grad=True)
            labels = Variable(labels.cuda(), requires_grad=False)
        else:
            images = Variable(images, requires_grad=True)
            labels = Variable(labels, requires_grad=False)

        optimizer.zero_grad()
        net_out = model(images)

        if random.random() < 0.99:
            train_loss = loss_fn(input_data=net_out, target=labels, weight=class_weight)
        else:
            train_loss = loss_fn(input_data=net_out, target=labels, weight=None)

        last_loss = train_loss.data[0]

        train_loss.backward()
        optimizer.step()

        if (train_i + 1) % 200 == 0:
            loss_log = "Epoch [%d/%d], Iter: %d Loss: %.4f" % (epoch + 1, args.num_epoch,
                                                               train_i + 1, last_loss)

            net_out = F.softmax(net_out)
            pred = net_out.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()

            running_metrics.update(gt, pred)
            score, class_iou = running_metrics.get_scores()

            metric_log = ""
            for k, v in score.items():
                metric_log += " {}: %.4f, ".format(k) % v
            running_metrics.reset()

            logs = loss_log + metric_log
            print(logs)

    #################################################################

    print("> Validation for Epoch [%d/%d]:" % (epoch + 1, args.num_epoch))
    model.eval()

    mval_loss = 0.0
    vali_count = 0
    for i_val, (images_val, labels_val) in enumerate(valid_loader):
        vali_count += 1

        images_val = Variable(images_val.cuda(), volatile=True)
        labels_val = Variable(labels_val.cuda(), volatile=True)

        net_out = model(images_val)

        val_loss = loss_fn(input_data=net_out, target=labels_val, weight=None)

        mval_loss += val_loss.data[0]

        net_out = F.softmax(net_out)
        pred = net_out.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics.update(gt, pred)

    mval_loss /= vali_count

    loss_log = "Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args.num_epoch, mval_loss)
    metric_log = ""
    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        metric_log += " {} %.4f, ".format(k) % v
    running_metrics.reset()

    logs = loss_log + metric_log
    print(logs)

    cur_time = datetime.datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time_str)
    print('save_lr_: {}'.format(optimizer.param_groups[0]['lr']))
    if score['Mean_IoU'] >= args.best_iou:
        args.best_iou = score['Mean_IoU']
    state = {'epoch': epoch + 1,
             "best_iou": args.best_iou,
             'model_state': model.state_dict(),
             'optimizer_state': optimizer.state_dict()}
    print('save_lr_: {}'.format(optimizer.param_groups[0]['lr']))
    torch.save(state, "{}{}_best_model.pkl".format(weight_dir, args.dataset))
