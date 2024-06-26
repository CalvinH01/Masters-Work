import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import struct
import fine_tune
import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def generate_random_numbers():
    #bit_to_change = random.randint(0, 31)
    #first_random_number = random.randint(0, 2)
    #second_random_number = random.randint(0, 2)
    #third_random_number = random.randint(0, 2)
    bit_to_change = 2
    first_random_number = 1
    second_random_number = 1
    third_random_number = 1
    return [bit_to_change, first_random_number, second_random_number, third_random_number]

def keep_watermark(model, bit_to_change, first_random_number, second_random_number, third_random_number, shape2values, shape4values):
    shape2count = 0
    shape4count = 0
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 4:
                num_channels = param.data.size(0)
                for i in range(0, num_channels, 1):
                    int_var = struct.unpack('!I', struct.pack('!f', param.data[ i ][first_random_number][second_random_number, third_random_number]))[0]
                    binary_representation = bin(int_var)[2:].zfill(32)
                    bit_list  =[int(bit) for bit in binary_representation]
                    bit_list[bit_to_change] = shape4values[shape4count]
                    new_int_var = int(''.join(map(str, bit_list)), 2)
                    modified_float = struct.unpack('!f', struct.pack('!I', new_int_var))[0]
                    param.data[ i ][first_random_number][second_random_number, third_random_number]= torch.tensor(modified_float)
                    shape4count += 1
            if len(param.shape) == 2:
                num_channels = param.data.size(0)
                for i in range(0, num_channels, 1):
                    int_var = struct.unpack('!I', struct.pack('!f', param.data[ i ][first_random_number]))[0]
                    binary_representation = bin(int_var)[2:].zfill(32)
                    bit_list  =[int(bit) for bit in binary_representation]
                    bit_list[bit_to_change] = shape2values[shape2count]
                    new_int_var = int(''.join(map(str, bit_list)), 2)
                    modified_float = struct.unpack('!f', struct.pack('!I', new_int_var))[0]
                    param.data[ i ][first_random_number]= torch.tensor(modified_float)
                    shape2count += 1

def insert_watermark(model, bit_to_change, first_random_number, second_random_number, third_random_number, shape2values, shape4values):
    byte_index = bit_to_change // 8
    bit_index = 7- (bit_to_change % 8)
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 4:
                num_channels = param.data.size(0)
                for i in range(0, num_channels, 1):
                    binary_representation = struct.pack('!f', param.data[ i ][first_random_number][second_random_number, third_random_number])
                    changed_binary_representation = bytearray(binary_representation)
                    changed_binary_representation[byte_index] ^= (1 << bit_index)
                    modified_binary_string = ''.join(format(byte, '08b') for byte in changed_binary_representation)
                    modified_number_float32 = struct.unpack('!f', bytes(changed_binary_representation))[0]
                    shape4values.append(modified_binary_string[bit_to_change])
                    param.data[ i ][first_random_number][second_random_number, third_random_number] = torch.tensor(modified_number_float32)
            if len(param.shape) == 2:
                num_channels = param.data.size(0)
                for i in range(0, num_channels, 1):
                    binary_representation = struct.pack('!f', param.data[ i ][first_random_number])
                    changed_binary_representation = bytearray(binary_representation)
                    changed_binary_representation[byte_index] ^= (1 << bit_index)
                    modified_binary_string = ''.join(format(byte, '08b') for byte in changed_binary_representation)
                    shape2values.append(modified_binary_string[bit_to_change])
                    modified_number_float32 = struct.unpack('!f', bytes(changed_binary_representation))[0]
                    param.data[ i ][first_random_number]= torch.tensor(modified_number_float32)

    return shape2values, shape4values

def compare_models(model, test_model, bit_to_change, first_random_number, second_random_number, third_random_number):
    data_model1 = []
    data_model2 = []
    changed_bits = 0
    unchanged_bits = 0

    for (param_original, param_fine_tuned) in zip(model.parameters(), test_model.parameters()):
        if param_original.requires_grad:
            if len(param_original.shape) == 4:
                num_channels = param_original.data.size(0)
                for i in range(0, num_channels, 1):
                    # Extracting the bit from the original model's parameter
                    original_int = struct.unpack('!I', struct.pack('!f', param_original.data[i][first_random_number][second_random_number, third_random_number]))[0]
                    original_bit = bin(original_int)[2:].zfill(32)[bit_to_change]
                    data_model1.append(int(original_bit))

                    # Extracting the bit from the fine-tuned model's parameter
                    fine_tuned_int = struct.unpack('!I', struct.pack('!f', param_fine_tuned.data[i][first_random_number][second_random_number, third_random_number]))[0]
                    fine_tuned_bit = bin(fine_tuned_int)[2:].zfill(32)[bit_to_change]
                    data_model2.append(int(fine_tuned_bit))

                    # Counting changed and unchanged bits
                    if original_bit == fine_tuned_bit:
                        unchanged_bits += 1
                    else:
                        changed_bits += 1

            elif len(param_original.shape) == 2:
                num_channels = param_original.data.size(0)
                for i in range(0, num_channels, 1):
                    # Similar extraction and counting for 2D parameters
                    original_int = struct.unpack('!I', struct.pack('!f', param_original.data[i][first_random_number]))[0]
                    original_bit = bin(original_int)[2:].zfill(32)[bit_to_change]
                    data_model1.append(int(original_bit))

                    fine_tuned_int = struct.unpack('!I', struct.pack('!f', param_fine_tuned.data[i][first_random_number]))[0]
                    fine_tuned_bit = bin(fine_tuned_int)[2:].zfill(32)[bit_to_change]
                    data_model2.append(int(fine_tuned_bit))

                    if original_bit == fine_tuned_bit:
                        unchanged_bits += 1
                    else:
                        changed_bits += 1

    # Calculate the Pearson correlation coefficient
    if np.std(data_model1) == 0 or np.std(data_model2) == 0:
        coeff = None
        print("Data is constant; skipping Pearson correlation calculation.")
    else:
        coeff, _ = pearsonr(data_model1, data_model2)

    print(f"Number of changed bits: {changed_bits}")
    print(f"Number of unchanged bits: {unchanged_bits}")
    print(f"Pearson Correlation Coefficient: {coeff}")

    return coeff


model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0

train_accuracies = []
val_accuracies = []

def main():
    global args, best_prec1
    args = parser.parse_args()

    shape2values = []
    shape4values = []

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()
    [bit_to_change, first_random_number, second_random_number, third_random_number] = generate_random_numbers()
    print("this is main function. random numbers are; ", bit_to_change, first_random_number, second_random_number, third_random_number)
    shape2values, shape4values = insert_watermark(model, bit_to_change, first_random_number, second_random_number, third_random_number, shape2values, shape4values)

    # Assuming you have the values for these variables
    config = {
        "bit_to_change": bit_to_change,
        "first_random_number": first_random_number,
        "second_random_number": second_random_number,
        "third_random_number": third_random_number
    }

    with open('config.json', 'w') as file:
        json.dump(config, file)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
    

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    #log_weight_statistics(model, 0)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

        # Replace weights
        keep_watermark(model, bit_to_change, first_random_number, second_random_number, third_random_number, shape2values, shape4values)

        #log_weight_statistics(model, epoch)

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    state_dict = model.state_dict()
    torch.save(state_dict, 'resnet34_weights.pth')

    fine_tune.main()

    test_model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    weights = torch.load('fine_tuned_model.pth')
    test_model.load_state_dict(weights)
    #compare_models(model, test_model, bit_to_change, first_random_number, second_random_number, third_random_number)


def log_weight_statistics(model, epoch):
    print(f"Epoch: {epoch}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} - Mean: {param.data.mean():.6f}, Std: {param.data.std():.6f}, Max: {param.data.max():.6f}, Min: {param.data.min():.6f}")

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    train_accuracies.append(top1.avg)

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    val_accuracies.append(top1.avg)

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
