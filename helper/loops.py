import sys
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from helper.util import AverageMeter, topk_accuracy


def train_one_epoch(epoch, train_loader, model, criterion, optimizer, scaler, opt):
    'train the model in one epoch'
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, batch in enumerate(train_loader):
        if len(batch) == 3:
            acoustic, vibration, target = batch
            acoustic, vibration = acoustic.float(), vibration.float()
        else:
            input, target = batch
            input = input.float()
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            if len(batch) == 3:
                acoustic, vibration = acoustic.cuda(), vibration.cuda()
            else:
                input = input.cuda()
            target = target.cuda()
        # Keep a batch dimension even for a final batch of size 1.
        target = target.reshape(-1)
        for parameters in optimizer.param_groups:
            cur_lr = parameters["lr"]
        optimizer.zero_grad()
        if opt.amp:
            with autocast():
                output = model(acoustic, vibration) if len(batch) == 3 else model(input)
                loss = criterion(output, target)
                # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            output = model(acoustic, vibration) if len(batch) == 3 else model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        acc1 = topk_accuracy(output, target, topk=(1,))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0].item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.4f} ({top1.avg:.4f})'.format(epoch,
                                                               idx,
                                                               len(train_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses,
                                                               top1=top1))
            sys.stdout.flush()

            print('* the train acc of this epoch is @{top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg, cur_lr


@torch.no_grad()
def validate(val_loader, model, criterion, opt, split_name="Test"):
    'evaluate a validation or test split'

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(val_loader):
            if len(batch) == 3:
                acoustic, vibration, target = batch
                acoustic, vibration = acoustic.float(), vibration.float()
            else:
                input, target = batch
                input = input.float()
            if torch.cuda.is_available():
                if len(batch) == 3:
                    acoustic, vibration = acoustic.cuda(), vibration.cuda()
                else:
                    input = input.cuda()
                target = target.cuda()
            # Keep a batch dimension even for a final batch of size 1.
            target = target.reshape(-1)
            output = model(acoustic, vibration) if len(batch) == 3 else model(input)
            loss = criterion(output, target)

            acc1 = topk_accuracy(output, target, topk=(1,))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0].item(), target.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('{2}: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.4f} (top1.avg:.4f)'.format(
                    idx, len(val_loader), split_name, batch_time=batch_time, loss=losses,
                    top1=top1))

        print(f'the {split_name.lower()} acc of this epoch is {top1.avg:.4f}')
    return top1.avg, losses.avg
