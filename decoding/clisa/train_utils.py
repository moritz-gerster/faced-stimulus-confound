"""CLISA training utilities (from Chen et al. 2023, unchanged)."""
import torch
import numpy as np
import os
from .io_utils import save_checkpoint


def train_earlyStopping(args, train_loader, val_loader, model, criterion, optimizer, scheduler, saveModel):
    bad_count = 0
    best_acc, best_loss = -1, 1000

    val_acc = 0
    val_loss = 0
    model.eval()
    for counter, (x_batch, y_batch) in enumerate(val_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        top1 = accuracy(logits, y_batch, topk=(1,))
        val_acc += top1[0]
        val_loss += loss.data.cpu().numpy()

    val_acc /= (counter + 1)
    val_loss /= (counter + 1)

    train_loss_history, val_loss_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)
    train_acc_history, val_acc_history = np.zeros(args.epochs_finetune), np.zeros(args.epochs_finetune)

    model_epochs, optimizer_epochs = {}, {}
    for epoch in range(args.epochs_finetune):
        train_acc = 0
        train_loss = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            train_acc += top1[0]
            train_loss += loss.data.cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_acc /= (counter + 1)
        train_loss /= (counter + 1)

        val_acc = 0
        val_loss = 0
        confusionMat = torch.zeros((9,9))
        model.eval()
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            confusionMat = confusionMat + get_confusionMat(logits, y_batch, 9)
            val_acc += top1[0]
            val_loss += loss.data.cpu().numpy()

        val_acc /= (counter + 1)
        val_loss /= (counter + 1)
        train_loss_history[epoch] = train_loss
        val_loss_history[epoch] = val_loss
        train_acc_history[epoch] = train_acc
        val_acc_history[epoch] = val_acc

        model_epochs[epoch] = model
        optimizer_epochs[epoch] = optimizer

        if val_acc > best_acc:
            bad_count = 0
            best_loss = val_loss
            best_acc = val_acc
            best_epoch = epoch
            best_confusion = confusionMat
        else:
            bad_count += 1

        if bad_count > args.max_tol:
            break

    if saveModel:
        checkpoint_name = 'finetune_checkpoint_{:04d}.pth.tar'.format(best_epoch)
        save_checkpoint({
            'epoch': best_epoch,
            'state_dict': model_epochs[best_epoch].state_dict(),
            'optimizer': optimizer_epochs[best_epoch].state_dict(),
        }, is_best=False, filename=os.path.join(args.save_dir_ft, checkpoint_name))

        checkpoint_name = 'finetune_checkpoint_{:04d}.pth.tar'.format(epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model_epochs[epoch].state_dict(),
            'optimizer': optimizer_epochs[epoch].state_dict(),
        }, is_best=False, filename=os.path.join(args.save_dir_ft, checkpoint_name))

    return best_epoch, train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_confusion


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_confusionMat(output, target, n_class):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        confusionMat = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                confusionMat[i, j] = torch.sum((pred==i) & (target==j))
        return confusionMat
