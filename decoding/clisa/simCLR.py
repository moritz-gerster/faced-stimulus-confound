"""CLISA SimCLR contrastive learning (from Chen et al. 2023, unchanged)."""
import torch
import torch.nn.functional as F
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from .io_utils import save_config_file, save_checkpoint
from .train_utils import accuracy


class SimCLR(object):

    def __init__(self, args, model, optimizer, scheduler, log_dir, stratified):
        self.args = args
        self.model = model.to(args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.stratified = stratified
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def info_nce_loss(self, features, stratified):
        bs = int(features.shape[0] // 2)
        labels = torch.cat([torch.arange(bs) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        if stratified == 'stratified':
            features_str = features.clone()
            features_str[:bs, :] = (features[:bs, :] - features[:bs, :].mean(
                dim=0)) / (features[:bs, :].std(dim=0) + 1e-3)
            features_str[bs:, :] = (features[bs:, :] - features[bs:, :].mean(
                dim=0)) / (features[bs:, :].std(dim=0) + 1e-3)
            features = F.normalize(features_str, dim=1)
        elif stratified == 'bn':
            features_str = features.clone()
            features_str = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-3)
            features = F.normalize(features_str, dim=1)
        elif stratified == 'no':
            features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([negatives, positives], dim=1)
        labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
        labels = labels.to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, val_loader, n_times):
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs_pretrain} epochs.")

        bad_count = 0
        best_acc, best_loss = -1, 1000
        model_epochs, optimizer_epochs = {}, {}
        train_top1_history, val_top1_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
        train_top5_history, val_top5_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
        train_loss_history, val_loss_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
        for epoch_counter in range(self.args.epochs_pretrain):
            train_loss = 0
            train_acc = 0
            train_acc5 = 0
            self.model.train()
            for count, (data, labels) in enumerate(train_loader):
                data = data.to(self.args.device)
                features = self.model(data)
                logits, labels = self.info_nce_loss(features, self.stratified)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                top1, top5 = accuracy(logits, labels, topk=(1,5))
                n_iter += 1
                train_loss = train_loss + loss.data.cpu().numpy()
                train_acc = train_acc + top1[0].cpu().numpy()
                train_acc5 = train_acc5 + top5[0].cpu().numpy()

            train_loss = train_loss / (count + 1)
            train_acc = train_acc / (count + 1)
            train_acc5 = train_acc5 / (count + 1)

            val_loss = 0
            val_acc = 0
            val_acc5 = 0
            self.model.eval()
            for count, (data, labels) in enumerate(val_loader):
                data = data.to(self.args.device)
                features = self.model(data)
                logits, labels = self.info_nce_loss(features, self.stratified)
                loss = self.criterion(logits, labels)
                top1, top5 = accuracy(logits, labels, topk=(1,5))
                val_loss = val_loss + loss.data.cpu().numpy()
                val_acc = val_acc + top1[0].cpu().numpy()
                val_acc5 = val_acc5 + top5[0].cpu().numpy()

            val_loss = val_loss / (count + 1)
            val_acc = val_acc / (count + 1)
            val_acc5 = val_acc5 / (count + 1)

            train_top1_history[epoch_counter] = train_acc
            val_top1_history[epoch_counter] = val_acc
            train_top5_history[epoch_counter] = train_acc5
            val_top5_history[epoch_counter] = val_acc5
            train_loss_history[epoch_counter] = train_loss
            val_loss_history[epoch_counter] = val_loss

            model_epochs[epoch_counter] = self.model
            optimizer_epochs[epoch_counter] = self.optimizer

            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tTrain loss: {train_loss}\tTrain top1 accuracy: {train_acc}")
            logging.debug(f"\tVal loss: {val_loss}\tVal top1 accuracy: {val_acc}")

            if val_acc > best_acc:
                bad_count = 0
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch_counter
            else:
                bad_count += 1

            if bad_count > self.args.max_tol_pretrain:
                break

        self.best_model = model_epochs[best_epoch]
        self.best_optimizer = optimizer_epochs[best_epoch]

        logging.info("Training has finished.")
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(best_epoch)
        save_checkpoint({
            'epoch': best_epoch,
            'state_dict': self.best_model.state_dict(),
            'optimizer': self.best_optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
        save_checkpoint({
            'epoch': epoch_counter,
            'state_dict': model_epochs[epoch_counter].state_dict(),
            'optimizer': optimizer_epochs[epoch_counter].state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        return self.best_model, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history
