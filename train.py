import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from dataset import ECGDataset
from vHeat import vHeat1D
from utils import cal_f1s, cal_aucs, split_data

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=16, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()

def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    logging.info(f'Training epoch {epoch}:')
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for data, labels, _ in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()
        output = net(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    logging.info(f'Training Loss: {running_loss:.4f}')


def evaluate(dataloader, net, args, criterion, device):
    logging.info('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for data, labels, _ in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())

    logging.info(f'Validation Loss: {running_loss:.4f}')
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    logging.info(f'F1s: {f1s}')
    logging.info(f'Avg F1: {avg_f1:.4f}')

    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        torch.save(net.state_dict(), args.model_path)
        logging.info(f"New best model saved to {args.model_path} with Avg F1: {avg_f1:.4f}")
    else:
        aucs = cal_aucs(y_trues, y_scores)
        avg_auc = np.mean(aucs)
        logging.info(f'AUCs: {aucs}')
        logging.info(f'Avg AUC: {avg_auc:.4f}')


if __name__ == "__main__":
    args = parse_args()
    print(args)
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

    if not args.model_path:
        args.model_path = f'model/vHeat_base_{database}_{args.leads}_{args.seed}_{args.epochs}.pth'

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'vHeat_base_{database}_{args.leads}_{args.seed}_{args.epochs}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Training started with args: {args}")

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'

    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)

    label_csv = os.path.join(data_dir, 'label.csv')

    train_folds, val_folds, test_folds = split_data(seed=args.seed)

    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    net = vHeat1D().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    from loss import FocalLoss

    criterion = FocalLoss(alpha=0.85, gamma=2.0, reduction="mean").to(device)

    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
            logging.info(f"Model resumed from {args.model_path}")
        for epoch in range(args.epochs):
            train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            evaluate(val_loader, net, args, criterion, device)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Model loaded for testing from {args.model_path}")
        evaluate(test_loader, net, args, criterion, device)
