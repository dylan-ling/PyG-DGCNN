#!/usr/bin/env python

"""
Training and evalution using PyG DGCNN_PyG
"""

import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
import sklearn.metrics as metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from pyg_data import ModelNet40PyG
from newpyg_model import DGCNN_PyG
from pyg_util import cal_loss, IOStream

def _init_(exp_name):
    checkpoint_directory = os.path.join('checkpoints', exp_name)
    models_dir = os.path.join(checkpoint_directory, 'models')
    os.makedirs(models_dir, exist_ok = True)
    for fname in ('newpyg_main.py', 'pyg_data.py', 'pyg_util.py', 'newpyg_model.py'):
        src = os.path.abspath(fname)
        dst = os.path.join(checkpoint_directory, f'{fname}.backup')
        shutil.copy(src,dst)


def train(args, io, device):
    train_ds = ModelNet40PyG(partition = 'train', num_points = args.num_points)
    test_ds = ModelNet40PyG(partition = 'test', num_points = args.num_points)

    train_loader = DataLoader(train_ds, num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory = args.cuda)
    test_loader = DataLoader(test_ds, num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False, pin_memory = args.cuda)

    model = DGCNN_PyG(args).to(device)
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Using SGD optimizer")
        optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr)
    criterion = cal_loss

    best_acc = 0

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    io.cprint(" ===Starting Training===")
    for epoch in range(args.epochs):
        optimizer.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        data_time = 0.0
        forward_time = 0.0
        loss_time = 0.0
        backward_time = 0.0
        model.train()
        all_pred = []
        all_true = []
        epoch_start = time.time()
        for data in train_loader:
            t0 = time.time()
            data = data.to(device, non_blocking = True)
            data_time += time.time() - t0

            t1 = time.time()
            optimizer.zero_grad()
            logits = model(data)
            forward_time += time.time() - t1

            t2 = time.time()
            loss = criterion(logits, data.y)
            loss_time += time.time() -t2

            t3 = time.time()
            loss.backward()
            scheduler.step()
            backward_time += time.time() - t3

            train_loss += loss.detach() * data.num_graphs
            preds = logits.max(dim=1)[1]
            all_true.append(data.y.detach())
            all_pred.append(preds.detach())

        epoch_time = time.time() - epoch_start
        true_flat = torch.cat(all_true).cpu().numpy()
        pred_flat = torch.cat(all_pred).cpu().numpy()

        train_loss = train_loss.item() / len(true_flat)
        acc = metrics.accuracy_score(true_flat, pred_flat)
        avg_acc = metrics.balanced_accuracy_score(true_flat, pred_flat)
        io.cprint(
            f"[Epoch {epoch}]   Train loss {train_loss:.4f}"
            f"  Acc: {acc:.4f}   Avg Acc: {avg_acc:.4f}"
            )

        io.cprint(
                f"Epoch Time: {epoch_time:.2f}s "
                f"(data: {data_time:.2f}s, "
                f"forward: {forward_time:.2f}s, "
                f"loss: {loss_time:.2f}s, "
                f"backward: {backward_time:.2f}s)"
        )

        train_losses.append(train_loss)
        train_accs.append(avg_acc)
        ####################
        # EVAL
        ####################
        eval_loss = 0.0
        eval_data_time = 0.0
        eval_forward_time = 0.0
        eval_loss_time = 0.0
        model.eval()
        all_pred = []
        all_true = []
        eval_start = time.time()
        with torch.no_grad():
            for data in test_loader:
                t4 = time.time()
                data = data.to(device, non_blocking = True)
                eval_data_time += time.time() - t4

                t5 = time.time()
                logits = model(data)
                eval_forward_time += time.time() - t5

                t6 = time.time()
                loss = criterion(logits, data.y)
                eval_loss_time += time.time() - t6

                eval_loss += loss.detach() * data.num_graphs
                preds = logits.argmax(dim=1)
                all_true.append(data.y.detach())
                all_pred.append(preds.detach())

        eval_time = time.time() - eval_start
        true_flat = torch.cat(all_true).cpu().numpy()
        pred_flat = torch.cat(all_pred).cpu().numpy()

        eval_loss = eval_loss.item() / len(true_flat)
        acc = metrics.accuracy_score(true_flat, pred_flat)
        avg_acc = metrics.balanced_accuracy_score(true_flat, pred_flat)

        io.cprint(
            f"[Epoch {epoch}]    Test loss: {eval_loss / len(true_flat):.4f}"
            f"   Acc: {acc:.4f}    Avg Acc: {avg_acc:.4f}"
        )
        io.cprint(
        f"Epoch Time: {eval_time:.2f}s "
        f"(data: {eval_data_time:.2f}s, "
        f"forward: {eval_forward_time:.2f}s, "
        f"loss: {eval_loss_time:.2f}s) "
        )

        test_losses.append(eval_loss)
        test_accs.append(avg_acc)

        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"checkpoints/{args.exp_name}/models/model.t7")

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(f"checkpoints/{args.exp_name}/loss_curve.png")

    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(f"checkpoints/{args.exp_name}/accuracy_curve.png")

def test(args, io, device):

    test_ds = ModelNet40PyG(partition = 'test', num_points = args.num_points)
    test_loader = DataLoader(test_ds, num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False, pin_memory = args.cuda)

    #Try to load models
    model = DGCNN_PyG(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device, non_blocking = True)
            logits = model(data)
            preds = logits.argmax(dim=1)
            all_true.append(data.y.detach())
            all_pred.append(preds.detach())
    true_flat = torch.cat(all_true).cpu().numpy()
    pred_flat = torch.cat(all_pred).cpu().numpy()
    acc = metrics.accuracy_score(true_flat, pred_flat)
    avg_acc = metrics.balanced_accuracy_score(true_flat, pred_flat)
    io.cprint(f"Test::    Acc: {acc:.4f}    AvgAcc: {avg_acc:.4f}")

if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp_pyg', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use: dgcnn')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action = 'store_true',
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action = 'store_true',
                        help='disables CUDA even if available')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    _init_(args.exp_name)

    if args.cuda:
        torch.backends.cudnn.benchmark = True

    io = IOStream(f"checkpoints/{args.exp_name}/run.log")
    io.cprint(str(args))

    device = torch.device('cuda' if args.cuda else 'cpu')
    torch.manual_seed(0)
    if args.cuda:
        torch.cuda.manual_seed_all(0)
        io.cprint(f"Using GPU(s): {torch.cuda.device_count()} ")

    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io, device)
    else:
        test(args, io, device)

              
