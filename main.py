import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from torchvision import transforms
from skopt import gp_minimize
from skopt.space import Real, Integer

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

class ImprovedConvClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels):
        super(ImprovedConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(32 * (seq_len // 2) * (num_channels // 2), num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = x.view(-1, 32 * (x.size(2) // 2) * (x.size(3) // 2))
        x = self.fc1(x)
        return x

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    # デフォルト値の設定
    seq_len = getattr(args, "seq_len", 128)
    num_channels = getattr(args, "num_channels", 64)
    
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((seq_len, num_channels))
    ])

    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args)

    model = ImprovedConvClassifier(train_set.num_classes, seq_len, num_channels).to(args.device)

    def objective(params):
        lr, batch_size = params
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        for epoch in range(args.epochs):
            model.train()
            for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
                X, y = X.to(args.device), y.to(args.device)
                y_pred = model(X)
                loss = F.cross_entropy(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        val_acc = []
        for X, y, subject_idxs in val_loader:
            X, y = X.to(args.device), y.to(args.device)
            with torch.no_grad():
                y_pred = model(X)
            val_acc.append(accuracy(y_pred, y).item())

        return -np.mean(val_acc)

    search_space = [Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
                    Integer(16, 128, name='batch_size')]

    res = gp_minimize(objective, search_space, n_calls=30, random_state=0)
    best_lr, best_batch_size = res.x

    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    early_stopping_patience = 10
    best_val_acc = 0
    epochs_no_improve = 0

    accuracy = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)
            y_pred = model(X)
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc.append(accuracy(y_pred, y).item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            with torch.no_grad():
                y_pred = model(X)
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        val_acc_mean = np.mean(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {val_acc_mean:.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": val_acc_mean})

        if val_acc_mean > best_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            best_val_acc = val_acc_mean
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Testing"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
