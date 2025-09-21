import argparse
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm

# -----------------------------
# Argparse
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Forward-Forward + Barlow Twins"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashionmnist", "cifar10", "cifar100"],
        default="cifar10"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--vanilla_loss", action="store_true", default=True)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=4234)
    parser.add_argument("--save-path", default=None)

    parser.add_argument("--hidden", nargs="+", type=int, default=[500, 500, 500])

    # Hybrid loss weights
    parser.add_argument("--alpha_ff", type=float, default=0.995)
    parser.add_argument("--beta_bt", type=float, default=0.005)
    parser.add_argument("--bt_lambda", type=float, default=0.001)

    # Augmentations
    parser.add_argument("--aug-noise-std", type=float, default=0.075)
    parser.add_argument("--aug-dropout-p", type=float, default=0.125)

    return parser.parse_args()


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class Config:
    mode: str
    dataset: str
    batch_size: int
    epochs: int
    lr: float
    threshold: float
    vanilla_loss: bool
    step_size: int
    optimizer: str
    device: str
    seed: int
    save_path: Optional[str]
    hidden: List[int]
    alpha_ff: float
    beta_bt: float
    bt_lambda: float
    aug_noise_std: float
    aug_dropout_p: float


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float, eps: float = 1e-12) -> torch.Tensor:
    N, D = z1.shape
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + eps)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + eps)

    c = (z1_norm.T @ z2_norm) / N
    on_diag = torch.diagonal(c) - 1
    off_diag = off_diagonal(c)
    return (on_diag.pow(2).sum()) + lambd * (off_diag.pow(2).sum())


def augment_views(x: torch.Tensor, noise_std: float, drop_p: float):
    if noise_std > 0:
        n1 = torch.randn_like(x) * noise_std
        n2 = torch.randn_like(x) * noise_std
    else:
        n1 = torch.zeros_like(x)
        n2 = torch.zeros_like(x)
    if drop_p > 0:
        m1 = (torch.rand_like(x) > drop_p).float()
        m2 = (torch.rand_like(x) > drop_p).float()
    else:
        m1 = torch.ones_like(x)
        m2 = torch.ones_like(x)
    return (x + n1) * m1, (x + n2) * m2


# -----------------------------
# Datasets
# -----------------------------
def load_data(name: str, seed: int):
    torch.manual_seed(seed)
    if name == "mnist":
        trans = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: x.view(-1))
        ])
        ds = MNIST
    elif name == "fashionmnist":
        trans = Compose([
            ToTensor(),
            Normalize((0.2860,), (0.3530,)),  # common stats for Fashion-MNIST
            Lambda(lambda x: x.view(-1))
        ])
        ds = FashionMNIST
    elif name == "cifar10":
        trans = Compose([
            ToTensor(),
            Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2471, 0.2435, 0.2616)),
            Lambda(lambda x: x.view(-1))
        ])
        ds = CIFAR10
    elif name == "cifar100":
        trans = Compose([
            ToTensor(),
            Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2471, 0.2435, 0.2616)),
            Lambda(lambda x: x.view(-1))
        ])
        ds = CIFAR100
    else:
        raise ValueError(name)
    return (
        ds('./data', train=True, download=True, transform=trans),
        ds('./data', train=False, download=True, transform=trans)
    )


def infer_input_dim(dataset: str):
    if dataset in ("mnist", "fashionmnist"):
        return 28 * 28
    elif dataset in ("cifar10", "cifar100"):
        return 32 * 32 * 3
    else:
        raise ValueError(dataset)


# -----------------------------
# FF Layer
# -----------------------------
class Layer(nn.Module):
    def __init__(self, in_dim, out_dim, lr, threshold, vanilla_loss, step_size, optimizer_type, device):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        if optimizer_type.lower() == "sgd":
            self.optimizer = SGD(self.parameters(), lr=lr)
        else:
            self.optimizer = Adam(self.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=0.1)
        self.threshold = threshold
        self.vanilla_loss = vanilla_loss
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.activation(self.linear(x))

    def balanced_loss(self, pos, neg, alpha=4.0):
        return torch.log1p(torch.exp(-alpha * (pos - neg))).mean()

    def soft_plus_loss(self, pos, neg, th):
        terms = torch.cat([-pos + th, neg - th])
        return torch.log1p(torch.exp(terms)).mean()

    def train_layer_hybrid(self, pos_data, neg_data, epochs, batch_size,
                           alpha_ff, beta_bt, bt_lambda, aug_noise_std, aug_dropout_p):
        loader = DataLoader(TensorDataset(pos_data, neg_data), batch_size=batch_size, shuffle=True)
        self.train()
        for _ in range(epochs):
            for x_pos, x_neg in loader:
                x_pos, x_neg = x_pos.to(self.device), x_neg.to(self.device)

                out_pos, out_neg = self(x_pos), self(x_neg)
                g_pos = out_pos.pow(2).mean(dim=1)
                g_neg = out_neg.pow(2).mean(dim=1)
                ff_loss = (
                    self.soft_plus_loss(g_pos, g_neg, self.threshold)
                    if self.vanilla_loss else self.balanced_loss(g_pos, g_neg)
                )

                bt_loss = torch.tensor(0.0, device=self.device)
                if beta_bt > 0:
                    v1, v2 = augment_views(x_pos, noise_std=aug_noise_std, drop_p=aug_dropout_p)
                    z1, z2 = self(v1), self(v2)
                    bt_loss = barlow_twins_loss(z1, z2, bt_lambda)

                loss = alpha_ff * ff_loss + beta_bt * bt_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        with torch.no_grad():
            pos_out = self(pos_data.to(self.device))
            neg_out = self(neg_data.to(self.device))
        return pos_out.detach(), neg_out.detach()


# -----------------------------
# FF Network
# -----------------------------
class Network(nn.Module):
    def __init__(self, dims, lr, threshold, vanilla_loss, step_size, optimizer_type, device, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList([
            Layer(dims[i], dims[i+1], lr, threshold, vanilla_loss, step_size, optimizer_type, device)
            for i in range(len(dims)-1)
        ])
        self.device = device

    def mark(self, data, labels):
        one_hot = torch.zeros(data.size(0), self.num_classes, device=data.device)
        one_hot[torch.arange(data.size(0)), labels] = 1.0
        return torch.cat([data.to(self.device), one_hot], dim=1)

    def train_network_hybrid(self, pos_data, neg_data, epochs_per_layer, batch_size,
                             alpha_ff, beta_bt, bt_lambda, aug_noise_std, aug_dropout_p):
        pos, neg = pos_data, neg_data
        for i, layer in enumerate(self.layers):
            print(f"Training layer {i+1}/{len(self.layers)}")
            pos, neg = layer.train_layer_hybrid(pos, neg, epochs_per_layer, batch_size,
                                                alpha_ff, beta_bt, bt_lambda, aug_noise_std, aug_dropout_p)

    def predict(self, data):
        x = data.to(self.device)
        scores = []
        for label in range(self.num_classes):
            m = self.mark(x, torch.full((x.size(0),), label, dtype=torch.long, device=self.device))
            layer_scores = []
            for layer in self.layers:
                m = layer(m)
                layer_scores.append(m.pow(2).mean(dim=1))
            scores.append(torch.stack(layer_scores).sum(0))
        return torch.stack(scores, dim=1).argmax(dim=1)


# -----------------------------
# Pos/Neg constructors
# -----------------------------
def make_positive(data, labels, num_classes):
    one_hot = torch.zeros(data.size(0), num_classes)
    one_hot[torch.arange(data.size(0)), labels] = 1.0
    return torch.cat([data, one_hot], dim=1)


def make_negative(data, labels, num_classes, seed=None):
    rnd = random.Random(seed)
    one_hot = torch.zeros(data.size(0), num_classes)
    for i, lbl in enumerate(labels):
        choices = list(range(num_classes))
        choices.remove(int(lbl))
        one_hot[i, rnd.choice(choices)] = 1.0
    return torch.cat([data, one_hot], dim=1)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    cfg = Config(**vars(args))
    set_seed(cfg.seed)

    num_classes = 100 if cfg.dataset == "cifar100" else 10
    input_dim = infer_input_dim(cfg.dataset)
    dims = [input_dim + num_classes] + cfg.hidden

    device = torch.device(cfg.device)
    print("Configuration:", cfg)

    train_ds, test_ds = load_data(cfg.dataset, cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))

    pos_train = make_positive(train_data, train_labels, num_classes)
    neg_train = make_negative(train_data, train_labels, num_classes, seed=cfg.seed)

    net = Network(dims, cfg.lr, cfg.threshold, cfg.vanilla_loss,
                  cfg.step_size, cfg.optimizer, cfg.device, num_classes)

    if cfg.mode == "train":
        best_acc = 0.0
        for epoch in tqdm(range(1, cfg.epochs+1), desc="Epochs"):
            net.train_network_hybrid(pos_train.to(device), neg_train.to(device),
                                     epochs_per_layer=1, batch_size=cfg.batch_size,
                                     alpha_ff=cfg.alpha_ff, beta_bt=cfg.beta_bt,
                                     bt_lambda=cfg.bt_lambda,
                                     aug_noise_std=cfg.aug_noise_std,
                                     aug_dropout_p=cfg.aug_dropout_p)

            with torch.no_grad():
                net.eval()
                train_preds = net.predict(train_data.to(device))
                test_preds = net.predict(test_data.to(device))
                train_acc = (train_preds == train_labels.to(device)).float().mean().item()
                test_acc = (test_preds == test_labels.to(device)).float().mean().item()
                best_acc = max(best_acc, test_acc)
            tqdm.write(f"Epoch {epoch}: Train {train_acc:.4f}, Test {test_acc:.4f}, Best {best_acc:.4f}")

        if cfg.save_path:
            torch.save(net.state_dict(), cfg.save_path)
    else:
        if cfg.save_path:
            net.load_state_dict(torch.load(cfg.save_path, map_location=device))
        with torch.no_grad():
            preds = net.predict(test_data.to(device))
            acc = (preds == test_labels.to(device)).float().mean().item()
        print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
