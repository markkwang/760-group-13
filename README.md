# Beyond Backpropagation: Hybrid Forward-Forward + Barlow Twins

This repository contains our implementation of a **forward-only training algorithm** that combines **Forward-Forward (FF)** learning with a **Barlow Twins (BT)** regularizer.  
It is part of our research project *“Beyond Backpropagation”* (Group XX).

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/markkwang/760-group-13.git
cd 760-group-13
```

### 2. Running

MNIST (3 layers × 500 neurons each)
```bash
python main.py \
  --dataset mnist \
  --hidden 500 500 500 \
  --epochs 20 \
  --batch-size 128
```

Fashion-MNIST
```bash
python main.py \
  --dataset mnist \
  --hidden 800 800 800 800 \
  --epochs 20 \
  --batch-size 128
```

CIFAR-10
```bash
python main.py \
  --dataset cifar10 \
  --hidden 500 500 500 \
  --epochs 20 \
  --batch-size 256
```

## ⚙️ Command-line Arguments

| Argument            | Default      | Description                                    |
|---------------------|--------------|------------------------------------------------|
| `--dataset`         | `cifar10`    | Dataset: `mnist`, `cifar10`, `cifar100`        |
| `--hidden`          | `500 500 500`| Hidden layer sizes (M/N notation)              |
| `--epochs`          | `100`        | Number of training epochs                      |
| `--batch-size`      | `256`        | Batch size                                     |
| `--lr`              | `0.0005`     | Learning rate                                  |
| `--optimizer`       | `adam`       | Optimizer: `adam` or `sgd`                     |
| `--threshold`       | `2.0`        | FF threshold                                   |
| `--alpha_ff`        | `0.995`    | Weight for FF loss                             |
| `--beta_bt`         | `0.005`    | Weight for BT loss (keep small!)               |
| `--bt_lambda`       | `0.01`       | Weight for BT off-diagonal penalty             |
| `--aug-noise-std`   | `0.05`       | Gaussian noise std for augmentations           |
| `--aug-dropout-p`   | `0.0`        | Dropout probability for augmentations          |
| `--save-path`       | `None`       | Path to save model state                       |
| `--mode`            | `train`      | Run mode: `train` or `test`                    |

