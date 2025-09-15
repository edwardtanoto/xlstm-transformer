# Hybrid xLSTM-Transformer for Financial Time Series Prediction
[Full Dissertation](https://drive.google.com/drive/u/0/my-drive).

## Overview

`main.py` implements a state-of-the-art hybrid neural network for time series prediction, combining the strengths of xLSTM and Transformer architectures. This model is designed to capture both local and global dependencies in financial time series, such as stock or cryptocurrency prices, and supports multi-ticker (multi-asset) prediction.

## Features
- **Hybrid Model**: Combines xLSTM (advanced RNN) and Transformer (self-attention) branches with a fusion layer.
- **Multi-Ticker Support**: Learns from multiple assets simultaneously.
- **Robust Data Handling**: Includes quantile clipping, feature scaling, and data integrity checks.
- **Modern Training Techniques**: Gradient clipping, early stopping, learning rate scheduling, and mixed-precision (AMP) support.
- **Experiment Tracking**: Integrated with [Weights & Biases (wandb)](https://wandb.ai/).
- **Hardware Acceleration**: Supports CUDA, MPS (Apple Silicon), and CPU.

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd xlstm
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

- The script expects a CSV file with columns: `code`, `open`, `close`, `volume`, `turnover`, `time`, `high`, `low`.
- Example: `filtered_stock_data.csv` or your own stock/crypto dataset.
- Place your data file in the project directory.

## Usage

Run the main script with your data:

```bash
python main.py --data_path <your_data.csv> --epochs 100 --batch_size 64
```

**Common arguments:**
- `--data_path`: Path to your CSV data file.
- `--epochs`: Number of training epochs (default: 100).
- `--batch_size`: Batch size for training (default: 64).
- `--seq_len`: Sequence length for time series windows (default: 30).
- `--device`: `cuda`, `mps`, or `cpu` (auto-detected if not specified).

For all options, run:

```bash
python xlstm-tf.py --help
```

## Outputs
- Training and validation metrics (MSE, MAE, R²) are logged.
- Model checkpoints and prediction plots are saved in the working directory.
- Integration with Weights & Biases (wandb) for experiment tracking.

## Model Impact

### Why Hybrid xLSTM-Transformer?
- **xLSTM**: An advanced RNN architecture that overcomes traditional LSTM limitations, providing better long-term memory and stability ([xLSTM paper](https://arxiv.org/abs/2405.04517)).
- **Transformer**: Excels at modeling global dependencies via self-attention.
- **Fusion Layer**: Combines both representations, leveraging the strengths of each.

### Key Benefits
- **Superior Sequence Modeling**: Captures both short-term and long-term patterns in financial data.
- **Multi-Ticker Generalization**: Learns from multiple assets, improving robustness and generalization.
- **Research-Backed**: Implements the latest advances in recurrent and attention-based architectures.
- **Flexible & Extensible**: Easily adaptable to other time series domains (energy, weather, etc.).

### Real-World Impact
- **Improved Forecasting**: Outperforms traditional LSTM and vanilla Transformer models on financial time series.
- **Robust Training**: Modern best practices for stability and reproducibility.
- **Open Source**: Enables further research and practical applications in finance and beyond.

## References
- [xLSTM: Extended Long Short-Term Memory (arXiv:2405.04517)](https://arxiv.org/abs/2405.04517)
- [xLSTM Python Package](https://github.com/NX-AI/xlstm)
- [Transformers (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
