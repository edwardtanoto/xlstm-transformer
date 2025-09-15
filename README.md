# Hybrid xLSTM-Transformer for Financial Time Series Prediction
[Full Dissertation](https://drive.google.com/drive/u/0/my-drive).

## Abstract

Transformer and LSTM models have made great strides in time-series
forecasting but have their respective disadvantages. LSTMs have difficulties
in parallelization while transformers find it hard to handle long sequences.
Hybrid models, which utilize both architectures but still suffer from drawbacks
such as, computational overhead and dependency on datasets have emerged.
In this study, we introduce a new hybrid architecture xLSTM-Transformer,
which combines scalar LSTM (sLSTM) for adaptive gating and matrix LSTM
(mLSTM) for augmentation of memory capacity. The model enhances both
feature extraction and attention mechanisms, at the same time achieves
computational efficiency. Evaluations on climate, finance, and environment
datasets show better accuracy than standalone models, providing a powerful
approach for complex time-series forecasting.

## Overview

`main.py` implements a state-of-the-art hybrid neural network for time series prediction, combining the strengths of xLSTM and Transformer architectures. This model is designed to capture both local and global dependencies in financial time series, such as stock or cryptocurrency prices, and supports multi-ticker (multi-asset) prediction.

## Architecture
<img width="277" height="793" alt="Screenshot 2025-09-15 at 11 37 06" src="https://github.com/user-attachments/assets/c2866e4f-c649-4057-ae4a-5f62d6aa3a6e" />


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

### Benchmark

<img width="442" height="302" alt="Screenshot 2025-09-15 at 11 38 27" src="https://github.com/user-attachments/assets/adc2682b-4f26-4a21-8619-840eb2f92946" />


The empirical results demonstrate the superior performance of the proposed
xLSTM-Transformer hybrid architecture on the stock price prediction task.
The xLSTM-Transformer achieved the lowest Mean Absolute Error (MAE) of
3.304, indicating the highest average accuracy in predicting stock prices. This32
represents an improvement over the LSTM baseline (MAE of 3.806) and a
significant improvement over the Transformer baseline (MAE of 25.346).
Furthermore, the xLSTM-Transformer's substantial reductions in Mean
Squared Error (MSE) to 57.864 and Root Mean Squared Error (RMSE) to
7.607, compared to LSTM (MSE of 77.088, RMSE of 8.780) and Transformer
(MSE of 2621.566, RMSE of 51.201), highlight its enhanced ability to avoid
large prediction errors, a critical advantage in financial forecasting. The RMSE
suggests a typical prediction error of approximately $7.61 for the hybrid model,
lower than the $8.78 of LSTM. The xLSTM-Transformer also exhibits the
lowest scaled test loss (0.0005075), showing 13.19% improvement over LSTM and 86.97% over Transformers. While it has a higher runtime and a lower step than
LSTM, it achieves this with a lower number of epochs, indicating a more
efficient convergence. LSTM's reasonable performance confirms the value of
recurrence in this time-series prediction task. In contrast, the Transformer's
struggle highlights potential challenges for pure self-attention models in
Transformer, indicating that pure self-attention architectures such as the
Transformer may not be capable of handling noisy financial data with similar
performance to specialized models or hybrid architectures that allow for
temporal processing like the xLSTM.

Overall, a strong hypothesis is made that
the synergy of xLSTM and Transformer can be more effective in stock market
prediction than traditional paradigms alone. 


### Real-World Impact
- **Improved Forecasting**: Outperforms traditional LSTM and vanilla Transformer models on financial time series.
- **Robust Training**: Modern best practices for stability and reproducibility.
- **Open Source**: Enables further research and practical applications in finance and beyond.

## References
- [xLSTM: Extended Long Short-Term Memory (arXiv:2405.04517)](https://arxiv.org/abs/2405.04517)
- [xLSTM Python Package](https://github.com/NX-AI/xlstm)
- [Transformers (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
