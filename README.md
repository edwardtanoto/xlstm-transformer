# Stock Price Prediction with Advanced Neural Networks

This project implements and compares different neural network architectures for stock price prediction, including a novel hybrid xLSTM-Transformer model. The implementation combines traditional approaches with cutting-edge architectures to analyze and predict stock price movements.

## Models Implemented

1. **Hybrid xLSTM-Transformer**
   - Combines the strengths of xLSTM and Transformer architectures
   - Uses an embedding layer followed by transformer encoding and xLSTM processing
   - Designed for capturing both local and global patterns in time series data

2. **Basic LSTM**
   - Traditional LSTM implementation
   - Serves as a baseline model for comparison
   - Well-suited for sequential data processing

3. **Basic Transformer**
   - Pure transformer-based architecture
   - Utilizes self-attention mechanisms
   - Effective at capturing long-range dependencies

4. **Basic xLSTM**
   - Implementation of the xLSTM architecture
   - Combines modern LSTM variants with advanced processing techniques
   - Designed for enhanced temporal feature extraction

## Key Features

- Multi-ticker support for processing multiple stock symbols
- Custom dataset implementation for handling time series data
- Comprehensive evaluation metrics (MSE and MAE in both normalized and original scales)
- Visualization of predictions vs actual values
- Flexible model configurations and hyperparameter settings

## Technical Architecture

### Data Processing
- Uses `StockDataset` class for creating sequential data samples
- Implements MinMaxScaler for data normalization
- Supports batch processing through PyTorch's DataLoader

### Model Components
- Embedding layers for initial data transformation
- Transformer encoder layers with multi-head attention
- xLSTM blocks with configurable architectures
- Feed-forward networks for final predictions

## Configuration

The models use the following default hyperparameters:
- Sequence Length: 30
- Hidden Dimension: 64
- Number of Transformer Layers: 2
- Number of Attention Heads: 4
- Number of xLSTM Blocks: 3
- Batch Size: 32
- Training Epochs: 30

## Usage

The main script handles:
1. Data loading and preprocessing
2. Model initialization and training
3. Evaluation and performance comparison
4. Visualization of results

## Requirements

- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- xLSTM package

## Model Evaluation

The system evaluates models using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Both normalized and original scale metrics
- Visual comparison through prediction plots

## Output

Each model generates:
- Training loss metrics
- Evaluation metrics in both normalized and original scales
- Prediction plots saved as PNG files
- Comparative performance analysis

## Hardware Acceleration

The code automatically detects and utilizes available hardware acceleration:
- MPS (Metal Performance Shaders) for Apple Silicon
- Falls back to CPU if MPS is not available

## Results

The project generates several visualization files to help analyze and compare model performances:

### Individual Model Predictions
Each model generates a prediction plot (`prediction_plot_[ModelName].png`) that shows:
- Actual stock prices vs. predicted values
- Time series visualization of prediction accuracy
- Available for all four models:
  - Hybrid xLSTM-Transformer
  - Basic LSTM
  - Basic Transformer
  - Basic xLSTM

### Model Comparison
The `model_comparison_mse.png` provides a comparative analysis of model performances:
- Mean Squared Error (MSE) comparison across all models
- Visual representation of relative model accuracies
- Higher resolution (1200x600) for detailed analysis

These visualizations help in:
- Evaluating each model's prediction accuracy
- Comparing performance across different architectures
- Identifying strengths and weaknesses of each approach
- Understanding the impact of different neural network architectures on stock price prediction

## Model Improvements

### Initial Implementation Issues
The initial Hybrid xLSTM-Transformer model showed suboptimal performance due to several limitations:

1. **Architecture Limitations**
   - Sequential processing (Transformer → xLSTM)
   - Limited information flow between components
   - Potential loss of features in the pipeline

2. **Hyperparameter Constraints**
   - Hidden dimension: 64
   - Transformer layers: 2
   - Attention heads: 4
   - xLSTM blocks: 3
   - Batch size: 32
   - Training epochs: 30

3. **Data Utilization**
   - Only using closing price
   - Limited feature engineering
   - No multi-feature analysis

### Improvements Implemented

1. **Enhanced Architecture**
   - Parallel processing branches for Transformer and xLSTM
   - Feature fusion layer for combining outputs
   - Improved information flow and feature preservation

2. **Optimized Hyperparameters**
   - Hidden dimension: 128 (2x increase)
   - Transformer layers: 4 (2x increase)
   - Attention heads: 8 (2x increase)
   - xLSTM blocks: 4 (33% increase)
   - Batch size: 64 (2x increase)
   - Training epochs: 100 (3.3x increase)

3. **Enhanced Data Processing**
   - Multiple input features:
     - Closing price
     - Trading volume
     - RSI (Relative Strength Index)
     - MACD (Moving Average Convergence Divergence)
     - 1-day returns
     - 5-day volatility

4. **Advanced Training Configuration**
   - Learning rate scheduling with ReduceLROnPlateau
   - Gradient clipping (max norm: 1.0)
   - Early stopping mechanism
   - Improved optimization strategy

These improvements aim to address the root causes of poor performance and enhance the model's prediction capabilities # xlstm-transformer
