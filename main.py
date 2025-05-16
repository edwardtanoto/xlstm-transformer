# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
import os
from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,

        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )

try:
    import wandb
    # WandB Initialization
    try:
        wandb.init(project="fyp-time-series-0", entity="edwardtanoto")
    except Exception as e:
        class WandB:
            def __init__(self): self.config = {}
            def log(self, *args, **kwargs): pass
            def watch(self, *args, **kwargs): pass
            def finish(self, *args, **kwargs): pass
            def save(self, *args, **kwargs): pass
        wandb = WandB()
except ImportError:
    print("wandb not installed. Running without logging. To install: pip install wandb")
    class WandB:
        def __init__(self): self.config = {}
        def log(self, *args, **kwargs): pass
        def watch(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
    wandb = WandB()


# --- Determine Device (with MPS support) ---
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
             print("MPS not available: PyTorch install wasn't built with MPS enabled.")
             return "cpu"
        # Basic check for CUDA capability requirement mentioned for kernels
        # If MPS is available, we likely *cannot* use 'cuda' backend for sLSTM
        print("MPS device detected. Will use 'native' backend for sLSTM.")
        return "mps"
    else:
        return "cpu"

device_name = get_device()

# --- Configuration / Hyperparameters (aligned with xLSTMBlockStackConfig) ---
config = {
    # Data/Training Params
    "sequence_length": 60,     # Input sequence length (context_length for xLSTM)
    "batch_size": 64,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "patience": 15,
    "test_size": 0.15,
    "val_size": 0.15,
    "device": device_name,
    "data_file": "filtered_stock_data.csv",
    "num_workers": 0,
    "pin_memory": False,

    # Model Dimensions
    "num_features": 5,         # open, high, low, close, volume (input to projection)
    "hidden_dim": 128,         # Dimension after input projection, used as embedding_dim for xLSTM and d_model for Transformer
    "output_dim": 1,           # Predicting 'close' price

    # xLSTM Block Stack Config
    "xlstm_num_blocks": 4,     # Total number of xLSTM blocks in the stack
    "slstm_at": [1, 3],        # Indices (0-based) where sLSTM blocks should be placed (must be < xlstm_num_blocks)
    # mLSTM Block Template Config
    "mlstm_conv_kernel": 4,    # Kernel size for conv1d in mLSTM layers
    "mlstm_num_heads": 4,      # Number of heads for mLSTM attention mechanism
    "mlstm_qkv_proj_blocksize": 4, # Block size for QKV projection in mLSTM
    # sLSTM Block Template Config
    "slstm_num_heads": 4,      # Number of heads for sLSTM attention
    "slstm_conv_kernel": 4,    # Kernel size for conv1d in sLSTM layers
    "slstm_bias_init": "powerlaw_blockdependent", # Bias initialization strategy for sLSTM
    # Feedforward Config (within sLSTM block template, as per example)
    "ff_proj_factor": 1.3,     # Projection factor for feedforward layer (e.g., 1.3 * hidden_dim)
    "ff_act_fn": "gelu",       # Activation function for feedforward layer

    # Transformer Config
    "transformer_layers": 2,   # Number of Transformer Encoder layers *after* xLSTM stack
    "transformer_nhead": 8,    # Number of attention heads in Transformer (hidden_dim % nhead == 0)
    "transformer_dim_feedforward": 512, # Feedforward dimension in Transformer
    "dropout": 0.15,           # Dropout rate used in Transformer and after xLSTM

}

# Dynamic adjustments based on device
if config["device"] == "cuda":
    config["num_workers"] = 2 #
    config["pin_memory"] = True
    config["slstm_backend"] = "cuda" # Use CUDA backend if available
else:
    config["slstm_backend"] = "vanilla" # Use native PyTorch for MPS/CPU

# Validation Checks for Config
if not all(i < config["xlstm_num_blocks"] for i in config["slstm_at"]):
    raise ValueError(f"Indices in 'slstm_at' ({config['slstm_at']}) must be less than 'xlstm_num_blocks' ({config['xlstm_num_blocks']}).")
if config["hidden_dim"] % config["mlstm_num_heads"] != 0:
    print(f"Warning: hidden_dim ({config['hidden_dim']}) is not perfectly divisible by mlstm_num_heads ({config['mlstm_num_heads']}).")
if config["hidden_dim"] % config["slstm_num_heads"] != 0:
     print(f"Warning: hidden_dim ({config['hidden_dim']}) is not perfectly divisible by slstm_num_heads ({config['slstm_num_heads']}).")
if config["hidden_dim"] % config["transformer_nhead"] != 0:
    raise ValueError(f"hidden_dim ({config['hidden_dim']}) must be divisible by transformer_nhead ({config['transformer_nhead']}).")


# Log final configuration
wandb.config.update(config)
print(f"Using device: {config['device']}")
print("Final Configuration:", config)
print(f"Attempting to load data from: {config['data_file']}")


# --- 2. Data Loading & Preprocessing ---
if not os.path.exists(config["data_file"]):
    raise FileNotFoundError(f"Error: Data file '{config['data_file']}' not found.")
try:
    df = pd.read_csv(config["data_file"])
    print(f"Loaded data from {config['data_file']}. Shape: {df.shape}")
except Exception as e:
    raise IOError(f"Error reading CSV file '{config['data_file']}': {e}")
required_cols = ["code", "open", "close", "volume", "time", "high", "low"]
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"CSV must contain: {required_cols}. Found: {df.columns.tolist()}")
try:
    df["time"] = pd.to_datetime(df["time"])
except Exception as e:
    raise ValueError(f"Error converting 'time' column: {e}.")
df = df.sort_values(by=["code", "time"]).set_index("time")
features = ["open", "high", "low", "close", "volume"]
target = "close"
all_sequences, all_targets, all_codes, target_scalers = [], [], [], {}
try:
    from tqdm import tqdm

    use_tqdm = True
except ImportError:
    use_tqdm = False
    print("tqdm not found, proceeding without progress bars.")
print("Processing stocks and creating sequences...")
grouped_data = df.groupby("code")
iterator = (
    tqdm(grouped_data, total=len(grouped_data), desc="Processing Stocks")
    if use_tqdm
    else grouped_data
)
for code, group in iterator:
    if len(group) <= config["sequence_length"]:
        continue
    feature_scaler = MinMaxScaler()
    group_features_clean = (
        group[features]
        .fillna(method="ffill")
        .fillna(method="bfill")
        .replace([np.inf, -np.inf], 0)
    )
    if group_features_clean.isnull().values.any():
        continue
    group_scaled_features = feature_scaler.fit_transform(group_features_clean)
    target_scaler = MinMaxScaler()
    group_target_clean = (
        group[[target]]
        .fillna(method="ffill")
        .fillna(method="bfill")
        .replace([np.inf, -np.inf], 0)
    )
    if group_target_clean.isnull().values.any():
        continue
    group_scaled_target = target_scaler.fit_transform(group_target_clean)
    target_scalers[code] = target_scaler
    X, y = [], []
    group_scaled_features_vals = group_scaled_features
    group_scaled_target_vals = group_scaled_target.flatten()
    for i in range(len(group_scaled_features_vals) - config["sequence_length"]):
        X.append(group_scaled_features_vals[i : i + config["sequence_length"]])
        y.append(group_scaled_target_vals[i + config["sequence_length"]])
    if not X:
        continue
    n_samples = len(X)
    n_test = int(n_samples * config["test_size"])
    n_val = int((n_samples - n_test) * config["val_size"])
    n_train = n_samples - n_test - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        continue
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    codes_train = [code] * len(X_train)
    codes_val = [code] * len(X_val)
    codes_test = [code] * len(X_test)
    if "X_train_all" not in locals():
        X_train_all, y_train_all, codes_train_all = X_train, y_train, codes_train
        X_val_all, y_val_all, codes_val_all = X_val, y_val, codes_val
        X_test_all, y_test_all, codes_test_all = X_test, y_test, codes_test
    else:
        X_train_all.extend(X_train)
        y_train_all.extend(y_train)
        codes_train_all.extend(codes_train)
        X_val_all.extend(X_val)
        y_val_all.extend(y_val)
        codes_val_all.extend(codes_val)
        X_test_all.extend(X_test)
        y_test_all.extend(y_test)
        codes_test_all.extend(codes_test)
if "X_train_all" not in locals():
    raise ValueError("No data sequences generated.")
print("\nConverting data to PyTorch tensors...")
X_train_t = torch.tensor(np.array(X_train_all), dtype=torch.float32)
y_train_t = torch.tensor(np.array(y_train_all), dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(np.array(X_val_all), dtype=torch.float32)
y_val_t = torch.tensor(np.array(y_val_all), dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(np.array(X_test_all), dtype=torch.float32)
y_test_t = torch.tensor(np.array(y_test_all), dtype=torch.float32).unsqueeze(1)


# --- PyTorch Datasets and DataLoaders ---
class TimeSeriesDataset(Dataset): # No changes needed
    def __init__(self, X, y, codes): self.X, self.y, self.codes = X, y, codes
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.codes[idx]

train_dataset = TimeSeriesDataset(X_train_t, y_train_t, codes_train_all); val_dataset = TimeSeriesDataset(X_val_t, y_val_t, codes_val_all); test_dataset = TimeSeriesDataset(X_test_t, y_test_t, codes_test_all)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'])
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'])
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'])
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
if len(train_dataset) == 0: raise ValueError("No training data generated after processing.")


# --- 3. Model Definition (Using xLSTMBlockStack) ---
class XLSTMTransformer(nn.Module):
    def __init__(self, cfg): # Pass the main config dictionary
        super().__init__()
        self.hidden_dim = cfg['hidden_dim']

        # Input projection layer (maps original features to hidden_dim)
        self.input_proj = nn.Linear(cfg['num_features'], self.hidden_dim)
        self.input_activation = nn.ReLU()

        # --- Configure and Instantiate xLSTMBlockStack ---
        # 1. Define mLSTM Layer/Block Config Template
        mlstm_layer_cfg = mLSTMLayerConfig(
            conv1d_kernel_size=cfg['mlstm_conv_kernel'],
            qkv_proj_blocksize=cfg['mlstm_qkv_proj_blocksize'],
            num_heads=cfg['mlstm_num_heads']
        )
        mlstm_block_cfg = mLSTMBlockConfig(mlstm=mlstm_layer_cfg) # Assuming no FeedForward in mLSTM block per example

        # 2. Define FeedForward Config Template (used in sLSTM block)
        ff_cfg = FeedForwardConfig(
            proj_factor=cfg['ff_proj_factor'],
            act_fn=cfg['ff_act_fn']
        )

        # 3. Define sLSTM Layer/Block Config Template
        slstm_layer_cfg = sLSTMLayerConfig(
            backend=cfg['slstm_backend'], # Use 'native' for MPS/CPU, 'cuda' for CUDA
            num_heads=cfg['slstm_num_heads'],
            conv1d_kernel_size=cfg['slstm_conv_kernel'],
            bias_init=cfg['slstm_bias_init'],
        )
        slstm_block_cfg = sLSTMBlockConfig(slstm=slstm_layer_cfg, feedforward=ff_cfg)

        # 4. Define the main xLSTMBlockStack Config
        xlstm_stack_cfg = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block_cfg,            # Template for mLSTM blocks
            slstm_block=slstm_block_cfg,            # Template for sLSTM blocks
            num_blocks=cfg['xlstm_num_blocks'],     # Total number of blocks
            embedding_dim=self.hidden_dim,          # Internal dimension of the stack
            slstm_at=cfg['slstm_at'],               # Indices where sLSTM blocks are placed
            context_length=cfg['sequence_length']   # Expected sequence length
        )
        print(f"Initializing xLSTMBlockStack with generated config: {xlstm_stack_cfg}")
        self.xlstm = xLSTMBlockStack(xlstm_stack_cfg)
        # --- End of xLSTMBlockStack Configuration ---


        # Transformer Encoder Layer (processes output of xLSTM stack)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, # Input dim is the output dim of xLSTM stack
            nhead=cfg['transformer_nhead'],
            dim_feedforward=cfg['transformer_dim_feedforward'],
            dropout=cfg['dropout'],
            activation='relu', # Or 'gelu'
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=cfg['transformer_layers']
        )

        # Output Layer
        self.fc_out = nn.Linear(self.hidden_dim, cfg['output_dim'])
        self.dropout_layer = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim=num_features)
        x = self.input_proj(x)
        x = self.input_activation(x)
        # x shape: (batch_size, seq_len, hidden_dim)

        # xLSTM Block Stack processing
        # Expects (batch, seq_len, embedding_dim=hidden_dim)
        # Returns (batch, seq_len, embedding_dim=hidden_dim) according to example
        xlstm_out = self.xlstm(x)
        # xlstm_out shape: (batch_size, seq_len, hidden_dim)

        # Apply dropout after xLSTM stack
        xlstm_out = self.dropout_layer(xlstm_out)

        # Transformer Encoder processing
        # Needs (batch, seq_len, features=hidden_dim) if batch_first=True
        transformer_out = self.transformer_encoder(xlstm_out)
        # transformer_out shape: (batch_size, seq_len, hidden_dim)

        # Aggregate Transformer output - Use the output of the LAST time step
        agg_out = transformer_out[:, -1, :]
        # agg_out shape: (batch_size, hidden_dim)

        # Final Prediction Layer
        output = self.fc_out(agg_out)
        # output shape: (batch_size, output_dim)
        return output

# Instantiate the model using the main config dictionary
model = XLSTMTransformer(config).to(config['device']) # Pass the whole config

print("\nModel Architecture:")
print(model)
# You might want to log the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")
try: wandb.config.update({"total_params": total_params}) # Log param count to wandb
except Exception as e: print(f"Wandb param count log failed: {e}")


# --- 4. Training Loop ---
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['patience'] // 2, factor=0.5, verbose=True)
best_val_loss = float('inf'); epochs_no_improve = 0; best_model_state = None

print("\n--- Starting Training ---")
for epoch in range(config['num_epochs']):
    model.train(); train_loss_epoch = 0.0
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} Training", leave=False) if use_tqdm else train_loader
    for batch_X, batch_y, _ in train_iterator:
        batch_X, batch_y = batch_X.to(config['device']), batch_y.to(config['device'])
        optimizer.zero_grad(); outputs = model(batch_X); loss = criterion(outputs, batch_y)
        loss.backward(); optimizer.step(); train_loss_epoch += loss.item()
        if use_tqdm: train_iterator.set_postfix(loss=loss.item())
    train_loss_epoch /= len(train_loader)

    model.eval(); val_loss_epoch = 0.0
    val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} Validation", leave=False) if use_tqdm else val_loader
    with torch.no_grad():
        for batch_X, batch_y, _ in val_iterator:
            batch_X, batch_y = batch_X.to(config['device']), batch_y.to(config['device'])
            outputs = model(batch_X); loss = criterion(outputs, batch_y); val_loss_epoch += loss.item()
    val_loss_epoch /= len(val_loader)
    print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss_epoch:.6f}")
    try: wandb.log({"epoch": epoch+1, "train_loss": train_loss_epoch, "val_loss": val_loss_epoch, "learning_rate": optimizer.param_groups[0]['lr']})
    except Exception as e: print(f"Wandb logging failed: {e}")
    scheduler.step(val_loss_epoch)
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch; epochs_no_improve = 0; best_model_state = copy.deepcopy(model.state_dict())
        print(f"Val loss improved to {best_val_loss:.6f}. Saving model state.")
        # Optional: Save checkpoint to disk
        # torch.save(model.state_dict(), f"xlstm_transformer_epoch_{epoch+1}_valloss_{best_val_loss:.4f}.pth")
    else:
        epochs_no_improve += 1; print(f"Val loss didn't improve for {epochs_no_improve} epoch(s). Best: {best_val_loss:.6f}")
    if epochs_no_improve >= config['patience']: print(f"Early stopping triggered after {epoch + 1} epochs."); break

if best_model_state: print("Loading best model for evaluation."); model.load_state_dict(best_model_state)
else: print("Warning: No best model state saved. Using last state.")


# --- 5. Evaluation ---
print("\n--- Starting Evaluation ---")
model.eval(); test_loss = 0.0; all_preds_scaled, all_targets_scaled, all_test_codes = [], [], []
test_iterator = tqdm(test_loader, desc="Evaluating Test Set", leave=False) if use_tqdm else test_loader
with torch.no_grad():
    for batch_X, batch_y, batch_codes in test_iterator:
        batch_X, batch_y = batch_X.to(config['device']), batch_y.to(config['device'])
        outputs = model(batch_X); loss = criterion(outputs, batch_y); test_loss += loss.item()
        all_preds_scaled.extend(outputs.cpu().numpy().flatten()); all_targets_scaled.extend(batch_y.cpu().numpy().flatten())
        all_test_codes.extend(batch_codes)
test_loss /= len(test_loader); print(f"Test Loss (Scaled, Avg/Batch): {test_loss:.6f}")


# --- Inverse Transform and Final Metrics ---
print("Inverse transforming predictions...")
all_preds_original, all_targets_original = [], []
inverse_iterator = tqdm(range(len(all_preds_scaled)), desc="Inverse Scaling", leave=False) if use_tqdm else range(len(all_preds_scaled))
for i in inverse_iterator:
    code = all_test_codes[i]
    if code not in target_scalers: continue
    pred_scaled = np.array([[all_preds_scaled[i]]]); target_scaled = np.array([[all_targets_scaled[i]]])
    try: pred_original = target_scalers[code].inverse_transform(pred_scaled); target_original = target_scalers[code].inverse_transform(target_scaled); all_preds_original.append(pred_original.flatten()[0]); all_targets_original.append(target_original.flatten()[0])
    except Exception as e: print(f"Inv transform error: code {code}, idx {i}: {e}"); pass
if not all_preds_original: print("Error: No predictions inverse-transformed."); mae, mse, rmse = float('nan'), float('nan'), float('nan')
else:
    all_preds_original_np = np.array(all_preds_original); all_targets_original_np = np.array(all_targets_original)
    mae = np.mean(np.abs(all_preds_original_np - all_targets_original_np)); mse = np.mean((all_preds_original_np - all_targets_original_np)**2); rmse = np.sqrt(mse)
    print(f"\n--- Final Test Metrics (Original Scale) ---\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}")
try: wandb.log({"test_loss_scaled": test_loss, "test_mae_original": mae, "test_mse_original": mse, "test_rmse_original": rmse})
except Exception as e: print(f"Wandb final log failed: {e}")


# --- Optional: Log Predictions Plot ---
if all_preds_original:
    try:
        import matplotlib.pyplot as plt
        print("Generating prediction plot...")
        num_plot = min(500, len(all_preds_original)); plot_indices = np.random.choice(len(all_preds_original), num_plot, replace=False)
        plot_actual = all_targets_original_np[plot_indices]; plot_predicted = all_preds_original_np[plot_indices]
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.scatter(range(num_plot), plot_actual, label='Actual', marker='.', alpha=0.7); ax.scatter(range(num_plot), plot_predicted, label='Predicted', marker='x', alpha=0.7)
        ax.set_title(f'Test Predictions vs Actual (Original Scale, Random Sample {num_plot})'); ax.set_xlabel('Sample Index (Random)'); ax.set_ylabel('Close Price'); ax.legend(); plt.grid(True)
        try: wandb.log({"test_predictions_plot": wandb.Image(fig)}); print("Logged prediction plot to WandB.")
        except Exception as e: print(f"Wandb plot log failed: {e}")
        plt.close(fig)
    except ImportError: print("Matplotlib not found. Skipping prediction plot.")
    except Exception as e: print(f"Error generating plot: {e}")


# --- Finish WandB Run ---
try: wandb.finish()
except Exception as e: print(f"Wandb finish failed: {e}")
print("\n--- Script Finished ---")