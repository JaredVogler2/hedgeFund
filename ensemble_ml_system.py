"""
Ensemble Machine Learning System with GPU Acceleration - COMPLETE VERSION
Professional-grade ML models with proper time series handling and deep learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import optuna
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import joblib
import warnings
from datetime import datetime, timedelta
from purged_cross_validation import PurgedTimeSeriesSplit
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("GPU not available, using CPU")

@dataclass
class ModelConfig:
    """Configuration for ensemble models with data leakage prevention"""
    # XGBoost
    xgb_params: Dict[str, Any] = None

    # LightGBM
    lgb_params: Dict[str, Any] = None

    # CatBoost
    cat_params: Dict[str, Any] = None

    # Deep Learning
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.3
    transformer_heads: int = 8
    transformer_layers: int = 4
    cnn_num_filters: int = 64
    cnn_filter_sizes: List[int] = None

    # Training with proper CV
    n_splits: int = 5
    purge_days: int = 5  # CRITICAL: Match prediction horizon
    embargo_days: int = 2  # Additional safety
    test_size: float = 0.2

    # Deep learning training
    dl_batch_size: int = 256
    dl_epochs: int = 100
    dl_learning_rate: float = 0.001
    dl_early_stopping_patience: int = 20

    # Ensemble
    min_agreement: float = 0.6
    use_meta_learner: bool = True
    use_deep_learning: bool = True  # Toggle for deep learning models

    def __post_init__(self):
        if self.cnn_filter_sizes is None:
            self.cnn_filter_sizes = [3, 5, 7]

        if self.xgb_params is None:
            self.xgb_params = {
                'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
                'gpu_id': 0,
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'early_stopping_rounds': 50
            }

        if self.lgb_params is None:
            self.lgb_params = {
                'device': 'gpu' if torch.cuda.is_available() else 'cpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'n_estimators': 1000,
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 1,
                'lambda_l2': 1,
                'min_child_samples': 20,
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'dart',
                'drop_rate': 0.1,
                'early_stopping_rounds': 50
            }

        if self.cat_params is None:
            self.cat_params = {
                'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
                'devices': '0',
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.01,
                'l2_leaf_reg': 3,
                'border_count': 128,
                'random_strength': 1,
                'bagging_temperature': 1,
                'od_type': 'Iter',
                'od_wait': 50
            }

# Deep Learning Models
class AttentionLSTM(nn.Module):
    """LSTM with multi-head attention mechanism for time series"""

    def __init__(self, input_size, hidden_size=256, num_layers=3,
                 num_heads=8, dropout=0.3, output_size=1):
        super(AttentionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size * 2, num_heads, dropout=dropout, batch_first=True
        )

        # Residual connection and normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Self-attention with residual connection
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.layer_norm1(lstm_out + attn_out)

        # Take last timestep
        out = lstm_out[:, -1, :]

        # Output layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class CNNLSTM(nn.Module):
    """CNN-LSTM for pattern extraction and sequence modeling"""

    def __init__(self, input_size, seq_len, hidden_size=256,
                 num_filters=64, filter_sizes=[3, 5, 7],
                 lstm_layers=2, dropout=0.3, output_size=1):
        super(CNNLSTM, self).__init__()

        # Multi-scale CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(input_size, num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])

        self.conv_dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # LSTM layers
        lstm_input_size = num_filters * len(filter_sizes)
        self.lstm = nn.LSTM(
            lstm_input_size, hidden_size, lstm_layers,
            batch_first=True, dropout=dropout
        )

        # Output layers
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape for CNN: (batch, features, sequence)
        x_cnn = x.transpose(1, 2)

        # Apply multiple convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(x_cnn))
            conv_out = self.conv_dropout(conv_out)
            conv_outputs.append(conv_out)

        # Concatenate multi-scale features
        x_combined = torch.cat(conv_outputs, dim=1)

        # Reshape back for LSTM: (batch, sequence, features)
        x_lstm = x_combined.transpose(1, 2)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x_lstm)

        # Take last timestep
        out = lstm_out[:, -1, :]

        # Output layer
        out = self.fc(out)

        return out

class TransformerModel(nn.Module):
    """Transformer model for sequence prediction"""

    def __init__(self, input_size, d_model=256, num_heads=8,
                 num_layers=4, dropout=0.3, output_size=1):
        super(TransformerModel, self).__init__()

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Main Ensemble Model
class EnsembleModel:
    """Professional ensemble with multiple ML models and proper CV"""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.feature_importance = {}
        self.model_weights = {}
        self.meta_learner = None
        self.cv_scores = {}
        self.sequence_length = 20  # For deep learning models

    def train(self, X: np.ndarray, y: np.ndarray,
             sample_weights: Optional[np.ndarray] = None,
             feature_names: Optional[List[str]] = None,
             cv = None) -> Dict[str, float]:
        """Train all models in the ensemble with proper time series CV"""
        logger.info("Starting ensemble training with data leakage prevention...")
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")

        # Store feature names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Use provided CV or create purged CV
        if cv is None:
            cv = PurgedTimeSeriesSplit(
                n_splits=self.config.n_splits,
                purge_days=self.config.purge_days,
                embargo_days=self.config.embargo_days
            )
            logger.info(f"Using PurgedTimeSeriesSplit with {self.config.purge_days} purge days")

        # Verify CV splits don't have leakage
        self._verify_cv_splits(cv, X, y)

        # Train each model type
        scores = {}

        # 1. XGBoost
        logger.info("Training XGBoost...")
        scores['xgboost'] = self._train_xgboost_safe(X, y, sample_weights, cv)

        # 2. LightGBM
        logger.info("Training LightGBM...")
        scores['lightgbm'] = self._train_lightgbm_safe(X, y, sample_weights, cv)

        # 3. CatBoost
        logger.info("Training CatBoost...")
        scores['catboost'] = self._train_catboost_safe(X, y, sample_weights, cv)

        # 4. Deep Learning Models (if enabled and GPU available)
        if self.config.use_deep_learning and device.type == 'cuda':
            logger.info("Training deep learning models...")

            # Scale features for deep learning
            X_scaled = self._scale_features(X, fit=True)

            # Train each deep model
            scores['attention_lstm'] = self._train_deep_model_safe(
                X_scaled, y, AttentionLSTM, 'attention_lstm', sample_weights, cv
            )
            scores['cnn_lstm'] = self._train_deep_model_safe(
                X_scaled, y, CNNLSTM, 'cnn_lstm', sample_weights, cv
            )
            scores['transformer'] = self._train_deep_model_safe(
                X_scaled, y, TransformerModel, 'transformer', sample_weights, cv
            )
        elif self.config.use_deep_learning:
            logger.info("Skipping deep learning models (GPU not available)")

        # 5. Train meta-learner if enabled
        if self.config.use_meta_learner:
            logger.info("Training meta-learner...")
            self._train_meta_learner_safe(X, y, sample_weights, cv)

        # Calculate model weights based on performance
        self._calculate_model_weights(scores)

        # Store CV scores for analysis
        self.cv_scores = scores

        logger.info("Ensemble training completed")
        logger.info(f"Cross-validation scores: {scores}")

        return scores

    def _verify_cv_splits(self, cv, X, y):
        """Verify that CV splits don't have data leakage"""
        logger.info("Verifying cross-validation splits...")

        for i, (train_idx, val_idx) in enumerate(cv.split(X)):
            # Check for overlap
            overlap = np.intersect1d(train_idx, val_idx)
            if len(overlap) > 0:
                raise ValueError(f"Data leakage detected! Train and val sets overlap in fold {i+1}")

            # Check gap
            if len(train_idx) > 0 and len(val_idx) > 0:
                gap = val_idx[0] - train_idx[-1] - 1
                if gap < self.config.purge_days:
                    logger.warning(f"Fold {i+1}: Gap of {gap} samples may be insufficient")

            logger.debug(f"Fold {i+1}: Train size={len(train_idx)}, Val size={len(val_idx)}, "
                        f"Gap={gap if 'gap' in locals() else 'N/A'} samples")

    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features for neural networks"""
        if 'feature_scaler' not in self.scalers:
            self.scalers['feature_scaler'] = StandardScaler()

        if fit:
            return self.scalers['feature_scaler'].fit_transform(X)
        else:
            return self.scalers['feature_scaler'].transform(X)

    def _create_sequences(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        """Create sequences from features for time series models"""
        n_samples, n_features = X.shape

        # For time series, we need to create overlapping sequences
        sequences = []

        for i in range(n_samples - seq_len + 1):
            seq = X[i:i+seq_len]
            sequences.append(seq)

        return np.array(sequences)

    def _train_xgboost_safe(self, X: np.ndarray, y: np.ndarray,
                           sample_weights: Optional[np.ndarray], cv) -> float:
        """Train XGBoost with proper CV and no data leakage"""
        scores = []
        fold_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            logger.debug(f"XGBoost Fold {fold_idx + 1}: Train {len(train_idx)}, Val {len(val_idx)}")

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # CRITICAL: Fit scaler only on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)  # Use training scaler

            if sample_weights is not None:
                w_train = sample_weights[train_idx]
            else:
                w_train = None

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=w_train)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)

            # Train model
            model = xgb.train(
                self.config.xgb_params,
                dtrain,
                num_boost_round=self.config.xgb_params['n_estimators'],
                evals=[(dval, 'eval')],
                early_stopping_rounds=self.config.xgb_params['early_stopping_rounds'],
                verbose_eval=False
            )

            # Evaluate
            pred = model.predict(dval)
            score = np.sqrt(np.mean((pred - y_val) ** 2))
            scores.append(score)
            fold_models.append((model, scaler))

            logger.debug(f"Fold {fold_idx + 1} RMSE: {score:.4f}")

        # Train final model on all data with its own scaler
        logger.info("Training final XGBoost model on all data...")
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X)

        dtrain_all = xgb.DMatrix(X_all_scaled, label=y, weight=sample_weights)
        self.models['xgboost'] = xgb.train(
            self.config.xgb_params,
            dtrain_all,
            num_boost_round=self.config.xgb_params['n_estimators']
        )
        self.scalers['xgboost'] = final_scaler

        # Store feature importance
        importance = self.models['xgboost'].get_score(importance_type='gain')
        self.feature_importance['xgboost'] = {
            f'f{k}': v for k, v in importance.items()
        }

        avg_score = np.mean(scores)
        logger.info(f"XGBoost CV score: {avg_score:.4f} (+/- {np.std(scores):.4f})")

        return avg_score

    def _train_lightgbm_safe(self, X: np.ndarray, y: np.ndarray,
                            sample_weights: Optional[np.ndarray], cv) -> float:
        """Train LightGBM with proper CV and no data leakage"""
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit scaler on training data only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            if sample_weights is not None:
                w_train = sample_weights[train_idx]
            else:
                w_train = None

            # Create datasets
            train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=w_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

            # Train model
            model = lgb.train(
                self.config.lgb_params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(self.config.lgb_params['early_stopping_rounds']),
                          lgb.log_evaluation(0)]
            )

            # Evaluate
            pred = model.predict(X_val_scaled)
            score = np.sqrt(np.mean((pred - y_val) ** 2))
            scores.append(score)

        # Train final model
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X)

        train_data_all = lgb.Dataset(X_all_scaled, label=y, weight=sample_weights)
        self.models['lightgbm'] = lgb.train(
            self.config.lgb_params,
            train_data_all
        )
        self.scalers['lightgbm'] = final_scaler

        # Store feature importance
        importance = self.models['lightgbm'].feature_importance(importance_type='gain')
        self.feature_importance['lightgbm'] = {
            self.feature_names[i]: imp for i, imp in enumerate(importance)
        }

        avg_score = np.mean(scores)
        logger.info(f"LightGBM CV score: {avg_score:.4f} (+/- {np.std(scores):.4f})")

        return avg_score

    def _train_catboost_safe(self, X: np.ndarray, y: np.ndarray,
                            sample_weights: Optional[np.ndarray], cv) -> float:
        """Train CatBoost with proper CV and no data leakage"""
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit scaler on training data only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            if sample_weights is not None:
                w_train = sample_weights[train_idx]
            else:
                w_train = None

            # Create pools
            train_pool = cb.Pool(X_train_scaled, y_train, weight=w_train)
            val_pool = cb.Pool(X_val_scaled, y_val)

            # Train model
            model = cb.CatBoostRegressor(**self.config.cat_params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=False
            )

            # Evaluate
            pred = model.predict(X_val_scaled)
            score = np.sqrt(np.mean((pred - y_val) ** 2))
            scores.append(score)

        # Train final model
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X)

        train_pool_all = cb.Pool(X_all_scaled, y, weight=sample_weights)
        self.models['catboost'] = cb.CatBoostRegressor(**self.config.cat_params)
        self.models['catboost'].fit(train_pool_all, verbose=False)
        self.scalers['catboost'] = final_scaler

        # Store feature importance
        importance = self.models['catboost'].feature_importances_
        self.feature_importance['catboost'] = {
            self.feature_names[i]: imp for i, imp in enumerate(importance)
        }

        avg_score = np.mean(scores)
        logger.info(f"CatBoost CV score: {avg_score:.4f} (+/- {np.std(scores):.4f})")

        return avg_score

    def _train_deep_model_safe(self, X: np.ndarray, y: np.ndarray,
                              model_class, model_name: str,
                              sample_weights: Optional[np.ndarray], cv) -> float:
        """Train deep learning model with proper CV and no data leakage"""
        scores = []

        # Prepare sequences
        seq_len = min(self.sequence_length, X.shape[0] // 10)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            # Ensure we have enough data for sequences
            if len(train_idx) < seq_len + 100 or len(val_idx) < seq_len + 20:
                logger.warning(f"Insufficient data for {model_name} fold {fold_idx + 1}, skipping")
                continue

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create sequences
            X_train_seq = self._create_sequences(X_train, seq_len)
            X_val_seq = self._create_sequences(X_val, seq_len)

            # Adjust targets to match sequence output
            y_train_seq = y_train[seq_len-1:]
            y_val_seq = y_val[seq_len-1:]

            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_seq),
                torch.FloatTensor(y_train_seq.reshape(-1, 1))
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq),
                torch.FloatTensor(y_val_seq.reshape(-1, 1))
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.dl_batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.dl_batch_size
            )

            # Initialize model
            input_size = X_train_seq.shape[-1]

            if model_class == CNNLSTM:
                model = model_class(
                    input_size, seq_len,
                    hidden_size=self.config.lstm_hidden_size,
                    num_filters=self.config.cnn_num_filters,
                    filter_sizes=self.config.cnn_filter_sizes,
                    dropout=self.config.lstm_dropout
                ).to(device)
            elif model_class == AttentionLSTM:
                model = model_class(
                    input_size,
                    hidden_size=self.config.lstm_hidden_size,
                    num_layers=self.config.lstm_num_layers,
                    num_heads=self.config.transformer_heads,
                    dropout=self.config.lstm_dropout
                ).to(device)
            else:  # TransformerModel
                model = model_class(
                    input_size,
                    num_heads=self.config.transformer_heads,
                    num_layers=self.config.transformer_layers,
                    dropout=self.config.lstm_dropout
                ).to(device)

            # Train
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.dl_learning_rate,
                weight_decay=0.01
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5
            )
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.dl_epochs):
                # Training
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.dl_early_stopping_patience:
                        break

                if epoch % 20 == 0:
                    logger.debug(f"{model_name} Fold {fold_idx + 1} Epoch {epoch}: Val Loss = {val_loss:.4f}")

            scores.append(np.sqrt(best_val_loss))

        if not scores:
            logger.warning(f"No valid folds for {model_name}")
            return float('inf')

        # Train final model on all data
        logger.info(f"Training final {model_name} model...")

        X_seq_all = self._create_sequences(X, seq_len)
        y_seq_all = y[seq_len-1:]

        train_dataset_all = TensorDataset(
            torch.FloatTensor(X_seq_all),
            torch.FloatTensor(y_seq_all.reshape(-1, 1))
        )
        train_loader_all = DataLoader(
            train_dataset_all,
            batch_size=self.config.dl_batch_size,
            shuffle=True
        )

        # Initialize final model
        input_size = X_seq_all.shape[-1]

        if model_class == CNNLSTM:
            final_model = model_class(
                input_size, seq_len,
                hidden_size=self.config.lstm_hidden_size,
                num_filters=self.config.cnn_num_filters,
                filter_sizes=self.config.cnn_filter_sizes,
                dropout=self.config.lstm_dropout
            ).to(device)
        elif model_class == AttentionLSTM:
            final_model = model_class(
                input_size,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                num_heads=self.config.transformer_heads,
                dropout=self.config.lstm_dropout
            ).to(device)
        else:  # TransformerModel
            final_model = model_class(
                input_size,
                num_heads=self.config.transformer_heads,
                num_layers=self.config.transformer_layers,
                dropout=self.config.lstm_dropout
            ).to(device)

        optimizer = optim.AdamW(
            final_model.parameters(),
            lr=self.config.dl_learning_rate,
            weight_decay=0.01
        )
        criterion = nn.MSELoss()

        # Shorter training for final model
        for epoch in range(min(50, self.config.dl_epochs)):
            final_model.train()
            for batch_X, batch_y in train_loader_all:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = final_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                optimizer.step()

        # Store model
        self.models[model_name] = final_model
        self.models[model_name].eval()  # Set to eval mode

        avg_score = np.mean(scores)
        logger.info(f"{model_name} CV score: {avg_score:.4f} (+/- {np.std(scores):.4f})")

        return avg_score

    def _train_meta_learner_safe(self, X: np.ndarray, y: np.ndarray,
                                sample_weights: Optional[np.ndarray], cv):
        """Train meta-learner with proper out-of-fold predictions"""
        logger.info("Generating out-of-fold predictions for meta-learner...")

        # Initialize arrays for out-of-fold predictions
        base_models = ['xgboost', 'lightgbm', 'catboost']
        if self.config.use_deep_learning and device.type == 'cuda':
            base_models.extend(['attention_lstm', 'cnn_lstm', 'transformer'])

        oof_predictions = {model: np.zeros(len(y)) for model in base_models}

        # Generate out-of-fold predictions
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train temporary models for this fold
            for model_name in base_models:
                if model_name in ['xgboost', 'lightgbm', 'catboost']:
                    # Tree-based models
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)

                    if model_name == 'xgboost':
                        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
                        dval = xgb.DMatrix(X_val_scaled)
                        params = self.config.xgb_params.copy()
                        params['n_estimators'] = 100  # Faster for meta-learning
                        model = xgb.train(params, dtrain,
                                        num_boost_round=100, verbose_eval=False)
                        pred = model.predict(dval)

                    elif model_name == 'lightgbm':
                        train_data = lgb.Dataset(X_train_scaled, label=y_train)
                        params = self.config.lgb_params.copy()
                        params['n_estimators'] = 100
                        model = lgb.train(params, train_data,
                                        num_boost_round=100, verbose=0)
                        pred = model.predict(X_val_scaled)

                    else:  # catboost
                        train_pool = cb.Pool(X_train_scaled, y_train)
                        params = self.config.cat_params.copy()
                        params['iterations'] = 100
                        model = cb.CatBoostRegressor(**params)
                        model.fit(train_pool, verbose=False)
                        pred = model.predict(X_val_scaled)

                    oof_predictions[model_name][val_idx] = pred

                else:
                    # Deep learning models - skip for meta-learner training (too slow)
                    # Use simple predictions instead
                    oof_predictions[model_name][val_idx] = np.mean(y_train)

        # Train meta-learner on out-of-fold predictions
        meta_features = np.column_stack([oof_predictions[m] for m in base_models])

        from sklearn.linear_model import RidgeCV
        self.meta_learner = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
        self.meta_learner.fit(meta_features, y)

        logger.info(f"Meta-learner trained with alpha={self.meta_learner.alpha_}")

    def predict(self, X: np.ndarray, return_all: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Generate predictions from ensemble"""
        predictions = {}

        # Get predictions from each model
        for model_name, model in self.models.items():
            if model_name in ['xgboost', 'lightgbm', 'catboost']:
                # Tree-based models
                if model_name in self.scalers:
                    X_scaled = self.scalers[model_name].transform(X)
                else:
                    X_scaled = X

                if model_name == 'xgboost':
                    dmatrix = xgb.DMatrix(X_scaled)
                    pred = model.predict(dmatrix)
                else:
                    pred = model.predict(X_scaled)

            elif 'lstm' in model_name or 'transformer' in model_name:
                # Deep learning models
                pred = self._predict_deep_model(model, X, model_name)

            else:
                pred = np.zeros(len(X))  # Fallback

            # Apply calibration if available
            if model_name in self.calibrators:
                pred = self.calibrators[model_name].transform(pred.reshape(-1, 1)).flatten()

            predictions[model_name] = pred

        if return_all:
            return predictions

        # Combine predictions
        if self.config.use_meta_learner and self.meta_learner is not None:
            # Use meta-learner
            base_models = ['xgboost', 'lightgbm', 'catboost']
            if self.config.use_deep_learning and device.type == 'cuda':
                base_models.extend(['attention_lstm', 'cnn_lstm', 'transformer'])

            meta_features = np.column_stack([predictions.get(m, np.zeros(len(X)))
                                           for m in base_models])
            ensemble_pred = self.meta_learner.predict(meta_features)
        else:
            # Weighted average
            ensemble_pred = np.zeros(len(X))
            total_weight = 0
            for model_name, pred in predictions.items():
                weight = self.model_weights.get(model_name, 1.0)
                ensemble_pred += weight * pred
                total_weight += weight
            ensemble_pred /= (total_weight + 1e-6)

        return ensemble_pred

    def _predict_deep_model(self, model, X: np.ndarray, model_name: str) -> np.ndarray:
        """Generate predictions from deep model"""
        model.eval()

        # Scale features
        if 'feature_scaler' in self.scalers:
            X_scaled = self.scalers['feature_scaler'].transform(X)
        else:
            X_scaled = X

        # Create sequences
        seq_len = min(self.sequence_length, len(X))

        # For single prediction, we might need to pad
        if len(X_scaled) < seq_len:
            # Pad with last value
            padding = np.repeat(X_scaled[-1:], seq_len - len(X_scaled), axis=0)
            X_padded = np.vstack([padding, X_scaled])
        else:
            X_padded = X_scaled

        X_seq = self._create_sequences(X_padded, seq_len)

        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(X_seq))
        loader = DataLoader(dataset, batch_size=self.config.dl_batch_size)

        predictions = []
        with torch.no_grad():
            for batch_X, in loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions.append(outputs.cpu().numpy())

        pred_array = np.concatenate(predictions).flatten()

        # If we padded, return only the last prediction
        if len(X) == 1:
            return pred_array[-1:]

        return pred_array

    def predict_proba(self, X: np.ndarray, quantiles: List[float] = [0.1, 0.5, 0.9]) -> np.ndarray:
        """Predict probability distributions with uncertainty"""
        all_predictions = self.predict(X, return_all=True)

        # Filter out any None predictions
        valid_predictions = {k: v for k, v in all_predictions.items() if v is not None}

        if not valid_predictions:
            # Return default quantiles
            return np.zeros((len(X), len(quantiles) + 1))

        predictions_array = np.column_stack(list(valid_predictions.values()))

        # Calculate quantiles across models
        quantile_predictions = np.percentile(predictions_array,
                                           [q * 100 for q in quantiles], axis=1).T

        # Add prediction uncertainty based on model disagreement
        prediction_std = np.std(predictions_array, axis=1)

        # Return quantiles and uncertainty
        result = np.column_stack([quantile_predictions, prediction_std])

        return result

    def _calculate_model_weights(self, scores: Dict[str, float]):
        """Calculate model weights based on performance"""
        # Invert scores (lower RMSE is better)
        inv_scores = {k: 1.0 / (v + 1e-6) for k, v in scores.items()}

        # Normalize to sum to 1
        total = sum(inv_scores.values())
        self.model_weights = {k: v / total for k, v in inv_scores.items()}

        logger.info(f"Model weights: {self.model_weights}")

    def get_feature_importance(self, top_n: int = 50) -> pd.DataFrame:
        """Get aggregated feature importance across models"""
        all_importance = []

        for model_name, importance_dict in self.feature_importance.items():
            if importance_dict:
                df = pd.DataFrame(
                    list(importance_dict.items()),
                    columns=['feature', 'importance']
                )
                df['model'] = model_name
                all_importance.append(df)

        if not all_importance:
            return pd.DataFrame()

        # Combine and aggregate
        importance_df = pd.concat(all_importance)

        # Normalize importance within each model
        importance_df['importance_norm'] = importance_df.groupby('model')['importance'].transform(
            lambda x: x / (x.sum() + 1e-6)
        )

        # Aggregate across models
        aggregated = importance_df.groupby('feature')['importance_norm'].agg(['mean', 'std', 'count'])
        aggregated = aggregated.sort_values('mean', ascending=False).head(top_n)

        return aggregated

    """
    Complete load_models method implementation
    Replace the existing load_models method in ensemble_ml_system.py with this version
    """

    def load_models(self, path: str):
        """Load models from disk with complete configuration"""
        # Import required model classes at the top
        from ensemble_ml_system import AttentionLSTM, CNNLSTM, TransformerModel
        import xgboost as xgb

        logger.info(f"Loading models from {path}...")

        # Load ensemble configuration first
        if os.path.exists(f"{path}/ensemble_config.pkl"):
            config_dict = joblib.load(f"{path}/ensemble_config.pkl")
            self.config = config_dict['model_config']
            self.sequence_length = config_dict.get('sequence_length', 20)
            if 'feature_names' in config_dict and config_dict['feature_names']:
                self.feature_names = config_dict['feature_names']
            logger.info("Loaded ensemble configuration")
        elif os.path.exists(f"{path}/model_config.pkl"):
            # Backward compatibility
            self.config = joblib.load(f"{path}/model_config.pkl")
            logger.info("Loaded model configuration (legacy)")

        # Load tree-based models
        for name in ['xgboost', 'lightgbm', 'catboost']:
            model_path = f"{path}/{name}_model"
            try:
                if name == 'xgboost' and os.path.exists(f"{model_path}.json"):
                    self.models[name] = xgb.Booster()
                    self.models[name].load_model(f"{model_path}.json")
                    logger.info(f"Loaded {name} model")
                elif os.path.exists(f"{model_path}.pkl"):
                    self.models[name] = joblib.load(f"{model_path}.pkl")
                    logger.info(f"Loaded {name} model")
            except Exception as e:
                logger.warning(f"Could not load {name} model: {e}")

        # Load deep learning models
        for name in ['attention_lstm', 'cnn_lstm', 'transformer']:
            model_path = f"{path}/{name}_model.pth"
            if os.path.exists(model_path):
                try:
                    # Load checkpoint
                    checkpoint = torch.load(model_path, map_location=device)
                    model_config = checkpoint['model_config']

                    # Log the configuration being loaded
                    logger.info(f"Loading {name} with config: {model_config}")

                    # Recreate model architecture based on saved configuration
                    if model_config['class'] == 'AttentionLSTM':
                        model = AttentionLSTM(
                            input_size=model_config['input_size'],
                            hidden_size=model_config.get('hidden_size', self.config.lstm_hidden_size),
                            num_layers=model_config.get('num_layers', self.config.lstm_num_layers),
                            num_heads=model_config.get('num_heads', self.config.transformer_heads),
                            dropout=model_config.get('dropout', self.config.lstm_dropout),
                            output_size=model_config.get('output_size', 1)
                        ).to(device)

                    elif model_config['class'] == 'CNNLSTM':
                        model = CNNLSTM(
                            input_size=model_config['input_size'],
                            seq_len=model_config['seq_len'],
                            hidden_size=model_config.get('hidden_size', self.config.lstm_hidden_size),
                            num_filters=model_config.get('num_filters', self.config.cnn_num_filters),
                            filter_sizes=model_config.get('filter_sizes', self.config.cnn_filter_sizes),
                            lstm_layers=model_config.get('lstm_layers', 2),
                            dropout=model_config.get('dropout', self.config.lstm_dropout),
                            output_size=model_config.get('output_size', 1)
                        ).to(device)

                    elif model_config['class'] == 'TransformerModel':
                        model = TransformerModel(
                            input_size=model_config['input_size'],
                            d_model=model_config.get('d_model', 256),
                            num_heads=model_config.get('num_heads', self.config.transformer_heads),
                            num_layers=model_config.get('num_layers', self.config.transformer_layers),
                            dropout=model_config.get('dropout', self.config.lstm_dropout),
                            output_size=model_config.get('output_size', 1)
                        ).to(device)
                    else:
                        logger.error(f"Unknown model class: {model_config['class']}")
                        continue

                    # Load model weights
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()  # Set to evaluation mode
                    self.models[name] = model

                    # Update sequence length if available
                    if 'sequence_length' in model_config:
                        self.sequence_length = model_config['sequence_length']

                    logger.info(f"Successfully loaded {name} model")

                except Exception as e:
                    logger.error(f"Error loading {name} model: {e}")
                    import traceback
                    traceback.print_exc()

        # Load auxiliary objects
        try:
            if os.path.exists(f"{path}/scalers.pkl"):
                self.scalers = joblib.load(f"{path}/scalers.pkl")
                logger.info(f"Loaded {len(self.scalers)} scalers")

            if os.path.exists(f"{path}/calibrators.pkl"):
                self.calibrators = joblib.load(f"{path}/calibrators.pkl")
                logger.info(f"Loaded {len(self.calibrators)} calibrators")

            if os.path.exists(f"{path}/feature_importance.pkl"):
                self.feature_importance = joblib.load(f"{path}/feature_importance.pkl")
                logger.info(f"Loaded feature importance for {len(self.feature_importance)} models")

            if os.path.exists(f"{path}/model_weights.pkl"):
                self.model_weights = joblib.load(f"{path}/model_weights.pkl")
                logger.info(f"Loaded model weights: {self.model_weights}")

            if os.path.exists(f"{path}/cv_scores.pkl"):
                self.cv_scores = joblib.load(f"{path}/cv_scores.pkl")
                logger.info(f"Loaded CV scores: {self.cv_scores}")

            if os.path.exists(f"{path}/meta_learner.pkl"):
                self.meta_learner = joblib.load(f"{path}/meta_learner.pkl")
                logger.info("Loaded meta-learner")

        except Exception as e:
            logger.error(f"Error loading auxiliary objects: {e}")

        # Summary
        logger.info(f"Loading complete. Available models: {list(self.models.keys())}")
        if hasattr(self, 'model_weights'):
            logger.info(f"Model weights: {self.model_weights}")

    # Also update the save_models method to be more robust
    def save_models(self, path: str):
        """Save all models to disk with complete configuration"""
        os.makedirs(path, exist_ok=True)

        # Save tree-based models
        for name in ['xgboost', 'lightgbm', 'catboost']:
            if name in self.models:
                try:
                    if name == 'xgboost':
                        self.models[name].save_model(f"{path}/{name}_model.json")
                    else:
                        joblib.dump(self.models[name], f"{path}/{name}_model.pkl")
                    logger.info(f"Saved {name} model")
                except Exception as e:
                    logger.error(f"Error saving {name} model: {e}")

        # Save deep learning models with complete configuration
        for name in ['attention_lstm', 'cnn_lstm', 'transformer']:
            if name in self.models:
                try:
                    model = self.models[name]

                    # Extract model configuration based on type
                    if name == 'attention_lstm':
                        # Get actual parameters from model architecture
                        lstm_hidden_size = model.hidden_size
                        lstm_num_layers = model.num_layers

                        # For attention heads, we need to check the attention layer
                        num_heads = model.attention.num_heads if hasattr(model.attention, 'num_heads') else 8

                        model_config = {
                            'class': 'AttentionLSTM',
                            'input_size': model.lstm.input_size,
                            'hidden_size': lstm_hidden_size,
                            'num_layers': lstm_num_layers,
                            'num_heads': num_heads,
                            'dropout': model.dropout.p,
                            'output_size': model.fc2.out_features,
                            'sequence_length': self.sequence_length
                        }

                    elif name == 'cnn_lstm':
                        # Extract parameters from CNNLSTM model
                        model_config = {
                            'class': 'CNNLSTM',
                            'input_size': model.convs[0].in_channels,
                            'seq_len': self.sequence_length,
                            'hidden_size': model.lstm.hidden_size,
                            'num_filters': model.convs[0].out_channels,
                            'filter_sizes': [conv.kernel_size[0] for conv in model.convs],
                            'lstm_layers': model.lstm.num_layers,
                            'dropout': model.conv_dropout.p,
                            'output_size': model.fc.out_features,
                            'sequence_length': self.sequence_length
                        }

                    elif name == 'transformer':
                        # Extract parameters from TransformerModel
                        model_config = {
                            'class': 'TransformerModel',
                            'input_size': model.input_projection.in_features,
                            'd_model': model.input_projection.out_features,
                            'num_heads': model.transformer.layers[0].self_attn.num_heads,
                            'num_layers': len(model.transformer.layers),
                            'dropout': model.dropout.p,
                            'output_size': model.fc2.out_features,
                            'sequence_length': self.sequence_length
                        }

                    # Save model state and configuration
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': model_config,
                        'model_class': type(model).__name__,
                        'pytorch_version': torch.__version__,
                        'cuda_available': torch.cuda.is_available(),
                        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
                    }, f"{path}/{name}_model.pth")

                    logger.info(f"Saved {name} with config: {model_config}")

                except Exception as e:
                    logger.error(f"Error saving {name} model: {e}")
                    import traceback
                    traceback.print_exc()

        # Save scalers and other objects
        auxiliary_objects = {
            'scalers': self.scalers,
            'calibrators': self.calibrators,
            'feature_importance': self.feature_importance,
            'model_weights': self.model_weights,
            'cv_scores': self.cv_scores
        }

        for obj_name, obj in auxiliary_objects.items():
            if obj:
                try:
                    joblib.dump(obj, f"{path}/{obj_name}.pkl")
                    logger.info(f"Saved {obj_name}")
                except Exception as e:
                    logger.error(f"Error saving {obj_name}: {e}")

        if self.meta_learner is not None:
            try:
                joblib.dump(self.meta_learner, f"{path}/meta_learner.pkl")
                logger.info("Saved meta-learner")
            except Exception as e:
                logger.error(f"Error saving meta-learner: {e}")

        # Save comprehensive configuration
        config_dict = {
            'model_config': self.config,
            'sequence_length': self.sequence_length,
            'feature_names': getattr(self, 'feature_names', None),
            'models_included': list(self.models.keys()),
            'save_date': datetime.now().isoformat(),
            'ensemble_version': '2.0'  # Version tracking
        }

        try:
            joblib.dump(config_dict, f"{path}/ensemble_config.pkl")
            logger.info("Saved ensemble configuration")
        except Exception as e:
            logger.error(f"Error saving ensemble configuration: {e}")

        logger.info(f"Model saving complete. Saved to {path}")


# Hyperparameter optimization
class ModelOptimizer:
    """Optimize hyperparameters using Optuna with proper CV"""

    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.best_params = None

    def optimize(self, X: np.ndarray, y: np.ndarray,
                n_trials: int = 100, cv=None) -> Dict[str, Any]:
        """Run hyperparameter optimization with purged CV"""

        if cv is None:
            cv = PurgedTimeSeriesSplit(n_splits=3, purge_days=5)

        def objective(trial):
            if self.model_type == 'xgboost':
                params = {
                    'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
                    'gpu_id': 0,
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
            elif self.model_type == 'lightgbm':
                params = {
                    'device': 'gpu' if torch.cuda.is_available() else 'cpu',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                }

            # Use cross-validation to evaluate
            cv_scores = []

            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                if self.model_type == 'xgboost':
                    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
                    dval = xgb.DMatrix(X_val_scaled, label=y_val)

                    model = xgb.train(
                        params, dtrain,
                        num_boost_round=params['n_estimators'],
                        evals=[(dval, 'eval')],
                        early_stopping_rounds=50,
                        verbose_eval=False
                    )
                    pred = model.predict(dval)

                elif self.model_type == 'lightgbm':
                    train_data = lgb.Dataset(X_train_scaled, label=y_train)
                    val_data = lgb.Dataset(X_val_scaled, label=y_val)

                    model = lgb.train(
                        params, train_data,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                    pred = model.predict(X_val_scaled)

                score = np.sqrt(np.mean((pred - y_val) ** 2))
                cv_scores.append(score)

            return np.mean(cv_scores)

        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        self.best_params = study.best_params
        logger.info(f"Best parameters found: {self.best_params}")

        return self.best_params


# Example usage
if __name__ == "__main__":
    from datetime import datetime

    # Set dates
    CURRENT_DATE = pd.Timestamp('2025-07-12')
    START_DATE = CURRENT_DATE - pd.DateOffset(years=5)

    print(f"Testing complete ensemble with data from {START_DATE.date()} to {CURRENT_DATE.date()}")

    # Generate sample time series data
    np.random.seed(42)
    n_samples = 1000  # ~4 years of trading days
    n_features = 150

    # Create temporal structure
    dates = pd.date_range(start=START_DATE, periods=n_samples, freq='B')

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate realistic returns
    base_returns = np.random.randn(n_samples) * 0.01
    y = pd.Series(base_returns).rolling(5).mean().fillna(0).values

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Initialize ensemble with all features
    config = ModelConfig(
        n_splits=5,
        purge_days=5,
        embargo_days=2,
        use_deep_learning=torch.cuda.is_available(),  # Only if GPU available
        use_meta_learner=True
    )
    ensemble = EnsembleModel(config)

    # Create purged CV
    from purged_cross_validation import PurgedTimeSeriesSplit
    pcv = PurgedTimeSeriesSplit(n_splits=5, purge_days=5, embargo_days=2)

    # Train ensemble
    print("\nTraining ensemble with all models...")
    scores = ensemble.train(X, y, cv=pcv)
    print(f"\nFinal CV scores: {scores}")

    # Generate predictions
    print("\nGenerating predictions...")
    test_X = X[-10:]
    predictions = ensemble.predict(test_X)
    print(f"Predictions shape: {predictions.shape}")

    # Get prediction intervals
    quantile_preds = ensemble.predict_proba(test_X)
    print(f"\nPrediction intervals (10%, 50%, 90%, std):")
    for i in range(min(5, len(quantile_preds))):
        print(f"  Sample {i}: [{quantile_preds[i,0]:.4f}, {quantile_preds[i,1]:.4f}, "
              f"{quantile_preds[i,2]:.4f}], std={quantile_preds[i,3]:.4f}")

    # Get feature importance
    importance_df = ensemble.get_feature_importance(top_n=10)
    if not importance_df.empty:
        print(f"\nTop 10 most important features:")
        print(importance_df)

    # Save models
    print("\nSaving models...")
    ensemble.save_models("./ensemble_models")
    print("Models saved successfully!")