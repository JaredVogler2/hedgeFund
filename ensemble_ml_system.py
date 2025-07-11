"""
Ensemble Machine Learning System with GPU Acceleration
Professional-grade ML models for trading signal generation
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
from sklearn.model_selection import TimeSeriesSplit, PurgedGroupTimeSeriesSplit
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
    """Configuration for ensemble models"""
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
    
    # Training
    n_splits: int = 5
    purge_days: int = 10
    test_size: float = 0.2
    
    # Ensemble
    min_agreement: float = 0.6
    use_meta_learner: bool = True
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'tree_method': 'gpu_hist',
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
                'device': 'gpu',
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
                'task_type': 'GPU',
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
    """LSTM with multi-head attention mechanism"""
    
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
        
        # Residual connection
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
    """Professional ensemble with multiple ML models"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.feature_importance = {}
        self.model_weights = {}
        self.meta_learner = None
        
    def train(self, X: np.ndarray, y: np.ndarray, 
             sample_weights: Optional[np.ndarray] = None,
             feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Train all models in the ensemble"""
        logger.info("Starting ensemble training...")
        
        # Store feature names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Prepare data
        X_scaled = self._scale_features(X, fit=True)
        
        # Create time series splits
        tscv = self._create_cv_splitter(X.shape[0])
        
        # Train each model type
        scores = {}
        
        # 1. XGBoost
        logger.info("Training XGBoost...")
        scores['xgboost'] = self._train_xgboost(X, y, sample_weights, tscv)
        
        # 2. LightGBM
        logger.info("Training LightGBM...")
        scores['lightgbm'] = self._train_lightgbm(X, y, sample_weights, tscv)
        
        # 3. CatBoost
        logger.info("Training CatBoost...")
        scores['catboost'] = self._train_catboost(X, y, sample_weights, tscv)
        
        # 4. Deep Learning Models
        if device.type == 'cuda':
            logger.info("Training deep learning models...")
            scores['attention_lstm'] = self._train_deep_model(
                X_scaled, y, AttentionLSTM, sample_weights, tscv
            )
            scores['cnn_lstm'] = self._train_deep_model(
                X_scaled, y, CNNLSTM, sample_weights, tscv
            )
            scores['transformer'] = self._train_deep_model(
                X_scaled, y, TransformerModel, sample_weights, tscv
            )
        
        # 5. Train meta-learner if enabled
        if self.config.use_meta_learner:
            logger.info("Training meta-learner...")
            self._train_meta_learner(X, y, sample_weights, tscv)
        
        # Calculate model weights based on performance
        self._calculate_model_weights(scores)
        
        logger.info("Ensemble training completed")
        return scores
    
    def predict(self, X: np.ndarray, return_all: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Generate predictions from ensemble"""
        predictions = {}
        
        # Scale features
        X_scaled = self._scale_features(X, fit=False)
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if 'deep' in model_name:
                pred = self._predict_deep_model(model, X_scaled)
            else:
                pred = model.predict(X)
            
            # Apply calibration if available
            if model_name in self.calibrators:
                pred = self.calibrators[model_name].transform(pred.reshape(-1, 1)).flatten()
            
            predictions[model_name] = pred
        
        if return_all:
            return predictions
        
        # Combine predictions
        if self.config.use_meta_learner and self.meta_learner is not None:
            # Use meta-learner
            meta_features = np.column_stack(list(predictions.values()))
            ensemble_pred = self.meta_learner.predict(meta_features)
        else:
            # Weighted average
            ensemble_pred = np.zeros(len(X))
            for model_name, pred in predictions.items():
                weight = self.model_weights.get(model_name, 1.0)
                ensemble_pred += weight * pred
            ensemble_pred /= sum(self.model_weights.values())
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray, quantiles: List[float] = [0.1, 0.5, 0.9]) -> np.ndarray:
        """Predict probability distributions"""
        all_predictions = self.predict(X, return_all=True)
        predictions_array = np.column_stack(list(all_predictions.values()))
        
        # Calculate quantiles across models
        quantile_predictions = np.percentile(predictions_array, 
                                           [q * 100 for q in quantiles], axis=1).T
        
        return quantile_predictions
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features for neural networks"""
        if 'feature_scaler' not in self.scalers:
            self.scalers['feature_scaler'] = StandardScaler()
        
        if fit:
            return self.scalers['feature_scaler'].fit_transform(X)
        else:
            return self.scalers['feature_scaler'].transform(X)
    
    def _create_cv_splitter(self, n_samples: int):
        """Create purged time series cross-validator"""
        # Use simple TimeSeriesSplit for now
        # In production, use PurgedGroupTimeSeriesSplit to prevent leakage
        return TimeSeriesSplit(
            n_splits=self.config.n_splits,
            test_size=int(n_samples * self.config.test_size)
        )
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, 
                      sample_weights: Optional[np.ndarray], cv) -> float:
        """Train XGBoost model"""
        scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if sample_weights is not None:
                w_train = sample_weights[train_idx]
            else:
                w_train = None
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
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
        
        # Train final model on all data
        dtrain_all = xgb.DMatrix(X, label=y, weight=sample_weights)
        self.models['xgboost'] = xgb.train(
            self.config.xgb_params,
            dtrain_all,
            num_boost_round=self.config.xgb_params['n_estimators']
        )
        
        # Store feature importance
        importance = self.models['xgboost'].get_score(importance_type='gain')
        self.feature_importance['xgboost'] = {
            f'f{k}': v for k, v in importance.items()
        }
        
        return np.mean(scores)
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray,
                       sample_weights: Optional[np.ndarray], cv) -> float:
        """Train LightGBM model"""
        scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if sample_weights is not None:
                w_train = sample_weights[train_idx]
            else:
                w_train = None
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                self.config.lgb_params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(self.config.lgb_params['early_stopping_rounds']),
                          lgb.log_evaluation(0)]
            )
            
            # Evaluate
            pred = model.predict(X_val)
            score = np.sqrt(np.mean((pred - y_val) ** 2))
            scores.append(score)
        
        # Train final model
        train_data_all = lgb.Dataset(X, label=y, weight=sample_weights)
        self.models['lightgbm'] = lgb.train(
            self.config.lgb_params,
            train_data_all
        )
        
        # Store feature importance
        importance = self.models['lightgbm'].feature_importance(importance_type='gain')
        self.feature_importance['lightgbm'] = {
            self.feature_names[i]: imp for i, imp in enumerate(importance)
        }
        
        return np.mean(scores)
    
    def _train_catboost(self, X: np.ndarray, y: np.ndarray,
                       sample_weights: Optional[np.ndarray], cv) -> float:
        """Train CatBoost model"""
        scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if sample_weights is not None:
                w_train = sample_weights[train_idx]
            else:
                w_train = None
            
            # Create pool
            train_pool = cb.Pool(X_train, y_train, weight=w_train)
            val_pool = cb.Pool(X_val, y_val)
            
            # Train model
            model = cb.CatBoostRegressor(**self.config.cat_params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=False
            )
            
            # Evaluate
            pred = model.predict(X_val)
            score = np.sqrt(np.mean((pred - y_val) ** 2))
            scores.append(score)
        
        # Train final model
        train_pool_all = cb.Pool(X, y, weight=sample_weights)
        self.models['catboost'] = cb.CatBoostRegressor(**self.config.cat_params)
        self.models['catboost'].fit(train_pool_all, verbose=False)
        
        # Store feature importance
        importance = self.models['catboost'].feature_importances_
        self.feature_importance['catboost'] = {
            self.feature_names[i]: imp for i, imp in enumerate(importance)
        }
        
        return np.mean(scores)
    
    def _train_deep_model(self, X: np.ndarray, y: np.ndarray, 
                         model_class, sample_weights: Optional[np.ndarray], cv) -> float:
        """Train deep learning model"""
        scores = []
        
        # Prepare sequences (assuming last dimension is time)
        # For now, treat features as sequence
        seq_len = min(20, X.shape[1] // 10)  # Use subset of features as sequence
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Reshape for sequence models
            X_train_seq = self._create_sequences(X_train, seq_len)
            X_val_seq = self._create_sequences(X_val, seq_len)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_seq),
                torch.FloatTensor(y_train.reshape(-1, 1))
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq),
                torch.FloatTensor(y_val.reshape(-1, 1))
            )
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256)
            
            # Initialize model
            input_size = X_train_seq.shape[-1]
            if model_class == CNNLSTM:
                model = model_class(input_size, seq_len).to(device)
            else:
                model = model_class(input_size).to(device)
            
            # Train
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(100):
                # Training
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    if sample_weights is not None:
                        # Apply sample weights
                        batch_weights = torch.FloatTensor(
                            sample_weights[train_idx][::len(train_idx)//len(batch_y)][:len(batch_y)]
                        ).to(device)
                        loss = (loss * batch_weights).mean()
                    
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
                    if patience_counter >= 20:
                        break
            
            scores.append(np.sqrt(best_val_loss))
        
        # Train final model
        X_seq_all = self._create_sequences(X, seq_len)
        train_dataset_all = TensorDataset(
            torch.FloatTensor(X_seq_all),
            torch.FloatTensor(y.reshape(-1, 1))
        )
        train_loader_all = DataLoader(train_dataset_all, batch_size=256, shuffle=True)
        
        # Initialize and train final model
        input_size = X_seq_all.shape[-1]
        if model_class == CNNLSTM:
            final_model = model_class(input_size, seq_len).to(device)
        else:
            final_model = model_class(input_size).to(device)
        
        optimizer = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(50):
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
        model_name = f'deep_{model_class.__name__.lower()}'
        self.models[model_name] = final_model
        
        return np.mean(scores)
    
    def _create_sequences(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        """Create sequences from features"""
        # Simple approach: reshape features into sequences
        n_samples, n_features = X.shape
        n_seq_features = n_features // seq_len
        
        # Truncate features to fit sequence length
        X_truncated = X[:, :n_seq_features * seq_len]
        
        # Reshape to (samples, sequence_length, features)
        X_seq = X_truncated.reshape(n_samples, seq_len, n_seq_features)
        
        return X_seq
    
    def _predict_deep_model(self, model, X: np.ndarray) -> np.ndarray:
        """Generate predictions from deep model"""
        model.eval()
        
        # Get sequence length from model
        if hasattr(model, 'lstm'):
            seq_len = 20  # Default sequence length
        else:
            seq_len = 20
        
        # Create sequences
        X_seq = self._create_sequences(X, seq_len)
        
        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(X_seq))
        loader = DataLoader(dataset, batch_size=256)
        
        predictions = []
        with torch.no_grad():
            for batch_X, in loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions).flatten()
    
    def _train_meta_learner(self, X: np.ndarray, y: np.ndarray,
                           sample_weights: Optional[np.ndarray], cv):
        """Train meta-learner for stacking"""
        # Get out-of-fold predictions from base models
        meta_features = []
        
        for train_idx, val_idx in cv.split(X):
            X_val = X[val_idx]
            fold_predictions = []
            
            for model_name in self.models:
                if 'deep' in model_name:
                    pred = self._predict_deep_model(self.models[model_name], 
                                                   self._scale_features(X_val, fit=False))
                else:
                    if hasattr(self.models[model_name], 'predict'):
                        if model_name == 'xgboost':
                            dval = xgb.DMatrix(X_val)
                            pred = self.models[model_name].predict(dval)
                        else:
                            pred = self.models[model_name].predict(X_val)
                    else:
                        pred = np.zeros(len(X_val))  # Placeholder
                
                fold_predictions.append(pred)
            
            meta_features.append(np.column_stack(fold_predictions))
        
        # Train meta-learner
        if len(meta_features) > 0:
            meta_X = np.vstack(meta_features)
            meta_y = y[cv.get_n_splits() * len(y) // (cv.get_n_splits() + 1):]  # Approximate
            
            # Use simple linear regression as meta-learner
            from sklearn.linear_model import RidgeCV
            self.meta_learner = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
            
            if len(meta_X) == len(meta_y):
                self.meta_learner.fit(meta_X, meta_y)
    
    def _calculate_model_weights(self, scores: Dict[str, float]):
        """Calculate model weights based on performance"""
        # Invert scores (lower is better for RMSE)
        inv_scores = {k: 1.0 / (v + 1e-6) for k, v in scores.items()}
        
        # Normalize to sum to 1
        total = sum(inv_scores.values())
        self.model_weights = {k: v / total for k, v in inv_scores.items()}
        
        logger.info(f"Model weights: {self.model_weights}")
    
    def calibrate_predictions(self, X: np.ndarray, y: np.ndarray):
        """Calibrate model predictions using isotonic regression"""
        for model_name in self.models:
            if 'deep' in model_name:
                pred = self._predict_deep_model(self.models[model_name], 
                                               self._scale_features(X, fit=False))
            else:
                if model_name == 'xgboost':
                    dval = xgb.DMatrix(X)
                    pred = self.models[model_name].predict(dval)
                else:
                    pred = self.models[model_name].predict(X)
            
            # Fit isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(pred, y)
            self.calibrators[model_name] = iso_reg
    
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
            lambda x: x / x.sum()
        )
        
        # Aggregate across models
        aggregated = importance_df.groupby('feature')['importance_norm'].agg(['mean', 'std', 'count'])
        aggregated = aggregated.sort_values('mean', ascending=False).head(top_n)
        
        return aggregated
    
    def save_models(self, path: str):
        """Save all models to disk"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save tree-based models
        for name in ['xgboost', 'lightgbm', 'catboost']:
            if name in self.models:
                if name == 'xgboost':
                    self.models[name].save_model(f"{path}/{name}_model.json")
                else:
                    joblib.dump(self.models[name], f"{path}/{name}_model.pkl")
        
        # Save deep learning models
        for name, model in self.models.items():
            if 'deep' in name:
                torch.save(model.state_dict(), f"{path}/{name}_model.pth")
        
        # Save scalers and calibrators
        joblib.dump(self.scalers, f"{path}/scalers.pkl")
        joblib.dump(self.calibrators, f"{path}/calibrators.pkl")
        joblib.dump(self.feature_importance, f"{path}/feature_importance.pkl")
        joblib.dump(self.model_weights, f"{path}/model_weights.pkl")
        
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, f"{path}/meta_learner.pkl")
    
    def load_models(self, path: str):
        """Load models from disk"""
        # Load tree-based models
        for name in ['xgboost', 'lightgbm', 'catboost']:
            model_path = f"{path}/{name}_model"
            if name == 'xgboost' and os.path.exists(f"{model_path}.json"):
                self.models[name] = xgb.Booster()
                self.models[name].load_model(f"{model_path}.json")
            elif os.path.exists(f"{model_path}.pkl"):
                self.models[name] = joblib.load(f"{model_path}.pkl")
        
        # Load deep learning models
        # Note: You'll need to initialize the models with correct architecture first
        
        # Load auxiliary objects
        if os.path.exists(f"{path}/scalers.pkl"):
            self.scalers = joblib.load(f"{path}/scalers.pkl")
        if os.path.exists(f"{path}/calibrators.pkl"):
            self.calibrators = joblib.load(f"{path}/calibrators.pkl")
        if os.path.exists(f"{path}/feature_importance.pkl"):
            self.feature_importance = joblib.load(f"{path}/feature_importance.pkl")
        if os.path.exists(f"{path}/model_weights.pkl"):
            self.model_weights = joblib.load(f"{path}/model_weights.pkl")
        if os.path.exists(f"{path}/meta_learner.pkl"):
            self.meta_learner = joblib.load(f"{path}/meta_learner.pkl")


# Hyperparameter optimization
class ModelOptimizer:
    """Optimize hyperparameters using Optuna"""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.best_params = None
        
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                n_trials: int = 100) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        def objective(trial):
            if self.model_type == 'xgboost':
                params = {
                    'tree_method': 'gpu_hist',
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
                    'device': 'gpu',
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
            tscv = TimeSeriesSplit(n_splits=3)
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if self.model_type == 'xgboost':
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    
                    model = xgb.train(
                        params, dtrain,
                        num_boost_round=params['n_estimators'],
                        evals=[(dval, 'eval')],
                        early_stopping_rounds=50,
                        verbose_eval=False
                    )
                    pred = model.predict(dval)
                
                elif self.model_type == 'lightgbm':
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val)
                    
                    model = lgb.train(
                        params, train_data,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                    pred = model.predict(X_val)
                
                score = np.sqrt(np.mean((pred - y_val) ** 2))
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        self.best_params = study.best_params
        return self.best_params


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 5000
    n_features = 150
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :10], axis=1) + np.random.randn(n_samples) * 0.1
    
    # Add some temporal structure
    y = pd.Series(y).rolling(10).mean().fillna(method='bfill').values
    
    # Create sample weights with time decay
    sample_weights = np.exp(-np.arange(n_samples)[::-1] / 1000)
    
    # Initialize ensemble
    config = ModelConfig()
    ensemble = EnsembleModel(config)
    
    # Train ensemble
    print("Training ensemble model...")
    scores = ensemble.train(X, y, sample_weights)
    print(f"\nModel scores: {scores}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = ensemble.predict(X[-100:])
    print(f"Predictions shape: {predictions.shape}")
    
    # Get quantile predictions
    quantiles = ensemble.predict_proba(X[-100:], quantiles=[0.1, 0.5, 0.9])
    print(f"Quantile predictions shape: {quantiles.shape}")
    
    # Get feature importance
    if ensemble.feature_importance:
        importance_df = ensemble.get_feature_importance(top_n=20)
        print(f"\nTop 20 features:")
        print(importance_df)
    
    # Optimize hyperparameters (example)
    print("\nOptimizing XGBoost hyperparameters...")
    optimizer = ModelOptimizer('xgboost')
    best_params = optimizer.optimize(X[:1000], y[:1000], n_trials=10)
    print(f"Best parameters: {best_params}")
