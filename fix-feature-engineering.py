"""
Fixed feature engineering to handle yfinance data correctly
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    return_periods: List[int] = None
    ma_periods: List[int] = None
    n_features: int = 150
    use_microstructure: bool = True
    rsi_periods: List[int] = None
    bb_periods: List[int] = None
    atr_periods: List[int] = None
    volume_ma_periods: List[int] = None
    macd_params: List[tuple] = None
    
    def __post_init__(self):
        if self.return_periods is None:
            self.return_periods = [1, 2, 3, 5, 10, 20]
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50]
        if self.rsi_periods is None:
            self.rsi_periods = [14, 21]
        if self.bb_periods is None:
            self.bb_periods = [20]
        if self.atr_periods is None:
            self.atr_periods = [14]
        if self.volume_ma_periods is None:
            self.volume_ma_periods = [5, 10, 20]
        if self.macd_params is None:
            self.macd_params = [(12, 26, 9)]

class EnhancedFeatureEngineer:
    """Feature engineering that properly handles yfinance data"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names = []
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using pandas operations - FIXED"""
        features = pd.DataFrame(index=data.index)
        
        # FIX: Handle multi-column data from yfinance download
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-index columns
            data = data.droplevel(1, axis=1)
        
        # Ensure we have single-column series for each price type
        if 'Close' in data.columns:
            close_prices = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close'].squeeze()
            open_prices = data['Open'] if isinstance(data['Open'], pd.Series) else data['Open'].squeeze()
            high_prices = data['High'] if isinstance(data['High'], pd.Series) else data['High'].squeeze()
            low_prices = data['Low'] if isinstance(data['Low'], pd.Series) else data['Low'].squeeze()
            volume = data['Volume'] if isinstance(data['Volume'], pd.Series) else data['Volume'].squeeze()
        else:
            logger.error(f"Missing required columns. Have: {data.columns.tolist()}")
            return features
        
        try:
            # Price features
            logger.info("Generating price-based features...")
            
            # Returns
            for period in self.config.return_periods:
                features[f'return_{period}d'] = close_prices.pct_change(period)
                features[f'log_return_{period}d'] = np.log(close_prices / close_prices.shift(period))
            
            # Moving averages
            for period in self.config.ma_periods:
                sma = close_prices.rolling(window=period).mean()
                features[f'sma_{period}'] = sma
                features[f'price_to_sma_{period}'] = close_prices / sma
                features[f'sma_{period}_slope'] = sma.diff()
            
            # Price ratios
            features['high_low_ratio'] = high_prices / (low_prices + 1e-10)
            features['close_open_ratio'] = close_prices / (open_prices + 1e-10)
            features['daily_range'] = (high_prices - low_prices) / (close_prices + 1e-10)
            features['close_position'] = (close_prices - low_prices) / (high_prices - low_prices + 1e-10)
            
            # Gaps
            features['gap_up'] = (open_prices - close_prices.shift(1)) / (close_prices.shift(1) + 1e-10)
            features['gap_size'] = np.abs(features['gap_up'])
            
            # Volume features
            logger.info("Generating volume features...")
            
            features['volume'] = volume
            features['log_volume'] = np.log(volume + 1)
            features['dollar_volume'] = close_prices * volume
            
            # Volume averages
            for period in self.config.volume_ma_periods:
                vol_sma = volume.rolling(window=period).mean()
                features[f'volume_sma_{period}'] = vol_sma
                features[f'volume_ratio_{period}'] = volume / (vol_sma + 1e-10)
            
            # OBV
            features['obv'] = (np.sign(close_prices.diff()) * volume).cumsum()
            
            # Technical indicators
            logger.info("Generating technical indicators...")
            
            # RSI
            for period in self.config.rsi_periods:
                delta = close_prices.diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / (loss + 1e-10)
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            for period in self.config.bb_periods:
                sma = close_prices.rolling(window=period).mean()
                std = close_prices.rolling(window=period).std()
                features[f'bb_upper_{period}'] = sma + (2 * std)
                features[f'bb_lower_{period}'] = sma - (2 * std)
                features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / (sma + 1e-10)
                features[f'bb_position_{period}'] = (close_prices - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 1e-10)
            
            # MACD
            for fast, slow, signal in self.config.macd_params:
                ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
                ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal, adjust=False).mean()
                features[f'macd_{fast}_{slow}'] = macd_line
                features[f'macd_signal_{fast}_{slow}'] = signal_line
                features[f'macd_hist_{fast}_{slow}'] = macd_line - signal_line
            
            # Volatility features
            logger.info("Generating volatility features...")
            
            returns = close_prices.pct_change()
            for period in [5, 10, 20]:
                features[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            
            # ATR
            for period in self.config.atr_periods:
                high_low = high_prices - low_prices
                high_close = np.abs(high_prices - close_prices.shift(1))
                low_close = np.abs(low_prices - close_prices.shift(1))
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                features[f'atr_{period}'] = true_range.rolling(window=period).mean()
                features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / (close_prices + 1e-10)
            
            # Pattern features
            logger.info("Generating pattern features...")
            
            features['body_size'] = np.abs(close_prices - open_prices) / (high_prices - low_prices + 1e-10)
            features['upper_shadow'] = (high_prices - np.maximum(close_prices, open_prices)) / (high_prices - low_prices + 1e-10)
            features['lower_shadow'] = (np.minimum(close_prices, open_prices) - low_prices) / (high_prices - low_prices + 1e-10)
            
            # Trend patterns
            features['higher_highs'] = ((high_prices > high_prices.shift(1)) & 
                                       (high_prices.shift(1) > high_prices.shift(2))).astype(int)
            features['lower_lows'] = ((low_prices < low_prices.shift(1)) & 
                                     (low_prices.shift(1) < low_prices.shift(2))).astype(int)
            
            # Price channels
            for period in [10, 20]:
                features[f'high_{period}d'] = high_prices.rolling(window=period).max()
                features[f'low_{period}d'] = low_prices.rolling(window=period).min()
                features[f'channel_pos_{period}'] = (close_prices - features[f'low_{period}d']) / (features[f'high_{period}d'] - features[f'low_{period}d'] + 1e-10)
            
            # Microstructure features
            if self.config.use_microstructure:
                logger.info("Generating microstructure features...")
                
                features['hl_spread'] = (high_prices - low_prices) / (close_prices + 1e-10)
                features['co_spread'] = np.abs(close_prices - open_prices) / (close_prices + 1e-10)
                features['turnover'] = volume / (volume.rolling(20).mean() + 1e-10)
                
                # Return autocorrelation
                features['return_autocorr'] = returns.rolling(20).apply(lambda x: x.autocorr() if len(x) > 1 else 0)
            
            # Additional features to reach target count
            logger.info("Generating additional features...")
            
            # More return-based features
            for period in [30, 60]:
                features[f'return_{period}d'] = close_prices.pct_change(period)
            
            # More volatility measures
            features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-10)
            features['volatility_change'] = features['volatility_20'].diff()
            
            # Volume-price features
            features['volume_price_corr'] = close_prices.rolling(20).corr(volume)
            features['money_flow'] = ((close_prices - low_prices) - (high_prices - close_prices)) / (high_prices - low_prices + 1e-10) * volume
            
            # More price position features
            for period in [5, 10, 20, 50]:
                highest = high_prices.rolling(period).max()
                lowest = low_prices.rolling(period).min()
                features[f'price_position_{period}'] = (close_prices - lowest) / (highest - lowest + 1e-10)
            
            # Rate of change
            for period in [5, 10, 20]:
                features[f'roc_{period}'] = (close_prices - close_prices.shift(period)) / close_prices.shift(period)
            
            # Efficiency ratio
            for period in [10, 20]:
                change = np.abs(close_prices - close_prices.shift(period))
                volatility = np.abs(close_prices.diff()).rolling(period).sum()
                features[f'efficiency_ratio_{period}'] = change / (volatility + 1e-10)
            
        except Exception as e:
            logger.error(f"Error in feature generation: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(method='ffill', limit=5)
        features = features.fillna(0)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Generated {len(features.columns)} features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names


# Fixed data preparation function
def prepare_training_data_fixed(feature_data, market_data, prediction_horizon=5):
    """
    Fixed version of prepare_training_data that properly handles the data
    """
    all_X = []
    all_y = []
    all_dates = []
    
    logger.info("\nPreparing training data (FIXED)...")
    
    for symbol in feature_data:
        if symbol not in market_data:
            continue
            
        features = feature_data[symbol]
        data = market_data[symbol]
        
        # Handle multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
        
        # Get close prices
        close_prices = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close'].squeeze()
        
        # Align dates
        common_dates = features.index.intersection(close_prices.index)
        features_aligned = features.loc[common_dates]
        prices_aligned = close_prices.loc[common_dates]
        
        logger.info(f"\n{symbol}:")
        logger.info(f"  Original features: {len(features)}")
        logger.info(f"  After alignment: {len(features_aligned)}")
        
        # Calculate forward returns
        forward_returns = prices_aligned.pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Remove last prediction_horizon days
        features_clean = features_aligned.iloc[:-prediction_horizon]
        returns_clean = forward_returns.iloc[:-prediction_horizon]
        dates_clean = common_dates[:-prediction_horizon]
        
        logger.info(f"  After removing future: {len(features_clean)}")
        
        # Create mask for NaN removal
        feature_nan_mask = ~features_clean.isna().any(axis=1)
        return_nan_mask = ~returns_clean.isna()
        combined_mask = feature_nan_mask & return_nan_mask
        
        valid_samples = combined_mask.sum()
        logger.info(f"  After NaN removal: {valid_samples}")
        logger.info(f"  Percentage kept: {valid_samples / len(features) * 100:.1f}%")
        
        if valid_samples > 0:
            all_X.append(features_clean[combined_mask].values)
            all_y.append(returns_clean[combined_mask].values)
            all_dates.append(dates_clean[combined_mask])
    
    # Combine all data
    if all_X:
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        dates = np.hstack(all_dates)
        
        # Sort by date
        sort_idx = np.argsort(dates)
        X = X[sort_idx]
        y = y[sort_idx]
        dates = dates[sort_idx]
        
        logger.info(f"\nFinal combined data:")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  Date range: {dates[0]} to {dates[-1]}")
        logger.info(f"  Memory size: {X.nbytes / 1024 / 1024:.2f} MB")
    else:
        X, y, dates = np.array([]).reshape(0, 0), np.array([]), np.array([])
        logger.warning("No valid training data after processing!")
    
    return X, y, dates


# Test the fixes
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing fixed feature engineering...")
    print("=" * 60)
    
    # Test with proper yfinance data
    symbol = 'AAPL'
    start_date = '2020-07-12'
    end_date = '2025-07-12'
    
    # Method 1: Using Ticker (recommended)
    print(f"\n1. Testing with yf.Ticker (recommended method):")
    ticker = yf.Ticker(symbol)
    data_ticker = ticker.history(start=start_date, end=end_date)
    print(f"Data shape: {data_ticker.shape}")
    print(f"Columns: {data_ticker.columns.tolist()}")
    
    # Method 2: Using download (can cause issues)
    print(f"\n2. Testing with yf.download:")
    data_download = yf.download(symbol, start=start_date, end=end_date, progress=False)
    print(f"Data shape: {data_download.shape}")
    print(f"Columns: {data_download.columns.tolist()}")
    
    # Test feature engineering with both
    engineer = EnhancedFeatureEngineer()
    
    print(f"\n3. Feature engineering with Ticker data:")
    features_ticker = engineer.engineer_features(data_ticker)
    print(f"Features shape: {features_ticker.shape}")
    print(f"First 5 features: {features_ticker.columns[:5].tolist()}")
    
    print(f"\n4. Feature engineering with download data:")
    features_download = engineer.engineer_features(data_download)
    print(f"Features shape: {features_download.shape}")
    
    # Test data preparation
    print(f"\n5. Testing data preparation:")
    feature_data = {symbol: features_ticker}
    market_data = {symbol: data_ticker}
    
    X, y, dates = prepare_training_data_fixed(feature_data, market_data)
    
    print(f"\nFinal results:")
    print(f"  Training samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1] if len(X) > 0 else 0}")
    print(f"  Target statistics: mean={np.mean(y):.4f}, std={np.std(y):.4f}" if len(y) > 0 else "  No data")
