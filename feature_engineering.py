"""
Feature Engineering for ML Trading System - Working Version
This version properly handles pandas/numpy operations
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
    """Feature engineering that works with yfinance data"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names = []
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using pandas operations"""
        features = pd.DataFrame(index=data.index)
        
        # Make sure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required):
            logger.error(f"Missing required columns. Have: {data.columns.tolist()}")
            return features
        
        try:
            # Price features
            logger.info("Generating price-based features...")
            
            # Returns - using pandas
            for period in self.config.return_periods:
                features[f'return_{period}d'] = data['Close'].pct_change(period)
                features[f'log_return_{period}d'] = np.log(data['Close'] / data['Close'].shift(period))
            
            # Moving averages - using pandas
            for period in self.config.ma_periods:
                features[f'sma_{period}'] = data['Close'].rolling(window=period).mean()
                features[f'price_to_sma_{period}'] = data['Close'] / features[f'sma_{period}']
                features[f'sma_{period}_slope'] = features[f'sma_{period}'].diff()
            
            # Price ratios
            features['high_low_ratio'] = data['High'] / (data['Low'] + 1e-10)
            features['close_open_ratio'] = data['Close'] / (data['Open'] + 1e-10)
            features['daily_range'] = (data['High'] - data['Low']) / (data['Close'] + 1e-10)
            features['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
            
            # Gaps
            features['gap_up'] = (data['Open'] - data['Close'].shift(1)) / (data['Close'].shift(1) + 1e-10)
            features['gap_size'] = np.abs(features['gap_up'])
            
            # Volume features
            logger.info("Generating volume features...")
            
            features['volume'] = data['Volume']
            features['log_volume'] = np.log(data['Volume'] + 1)
            features['dollar_volume'] = data['Close'] * data['Volume']
            
            # Volume averages - using pandas
            for period in self.config.volume_ma_periods:
                features[f'volume_sma_{period}'] = data['Volume'].rolling(window=period).mean()
                features[f'volume_ratio_{period}'] = data['Volume'] / (features[f'volume_sma_{period}'] + 1e-10)
            
            # OBV - simplified
            features['obv'] = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
            
            # Technical indicators
            logger.info("Generating technical indicators...")
            
            # RSI - using pandas
            for period in self.config.rsi_periods:
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / (loss + 1e-10)
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands - using pandas
            for period in self.config.bb_periods:
                sma = data['Close'].rolling(window=period).mean()
                std = data['Close'].rolling(window=period).std()
                features[f'bb_upper_{period}'] = sma + (2 * std)
                features[f'bb_lower_{period}'] = sma - (2 * std)
                features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / (sma + 1e-10)
                features[f'bb_position_{period}'] = (data['Close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 1e-10)
            
            # MACD - using pandas
            for fast, slow, signal in self.config.macd_params:
                ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
                ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
                features[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
                features[f'macd_signal_{fast}_{slow}'] = features[f'macd_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
                features[f'macd_hist_{fast}_{slow}'] = features[f'macd_{fast}_{slow}'] - features[f'macd_signal_{fast}_{slow}']
            
            # Volatility features
            logger.info("Generating volatility features...")
            
            returns = data['Close'].pct_change()
            for period in [5, 10, 20]:
                features[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            
            # ATR simplified
            for period in self.config.atr_periods:
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift(1))
                low_close = np.abs(data['Low'] - data['Close'].shift(1))
                
                true_range = pd.DataFrame({
                    'hl': high_low,
                    'hc': high_close,
                    'lc': low_close
                }).max(axis=1)
                
                features[f'atr_{period}'] = true_range.rolling(window=period).mean()
                features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / (data['Close'] + 1e-10)
            
            # Pattern features
            logger.info("Generating pattern features...")
            
            features['body_size'] = np.abs(data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-10)
            features['upper_shadow'] = (data['High'] - np.maximum(data['Close'], data['Open'])) / (data['High'] - data['Low'] + 1e-10)
            features['lower_shadow'] = (np.minimum(data['Close'], data['Open']) - data['Low']) / (data['High'] - data['Low'] + 1e-10)
            
            # Trend patterns
            features['higher_highs'] = ((data['High'] > data['High'].shift(1)) & 
                                       (data['High'].shift(1) > data['High'].shift(2))).astype(int)
            features['lower_lows'] = ((data['Low'] < data['Low'].shift(1)) & 
                                     (data['Low'].shift(1) < data['Low'].shift(2))).astype(int)
            
            # Price channels
            for period in [10, 20]:
                features[f'high_{period}d'] = data['High'].rolling(window=period).max()
                features[f'low_{period}d'] = data['Low'].rolling(window=period).min()
                features[f'channel_pos_{period}'] = (data['Close'] - features[f'low_{period}d']) / (features[f'high_{period}d'] - features[f'low_{period}d'] + 1e-10)
            
            # Microstructure features
            if self.config.use_microstructure:
                logger.info("Generating microstructure features...")
                
                features['hl_spread'] = (data['High'] - data['Low']) / (data['Close'] + 1e-10)
                features['co_spread'] = np.abs(data['Close'] - data['Open']) / (data['Close'] + 1e-10)
                features['turnover'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1e-10)
                
                # Simple return autocorrelation
                features['return_autocorr'] = returns.rolling(20).apply(lambda x: x.autocorr() if len(x) > 1 else 0)
            
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
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, method: str = 'mutual_info') -> pd.DataFrame:
        """Basic feature selection"""
        return features  # Return all features for now


# Quick test
if __name__ == "__main__":
    print("Feature engineering module loaded successfully!")
    
    # Test with dummy data
    dates = pd.date_range('2023-01-01', periods=100)
    test_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 101,
        'Low': np.random.randn(100).cumsum() + 99,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    engineer = EnhancedFeatureEngineer()
    features = engineer.engineer_features(test_data)
    print(f"Test successful! Generated {len(features.columns)} features")
