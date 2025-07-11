"""
Enhanced Feature Engineering System
Comprehensive feature extraction with 30+ categories for ML trading
"""

import numpy as np
import pandas as pd
import talib
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Price-based
    return_periods: List[int] = None
    ma_periods: List[int] = None
    ema_periods: List[int] = None
    
    # Volume
    volume_ma_periods: List[int] = None
    
    # Volatility
    atr_periods: List[int] = None
    bb_periods: List[int] = None
    bb_stds: List[float] = None
    vol_lookbacks: List[int] = None
    
    # Technical indicators
    rsi_periods: List[int] = None
    macd_params: List[Tuple[int, int, int]] = None
    stoch_params: List[Tuple[int, int]] = None
    
    # Pattern recognition
    pattern_lookback: int = 20
    
    # Statistical
    stat_windows: List[int] = None
    
    def __post_init__(self):
        # Set defaults if not provided
        if self.return_periods is None:
            self.return_periods = [1, 2, 3, 5, 10, 20, 60, 120]
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 100, 200]
        if self.ema_periods is None:
            self.ema_periods = [8, 12, 21, 26, 50]
        if self.volume_ma_periods is None:
            self.volume_ma_periods = [5, 10, 20, 50]
        if self.atr_periods is None:
            self.atr_periods = [5, 10, 14, 20]
        if self.bb_periods is None:
            self.bb_periods = [10, 20, 30]
        if self.bb_stds is None:
            self.bb_stds = [1.5, 2.0, 2.5]
        if self.vol_lookbacks is None:
            self.vol_lookbacks = [5, 10, 20, 30, 60]
        if self.rsi_periods is None:
            self.rsi_periods = [7, 14, 21, 28]
        if self.macd_params is None:
            self.macd_params = [(12, 26, 9), (5, 35, 5), (8, 17, 9)]
        if self.stoch_params is None:
            self.stoch_params = [(14, 3), (21, 5), (5, 3)]
        if self.stat_windows is None:
            self.stat_windows = [5, 10, 20, 50]

class EnhancedFeatureEngineer:
    """Professional feature engineering with 30+ feature categories"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names = []
        self.scaler = RobustScaler()
        
    def engineer_features(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Generate all features for a single symbol"""
        features = pd.DataFrame(index=data.index)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # 1. Price-based features
        logger.info("Generating price-based features...")
        features = pd.concat([features, self._price_features(data)], axis=1)
        
        # 2. Volume features
        logger.info("Generating volume features...")
        features = pd.concat([features, self._volume_features(data)], axis=1)
        
        # 3. Volatility features
        logger.info("Generating volatility features...")
        features = pd.concat([features, self._volatility_features(data)], axis=1)
        
        # 4. Technical indicators
        logger.info("Generating technical indicators...")
        features = pd.concat([features, self._technical_indicators(data)], axis=1)
        
        # 5. Market microstructure
        logger.info("Generating microstructure features...")
        features = pd.concat([features, self._microstructure_features(data)], axis=1)
        
        # 6. Pattern recognition
        logger.info("Generating pattern features...")
        features = pd.concat([features, self._pattern_features(data)], axis=1)
        
        # 7. Statistical features
        logger.info("Generating statistical features...")
        features = pd.concat([features, self._statistical_features(data)], axis=1)
        
        # 8. Interaction features
        logger.info("Generating interaction features...")
        features = pd.concat([features, self._interaction_features(data, features)], axis=1)
        
        # 9. ML-discovered features
        logger.info("Generating ML-discovered features...")
        features = pd.concat([features, self._ml_discovered_features(data, features)], axis=1)
        
        # 10. Cross-asset features (if market data available)
        # features = pd.concat([features, self._cross_asset_features(data)], axis=1)
        
        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill', limit=2).fillna(0)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Scale features if requested
        if fit_scaler:
            features_scaled = pd.DataFrame(
                self.scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
            return features_scaled
        
        return features
    
    def _price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features"""
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        high = data['High']
        low = data['Low']
        open_ = data['Open']
        
        # Returns at multiple timeframes
        for period in self.config.return_periods:
            features[f'return_{period}d'] = close.pct_change(period)
            features[f'log_return_{period}d'] = np.log(close / close.shift(period))
        
        # Moving averages
        for period in self.config.ma_periods:
            ma = close.rolling(period).mean()
            features[f'sma_{period}'] = ma
            features[f'price_to_sma_{period}'] = close / ma - 1
        
        # Exponential moving averages
        for period in self.config.ema_periods:
            ema = close.ewm(span=period).mean()
            features[f'ema_{period}'] = ema
            features[f'price_to_ema_{period}'] = close / ema - 1
        
        # VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
        features['vwap'] = vwap
        features['price_to_vwap'] = close / vwap - 1
        
        # Support and Resistance levels
        for lookback in [10, 20, 50, 100]:
            features[f'resistance_{lookback}d'] = high.rolling(lookback).max()
            features[f'support_{lookback}d'] = low.rolling(lookback).min()
            features[f'price_to_resistance_{lookback}d'] = close / features[f'resistance_{lookback}d'] - 1
            features[f'price_to_support_{lookback}d'] = close / features[f'support_{lookback}d'] - 1
        
        # Fibonacci retracements
        for lookback in [20, 50]:
            swing_high = high.rolling(lookback).max()
            swing_low = low.rolling(lookback).min()
            diff = swing_high - swing_low
            
            for fib_level, fib_pct in [(0.236, '236'), (0.382, '382'), (0.5, '50'), (0.618, '618'), (0.786, '786')]:
                fib_price = swing_low + diff * fib_level
                features[f'fib_{fib_pct}_{lookback}d'] = fib_price
                features[f'price_to_fib_{fib_pct}_{lookback}d'] = close / fib_price - 1
        
        # Price channels
        for period in [10, 20, 50]:
            upper_channel = high.rolling(period).max()
            lower_channel = low.rolling(period).min()
            channel_width = upper_channel - lower_channel
            features[f'channel_position_{period}d'] = (close - lower_channel) / channel_width
            features[f'channel_width_{period}d'] = channel_width / close
        
        # High/Low/Close relationships
        features['high_low_spread'] = (high - low) / close
        features['close_to_high'] = (close - high) / close
        features['close_to_low'] = (close - low) / close
        features['body_size'] = abs(close - open_) / close
        
        return features
    
    def _volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features"""
        features = pd.DataFrame(index=data.index)
        volume = data['Volume']
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Volume moving averages and ratios
        for period in self.config.volume_ma_periods:
            vol_ma = volume.rolling(period).mean()
            features[f'volume_ma_{period}'] = vol_ma
            features[f'volume_ratio_{period}'] = volume / vol_ma
        
        # On-Balance Volume (OBV)
        obv = talib.OBV(close.values, volume.values)
        features['obv'] = obv
        features['obv_ma_20'] = pd.Series(obv).rolling(20).mean()
        features['obv_divergence'] = features['obv'] / features['obv_ma_20'] - 1
        
        # Accumulation/Distribution
        ad = talib.AD(high.values, low.values, close.values, volume.values)
        features['acc_dist'] = ad
        features['acc_dist_ma_20'] = pd.Series(ad).rolling(20).mean()
        features['acc_dist_signal'] = features['acc_dist'] / features['acc_dist_ma_20'] - 1
        
        # Chaikin Money Flow
        for period in [10, 20]:
            mfm = ((close - low) - (high - close)) / (high - low)
            mfv = mfm * volume
            cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
            features[f'cmf_{period}'] = cmf
        
        # Volume Price Trend
        vpt = volume * ((close - close.shift(1)) / close.shift(1))
        features['vpt'] = vpt.cumsum()
        features['vpt_ma_20'] = features['vpt'].rolling(20).mean()
        
        # Money Flow Index variations
        for period in [10, 14, 20]:
            mfi = talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period)
            features[f'mfi_{period}'] = mfi
        
        # Volume profile analysis
        for period in [20, 50]:
            # Volume-weighted average price over period
            vwap_period = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
            features[f'vwap_{period}d'] = vwap_period
            
            # Volume at price levels
            price_bins = pd.qcut(close.rolling(period).apply(lambda x: x.iloc[-1] if len(x) == period else np.nan), 
                                q=5, duplicates='drop')
            vol_profile = volume.groupby(price_bins).sum()
            features[f'volume_concentration_{period}d'] = volume.rolling(period).apply(
                lambda x: stats.entropy(x / x.sum()) if x.sum() > 0 else 0
            )
        
        # Volume-weighted momentum
        for period in [5, 10, 20]:
            vol_weight = volume / volume.rolling(period).sum()
            vol_momentum = (close.pct_change() * vol_weight).rolling(period).sum()
            features[f'volume_momentum_{period}d'] = vol_momentum
        
        return features
    
    def _volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility features"""
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        high = data['High']
        low = data['Low']
        open_ = data['Open']
        
        # ATR at multiple timeframes
        for period in self.config.atr_periods:
            atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            features[f'atr_{period}'] = atr
            features[f'atr_pct_{period}'] = atr / close * 100
        
        # Bollinger Bands
        for period in self.config.bb_periods:
            for std in self.config.bb_stds:
                upper, middle, lower = talib.BBANDS(close.values, timeperiod=period, nbdevup=std, nbdevdn=std)
                features[f'bb_upper_{period}_{int(std*10)}'] = upper
                features[f'bb_lower_{period}_{int(std*10)}'] = lower
                features[f'bb_width_{period}_{int(std*10)}'] = (upper - lower) / middle
                features[f'bb_position_{period}_{int(std*10)}'] = (close - lower) / (upper - lower)
        
        # Keltner Channels
        for period in [10, 20]:
            for mult in [1.5, 2.0]:
                ema = close.ewm(span=period).mean()
                atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
                kc_upper = ema + mult * atr
                kc_lower = ema - mult * atr
                features[f'kc_position_{period}_{int(mult*10)}'] = (close - kc_lower) / (kc_upper - kc_lower)
                
                # Squeeze detection (BB inside KC)
                bb_upper, _, bb_lower = talib.BBANDS(close.values, timeperiod=period, nbdevup=2, nbdevdn=2)
                squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)
                features[f'squeeze_{period}_{int(mult*10)}'] = squeeze
        
        # Historical volatility
        for lookback in self.config.vol_lookbacks:
            returns = close.pct_change()
            features[f'volatility_{lookback}d'] = returns.rolling(lookback).std() * np.sqrt(252)
        
        # Advanced volatility estimators
        for period in [10, 20, 30]:
            # Parkinson volatility
            park_vol = np.sqrt(252 / (4 * np.log(2))) * (np.log(high / low)).rolling(period).std()
            features[f'parkinson_vol_{period}'] = park_vol
            
            # Garman-Klass volatility
            gk_vol = np.sqrt(252 / period) * np.sqrt(
                ((np.log(high / low) ** 2) / 2 - (2 * np.log(2) - 1) * (np.log(close / open_) ** 2)).rolling(period).sum()
            )
            features[f'garman_klass_vol_{period}'] = gk_vol
            
            # Rogers-Satchell volatility
            rs_vol = np.sqrt(252 / period) * np.sqrt(
                (np.log(high / close) * np.log(high / open_) + 
                 np.log(low / close) * np.log(low / open_)).rolling(period).sum()
            )
            features[f'rogers_satchell_vol_{period}'] = rs_vol
        
        # Volatility regime detection
        short_vol = returns.rolling(10).std() * np.sqrt(252)
        long_vol = returns.rolling(60).std() * np.sqrt(252)
        features['volatility_regime'] = short_vol / long_vol
        features['high_volatility'] = (short_vol > short_vol.rolling(252).quantile(0.75)).astype(int)
        
        # Volatility of volatility
        for period in [20, 50]:
            vol_series = returns.rolling(period).std()
            features[f'vol_of_vol_{period}'] = vol_series.rolling(period).std()
        
        return features
    
    def _technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators"""
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # RSI with divergence detection
        for period in self.config.rsi_periods:
            rsi = talib.RSI(close.values, timeperiod=period)
            features[f'rsi_{period}'] = rsi
            
            # RSI divergence
            rsi_series = pd.Series(rsi, index=data.index)
            price_peaks, _ = find_peaks(close.values, distance=period)
            rsi_peaks, _ = find_peaks(rsi, distance=period)
            
            # Simplified divergence detection
            features[f'rsi_divergence_{period}'] = 0  # Placeholder for complex divergence logic
        
        # MACD variations
        for fast, slow, signal in self.config.macd_params:
            macd, macd_signal, macd_hist = talib.MACD(close.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            features[f'macd_{fast}_{slow}_{signal}'] = macd
            features[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            features[f'macd_hist_{fast}_{slow}_{signal}'] = macd_hist
            features[f'macd_hist_slope_{fast}_{slow}_{signal}'] = pd.Series(macd_hist).diff()
        
        # Stochastic oscillators
        for period, smooth in self.config.stoch_params:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values, 
                                       fastk_period=period, slowk_period=smooth, slowd_period=smooth)
            features[f'stoch_k_{period}_{smooth}'] = slowk
            features[f'stoch_d_{period}_{smooth}'] = slowd
            features[f'stoch_cross_{period}_{smooth}'] = slowk - slowd
        
        # Williams %R
        for period in [10, 14, 20]:
            willr = talib.WILLR(high.values, low.values, close.values, timeperiod=period)
            features[f'williams_r_{period}'] = willr
        
        # Commodity Channel Index
        for period in [14, 20]:
            cci = talib.CCI(high.values, low.values, close.values, timeperiod=period)
            features[f'cci_{period}'] = cci
        
        # Average Directional Index
        for period in [14, 20]:
            adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
            plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period)
            minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period)
            features[f'adx_{period}'] = adx
            features[f'plus_di_{period}'] = plus_di
            features[f'minus_di_{period}'] = minus_di
            features[f'di_diff_{period}'] = plus_di - minus_di
        
        # Aroon
        for period in [14, 25]:
            aroon_up, aroon_down = talib.AROON(high.values, low.values, timeperiod=period)
            features[f'aroon_up_{period}'] = aroon_up
            features[f'aroon_down_{period}'] = aroon_down
            features[f'aroon_osc_{period}'] = aroon_up - aroon_down
        
        # Ultimate Oscillator
        ultosc = talib.ULTOSC(high.values, low.values, close.values)
        features['ultimate_oscillator'] = ultosc
        
        # PPO (Percentage Price Oscillator)
        for fast, slow in [(12, 26), (5, 35)]:
            ppo = talib.PPO(close.values, fastperiod=fast, slowperiod=slow)
            features[f'ppo_{fast}_{slow}'] = ppo
        
        # TRIX
        for period in [14, 20]:
            trix = talib.TRIX(close.values, timeperiod=period)
            features[f'trix_{period}'] = trix
        
        # CMO (Chande Momentum Oscillator)
        for period in [14, 20]:
            cmo = talib.CMO(close.values, timeperiod=period)
            features[f'cmo_{period}'] = cmo
        
        return features
    
    def _microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market microstructure features"""
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        high = data['High']
        low = data['Low']
        open_ = data['Open']
        volume = data['Volume']
        
        # Bid-ask spread proxy (using high-low)
        features['spread_proxy'] = (high - low) / close
        features['spread_proxy_ma_20'] = features['spread_proxy'].rolling(20).mean()
        
        # Intraday momentum
        features['intraday_momentum'] = (close - open_) / open_
        features['intraday_momentum_ma_5'] = features['intraday_momentum'].rolling(5).mean()
        
        # Intraday volatility
        features['intraday_volatility'] = (high - low) / open_
        features['intraday_vol_ma_20'] = features['intraday_volatility'].rolling(20).mean()
        
        # Order flow imbalance estimation
        # Estimate using price movement and volume
        price_change = close.pct_change()
        features['order_flow_imbalance'] = price_change * volume
        features['order_flow_ma_10'] = features['order_flow_imbalance'].rolling(10).mean()
        
        # Amihud illiquidity measure
        for period in [5, 10, 20]:
            returns_abs = np.abs(close.pct_change())
            dollar_volume = close * volume
            illiquidity = (returns_abs / dollar_volume).rolling(period).mean() * 1e6
            features[f'amihud_illiquidity_{period}'] = illiquidity
        
        # Kyle's lambda (simplified)
        for period in [10, 20]:
            # Estimate as price impact of volume
            returns = close.pct_change()
            volume_change = volume.pct_change()
            
            # Rolling regression coefficient
            def rolling_beta(x, y, window):
                result = []
                for i in range(len(x)):
                    if i < window:
                        result.append(np.nan)
                    else:
                        x_window = x[i-window:i]
                        y_window = y[i-window:i]
                        if np.std(x_window) > 0:
                            beta = np.cov(x_window, y_window)[0, 1] / np.var(x_window)
                        else:
                            beta = 0
                        result.append(beta)
                return pd.Series(result, index=x.index)
            
            kyle_lambda = rolling_beta(volume_change, returns, period)
            features[f'kyle_lambda_{period}'] = kyle_lambda
        
        # Microstructure noise ratio
        # Ratio of short-term to long-term volatility
        short_vol = close.pct_change().rolling(5).std()
        long_vol = close.pct_change().rolling(20).std()
        features['noise_ratio'] = short_vol / long_vol
        
        # Effective spread proxy
        # Using Roll's model
        returns = close.pct_change()
        features['roll_spread'] = 2 * np.sqrt(np.abs(returns.rolling(20).cov(returns.shift(1))))
        
        # Price efficiency measures
        # Variance ratio test statistic
        for short, long in [(5, 20), (10, 50)]:
            short_var = returns.rolling(short).var()
            long_var = returns.rolling(long).var()
            features[f'variance_ratio_{short}_{long}'] = (short_var * long) / (long_var * short)
        
        return features
    
    def _pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pattern recognition features"""
        features = pd.DataFrame(index=data.index)
        open_ = data['Open']
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Candlestick patterns via TA-Lib
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'engulfing': talib.CDLENGULFING,
            'harami': talib.CDLHARAMI,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'inverted_hammer': talib.CDLINVERTEDHAMMER,
            'piercing': talib.CDLPIERCING,
            'dark_cloud': talib.CDLDARKCLOUDCOVER,
            'three_inside': talib.CDL3INSIDE,
            'three_outside': talib.CDL3OUTSIDE,
            'abandoned_baby': talib.CDLABANDONEDBABY
        }
        
        for name, func in patterns.items():
            pattern_signal = func(open_.values, high.values, low.values, close.values)
            features[f'pattern_{name}'] = pattern_signal / 100  # Normalize to -1, 0, 1
        
        # Custom patterns
        # Pin bar
        body = abs(close - open_)
        upper_wick = high - np.maximum(close, open_)
        lower_wick = np.minimum(close, open_) - low
        
        features['pin_bar_top'] = ((upper_wick > 2 * body) & (upper_wick > 2 * lower_wick)).astype(int)
        features['pin_bar_bottom'] = ((lower_wick > 2 * body) & (lower_wick > 2 * upper_wick)).astype(int)
        
        # Inside and outside bars
        features['inside_bar'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
        features['outside_bar'] = ((high > high.shift(1)) & (low < low.shift(1))).astype(int)
        
        # Consecutive up/down days
        up_days = (close > close.shift(1)).astype(int)
        down_days = (close < close.shift(1)).astype(int)
        
        for n in [3, 5, 7]:
            features[f'consecutive_up_{n}d'] = up_days.rolling(n).sum() == n
            features[f'consecutive_down_{n}d'] = down_days.rolling(n).sum() == n
        
        # Pattern strength scoring
        # Combine multiple patterns for stronger signals
        bullish_patterns = features[[col for col in features.columns if 'pattern_' in col]].copy()
        bullish_patterns[bullish_patterns < 0] = 0
        bearish_patterns = features[[col for col in features.columns if 'pattern_' in col]].copy()
        bearish_patterns[bearish_patterns > 0] = 0
        
        features['bullish_pattern_score'] = bullish_patterns.sum(axis=1)
        features['bearish_pattern_score'] = abs(bearish_patterns.sum(axis=1))
        
        # Gap patterns
        features['gap_up'] = ((open_ - close.shift(1)) / close.shift(1) > 0.002).astype(int)
        features['gap_down'] = ((open_ - close.shift(1)) / close.shift(1) < -0.002).astype(int)
        features['gap_filled'] = ((features['gap_up'].shift(1) == 1) & (low <= close.shift(2))) | \
                                ((features['gap_down'].shift(1) == 1) & (high >= close.shift(2)))
        
        return features
    
    def _statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features"""
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        returns = close.pct_change()
        
        # Rolling statistics
        for window in self.config.stat_windows:
            # Skewness and kurtosis
            features[f'skewness_{window}d'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}d'] = returns.rolling(window).kurt()
            
            # Jarque-Bera test statistic
            def jarque_bera(x):
                if len(x) < 4:
                    return np.nan
                n = len(x)
                s = stats.skew(x)
                k = stats.kurtosis(x)
                jb = n * (s**2 / 6 + (k - 3)**2 / 24)
                return jb
            
            features[f'jarque_bera_{window}d'] = returns.rolling(window).apply(jarque_bera)
        
        # Autocorrelation at multiple lags
        for lag in [1, 5, 10, 20]:
            features[f'autocorr_lag_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Hurst exponent (simplified calculation)
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent using R/S analysis"""
            if len(ts) < max_lag:
                return np.nan
            
            lags = range(2, min(max_lag, len(ts)//2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            
            if len(tau) > 0 and all(t > 0 for t in tau):
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] / 2.0
            return np.nan
        
        features['hurst_exponent'] = returns.rolling(100).apply(hurst_exponent)
        
        # Shannon entropy
        def shannon_entropy(x, bins=10):
            if len(x) < bins:
                return np.nan
            hist, _ = np.histogram(x, bins=bins)
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))
        
        for window in [20, 50]:
            features[f'shannon_entropy_{window}d'] = returns.rolling(window).apply(
                lambda x: shannon_entropy(x)
            )
        
        # Z-scores and percentile ranks
        for window in [20, 50, 100]:
            rolling_mean = close.rolling(window).mean()
            rolling_std = close.rolling(window).std()
            features[f'z_score_{window}d'] = (close - rolling_mean) / rolling_std
            
            features[f'percentile_rank_{window}d'] = close.rolling(window).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
            )
        
        # Run length encoding features
        # Length of current trend
        up_trend = (returns > 0).astype(int)
        trend_changes = up_trend.diff().fillna(0) != 0
        trend_id = trend_changes.cumsum()
        trend_lengths = up_trend.groupby(trend_id).cumsum()
        features['trend_length'] = trend_lengths
        
        # Extreme value statistics
        for window in [20, 50]:
            features[f'max_return_{window}d'] = returns.rolling(window).max()
            features[f'min_return_{window}d'] = returns.rolling(window).min()
            features[f'return_range_{window}d'] = features[f'max_return_{window}d'] - features[f'min_return_{window}d']
        
        return features
    
    def _interaction_features(self, data: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between different indicators"""
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        
        # Golden Cross / Death Cross
        if 'sma_50' in base_features.columns and 'sma_200' in base_features.columns:
            golden_cross = (base_features['sma_50'] > base_features['sma_200']) & \
                          (base_features['sma_50'].shift(1) <= base_features['sma_200'].shift(1))
            death_cross = (base_features['sma_50'] < base_features['sma_200']) & \
                         (base_features['sma_50'].shift(1) >= base_features['sma_200'].shift(1))
            
            features['golden_cross'] = golden_cross.astype(int)
            features['death_cross'] = death_cross.astype(int)
            
            # Days since cross
            for cross, name in [(golden_cross, 'golden'), (death_cross, 'death')]:
                days_since = pd.Series(index=data.index, dtype=float)
                last_cross_idx = None
                
                for idx in data.index:
                    if cross.loc[idx]:
                        last_cross_idx = idx
                    if last_cross_idx is not None:
                        days_since.loc[idx] = (idx - last_cross_idx).days
                
                features[f'days_since_{name}_cross'] = days_since
        
        # MA Crossovers
        ma_pairs = [(5, 20), (10, 30), (20, 50)]
        for short, long in ma_pairs:
            if f'sma_{short}' in base_features.columns and f'sma_{long}' in base_features.columns:
                cross_up = (base_features[f'sma_{short}'] > base_features[f'sma_{long}']) & \
                          (base_features[f'sma_{short}'].shift(1) <= base_features[f'sma_{long}'].shift(1))
                cross_down = (base_features[f'sma_{short}'] < base_features[f'sma_{long}']) & \
                            (base_features[f'sma_{short}'].shift(1) >= base_features[f'sma_{long}'].shift(1))
                
                features[f'ma_cross_up_{short}_{long}'] = cross_up.astype(int)
                features[f'ma_cross_down_{short}_{long}'] = cross_down.astype(int)
        
        # RSI Divergences (simplified)
        for period in [14, 21]:
            if f'rsi_{period}' in base_features.columns:
                # Bullish divergence: price makes new low, RSI doesn't
                price_new_low = close == close.rolling(20).min()
                rsi_higher_low = base_features[f'rsi_{period}'] > base_features[f'rsi_{period}'].rolling(20).min()
                features[f'rsi_bullish_div_{period}'] = (price_new_low & rsi_higher_low).astype(int)
                
                # Bearish divergence: price makes new high, RSI doesn't
                price_new_high = close == close.rolling(20).max()
                rsi_lower_high = base_features[f'rsi_{period}'] < base_features[f'rsi_{period}'].rolling(20).max()
                features[f'rsi_bearish_div_{period}'] = (price_new_high & rsi_lower_high).astype(int)
        
        # Volume-Price Confirmation
        if 'volume_ratio_20' in base_features.columns:
            price_up = close > close.shift(1)
            volume_up = base_features['volume_ratio_20'] > 1
            features['volume_price_confirm'] = (price_up == volume_up).astype(int)
            features['volume_price_diverge'] = (price_up != volume_up).astype(int)
        
        # Bollinger/Keltner Squeeze
        if 'bb_width_20_20' in base_features.columns and 'squeeze_20_20' in base_features.columns:
            features['squeeze_fired'] = (base_features['squeeze_20_20'].shift(1) == 1) & \
                                       (base_features['squeeze_20_20'] == 0)
        
        # Support/Resistance Interactions
        for lookback in [20, 50]:
            if f'support_{lookback}d' in base_features.columns and f'resistance_{lookback}d' in base_features.columns:
                # Distance to support/resistance
                dist_to_support = (close - base_features[f'support_{lookback}d']) / close
                dist_to_resistance = (base_features[f'resistance_{lookback}d'] - close) / close
                
                # Near support/resistance
                features[f'near_support_{lookback}d'] = (dist_to_support < 0.02).astype(int)
                features[f'near_resistance_{lookback}d'] = (dist_to_resistance < 0.02).astype(int)
                
                # Breakout/breakdown
                features[f'resistance_break_{lookback}d'] = (close > base_features[f'resistance_{lookback}d']).astype(int)
                features[f'support_break_{lookback}d'] = (close < base_features[f'support_{lookback}d']).astype(int)
        
        # Multi-timeframe Momentum Alignment
        momentum_periods = [5, 10, 20]
        momentum_aligned_up = True
        momentum_aligned_down = True
        
        for period in momentum_periods:
            if f'return_{period}d' in base_features.columns:
                momentum_aligned_up &= base_features[f'return_{period}d'] > 0
                momentum_aligned_down &= base_features[f'return_{period}d'] < 0
        
        features['momentum_aligned_up'] = momentum_aligned_up.astype(int)
        features['momentum_aligned_down'] = momentum_aligned_down.astype(int)
        
        # Composite Scores
        # Bull Market Score
        bull_score = 0
        if 'sma_50' in base_features.columns and 'sma_200' in base_features.columns:
            bull_score += (base_features['sma_50'] > base_features['sma_200']).astype(int)
        if 'rsi_14' in base_features.columns:
            bull_score += (base_features['rsi_14'] > 50).astype(int)
        if 'macd_12_26_9' in base_features.columns:
            bull_score += (base_features['macd_12_26_9'] > 0).astype(int)
        if 'adx_14' in base_features.columns:
            bull_score += (base_features['adx_14'] > 25).astype(int)
        
        features['bull_market_score'] = bull_score / 4  # Normalize to 0-1
        
        # Mean Reversion Setup Score
        mr_score = 0
        if 'rsi_14' in base_features.columns:
            mr_score += ((base_features['rsi_14'] < 30) | (base_features['rsi_14'] > 70)).astype(int)
        if 'bb_position_20_20' in base_features.columns:
            mr_score += ((base_features['bb_position_20_20'] < 0.2) | (base_features['bb_position_20_20'] > 0.8)).astype(int)
        if 'z_score_20d' in base_features.columns:
            mr_score += (abs(base_features['z_score_20d']) > 2).astype(int)
        
        features['mean_reversion_score'] = mr_score / 3
        
        # Breakout Setup Score
        breakout_score = 0
        if 'bb_width_20_20' in base_features.columns:
            breakout_score += (base_features['bb_width_20_20'] < base_features['bb_width_20_20'].rolling(50).quantile(0.25)).astype(int)
        if 'atr_pct_14' in base_features.columns:
            breakout_score += (base_features['atr_pct_14'] < base_features['atr_pct_14'].rolling(50).quantile(0.25)).astype(int)
        if 'volume_ratio_20' in base_features.columns:
            breakout_score += (base_features['volume_ratio_20'] > 1.5).astype(int)
        
        features['breakout_setup_score'] = breakout_score / 3
        
        # Trend Exhaustion Score
        exhaustion_score = 0
        if 'rsi_14' in base_features.columns:
            exhaustion_score += ((base_features['rsi_14'] > 70) | (base_features['rsi_14'] < 30)).astype(int)
        if 'consecutive_up_7d' in base_features.columns:
            exhaustion_score += (base_features['consecutive_up_7d'] | base_features['consecutive_down_7d']).astype(int)
        if 'return_20d' in base_features.columns:
            exhaustion_score += (abs(base_features['return_20d']) > 0.15).astype(int)
        
        features['trend_exhaustion_score'] = exhaustion_score / 3
        
        return features
    
    def _ml_discovered_features(self, data: pd.DataFrame, base_features: pd.DataFrame) -> pd.DataFrame:
        """Generate ML-discovered features using transformations"""
        features = pd.DataFrame(index=data.index)
        close = data['Close']
        returns = close.pct_change()
        
        # Polynomial features for key indicators
        key_features = []
        for col in ['rsi_14', 'macd_hist_12_26_9', 'bb_position_20_20', 'volume_ratio_20']:
            if col in base_features.columns:
                key_features.append(col)
        
        if key_features:
            for feat in key_features[:3]:  # Limit to avoid explosion
                if feat in base_features.columns:
                    features[f'{feat}_squared'] = base_features[feat] ** 2
                    features[f'{feat}_cubed'] = base_features[feat] ** 3
                    features[f'{feat}_sqrt'] = np.sqrt(np.abs(base_features[feat])) * np.sign(base_features[feat])
        
        # Interaction terms between key features
        for i, feat1 in enumerate(key_features[:3]):
            for feat2 in key_features[i+1:4]:
                if feat1 in base_features.columns and feat2 in base_features.columns:
                    features[f'{feat1}_x_{feat2}'] = base_features[feat1] * base_features[feat2]
        
        # Fourier transforms for cyclical patterns
        for period in [20, 50]:
            if len(returns) >= period * 2:
                # Simple sine/cosine features for cyclical patterns
                t = np.arange(len(returns))
                for freq in [1, 2, 3]:
                    features[f'fourier_sin_{period}_{freq}'] = np.sin(2 * np.pi * freq * t / period)
                    features[f'fourier_cos_{period}_{freq}'] = np.cos(2 * np.pi * freq * t / period)
        
        # Fractal dimension (simplified box-counting)
        def fractal_dimension(ts, min_box_size=2, max_box_size=20):
            if len(ts) < max_box_size:
                return np.nan
            
            # Normalize time series
            ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-10)
            
            # Box counting
            box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=10, dtype=int)
            counts = []
            
            for box_size in box_sizes:
                # Count boxes needed to cover the time series
                n_boxes = len(ts) // box_size
                box_count = 0
                
                for i in range(n_boxes):
                    box_data = ts_norm[i*box_size:(i+1)*box_size]
                    if len(box_data) > 0:
                        box_range = box_data.max() - box_data.min()
                        if box_range > 0:
                            box_count += int(np.ceil(box_range * box_size))
                
                if box_count > 0:
                    counts.append(box_count)
                else:
                    counts.append(1)
            
            # Fit log-log relationship
            if len(counts) > 1:
                coeffs = np.polyfit(np.log(box_sizes[:len(counts)]), np.log(counts), 1)
                return -coeffs[0]
            return np.nan
        
        features['fractal_dimension'] = returns.rolling(100).apply(fractal_dimension)
        
        # Sample entropy (simplified)
        def sample_entropy(ts, m=2, r=0.2):
            if len(ts) < m + 1:
                return np.nan
            
            # Normalize
            ts_norm = (ts - ts.mean()) / (ts.std() + 1e-10)
            tolerance = r * ts.std()
            
            # Count pattern matches
            def count_patterns(ts, m, tolerance):
                patterns = np.array([ts[i:i+m] for i in range(len(ts)-m+1)])
                count = 0
                
                for i in range(len(patterns)):
                    for j in range(i+1, len(patterns)):
                        if np.all(np.abs(patterns[i] - patterns[j]) <= tolerance):
                            count += 1
                
                return count
            
            phi_m = count_patterns(ts_norm, m, tolerance)
            phi_m1 = count_patterns(ts_norm, m + 1, tolerance)
            
            if phi_m > 0 and phi_m1 > 0:
                return -np.log(phi_m1 / phi_m)
            return np.nan
        
        features['sample_entropy'] = returns.rolling(50).apply(sample_entropy)
        
        # Wavelet-inspired features (simplified using rolling windows)
        for scale in [4, 8, 16]:
            # Approximate wavelet decomposition using difference of averages
            ma_short = close.rolling(scale).mean()
            ma_long = close.rolling(scale * 2).mean()
            features[f'wavelet_d{scale}'] = (ma_short - ma_long) / close
        
        # Recurrence features (simplified)
        # Time since last visit to current price level
        def time_since_level(ts, window=50, n_levels=10):
            if len(ts) < window:
                return np.nan
            
            recent = ts.iloc[-window:]
            current = ts.iloc[-1]
            
            # Discretize into levels
            levels = pd.qcut(recent, q=n_levels, duplicates='drop')
            current_level = pd.cut([current], bins=levels.cat.categories)[0]
            
            # Find last occurrence of current level
            for i in range(len(recent)-2, -1, -1):
                if levels.iloc[i] == current_level:
                    return len(recent) - i - 1
            
            return window
        
        features['time_since_price_level'] = close.rolling(100).apply(
            lambda x: time_since_level(x) if len(x) == 100 else np.nan
        )
        
        return features
    
    def select_features(self, features_df: pd.DataFrame, target: pd.Series, 
                       n_features: int = 150) -> List[str]:
        """Select top features using statistical tests"""
        # Remove any features with too many NaN values
        valid_features = features_df.columns[features_df.isnull().sum() < len(features_df) * 0.1]
        features_clean = features_df[valid_features].fillna(0)
        
        # Use SelectKBest for feature selection
        selector = SelectKBest(score_func=f_classif, k=min(n_features, len(valid_features)))
        selector.fit(features_clean, target)
        
        # Get selected feature names
        selected_features = features_clean.columns[selector.get_support()].tolist()
        
        return selected_features
    
    def create_feature_groups(self) -> Dict[str, List[str]]:
        """Organize features into logical groups for analysis"""
        groups = {
            'price_based': [],
            'volume_based': [],
            'volatility': [],
            'technical': [],
            'microstructure': [],
            'patterns': [],
            'statistical': [],
            'interactions': [],
            'ml_discovered': []
        }
        
        # Categorize features based on names
        for feat in self.feature_names:
            if any(x in feat for x in ['return_', 'sma_', 'ema_', 'price_to_', 'vwap', 'support', 'resistance', 'fib_', 'channel_']):
                groups['price_based'].append(feat)
            elif any(x in feat for x in ['volume', 'obv', 'acc_dist', 'cmf_', 'vpt', 'mfi_']):
                groups['volume_based'].append(feat)
            elif any(x in feat for x in ['atr_', 'bb_', 'kc_', 'volatility_', 'parkinson', 'garman', 'rogers']):
                groups['volatility'].append(feat)
            elif any(x in feat for x in ['rsi_', 'macd_', 'stoch_', 'williams_', 'cci_', 'adx_', 'aroon_', 'ppo_', 'trix_', 'cmo_']):
                groups['technical'].append(feat)
            elif any(x in feat for x in ['spread_', 'intraday_', 'order_flow', 'amihud', 'kyle_', 'noise_', 'roll_', 'variance_ratio']):
                groups['microstructure'].append(feat)
            elif any(x in feat for x in ['pattern_', 'pin_bar', 'inside_bar', 'consecutive_', 'gap_']):
                groups['patterns'].append(feat)
            elif any(x in feat for x in ['skewness', 'kurtosis', 'jarque_bera', 'autocorr', 'hurst', 'entropy', 'z_score', 'percentile']):
                groups['statistical'].append(feat)
            elif any(x in feat for x in ['golden_cross', 'death_cross', 'ma_cross', 'diverge', 'confirm', 'squeeze', 'aligned', 'score']):
                groups['interactions'].append(feat)
            elif any(x in feat for x in ['squared', 'cubed', 'sqrt', '_x_', 'fourier', 'fractal', 'wavelet', 'sample_entropy']):
                groups['ml_discovered'].append(feat)
        
        return groups


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    close = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
    high = close * (1 + np.abs(np.random.randn(len(dates)) * 0.005))
    low = close * (1 - np.abs(np.random.randn(len(dates)) * 0.005))
    open_ = close.shift(1).fillna(close[0]) * (1 + np.random.randn(len(dates)) * 0.002)
    volume = np.random.lognormal(15, 1, len(dates))
    
    data = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    # Initialize feature engineer
    engineer = EnhancedFeatureEngineer()
    
    # Generate features
    print("Generating features...")
    features = engineer.engineer_features(data, fit_scaler=True)
    
    print(f"\nGenerated {len(features.columns)} features")
    print(f"Feature shape: {features.shape}")
    print(f"Memory usage: {features.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Show feature groups
    feature_groups = engineer.create_feature_groups()
    print("\nFeature groups:")
    for group, feats in feature_groups.items():
        print(f"  {group}: {len(feats)} features")
    
    # Show sample features
    print("\nSample features (last 5 rows):")
    print(features.tail())
    
    # Check for NaN values
    nan_counts = features.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"\nFeatures with NaN values: {nan_counts[nan_counts > 0].to_dict()}")
