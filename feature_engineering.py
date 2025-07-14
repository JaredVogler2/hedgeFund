"""
Enhanced Feature Engineering with Chart Patterns, Feature Interactions, and ML Patterns
Includes all requested features for hedge fund-quality trading
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import talib
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Basic features
    return_periods: List[int] = None
    ma_periods: List[int] = None

    # Technical indicators
    rsi_periods: List[int] = None
    bb_periods: List[int] = None
    atr_periods: List[int] = None
    volume_ma_periods: List[int] = None
    macd_params: List[tuple] = None

    # Pattern detection
    pattern_lookback: int = 20
    support_resistance_periods: List[int] = None

    # Feature engineering
    n_features: int = 300  # Increased for more features
    use_microstructure: bool = True
    use_patterns: bool = True
    use_ml_features: bool = True
    use_interactions: bool = True

    def __post_init__(self):
        if self.return_periods is None:
            self.return_periods = [1, 2, 3, 5, 10, 20, 60]
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 100, 200]
        if self.rsi_periods is None:
            self.rsi_periods = [9, 14, 21]
        if self.bb_periods is None:
            self.bb_periods = [20, 50]
        if self.atr_periods is None:
            self.atr_periods = [14, 20]
        if self.volume_ma_periods is None:
            self.volume_ma_periods = [5, 10, 20, 50]
        if self.macd_params is None:
            self.macd_params = [(12, 26, 9), (5, 35, 5)]
        if self.support_resistance_periods is None:
            self.support_resistance_periods = [20, 50, 100]

class ChartPatternDetector:
    """Detects classical chart patterns"""

    @staticmethod
    def detect_head_shoulders(highs: pd.Series, lows: pd.Series, window: int = 30) -> pd.Series:
        """Detect head and shoulders pattern"""
        pattern = pd.Series(0, index=highs.index)

        if len(highs) < window:
            return pattern

        for i in range(window, len(highs)):
            window_highs = highs[i-window:i].values
            window_lows = lows[i-window:i].values

            # Find peaks
            peaks = []
            for j in range(1, len(window_highs)-1):
                if window_highs[j] > window_highs[j-1] and window_highs[j] > window_highs[j+1]:
                    peaks.append(j)

            # Check for head and shoulders pattern (3 peaks, middle highest)
            if len(peaks) >= 3:
                # Get the three most recent peaks
                recent_peaks = peaks[-3:]
                peak_values = [window_highs[p] for p in recent_peaks]

                # Check if middle peak is highest (head)
                if peak_values[1] > peak_values[0] and peak_values[1] > peak_values[2]:
                    # Check if shoulders are roughly equal
                    if abs(peak_values[0] - peak_values[2]) / peak_values[1] < 0.1:
                        pattern.iloc[i] = -1  # Bearish pattern

                # Inverse head and shoulders
                valley_values = [window_lows[p] for p in recent_peaks]
                if valley_values[1] < valley_values[0] and valley_values[1] < valley_values[2]:
                    if abs(valley_values[0] - valley_values[2]) / abs(valley_values[1]) < 0.1:
                        pattern.iloc[i] = 1  # Bullish pattern

        return pattern

    @staticmethod
    def detect_triangles(highs: pd.Series, lows: pd.Series, window: int = 20) -> pd.Series:
        """Detect triangle patterns (ascending, descending, symmetric)"""
        pattern = pd.Series(0, index=highs.index)

        for i in range(window, len(highs)):
            window_highs = highs[i-window:i]
            window_lows = lows[i-window:i]

            # Calculate trendlines
            x = np.arange(window)
            high_slope = np.polyfit(x, window_highs, 1)[0]
            low_slope = np.polyfit(x, window_lows, 1)[0]

            # Ascending triangle: flat top, rising bottom
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                pattern.iloc[i] = 1
            # Descending triangle: flat bottom, falling top
            elif abs(low_slope) < 0.001 and high_slope < -0.001:
                pattern.iloc[i] = -1
            # Symmetric triangle: converging lines
            elif high_slope < -0.001 and low_slope > 0.001:
                pattern.iloc[i] = 0.5

        return pattern

    @staticmethod
    def detect_double_tops_bottoms(prices: pd.Series, window: int = 30) -> pd.Series:
        """Detect double tops and bottoms"""
        pattern = pd.Series(0, index=prices.index)

        for i in range(window*2, len(prices)):
            window_prices = prices[i-window*2:i].values

            # Find local maxima and minima
            peaks = []
            valleys = []

            for j in range(1, len(window_prices)-1):
                if window_prices[j] > window_prices[j-1] and window_prices[j] > window_prices[j+1]:
                    peaks.append((j, window_prices[j]))
                elif window_prices[j] < window_prices[j-1] and window_prices[j] < window_prices[j+1]:
                    valleys.append((j, window_prices[j]))

            # Check for double top
            if len(peaks) >= 2:
                last_peaks = peaks[-2:]
                if abs(last_peaks[0][1] - last_peaks[1][1]) / last_peaks[0][1] < 0.02:
                    if last_peaks[1][0] - last_peaks[0][0] > window // 3:
                        pattern.iloc[i] = -1

            # Check for double bottom
            if len(valleys) >= 2:
                last_valleys = valleys[-2:]
                if abs(last_valleys[0][1] - last_valleys[1][1]) / last_valleys[0][1] < 0.02:
                    if last_valleys[1][0] - last_valleys[0][0] > window // 3:
                        pattern.iloc[i] = 1

        return pattern

    @staticmethod
    def detect_flags_pennants(prices: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Detect flag and pennant patterns"""
        pattern = pd.Series(0, index=prices.index)

        for i in range(window, len(prices)):
            if i < 50:  # Need history for pattern
                continue

            # Look for strong move (pole)
            pole_return = (prices.iloc[i-window] - prices.iloc[i-window-20]) / prices.iloc[i-window-20]

            # Look for consolidation after pole
            consolidation_prices = prices[i-window:i]
            consolidation_std = consolidation_prices.pct_change().std()

            # High volume on pole, lower volume on consolidation
            pole_volume = volume[i-window-20:i-window].mean()
            consolidation_volume = volume[i-window:i].mean()

            if abs(pole_return) > 0.1 and consolidation_std < 0.02:
                if consolidation_volume < pole_volume * 0.7:
                    pattern.iloc[i] = np.sign(pole_return)

        return pattern

class AdvancedFeatureEngineer(ChartPatternDetector):
    """Enhanced feature engineering with all requested features"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names = []
        self.scaler = StandardScaler()

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        features = pd.DataFrame(index=data.index)

        # Handle multi-index columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)

        # Extract price series
        close = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close'].squeeze()
        open_ = data['Open'] if isinstance(data['Open'], pd.Series) else data['Open'].squeeze()
        high = data['High'] if isinstance(data['High'], pd.Series) else data['High'].squeeze()
        low = data['Low'] if isinstance(data['Low'], pd.Series) else data['Low'].squeeze()
        volume = data['Volume'] if isinstance(data['Volume'], pd.Series) else data['Volume'].squeeze()

        try:
            # 1. BASIC PRICE FEATURES
            logger.info("Generating basic price features...")
            features.update(self._generate_price_features(close, open_, high, low))

            # 2. VOLUME FEATURES
            logger.info("Generating volume features...")
            features.update(self._generate_volume_features(close, volume))

            # 3. TECHNICAL INDICATORS
            logger.info("Generating technical indicators...")
            features.update(self._generate_technical_indicators(close, high, low, volume))

            # 4. VOLATILITY FEATURES
            logger.info("Generating volatility features...")
            features.update(self._generate_volatility_features(close, high, low))

            # 5. CHART PATTERNS
            if self.config.use_patterns:
                logger.info("Detecting chart patterns...")
                features.update(self._generate_pattern_features(close, open_, high, low, volume))

            # 6. SUPPORT/RESISTANCE LEVELS
            logger.info("Calculating support/resistance...")
            features.update(self._generate_support_resistance(close, high, low))

            # 7. CANDLESTICK PATTERNS
            logger.info("Detecting candlestick patterns...")
            features.update(self._generate_candlestick_patterns(open_, high, low, close))

            # 8. MARKET MICROSTRUCTURE
            if self.config.use_microstructure:
                logger.info("Generating microstructure features...")
                features.update(self._generate_microstructure_features(close, high, low, volume))

            # 9. FEATURE INTERACTIONS (Golden Cross, etc.)
            if self.config.use_interactions:
                logger.info("Generating feature interactions...")
                features.update(self._generate_feature_interactions(features, close, volume))

            # 10. ML-BASED FEATURES
            if self.config.use_ml_features:
                logger.info("Generating ML-based features...")
                features.update(self._generate_ml_features(features))

            # 11. MARKET REGIME FEATURES
            logger.info("Detecting market regimes...")
            features.update(self._generate_regime_features(close, volume))

        except Exception as e:
            logger.error(f"Error in feature generation: {e}")
            import traceback
            traceback.print_exc()

        # Clean up
        features = features.replace([np.inf, -np.inf], 0)
        features = features.ffill(limit=5)
        features = features.fillna(0)

        # Store feature names
        self.feature_names = features.columns.tolist()

        logger.info(f"Generated {len(features.columns)} features")

        return features

    def _generate_price_features(self, close, open_, high, low):
        """Generate price-based features"""
        features = {}

        # Returns at different intervals
        for period in self.config.return_periods:
            features[f'return_{period}d'] = close.pct_change(period)
            features[f'log_return_{period}d'] = np.log(close / close.shift(period))

        # Moving averages and related features
        for period in self.config.ma_periods:
            sma = close.rolling(window=period).mean()
            ema = close.ewm(span=period, adjust=False).mean()

            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            features[f'price_to_sma_{period}'] = close / sma
            features[f'price_to_ema_{period}'] = close / ema
            features[f'sma_{period}_slope'] = sma.diff()
            features[f'ema_sma_diff_{period}'] = ema - sma

        # Price channels and positions
        for period in [10, 20, 50]:
            features[f'high_{period}d'] = high.rolling(window=period).max()
            features[f'low_{period}d'] = low.rolling(window=period).min()
            features[f'channel_position_{period}'] = (close - features[f'low_{period}d']) / (features[f'high_{period}d'] - features[f'low_{period}d'] + 1e-10)
            features[f'channel_width_{period}'] = (features[f'high_{period}d'] - features[f'low_{period}d']) / close

        # Price ratios and gaps
        features['high_low_ratio'] = high / (low + 1e-10)
        features['close_open_ratio'] = close / (open_ + 1e-10)
        features['daily_range'] = (high - low) / (close + 1e-10)
        features['close_position'] = (close - low) / (high - low + 1e-10)
        features['gap_up'] = (open_ - close.shift(1)) / (close.shift(1) + 1e-10)
        features['gap_size'] = np.abs(features['gap_up'])

        return features

    def _generate_volume_features(self, close, volume):
        """Generate volume-based features"""
        features = {}

        # Basic volume features
        features['volume'] = volume
        features['log_volume'] = np.log(volume + 1)
        features['dollar_volume'] = close * volume

        # Volume moving averages and ratios
        for period in self.config.volume_ma_periods:
            vol_sma = volume.rolling(window=period).mean()
            features[f'volume_sma_{period}'] = vol_sma
            features[f'volume_ratio_{period}'] = volume / (vol_sma + 1e-10)
            features[f'relative_volume_{period}'] = (volume - vol_sma) / (vol_sma.rolling(period).std() + 1e-10)

        # VWAP (Volume Weighted Average Price)
        for period in [5, 10, 20]:
            typical_price = (close + high + low) / 3
            features[f'vwap_{period}'] = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
            features[f'price_to_vwap_{period}'] = close / features[f'vwap_{period}']

        # On Balance Volume (OBV) and variations
        features['obv'] = (np.sign(close.diff()) * volume).cumsum()
        features['obv_sma'] = features['obv'].rolling(20).mean()
        features['obv_divergence'] = features['obv'] - features['obv_sma']

        # Accumulation/Distribution
        features['acc_dist'] = ((close - low) - (high - close)) / (high - low + 1e-10) * volume
        features['acc_dist_cum'] = features['acc_dist'].cumsum()

        # Money Flow Index components
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        for period in [14, 20]:
            positive_sum = positive_flow.rolling(period).sum()
            negative_sum = negative_flow.rolling(period).sum()
            features[f'mfi_{period}'] = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))

        # Volume Rate of Change
        for period in [5, 10, 20]:
            features[f'volume_roc_{period}'] = volume.pct_change(period)

        return features

    def _generate_technical_indicators(self, close, high, low, volume):
        """Generate technical indicators using TA-Lib where available"""
        features = {}

        # Convert to numpy arrays for TA-Lib
        close_arr = close.values
        high_arr = high.values
        low_arr = low.values
        volume_arr = volume.values

        # RSI variations
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = pd.Series(talib.RSI(close_arr, timeperiod=period), index=close.index)

            # Stochastic RSI
            stoch_rsi = pd.Series(talib.STOCHRSI(close_arr, timeperiod=period)[0], index=close.index)
            features[f'stoch_rsi_{period}'] = stoch_rsi

        # MACD variations
        for fast, slow, signal in self.config.macd_params:
            macd, macd_signal, macd_hist = talib.MACD(close_arr, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            features[f'macd_{fast}_{slow}'] = pd.Series(macd, index=close.index)
            features[f'macd_signal_{fast}_{slow}'] = pd.Series(macd_signal, index=close.index)
            features[f'macd_hist_{fast}_{slow}'] = pd.Series(macd_hist, index=close.index)

        # Bollinger Bands
        for period in self.config.bb_periods:
            upper, middle, lower = talib.BBANDS(close_arr, timeperiod=period)
            features[f'bb_upper_{period}'] = pd.Series(upper, index=close.index)
            features[f'bb_middle_{period}'] = pd.Series(middle, index=close.index)
            features[f'bb_lower_{period}'] = pd.Series(lower, index=close.index)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / features[f'bb_middle_{period}']
            features[f'bb_position_{period}'] = (close - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 1e-10)

        # Stochastic Oscillator
        for period in [14, 21]:
            slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr, fastk_period=period)
            features[f'stoch_k_{period}'] = pd.Series(slowk, index=close.index)
            features[f'stoch_d_{period}'] = pd.Series(slowd, index=close.index)

        # Williams %R
        for period in [14, 21]:
            features[f'williams_r_{period}'] = pd.Series(talib.WILLR(high_arr, low_arr, close_arr, timeperiod=period), index=close.index)

        # ADX (Average Directional Index)
        for period in [14, 20]:
            features[f'adx_{period}'] = pd.Series(talib.ADX(high_arr, low_arr, close_arr, timeperiod=period), index=close.index)
            features[f'plus_di_{period}'] = pd.Series(talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=period), index=close.index)
            features[f'minus_di_{period}'] = pd.Series(talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=period), index=close.index)

        # CCI (Commodity Channel Index)
        for period in [14, 20]:
            features[f'cci_{period}'] = pd.Series(talib.CCI(high_arr, low_arr, close_arr, timeperiod=period), index=close.index)

        # Aroon
        for period in [25]:
            aroon_up, aroon_down = talib.AROON(high_arr, low_arr, timeperiod=period)
            features[f'aroon_up_{period}'] = pd.Series(aroon_up, index=close.index)
            features[f'aroon_down_{period}'] = pd.Series(aroon_down, index=close.index)
            features[f'aroon_oscillator_{period}'] = features[f'aroon_up_{period}'] - features[f'aroon_down_{period}']

        # Ultimate Oscillator
        features['ultimate_oscillator'] = pd.Series(talib.ULTOSC(high_arr, low_arr, close_arr), index=close.index)

        # ROC (Rate of Change)
        for period in [10, 20]:
            features[f'roc_{period}'] = pd.Series(talib.ROC(close_arr, timeperiod=period), index=close.index)

        # CMO (Chande Momentum Oscillator)
        for period in [14, 20]:
            features[f'cmo_{period}'] = pd.Series(talib.CMO(close_arr, timeperiod=period), index=close.index)

        return features

    def _generate_volatility_features(self, close, high, low):
        """Generate volatility features"""
        features = {}

        returns = close.pct_change()

        # Historical volatility at different periods
        for period in [5, 10, 20, 60]:
            features[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            features[f'volatility_skew_{period}'] = returns.rolling(window=period).skew()
            features[f'volatility_kurt_{period}'] = returns.rolling(window=period).kurt()

        # ATR (Average True Range) variations
        for period in self.config.atr_periods:
            atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
            features[f'atr_{period}'] = atr
            features[f'atr_ratio_{period}'] = atr / close
            features[f'natr_{period}'] = pd.Series(talib.NATR(high.values, low.values, close.values, timeperiod=period), index=close.index)

        # Volatility ratios
        features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-10)
        features['volatility_change'] = features['volatility_20'].diff()

        # Parkinson volatility (using high-low)
        for period in [10, 20]:
            hl_ratio = np.log(high / low)
            features[f'parkinson_vol_{period}'] = np.sqrt(252 / (4 * np.log(2))) * hl_ratio.rolling(period).std()

        # Garman-Klass volatility
        for period in [10, 20]:
            rs = np.log(high / close) * np.log(high / open)
            co = np.log(close / open)
            features[f'garman_klass_{period}'] = np.sqrt(
                252 / period * (
                    0.5 * (np.log(high / low) ** 2).rolling(period).sum() -
                    (2 * np.log(2) - 1) * (co ** 2).rolling(period).sum()
                )
            )

        # Yang-Zhang volatility (most accurate)
        for period in [20]:
            overnight = np.log(open / close.shift(1))
            open_close = np.log(close / open)

            overnight_var = overnight.rolling(period).var()
            open_close_var = open_close.rolling(period).var()
            rs_var = features[f'garman_klass_{period}'] ** 2

            k = 0.34 / (1.34 + (period + 1) / (period - 1))
            features[f'yang_zhang_{period}'] = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)

        return features

    def _generate_pattern_features(self, close, open_, high, low, volume):
        """Generate chart pattern features"""
        features = {}

        # Head and Shoulders
        features['head_shoulders'] = self.detect_head_shoulders(high, low, window=30)

        # Triangles
        features['triangle_pattern'] = self.detect_triangles(high, low, window=20)

        # Double Tops/Bottoms
        features['double_top_bottom'] = self.detect_double_tops_bottoms(close, window=30)

        # Flags and Pennants
        features['flag_pennant'] = self.detect_flags_pennants(close, volume, window=20)

        # Cup and Handle pattern
        features['cup_handle'] = self._detect_cup_handle(close, volume)

        # Wedges
        features['wedge_pattern'] = self._detect_wedges(high, low)

        # Breakout detection
        for period in [20, 50]:
            high_break = close > high.rolling(period).max().shift(1)
            low_break = close < low.rolling(period).min().shift(1)
            features[f'breakout_{period}d'] = high_break.astype(int) - low_break.astype(int)

            # Volume confirmation
            vol_ratio = volume / volume.rolling(period).mean()
            features[f'volume_breakout_{period}d'] = features[f'breakout_{period}d'] * (vol_ratio > 1.5).astype(int)

        return features

    def _detect_cup_handle(self, close: pd.Series, volume: pd.Series, window: int = 50) -> pd.Series:
        """Detect cup and handle pattern"""
        pattern = pd.Series(0, index=close.index)

        for i in range(window, len(close)):
            if i < 100:
                continue

            # Look for U-shaped price movement
            window_prices = close[i-window:i].values
            min_idx = np.argmin(window_prices[:window//2])

            # Check if we have a cup shape
            left_high = np.max(window_prices[:min_idx])
            bottom = window_prices[min_idx]
            right_high = np.max(window_prices[window//2:])

            if left_high > bottom * 1.1 and right_high > bottom * 1.1:
                if abs(left_high - right_high) / left_high < 0.05:
                    # Look for handle (small consolidation)
                    handle_prices = close[i-10:i]
                    if handle_prices.std() / handle_prices.mean() < 0.02:
                        pattern.iloc[i] = 1

        return pattern

    def _detect_wedges(self, high: pd.Series, low: pd.Series, window: int = 30) -> pd.Series:
        """Detect rising and falling wedges"""
        pattern = pd.Series(0, index=high.index)

        for i in range(window, len(high)):
            window_highs = high[i-window:i].values
            window_lows = low[i-window:i].values

            x = np.arange(window)
            high_slope = np.polyfit(x, window_highs, 1)[0]
            low_slope = np.polyfit(x, window_lows, 1)[0]

            # Rising wedge: both lines rising, converging
            if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                pattern.iloc[i] = -1  # Bearish
            # Falling wedge: both lines falling, converging
            elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
                pattern.iloc[i] = 1  # Bullish

        return pattern

    def _generate_support_resistance(self, close, high, low):
        """Generate support and resistance levels"""
        features = {}

        for period in self.config.support_resistance_periods:
            # Pivot points
            pivot = (high.rolling(period).max() + low.rolling(period).min() + close) / 3

            features[f'pivot_{period}'] = pivot
            features[f'resistance1_{period}'] = 2 * pivot - low.rolling(period).min()
            features[f'support1_{period}'] = 2 * pivot - high.rolling(period).max()
            features[f'resistance2_{period}'] = pivot + (high.rolling(period).max() - low.rolling(period).min())
            features[f'support2_{period}'] = pivot - (high.rolling(period).max() - low.rolling(period).min())

            # Distance from levels
            features[f'dist_to_pivot_{period}'] = (close - pivot) / close
            features[f'dist_to_resistance1_{period}'] = (features[f'resistance1_{period}'] - close) / close
            features[f'dist_to_support1_{period}'] = (close - features[f'support1_{period}']) / close

        # Fibonacci retracements
        for period in [50, 100]:
            period_high = high.rolling(period).max()
            period_low = low.rolling(period).min()
            diff = period_high - period_low

            features[f'fib_0.236_{period}'] = period_high - 0.236 * diff
            features[f'fib_0.382_{period}'] = period_high - 0.382 * diff
            features[f'fib_0.5_{period}'] = period_high - 0.5 * diff
            features[f'fib_0.618_{period}'] = period_high - 0.618 * diff

            # Distance to Fibonacci levels
            features[f'dist_to_fib_0.382_{period}'] = (close - features[f'fib_0.382_{period}']) / close
            features[f'dist_to_fib_0.618_{period}'] = (close - features[f'fib_0.618_{period}']) / close

        return features

    def _generate_candlestick_patterns(self, open_, high, low, close):
        """Generate candlestick pattern features using TA-Lib"""
        features = {}

        # Convert to numpy arrays
        o = open_.values
        h = high.values
        l = low.values
        c = close.values

        # Single candlestick patterns
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'inverted_hammer': talib.CDLINVERTEDHAMMER,
            'spinning_top': talib.CDLSPINNINGTOP,
            'marubozu': talib.CDLMARUBOZU,
            'long_line': talib.CDLLONGLINE,
            'short_line': talib.CDLSHORTLINE,
        }

        # Multi-candlestick patterns
        multi_patterns = {
            'engulfing': talib.CDLENGULFING,
            'harami': talib.CDLHARAMI,
            'harami_cross': talib.CDLHARAMICROSS,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'three_inside': talib.CDL3INSIDE,
            'three_outside': talib.CDL3OUTSIDE,
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER,
            'piercing': talib.CDLPIERCING,
        }

        # Apply all patterns
        for name, func in {**patterns, **multi_patterns}.items():
            features[f'cdl_{name}'] = pd.Series(func(o, h, l, c), index=close.index) / 100  # Normalize to -1, 0, 1

        return features

    def _generate_microstructure_features(self, close, high, low, volume):
        """Generate market microstructure features"""
        features = {}

        # Spread measures
        features['hl_spread'] = (high - low) / close
        features['co_spread'] = np.abs(close - open) / close

        # Liquidity measures
        features['turnover'] = volume / volume.rolling(20).mean()
        features['dollar_volume_log'] = np.log(close * volume + 1)

        # Price impact measures
        returns = close.pct_change()
        signed_volume = volume * np.sign(returns)

        for period in [5, 10, 20]:
            # Kyle's lambda (price impact)
            features[f'kyle_lambda_{period}'] = returns.rolling(period).sum() / (signed_volume.rolling(period).sum() + 1e-10)

            # Amihud illiquidity
            features[f'amihud_{period}'] = (np.abs(returns) / (close * volume + 1e-10)).rolling(period).mean() * 1e6

        # Return autocorrelation
        for lag in [1, 5, 10]:
            features[f'return_autocorr_lag{lag}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )

        # Variance ratio test for market efficiency
        for period in [5, 10]:
            var_1 = returns.rolling(period).var()
            var_k = returns.rolling(period * 5).apply(lambda x: x.values.sum() ** 2 / (len(x) * period)).rolling(1).mean()
            features[f'variance_ratio_{period}'] = var_k / (var_1 + 1e-10)

        # Tick statistics
        features['tick_up'] = (close > close.shift(1)).rolling(20).sum()
        features['tick_down'] = (close < close.shift(1)).rolling(20).sum()
        features['tick_ratio'] = features['tick_up'] / (features['tick_down'] + 1e-10)

        return features

    def _generate_feature_interactions(self, features, close, volume):
        """Generate feature interactions like Golden Cross"""
        interaction_features = {}

        # Moving Average Crossovers
        # Golden Cross / Death Cross
        if 'sma_50' in features and 'sma_200' in features:
            interaction_features['golden_cross'] = (
                (features['sma_50'] > features['sma_200']) &
                (features['sma_50'].shift(1) <= features['sma_200'].shift(1))
            ).astype(int)

            interaction_features['death_cross'] = (
                (features['sma_50'] < features['sma_200']) &
                (features['sma_50'].shift(1) >= features['sma_200'].shift(1))
            ).astype(int)

            interaction_features['ma_50_200_spread'] = (features['sma_50'] - features['sma_200']) / features['sma_200']

        # EMA crossovers
        if 'ema_10' in features and 'ema_20' in features:
            interaction_features['ema_10_20_cross'] = (
                (features['ema_10'] > features['ema_20']) &
                (features['ema_10'].shift(1) <= features['ema_20'].shift(1))
            ).astype(int) - (
                (features['ema_10'] < features['ema_20']) &
                (features['ema_10'].shift(1) >= features['ema_20'].shift(1))
            ).astype(int)

        # MACD crossovers
        if 'macd_12_26' in features and 'macd_signal_12_26' in features:
            interaction_features['macd_signal_cross'] = (
                (features['macd_12_26'] > features['macd_signal_12_26']) &
                (features['macd_12_26'].shift(1) <= features['macd_signal_12_26'].shift(1))
            ).astype(int) - (
                (features['macd_12_26'] < features['macd_signal_12_26']) &
                (features['macd_12_26'].shift(1) >= features['macd_signal_12_26'].shift(1))
            ).astype(int)

        # RSI divergences
        if 'rsi_14' in features:
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_lower_low = (close == close.rolling(20).min()) & (close < close.shift(20).rolling(20).min())
            rsi_higher_low = (features['rsi_14'] == features['rsi_14'].rolling(20).min()) & (features['rsi_14'] > features['rsi_14'].shift(20).rolling(20).min())
            interaction_features['rsi_bullish_divergence'] = (price_lower_low & rsi_higher_low).astype(int)

            # Bearish divergence: price makes higher high, RSI makes lower high
            price_higher_high = (close == close.rolling(20).max()) & (close > close.shift(20).rolling(20).max())
            rsi_lower_high = (features['rsi_14'] == features['rsi_14'].rolling(20).max()) & (features['rsi_14'] < features['rsi_14'].shift(20).rolling(20).max())
            interaction_features['rsi_bearish_divergence'] = (price_higher_high & rsi_lower_high).astype(int)

        # Volume and price interactions
        if 'volume_sma_20' in features:
            # Volume breakout with price breakout
            if 'breakout_20d' in features:
                interaction_features['volume_price_breakout'] = (
                    (features['breakout_20d'] != 0) &
                    (volume > features['volume_sma_20'] * 1.5)
                ).astype(int) * features['breakout_20d']

            # Accumulation/Distribution divergence
            if 'acc_dist_cum' in features:
                price_trend = close.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
                ad_trend = features['acc_dist_cum'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
                interaction_features['ad_divergence'] = np.sign(price_trend) != np.sign(ad_trend)

        # Bollinger Band squeeze
        if 'bb_width_20' in features and 'atr_14' in features:
            bb_squeeze = features['bb_width_20'] < features['bb_width_20'].rolling(100).quantile(0.2)
            low_volatility = features['atr_14'] < features['atr_14'].rolling(100).quantile(0.2)
            interaction_features['bb_squeeze'] = (bb_squeeze & low_volatility).astype(int)

        # Stochastic and RSI overbought/oversold
        if 'stoch_k_14' in features and 'rsi_14' in features:
            interaction_features['stoch_rsi_overbought'] = (
                (features['stoch_k_14'] > 80) & (features['rsi_14'] > 70)
            ).astype(int)

            interaction_features['stoch_rsi_oversold'] = (
                (features['stoch_k_14'] < 20) & (features['rsi_14'] < 30)
            ).astype(int)

        # ADX and DI crossovers
        if 'adx_14' in features and 'plus_di_14' in features and 'minus_di_14' in features:
            interaction_features['di_bullish_cross'] = (
                (features['plus_di_14'] > features['minus_di_14']) &
                (features['plus_di_14'].shift(1) <= features['minus_di_14'].shift(1)) &
                (features['adx_14'] > 25)
            ).astype(int)

            interaction_features['di_bearish_cross'] = (
                (features['plus_di_14'] < features['minus_di_14']) &
                (features['plus_di_14'].shift(1) >= features['minus_di_14'].shift(1)) &
                (features['adx_14'] > 25)
            ).astype(int)

        return interaction_features

    def _generate_ml_features(self, features):
        """Generate ML-based features using PCA and clustering"""
        ml_features = {}

        # Select numeric features for ML processing
        numeric_features = features.select_dtypes(include=[np.number])

        if len(numeric_features.columns) > 20:
            # PCA for dimensionality reduction
            feature_matrix = numeric_features.fillna(0).values

            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            # Apply PCA
            n_components = min(10, len(numeric_features.columns) // 2)
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(feature_matrix_scaled)

            # Add PCA features
            for i in range(n_components):
                ml_features[f'pca_{i}'] = pd.Series(pca_features[:, i], index=features.index)

            # Add explained variance ratios
            ml_features['pca_explained_variance'] = pd.Series(
                np.sum(pca.explained_variance_ratio_[:3]) * np.ones(len(features)),
                index=features.index
            )

        # Feature statistics
        if len(numeric_features.columns) > 10:
            # Rolling feature correlations
            for window in [20, 50]:
                feature_corr = numeric_features.rolling(window).corr()
                # Average correlation of each feature with others
                avg_corr = feature_corr.groupby(level=0).mean().mean(axis=1)
                ml_features[f'avg_feature_corr_{window}'] = avg_corr

        return ml_features

    def _generate_regime_features(self, close, volume):
        """Detect market regimes (trending, ranging, volatile)"""
        features = {}

        returns = close.pct_change()

        # Trend strength using ADX
        high = close.rolling(2).max()
        low = close.rolling(2).min()
        adx = pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=14), index=close.index)

        # Define regimes
        features['regime_trending'] = (adx > 25).astype(int)
        features['regime_ranging'] = ((adx <= 25) & (adx > 15)).astype(int)
        features['regime_no_trend'] = (adx <= 15).astype(int)

        # Volatility regimes
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = vol_20.rolling(252).rank(pct=True)

        features['regime_high_vol'] = (vol_percentile > 0.8).astype(int)
        features['regime_normal_vol'] = ((vol_percentile <= 0.8) & (vol_percentile > 0.2)).astype(int)
        features['regime_low_vol'] = (vol_percentile <= 0.2).astype(int)

        # Volume regimes
        vol_sma = volume.rolling(20).mean()
        vol_ratio = volume / vol_sma

        features['regime_high_volume'] = (vol_ratio > 1.5).astype(int)
        features['regime_normal_volume'] = ((vol_ratio <= 1.5) & (vol_ratio > 0.7)).astype(int)
        features['regime_low_volume'] = (vol_ratio <= 0.7).astype(int)

        # Market efficiency (using Hurst exponent approximation)
        for period in [20, 50]:
            # Simplified Hurst calculation
            lags = range(2, min(period//2, 10))
            tau = []
            for lag in lags:
                pp = np.log(close / close.shift(lag)).dropna()
                tau.append(pp.rolling(period).std().iloc[-1])

            if len(tau) > 2:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                features[f'hurst_exponent_{period}'] = poly[0] * 2.0
            else:
                features[f'hurst_exponent_{period}'] = 0.5

        # Regime duration
        regime_trend = features['regime_trending']
        regime_changes = regime_trend.diff().ne(0).cumsum()
        features['regime_duration'] = regime_trend.groupby(regime_changes).cumcount()

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names

    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from a trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return pd.DataFrame()


# Test the enhanced feature engineering
if __name__ == "__main__":
    import yfinance as yf

    print("Testing Enhanced Feature Engineering...")
    print("=" * 60)

    # Fetch test data
    ticker = yf.Ticker('AAPL')
    data = ticker.history(period='2y')

    print(f"Data shape: {data.shape}")

    # Initialize feature engineer
    config = FeatureConfig(
        use_patterns=True,
        use_ml_features=True,
        use_interactions=True
    )
    engineer = AdvancedFeatureEngineer(config)

    # Generate features
    print("\nGenerating features...")
    features = engineer.engineer_features(data)

    print(f"\nGenerated {len(features.columns)} features")
    print("\nFeature categories:")

    # Count features by category
    categories = {
        'Price': [f for f in features.columns if 'price' in f or 'sma' in f or 'ema' in f],
        'Volume': [f for f in features.columns if 'volume' in f or 'obv' in f or 'mfi' in f],
        'Technical': [f for f in features.columns if 'rsi' in f or 'macd' in f or 'bb_' in f],
        'Volatility': [f for f in features.columns if 'volatility' in f or 'atr' in f],
        'Patterns': [f for f in features.columns if 'pattern' in f or 'head' in f or 'triangle' in f],
        'Candlestick': [f for f in features.columns if 'cdl_' in f],
        'Support/Resistance': [f for f in features.columns if 'support' in f or 'resistance' in f or 'pivot' in f],
        'Interactions': [f for f in features.columns if 'cross' in f or 'divergence' in f],
        'ML Features': [f for f in features.columns if 'pca' in f or 'cluster' in f],
        'Regime': [f for f in features.columns if 'regime' in f]
    }

    for category, feature_list in categories.items():
        print(f"{category}: {len(feature_list)} features")

    # Show some example features
    print("\nExample features:")
    print(features[['golden_cross', 'rsi_14', 'macd_hist_12_26', 'head_shoulders', 'regime_trending']].tail(10))

    print("\n✅ Enhanced feature engineering complete!")