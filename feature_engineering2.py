"""
Advanced Feature Engineering for ML Trading System

This module provides comprehensive feature engineering capabilities for the ML trading system,
including technical indicators, microstructure features, and pattern recognition.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Price-based features
    return_periods: List[int] = None
    ma_periods: List[int] = None

    # Technical indicators
    rsi_periods: List[int] = None
    bb_periods: List[int] = None
    atr_periods: List[int] = None

    # Volume features
    volume_ma_periods: List[int] = None

    # Microstructure
    spread_periods: List[int] = None

    def __post_init__(self):
        # Set defaults if not provided
        if self.return_periods is None:
            self.return_periods = [1, 2, 3, 5, 10, 20, 60]
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 100, 200]
        if self.rsi_periods is None:
            self.rsi_periods = [7, 14, 21, 28]
        if self.bb_periods is None:
            self.bb_periods = [10, 20, 30]
        if self.atr_periods is None:
            self.atr_periods = [5, 10, 14, 20]
        if self.volume_ma_periods is None:
            self.volume_ma_periods = [5, 10, 20, 50]
        if self.spread_periods is None:
            self.spread_periods = [5, 10, 20]


class FeatureEngineer:
    """
    Advanced feature engineering for ML trading

    Generates 200+ features across multiple categories:
    - Price-based features
    - Technical indicators
    - Volume analysis
    - Volatility measures
    - Market microstructure
    - Pattern recognition
    - Sentiment indicators
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer

        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()
        self.logger = logging.getLogger(__name__)

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the given data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with all engineered features
        """
        try:
            # Extract price and volume data
            close = data['Close']
            high = data['High']
            low = data['Low']
            open_price = data['Open']  # Renamed from 'open' to avoid conflict with builtin
            volume = data['Volume']

            features = {}

            # Generate all feature categories
            self.logger.info("Generating basic price features...")
            features.update(self._generate_basic_features(close, high, low, open_price))

            self.logger.info("Generating volume features...")
            features.update(self._generate_volume_features(close, volume))

            self.logger.info("Generating technical indicators...")
            features.update(self._generate_technical_indicators(close, high, low, open_price, volume))

            self.logger.info("Generating volatility features...")
            features.update(self._generate_volatility_features(close, high, low, open_price))

            self.logger.info("Generating microstructure features...")
            features.update(self._generate_microstructure_features(close, high, low, open_price, volume))

            self.logger.info("Generating pattern features...")
            features.update(self._generate_pattern_features(close, high, low, open_price))

            self.logger.info("Generating sentiment features...")
            features.update(self._generate_sentiment_features(close, volume))

            # Convert to DataFrame
            feature_df = pd.DataFrame(features, index=data.index)

            # Add time-based features
            feature_df = self._add_time_features(feature_df)

            # Add interaction features
            feature_df = self._add_interaction_features(feature_df)

            # Handle missing values
            feature_df = self._handle_missing_values(feature_df)

            self.logger.info(f"Generated {len(feature_df.columns)} features")

            return feature_df

        except Exception as e:
            self.logger.error(f"Error in feature generation: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_basic_features(self, close, high, low, open_price):
        """Generate basic price-based features"""
        features = {}

        # Returns
        for period in self.config.return_periods:
            features[f'return_{period}'] = close.pct_change(period)
            features[f'log_return_{period}'] = np.log(close / close.shift(period))

        # Moving averages
        for period in self.config.ma_periods:
            features[f'sma_{period}'] = close.rolling(period).mean()
            features[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            features[f'price_to_sma_{period}'] = close / features[f'sma_{period}']
            features[f'price_to_ema_{period}'] = close / features[f'ema_{period}']

        # Moving average slopes
        for period in [20, 50, 200]:
            if f'sma_{period}' in features:
                features[f'sma_{period}_slope'] = features[f'sma_{period}'].pct_change(5)

        # Price ratios
        features['high_low_ratio'] = high / low
        features['close_open_ratio'] = close / open_price  # Fixed: using open_price

        # Price position within daily range
        features['price_position'] = (close - low) / (high - low)

        # Gap features
        features['gap'] = open_price / close.shift(1) - 1  # Fixed: using open_price
        features['gap_up'] = (features['gap'] > 0).astype(int)
        features['gap_down'] = (features['gap'] < 0).astype(int)

        # Cumulative returns
        for period in [5, 10, 20]:
            features[f'cum_return_{period}'] = close.pct_change(period)

        # Price acceleration
        returns = close.pct_change()
        features['return_acceleration'] = returns - returns.shift(1)

        # Highs and lows
        for period in [5, 10, 20, 50]:
            features[f'high_{period}'] = high.rolling(period).max()
            features[f'low_{period}'] = low.rolling(period).min()
            features[f'high_low_range_{period}'] = features[f'high_{period}'] - features[f'low_{period}']
            features[f'close_to_high_{period}'] = close / features[f'high_{period}']
            features[f'close_to_low_{period}'] = close / features[f'low_{period}']

        return features

    def _generate_volume_features(self, close, volume):
        """Generate volume-based features"""
        features = {}

        # Volume moving averages
        for period in self.config.volume_ma_periods:
            features[f'volume_sma_{period}'] = volume.rolling(period).mean()
            features[f'volume_ratio_{period}'] = volume / features[f'volume_sma_{period}']

        # Volume trends
        features['volume_trend'] = volume.rolling(10).mean() / volume.rolling(30).mean()

        # VWAP
        typical_price = close  # Simplified VWAP
        features['vwap'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        features['price_to_vwap'] = close / features['vwap']

        # Volume rate of change
        for period in [5, 10, 20]:
            features[f'volume_roc_{period}'] = volume.pct_change(period)

        # Money Flow
        features['money_flow'] = close * volume
        for period in [5, 10, 20]:
            features[f'money_flow_sma_{period}'] = features['money_flow'].rolling(period).mean()

        # Volume-price correlation
        for period in [20, 60]:
            features[f'volume_price_corr_{period}'] = close.rolling(period).corr(volume)

        # Accumulation/Distribution
        features['acc_dist'] = ((close - low) - (high - close)) / (high - low) * volume
        features['acc_dist_sma'] = features['acc_dist'].rolling(20).mean()

        # Volume volatility
        features['volume_volatility'] = volume.rolling(20).std() / volume.rolling(20).mean()

        return features

    def _generate_technical_indicators(self, close, high, low, open_price, volume):
        """Generate technical indicators using TA-Lib"""
        features = {}

        # Convert to numpy arrays for TA-Lib
        close_arr = close.values
        high_arr = high.values
        low_arr = low.values
        open_arr = open_price.values  # Fixed: using open_price
        volume_arr = volume.values

        # RSI
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = talib.RSI(close_arr, timeperiod=period)
            # RSI divergence
            rsi = features[f'rsi_{period}']
            features[f'rsi_{period}_divergence'] = (close.pct_change(5) * 100) - (pd.Series(rsi).pct_change(5) * 100)

        # MACD
        macd, signal, hist = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        features['macd_cross'] = np.where(macd > signal, 1, -1)

        # Bollinger Bands
        for period in self.config.bb_periods:
            upper, middle, lower = talib.BBANDS(close_arr, timeperiod=period)
            features[f'bb_upper_{period}'] = upper
            features[f'bb_middle_{period}'] = middle
            features[f'bb_lower_{period}'] = lower
            features[f'bb_width_{period}'] = upper - lower
            features[f'bb_position_{period}'] = (close_arr - lower) / (upper - lower)
            # Bollinger Band squeeze
            features[f'bb_squeeze_{period}'] = features[f'bb_width_{period}'] / middle

        # Stochastic
        slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr,
                                   fastk_period=14, slowk_period=3, slowd_period=3)
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        features['stoch_cross'] = np.where(slowk > slowd, 1, -1)

        # ADX
        features['adx'] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=14)
        features['di_diff'] = features['plus_di'] - features['minus_di']

        # CCI
        features['cci'] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=14)

        # MFI
        features['mfi'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=14)

        # Williams %R
        features['willr'] = talib.WILLR(high_arr, low_arr, close_arr, timeperiod=14)

        # SAR
        features['sar'] = talib.SAR(high_arr, low_arr, acceleration=0.02, maximum=0.2)
        features['sar_signal'] = np.where(close_arr > features['sar'], 1, -1)

        # OBV
        features['obv'] = talib.OBV(close_arr, volume_arr)
        features['obv_ema'] = talib.EMA(features['obv'], timeperiod=20)
        features['obv_signal'] = features['obv'] - features['obv_ema']

        # CMF (Chaikin Money Flow)
        features['cmf'] = talib.ADOSC(high_arr, low_arr, close_arr, volume_arr, fastperiod=3, slowperiod=10)

        # ROC
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = talib.ROC(close_arr, timeperiod=period)

        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = talib.MOM(close_arr, timeperiod=period)

        # TRIX
        features['trix'] = talib.TRIX(close_arr, timeperiod=15)

        # Ultimate Oscillator
        features['ultosc'] = talib.ULTOSC(high_arr, low_arr, close_arr)

        # Aroon
        aroon_up, aroon_down = talib.AROON(high_arr, low_arr, timeperiod=14)
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_oscillator'] = aroon_up - aroon_down

        return features

    def _generate_volatility_features(self, close, high, low, open_price):
        """Generate volatility-based features"""
        features = {}

        # Historical volatility
        returns = close.pct_change()
        for period in [5, 10, 20, 60]:
            features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)

        # Average True Range (ATR)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        for period in self.config.atr_periods:
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / close

        # Bollinger Bands (additional volatility features)
        for period in [10, 20, 30]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'bb_upper_{period}'] = ma + 2 * std
            features[f'bb_lower_{period}'] = ma - 2 * std
            features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
            features[f'bb_position_{period}'] = (close - features[f'bb_lower_{period}']) / features[
                f'bb_width_{period}']

        # Keltner Channels
        for period in [10, 20]:
            ma = close.rolling(period).mean()
            atr = true_range.rolling(period).mean()
            features[f'kc_upper_{period}'] = ma + 2 * atr
            features[f'kc_lower_{period}'] = ma - 2 * atr
            features[f'kc_position_{period}'] = (close - features[f'kc_lower_{period}']) / (
                        features[f'kc_upper_{period}'] - features[f'kc_lower_{period}'])

        # Donchian Channels
        for period in [10, 20, 50]:
            features[f'dc_upper_{period}'] = high.rolling(period).max()
            features[f'dc_lower_{period}'] = low.rolling(period).min()
            features[f'dc_mid_{period}'] = (features[f'dc_upper_{period}'] + features[f'dc_lower_{period}']) / 2
            features[f'dc_position_{period}'] = (close - features[f'dc_lower_{period}']) / (
                        features[f'dc_upper_{period}'] - features[f'dc_lower_{period}'])

        # Chandelier Exit
        for period in [22]:
            features[f'chandelier_long_{period}'] = high.rolling(period).max() - 3 * true_range.rolling(period).mean()
            features[f'chandelier_short_{period}'] = low.rolling(period).min() + 3 * true_range.rolling(period).mean()

        # Ulcer Index
        for period in [14, 30]:
            max_close = close.rolling(period).max()
            drawdown = ((close - max_close) / max_close) * 100
            features[f'ulcer_index_{period}'] = np.sqrt((drawdown ** 2).rolling(period).mean())

        # Chaikin Volatility
        hl_spread = high - low
        for period in [10, 20]:
            features[f'chaikin_volatility_{period}'] = (hl_spread.rolling(period).mean() - hl_spread.rolling(
                period).mean().shift(period)) / hl_spread.rolling(period).mean().shift(period)

        # Normalized Average True Range
        for period in [14]:
            features[f'natr_{period}'] = (true_range.rolling(period).mean() / close) * 100

        # Relative Volatility Index
        for period in [10, 14]:
            std_dev = returns.rolling(period).std()
            rvi_up = pd.Series(0, index=returns.index)
            rvi_down = pd.Series(0, index=returns.index)
            rvi_up[returns > 0] = std_dev[returns > 0]
            rvi_down[returns <= 0] = std_dev[returns <= 0]
            rvi = 100 * rvi_up.rolling(period).mean() / (
                        rvi_up.rolling(period).mean() + rvi_down.rolling(period).mean())
            features[f'rvi_{period}'] = rvi

        # Historical Volatility Ratio
        for short, long in [(10, 100), (20, 100)]:
            features[f'volatility_ratio_{short}_{long}'] = features[f'volatility_{short}'] / features[
                f'volatility_{long}']

        # Volatility of Volatility
        for period in [20, 60]:
            vol_series = returns.rolling(20).std() * np.sqrt(252)
            features[f'vol_of_vol_{period}'] = vol_series.rolling(period).std()

        # Price Range Metrics
        features['daily_range'] = (high - low) / close
        features['daily_range_ma20'] = features['daily_range'].rolling(20).mean()
        features['range_expansion'] = features['daily_range'] / features['daily_range_ma20']

        # Garman-Klass Volatility (FIXED: using open_price instead of open)
        if len(high) > 0:
            rs = np.log(high / close) * np.log(high / open_price)  # Fixed line 470
            co = np.log(close / open_price)  # Fixed
            gk_vol = np.sqrt(
                252 / len(high) * (0.5 * (np.log(high / low) ** 2).sum() - (2 * np.log(2) - 1) * (co ** 2).sum()))
            features['garman_klass_vol'] = gk_vol

        # Parkinson Volatility
        if len(high) > 0:
            park_vol = np.sqrt(252 / (4 * len(high) * np.log(2)) * ((np.log(high / low) ** 2).sum()))
            features['parkinson_vol'] = park_vol

        # Rogers-Satchell Volatility
        if len(high) > 0:
            rs_vol = np.sqrt(252 / len(high) * ((np.log(high / close) * np.log(high / open_price) +
                                                 np.log(low / close) * np.log(low / open_price)).sum()))  # Fixed
            features['rogers_satchell_vol'] = rs_vol

        return features

    def _generate_microstructure_features(self, close, high, low, open_price, volume):
        """Generate market microstructure features"""
        features = {}

        # Spread estimators
        features['hl_spread'] = (high - low) / close
        features['co_spread'] = abs(close - open_price) / close  # Fixed: using open_price

        # Effective spread proxy
        features['effective_spread'] = 2 * abs(close - (high + low) / 2) / close

        # Roll's implied spread
        returns = close.pct_change()
        for period in self.config.spread_periods:
            autocov = returns.rolling(period).apply(lambda x: x.autocorr(lag=1), raw=False)
            features[f'roll_spread_{period}'] = 2 * np.sqrt(-autocov.clip(upper=0))

        # Amihud illiquidity
        features['amihud_illiquidity'] = abs(returns) / (volume * close)
        features['amihud_illiquidity_ma'] = features['amihud_illiquidity'].rolling(20).mean()

        # Kyle's lambda (simplified version)
        features['kyle_lambda'] = abs(returns) / np.log1p(volume)
        features['kyle_lambda_ma'] = features['kyle_lambda'].rolling(20).mean()

        # Hasbrouck's lambda
        # Simplified: using volume imbalance
        volume_imbalance = volume - volume.rolling(20).mean()
        features['hasbrouck_lambda'] = abs(returns) / abs(volume_imbalance + 1)

        # Price impact
        features['price_impact'] = abs(returns) / (volume / volume.rolling(20).mean())

        # Quote imbalance (using high-low as proxy for bid-ask)
        features['quote_imbalance'] = (close - (high + low) / 2) / (high - low)

        # Probability of informed trading (PIN) proxy
        # Using volume and price movement correlation
        features['pin_proxy'] = abs(returns).rolling(20).corr(volume)

        # Realized spread
        features['realized_spread'] = 2 * close * abs(returns)

        # Depth imbalance (using volume at different price levels)
        features['depth_imbalance'] = (volume * (close > close.shift())).rolling(20).sum() / volume.rolling(20).sum()

        return features

    def _generate_pattern_features(self, close, high, low, open_price):
        """Generate pattern recognition features"""
        features = {}

        # Convert to numpy arrays for TA-Lib
        open_arr = open_price.values  # Fixed: using open_price
        high_arr = high.values
        low_arr = low.values
        close_arr = close.values

        # Candlestick patterns
        # Reversal patterns
        features['cdl_hammer'] = talib.CDLHAMMER(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_inverted_hammer'] = talib.CDLINVERTEDHAMMER(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_hanging_man'] = talib.CDLHANGINGMAN(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_engulfing'] = talib.CDLENGULFING(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_harami'] = talib.CDLHARAMI(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_piercing'] = talib.CDLPIERCING(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_dark_cloud'] = talib.CDLDARKCLOUDCOVER(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_morning_star'] = talib.CDLMORNINGSTAR(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_evening_star'] = talib.CDLEVENINGSTAR(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_morning_doji'] = talib.CDLMORNINGDOJISTAR(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_evening_doji'] = talib.CDLEVENINGDOJISTAR(open_arr, high_arr, low_arr, close_arr) / 100

        # Continuation patterns
        features['cdl_three_white'] = talib.CDL3WHITESOLDIERS(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_three_black'] = talib.CDL3BLACKCROWS(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_three_inside'] = talib.CDL3INSIDE(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_three_outside'] = talib.CDL3OUTSIDE(open_arr, high_arr, low_arr, close_arr) / 100

        # Doji patterns
        features['cdl_doji'] = talib.CDLDOJI(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_long_doji'] = talib.CDLLONGLEGGEDDOJI(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(open_arr, high_arr, low_arr, close_arr) / 100
        features['cdl_gravestone_doji'] = talib.CDLGRAVESTONEDOJI(open_arr, high_arr, low_arr, close_arr) / 100

        # Support/Resistance levels
        for period in [20, 50, 100]:
            features[f'resistance_{period}'] = high.rolling(period).max()
            features[f'support_{period}'] = low.rolling(period).min()
            features[f'sr_range_{period}'] = features[f'resistance_{period}'] - features[f'support_{period}']
            features[f'sr_position_{period}'] = (close - features[f'support_{period}']) / features[f'sr_range_{period}']

            # Distance to support/resistance
            features[f'dist_to_resistance_{period}'] = (features[f'resistance_{period}'] - close) / close
            features[f'dist_to_support_{period}'] = (close - features[f'support_{period}']) / close

        # Pivot points
        pivot = (high + low + close) / 3
        features['pivot'] = pivot
        features['r1'] = 2 * pivot - low
        features['s1'] = 2 * pivot - high
        features['r2'] = pivot + (high - low)
        features['s2'] = pivot - (high - low)
        features['r3'] = high + 2 * (pivot - low)
        features['s3'] = low - 2 * (high - pivot)

        # Price relative to pivot levels
        features['close_to_pivot'] = close / pivot
        features['close_to_r1'] = close / features['r1']
        features['close_to_s1'] = close / features['s1']

        # Chart patterns (simplified detection)
        # Head and shoulders
        for period in [20, 40]:
            rolling_max = close.rolling(period).max()
            rolling_min = close.rolling(period).min()
            mid_point = (rolling_max + rolling_min) / 2

            # Simple pattern strength indicator
            features[f'pattern_strength_{period}'] = (close - mid_point).rolling(period).std() / mid_point

        # Trend line breaks
        for period in [20, 50]:
            # Simple linear regression as trend line
            x = np.arange(period)
            slopes = close.rolling(period).apply(
                lambda y: np.polyfit(x, y, 1)[0] if len(y) == period else np.nan,
                raw=True
            )
            features[f'trend_slope_{period}'] = slopes

            # Distance from trend
            trend_values = close.rolling(period).apply(
                lambda y: np.polyval(np.polyfit(x, y, 1), period - 1) if len(y) == period else np.nan,
                raw=True
            )
            features[f'dist_from_trend_{period}'] = (close - trend_values) / trend_values

        return features

    def _generate_sentiment_features(self, close, volume):
        """Generate market sentiment features"""
        features = {}

        # Put/Call ratio proxy (using volume patterns)
        # High volume on down moves vs up moves
        returns = close.pct_change()
        down_volume = volume[returns < 0].rolling(20).sum()
        up_volume = volume[returns > 0].rolling(20).sum()
        features['volume_put_call_ratio'] = down_volume / (up_volume + 1)

        # Market regime detection
        features['market_regime'] = returns.rolling(60).mean() / returns.rolling(60).std()

        # Trend strength
        for period in [20, 50, 100]:
            # ADX is already calculated in technical indicators
            # Here we add trend consistency
            features[f'trend_consistency_{period}'] = (returns > 0).rolling(period).mean()

            # Trend quality (how smooth is the trend)
            cumret = (1 + returns).rolling(period).apply(lambda x: x.prod(), raw=True) - 1
            path_length = returns.rolling(period).apply(lambda x: np.abs(x).sum(), raw=True)
            features[f'trend_quality_{period}'] = cumret / (path_length + 0.0001)

        # Momentum quality
        for period in [20, 60]:
            features[f'momentum_quality_{period}'] = returns.rolling(period).mean() / returns.rolling(period).std()

        # Fear index (simplified VIX proxy)
        features['fear_index'] = returns.rolling(20).std() * np.sqrt(252) * 100

        # Greed index (using price position and volume)
        features['greed_index'] = ((close / close.rolling(252).max()) *
                                   (volume / volume.rolling(20).mean())).rolling(20).mean()

        # Market breadth (using rolling correlation with market)
        # Simplified: correlation with own moving average
        features['market_breadth'] = close.rolling(20).corr(close.rolling(50).mean())

        # Sentiment momentum
        sentiment_proxy = features['fear_index'] - features['greed_index']
        features['sentiment_momentum'] = sentiment_proxy - sentiment_proxy.rolling(20).mean()

        # Volume sentiment
        features['volume_sentiment'] = (volume * returns).rolling(20).sum() / volume.rolling(20).sum()

        # Price-volume divergence
        for period in [20, 50]:
            price_change = close.pct_change(period)
            volume_change = volume.pct_change(period)
            features[f'price_volume_divergence_{period}'] = price_change - volume_change

        # Accumulation/Distribution sentiment
        ad_line = ((close - low) - (high - close)) / (high - low) * volume
        features['ad_sentiment'] = ad_line.rolling(20).mean() / volume.rolling(20).mean()

        return features

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Trading session features
        df['is_morning'] = (df['hour'] >= 9) & (df['hour'] < 12)
        df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] < 16)
        df['is_opening'] = (df['hour'] == 9) & (df.index.minute < 45)
        df['is_closing'] = (df['hour'] == 15) & (df.index.minute > 15)

        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Special days
        df['is_monday'] = df['day_of_week'] == 0
        df['is_friday'] = df['day_of_week'] == 4
        df['is_month_start'] = df['day_of_month'] <= 3
        df['is_month_end'] = df['day_of_month'] >= 27
        df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]) & df['is_month_end']

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key indicators"""
        # RSI and volume interaction
        if 'rsi_14' in df.columns and 'volume_ratio_20' in df.columns:
            df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio_20']
            df['rsi_volume_divergence'] = (df['rsi_14'] - 50) * (df['volume_ratio_20'] - 1)

        # Price position and volatility
        if 'bb_position_20' in df.columns and 'volatility_20' in df.columns:
            df['position_volatility_interaction'] = df['bb_position_20'] * df['volatility_20']
            df['extreme_position_high_vol'] = ((df['bb_position_20'] > 0.8) | (df['bb_position_20'] < 0.2)) * df[
                'volatility_20']

        # Momentum and trend
        if 'rsi_14' in df.columns and 'adx' in df.columns:
            df['momentum_trend_interaction'] = df['rsi_14'] * df['adx'] / 100

        # Volume and volatility
        if 'volume_ratio_20' in df.columns and 'atr_pct_14' in df.columns:
            df['volume_volatility_interaction'] = df['volume_ratio_20'] * df['atr_pct_14']

        # MACD and volume
        if 'macd_hist' in df.columns and 'volume_ratio_20' in df.columns:
            df['macd_volume_confirmation'] = np.sign(df['macd_hist']) * df['volume_ratio_20']

        # Support/resistance and volume
        if 'dist_to_resistance_20' in df.columns and 'volume_ratio_20' in df.columns:
            df['resistance_volume_interaction'] = df['dist_to_resistance_20'] * df['volume_ratio_20']

        # Fear/greed and volume
        if 'fear_index' in df.columns and 'greed_index' in df.columns and 'volume_ratio_20' in df.columns:
            df['sentiment_volume_interaction'] = (df['greed_index'] - df['fear_index']) * df['volume_ratio_20']

        # Time and volatility
        if 'hour' in df.columns and 'volatility_20' in df.columns:
            df['time_volatility_interaction'] = df['hour'] * df['volatility_20']

        # Trend and time
        if 'trend_consistency_20' in df.columns and 'is_month_end' in df.columns:
            df['trend_monthend_interaction'] = df['trend_consistency_20'] * df['is_month_end']

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # First, forward fill most features
        df = df.fillna(method='ffill')

        # For any remaining NaN values (typically at the beginning), use backward fill
        df = df.fillna(method='bfill')

        # For any still remaining NaN values, fill with zeros
        # This should be rare and only happen for features that can't be calculated
        df = df.fillna(0)

        # Drop any columns that are all NaN or all zeros
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.loc[:, df.notna().any(axis=0)]

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated"""
        # This is useful for debugging and understanding the feature space
        feature_names = []

        # Basic features
        for period in self.config.return_periods:
            feature_names.extend([f'return_{period}', f'log_return_{period}'])

        for period in self.config.ma_periods:
            feature_names.extend([
                f'sma_{period}', f'ema_{period}',
                f'price_to_sma_{period}', f'price_to_ema_{period}'
            ])

        # Add other feature categories...
        # This is a simplified version - you'd add all feature names

        return feature_names