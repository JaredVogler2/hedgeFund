"""
Signal Generation & Risk Management System
Professional signal generation with multi-factor scoring and risk controls
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    # Signal thresholds
    min_confidence: float = 0.6
    min_expected_return: float = 0.02  # 2%
    max_volatility: float = 0.5  # 50% annualized
    
    # Market regime filters
    max_vix_level: float = 30.0
    min_market_breadth: float = 0.3
    
    # Risk parameters
    max_positions: int = 20
    max_position_size: float = 0.10
    max_sector_exposure: float = 0.30
    max_correlation: float = 0.7
    
    # Kelly Criterion
    kelly_fraction: float = 0.25
    min_kelly_size: float = 0.01
    max_kelly_size: float = 0.10
    
    # Stop loss
    stop_loss_atr_mult: float = 2.0
    trailing_stop_atr_mult: float = 3.0
    time_stop_days: int = 20
    
    # Score weights
    ml_weight: float = 0.4
    technical_weight: float = 0.3
    sentiment_weight: float = 0.2
    regime_weight: float = 0.1

@dataclass
class Signal:
    """Trading signal with all relevant information"""
    symbol: str
    timestamp: datetime
    direction: str  # 'long' or 'short'
    
    # Predictions
    ml_score: float
    expected_return: float
    predicted_volatility: float
    confidence_interval: Tuple[float, float]
    
    # Technical scores
    technical_score: float
    feature_quality_score: float
    pattern_score: float
    
    # Risk metrics
    position_size: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    
    # Additional info
    sentiment_score: float = 0.0
    regime_score: float = 1.0
    catalyst: str = ""
    
    @property
    def total_score(self) -> float:
        """Calculate total signal score"""
        return (
            self.ml_score * 0.4 +
            self.technical_score * 0.3 +
            self.sentiment_score * 0.2 +
            self.regime_score * 0.1
        )
    
    @property
    def bayesian_score(self) -> float:
        """Calculate Bayesian score for ranking"""
        win_prob = (self.ml_score + 1) / 2  # Convert to probability
        return (win_prob * self.expected_return) / (self.predicted_volatility + 0.01)

class MarketRegimeAnalyzer:
    """Analyze market regime and conditions"""
    
    def __init__(self):
        self.regime_data = {}
        
    def analyze_regime(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze current market regime"""
        regime = {}
        
        # VIX level
        if 'VIX' in market_data:
            current_vix = market_data['VIX']['Close'].iloc[-1]
            vix_percentile = stats.percentileofscore(
                market_data['VIX']['Close'].tail(252), current_vix
            )
            regime['vix_level'] = current_vix
            regime['vix_percentile'] = vix_percentile / 100
            regime['vix_regime'] = 'high' if current_vix > 30 else 'normal' if current_vix > 20 else 'low'
        
        # Market trend
        if 'SPY' in market_data:
            spy = market_data['SPY']['Close']
            sma_50 = spy.rolling(50).mean().iloc[-1]
            sma_200 = spy.rolling(200).mean().iloc[-1]
            
            regime['spy_trend'] = 'bullish' if sma_50 > sma_200 else 'bearish'
            regime['spy_momentum'] = spy.pct_change(20).iloc[-1]
            
            # Market breadth
            advances = sum(1 for _, data in market_data.items() 
                         if data['Close'].pct_change().iloc[-1] > 0)
            regime['market_breadth'] = advances / len(market_data)
        
        # Volatility regime
        returns = pd.DataFrame({
            symbol: data['Close'].pct_change() 
            for symbol, data in market_data.items()
        })
        
        short_vol = returns.tail(20).std().mean() * np.sqrt(252)
        long_vol = returns.tail(60).std().mean() * np.sqrt(252)
        regime['volatility_ratio'] = short_vol / long_vol
        regime['volatility_regime'] = 'expanding' if regime['volatility_ratio'] > 1.2 else 'contracting'
        
        # Correlation regime
        recent_corr = returns.tail(20).corr().values
        avg_correlation = np.mean(recent_corr[np.triu_indices_from(recent_corr, k=1)])
        regime['avg_correlation'] = avg_correlation
        regime['correlation_regime'] = 'high' if avg_correlation > 0.7 else 'normal'
        
        # Calculate regime score (0-1, higher is better)
        regime_score = 1.0
        
        if regime.get('vix_level', 20) > 30:
            regime_score *= 0.5
        elif regime.get('vix_level', 20) > 25:
            regime_score *= 0.8
            
        if regime.get('spy_trend') == 'bearish':
            regime_score *= 0.7
            
        if regime.get('market_breadth', 0.5) < 0.3:
            regime_score *= 0.8
            
        if regime.get('correlation_regime') == 'high':
            regime_score *= 0.9
            
        regime['regime_score'] = regime_score
        
        self.regime_data = regime
        return regime
    
    def get_position_adjustment(self) -> float:
        """Get position size adjustment based on regime"""
        base_adjustment = 1.0
        
        # Reduce position size in high volatility
        if self.regime_data.get('vix_level', 20) > 25:
            base_adjustment *= 0.8
        
        # Reduce in high correlation regime
        if self.regime_data.get('avg_correlation', 0.5) > 0.7:
            base_adjustment *= 0.9
        
        # Reduce in bear market
        if self.regime_data.get('spy_trend') == 'bearish':
            base_adjustment *= 0.85
        
        return base_adjustment

class SignalGenerator:
    """Generate trading signals from ML predictions and features"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.signals_history = []
        
    def generate_signals(self, 
                        predictions: Dict[str, np.ndarray],
                        features: Dict[str, pd.DataFrame],
                        market_data: Dict[str, pd.DataFrame],
                        sentiment_scores: Optional[Dict[str, float]] = None) -> List[Signal]:
        """Generate trading signals from predictions and features"""
        
        # Analyze market regime
        regime = self.regime_analyzer.analyze_regime(market_data)
        logger.info(f"Market regime: {regime}")
        
        # Check if market conditions are suitable
        if not self._check_market_conditions(regime):
            logger.warning("Market conditions not suitable for trading")
            return []
        
        signals = []
        
        for symbol in predictions:
            if symbol not in features or symbol not in market_data:
                continue
                
            # Generate signal for symbol
            signal = self._generate_symbol_signal(
                symbol, 
                predictions[symbol],
                features[symbol],
                market_data[symbol],
                sentiment_scores.get(symbol, 0) if sentiment_scores else 0,
                regime
            )
            
            if signal and self._validate_signal(signal):
                signals.append(signal)
        
        # Rank and filter signals
        signals = self._rank_and_filter_signals(signals)
        
        # Apply portfolio-level risk checks
        signals = self._apply_portfolio_constraints(signals, market_data)
        
        # Store signals
        self.signals_history.extend(signals)
        
        logger.info(f"Generated {len(signals)} valid signals")
        return signals
    
    def _check_market_conditions(self, regime: Dict[str, float]) -> bool:
        """Check if market conditions are suitable for trading"""
        # Don't trade in extreme VIX
        if regime.get('vix_level', 20) > self.config.max_vix_level:
            return False
        
        # Don't trade in extreme correlation
        if regime.get('avg_correlation', 0.5) > 0.85:
            return False
        
        # Don't trade in extreme bear market
        if regime.get('market_breadth', 0.5) < self.config.min_market_breadth:
            return False
        
        return True
    
    def _generate_symbol_signal(self, 
                               symbol: str,
                               prediction: np.ndarray,
                               symbol_features: pd.DataFrame,
                               symbol_data: pd.DataFrame,
                               sentiment_score: float,
                               regime: Dict[str, float]) -> Optional[Signal]:
        """Generate signal for a single symbol"""
        
        # Extract predictions (assuming prediction contains [return, volatility, confidence])
        if len(prediction) >= 3:
            expected_return = prediction[0]
            predicted_volatility = prediction[1]
            ml_confidence = prediction[2]
        else:
            expected_return = prediction[0] if len(prediction) > 0 else 0
            predicted_volatility = 0.02  # Default 2% daily vol
            ml_confidence = 0.5
        
        # Check minimum thresholds
        if abs(expected_return) < self.config.min_expected_return:
            return None
        if predicted_volatility > self.config.max_volatility:
            return None
        if ml_confidence < self.config.min_confidence:
            return None
        
        # Calculate technical score
        technical_score = self._calculate_technical_score(symbol_features)
        
        # Calculate feature quality score
        feature_quality_score = self._calculate_feature_quality_score(symbol_features)
        
        # Calculate pattern score
        pattern_score = self._calculate_pattern_score(symbol_features)
        
        # Determine direction
        direction = 'long' if expected_return > 0 else 'short'
        
        # Calculate position size
        current_price = symbol_data['Close'].iloc[-1]
        atr = symbol_features.get('atr_14', pd.Series([current_price * 0.02])).iloc[-1]
        
        position_size = self._calculate_position_size(
            expected_return,
            predicted_volatility,
            ml_confidence,
            regime.get('regime_score', 1.0)
        )
        
        # Calculate stop loss and take profit
        if direction == 'long':
            stop_loss = current_price - (atr * self.config.stop_loss_atr_mult)
            take_profit = current_price + (abs(expected_return) * current_price * 3)
        else:
            stop_loss = current_price + (atr * self.config.stop_loss_atr_mult)
            take_profit = current_price - (abs(expected_return) * current_price * 3)
        
        # Calculate risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Create signal
        signal = Signal(
            symbol=symbol,
            timestamp=datetime.now(),
            direction=direction,
            ml_score=ml_confidence,
            expected_return=expected_return,
            predicted_volatility=predicted_volatility,
            confidence_interval=(
                expected_return - 2 * predicted_volatility,
                expected_return + 2 * predicted_volatility
            ),
            technical_score=technical_score,
            feature_quality_score=feature_quality_score,
            pattern_score=pattern_score,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            sentiment_score=sentiment_score,
            regime_score=regime.get('regime_score', 1.0)
        )
        
        return signal
    
    def _calculate_technical_score(self, features: pd.DataFrame) -> float:
        """Calculate technical indicator score"""
        score = 0.0
        count = 0
        
        # RSI
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14'].iloc[-1]
            if 30 < rsi < 70:
                score += 0.5
            elif rsi <= 30 or rsi >= 70:
                score += 1.0  # Oversold/overbought
            count += 1
        
        # MACD
        if 'macd_hist_12_26_9' in features.columns:
            macd_hist = features['macd_hist_12_26_9'].iloc[-1]
            macd_slope = features['macd_hist_slope_12_26_9'].iloc[-1] if 'macd_hist_slope_12_26_9' in features.columns else 0
            if macd_hist > 0 and macd_slope > 0:
                score += 1.0
            elif macd_hist < 0 and macd_slope < 0:
                score += 1.0
            else:
                score += 0.3
            count += 1
        
        # Moving average alignment
        if 'sma_50' in features.columns and 'sma_200' in features.columns:
            if features['Close'].iloc[-1] > features['sma_50'].iloc[-1] > features['sma_200'].iloc[-1]:
                score += 1.0  # Bullish alignment
            elif features['Close'].iloc[-1] < features['sma_50'].iloc[-1] < features['sma_200'].iloc[-1]:
                score += 1.0  # Bearish alignment
            else:
                score += 0.3
            count += 1
        
        # Bollinger Band position
        if 'bb_position_20_20' in features.columns:
            bb_pos = features['bb_position_20_20'].iloc[-1]
            if bb_pos < 0.2 or bb_pos > 0.8:
                score += 0.8  # Near bands
            else:
                score += 0.4
            count += 1
        
        # Volume confirmation
        if 'volume_ratio_20' in features.columns:
            vol_ratio = features['volume_ratio_20'].iloc[-1]
            if vol_ratio > 1.5:
                score += 0.8  # High volume
            elif vol_ratio > 1.0:
                score += 0.5
            else:
                score += 0.2
            count += 1
        
        return score / count if count > 0 else 0.5
    
    def _calculate_feature_quality_score(self, features: pd.DataFrame) -> float:
        """Calculate feature quality score based on key alignments"""
        quality_checks = []
        
        # Trend alignment
        if all(col in features.columns for col in ['return_5d', 'return_10d', 'return_20d']):
            returns = [features[col].iloc[-1] for col in ['return_5d', 'return_10d', 'return_20d']]
            if all(r > 0 for r in returns) or all(r < 0 for r in returns):
                quality_checks.append(1.0)
            else:
                quality_checks.append(0.3)
        
        # Momentum alignment
        if 'momentum_aligned_up' in features.columns:
            quality_checks.append(float(features['momentum_aligned_up'].iloc[-1]))
        elif 'momentum_aligned_down' in features.columns:
            quality_checks.append(float(features['momentum_aligned_down'].iloc[-1]))
        
        # Volatility regime
        if 'volatility_regime' in features.columns:
            vol_regime = features['volatility_regime'].iloc[-1]
            quality_checks.append(0.8 if vol_regime < 1.2 else 0.4)
        
        # Support/Resistance proximity
        if 'near_support_20d' in features.columns or 'near_resistance_20d' in features.columns:
            near_level = (features.get('near_support_20d', 0).iloc[-1] or 
                         features.get('near_resistance_20d', 0).iloc[-1])
            quality_checks.append(0.9 if near_level else 0.5)
        
        # Market microstructure
        if 'spread_proxy' in features.columns:
            spread = features['spread_proxy'].iloc[-1]
            spread_avg = features['spread_proxy'].rolling(20).mean().iloc[-1]
            quality_checks.append(0.8 if spread < spread_avg else 0.4)
        
        return np.mean(quality_checks) if quality_checks else 0.5
    
    def _calculate_pattern_score(self, features: pd.DataFrame) -> float:
        """Calculate pattern recognition score"""
        pattern_score = 0.0
        pattern_count = 0
        
        # Check candlestick patterns
        pattern_columns = [col for col in features.columns if col.startswith('pattern_')]
        
        for col in pattern_columns:
            pattern_value = features[col].iloc[-1]
            if pattern_value != 0:
                pattern_score += abs(pattern_value)
                pattern_count += 1
        
        # Check custom patterns
        if 'bullish_pattern_score' in features.columns:
            pattern_score += features['bullish_pattern_score'].iloc[-1]
            pattern_count += 1
        
        if 'bearish_pattern_score' in features.columns:
            pattern_score += features['bearish_pattern_score'].iloc[-1]
            pattern_count += 1
        
        # Breakout patterns
        if 'resistance_break_20d' in features.columns:
            if features['resistance_break_20d'].iloc[-1]:
                pattern_score += 0.8
                pattern_count += 1
        
        if 'support_break_20d' in features.columns:
            if features['support_break_20d'].iloc[-1]:
                pattern_score += 0.8
                pattern_count += 1
        
        return pattern_score / pattern_count if pattern_count > 0 else 0.5
    
    def _calculate_position_size(self, 
                                expected_return: float,
                                volatility: float,
                                confidence: float,
                                regime_score: float) -> float:
        """Calculate position size using Kelly Criterion with adjustments"""
        
        # Basic Kelly fraction
        # f = (p * b - q) / b, where p = win prob, b = win/loss ratio, q = loss prob
        win_prob = (confidence + 1) / 2  # Convert confidence to probability
        loss_prob = 1 - win_prob
        
        # Assume win/loss ratio based on expected return and volatility
        win_loss_ratio = abs(expected_return) / volatility if volatility > 0 else 1.0
        
        # Kelly fraction
        kelly = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        
        # Apply safety factor
        kelly *= self.config.kelly_fraction
        
        # Adjust for regime
        kelly *= regime_score
        
        # Adjust for volatility (reduce size in high vol)
        vol_adjustment = np.exp(-volatility * 10)
        kelly *= vol_adjustment
        
        # Apply bounds
        kelly = np.clip(kelly, self.config.min_kelly_size, self.config.max_kelly_size)
        
        return kelly
    
    def _validate_signal(self, signal: Signal) -> bool:
        """Validate signal meets all criteria"""
        # Check minimum score thresholds
        if signal.total_score < 0.6:
            return False
        
        # Check risk-reward ratio
        if signal.risk_reward_ratio < 1.5:
            return False
        
        # Check position size
        if signal.position_size < self.config.min_kelly_size:
            return False
        
        # Check expected return
        if abs(signal.expected_return) < self.config.min_expected_return:
            return False
        
        return True
    
    def _rank_and_filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank signals by Bayesian score and filter top candidates"""
        if not signals:
            return []
        
        # Sort by Bayesian score
        signals.sort(key=lambda x: x.bayesian_score, reverse=True)
        
        # Filter by minimum score
        signals = [s for s in signals if s.bayesian_score > 0.5]
        
        # Limit to max positions
        signals = signals[:self.config.max_positions]
        
        return signals
    
    def _apply_portfolio_constraints(self, 
                                   signals: List[Signal],
                                   market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Apply portfolio-level constraints"""
        filtered_signals = []
        sector_exposure = {}
        total_exposure = 0.0
        
        # Get correlations
        returns_df = pd.DataFrame({
            symbol: data['Close'].pct_change()
            for symbol, data in market_data.items()
            if symbol in [s.symbol for s in signals]
        })
        correlation_matrix = returns_df.tail(60).corr()
        
        for signal in signals:
            # Check total exposure
            if total_exposure + signal.position_size > 1.0:
                continue
            
            # Check sector exposure (simplified - in production, use real sector data)
            sector = self._get_sector(signal.symbol)
            current_sector_exposure = sector_exposure.get(sector, 0.0)
            if current_sector_exposure + signal.position_size > self.config.max_sector_exposure:
                continue
            
            # Check correlation with existing positions
            if filtered_signals:
                existing_symbols = [s.symbol for s in filtered_signals]
                if signal.symbol in correlation_matrix.columns:
                    correlations = [
                        correlation_matrix.loc[signal.symbol, sym]
                        for sym in existing_symbols
                        if sym in correlation_matrix.columns
                    ]
                    if correlations and max(correlations) > self.config.max_correlation:
                        continue
            
            # Add signal
            filtered_signals.append(signal)
            total_exposure += signal.position_size
            sector_exposure[sector] = current_sector_exposure + signal.position_size
        
        return filtered_signals
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified)"""
        # In production, use real sector mapping
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC']
        finance_symbols = ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C']
        healthcare_symbols = ['JNJ', 'PFE', 'UNH', 'CVS', 'ABBV', 'MRK']
        
        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in finance_symbols:
            return 'Finance'
        elif symbol in healthcare_symbols:
            return 'Healthcare'
        else:
            return 'Other'

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.portfolio_heat = 0.0
        self.position_risks = {}
        
    def calculate_portfolio_risk(self, 
                               positions: List[Dict[str, Any]],
                               market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not positions:
            return {
                'portfolio_heat': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_drawdown_risk': 0.0,
                'correlation_risk': 0.0,
                'concentration_risk': 0.0
            }
        
        # Calculate returns matrix
        symbols = [p['symbol'] for p in positions]
        returns_data = {}
        
        for symbol in symbols:
            if symbol in market_data:
                returns_data[symbol] = market_data[symbol]['Close'].pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data).tail(252)  # 1 year of data
        
        # Portfolio weights
        total_value = sum(p['market_value'] for p in positions)
        weights = np.array([p['market_value'] / total_value for p in positions])
        
        # Portfolio returns
        portfolio_returns = returns_df @ weights
        
        # Calculate risk metrics
        metrics = {}
        
        # 1. Portfolio Heat (total risk from stops)
        portfolio_heat = 0.0
        for position in positions:
            position_value = position['market_value']
            current_price = position['current_price']
            stop_loss = position.get('stop_loss', current_price * 0.95)
            
            risk_pct = (current_price - stop_loss) / current_price
            position_risk = (position_value / total_value) * risk_pct
            portfolio_heat += position_risk
        
        metrics['portfolio_heat'] = portfolio_heat
        
        # 2. Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)
        metrics['var_95'] = abs(var_95)
        
        # 3. Conditional Value at Risk
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        metrics['cvar_95'] = abs(cvar_95)
        
        # 4. Maximum Drawdown Risk
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown_risk'] = abs(drawdowns.min())
        
        # 5. Correlation Risk
        correlation_matrix = returns_df.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        metrics['correlation_risk'] = avg_correlation
        
        # 6. Concentration Risk (Herfindahl index)
        concentration = np.sum(weights ** 2)
        metrics['concentration_risk'] = concentration
        
        self.portfolio_heat = portfolio_heat
        return metrics
    
    def adjust_position_sizes(self, 
                            signals: List[Signal],
                            current_positions: List[Dict[str, Any]],
                            portfolio_value: float) -> List[Signal]:
        """Adjust position sizes based on portfolio risk"""
        
        # Calculate current portfolio risk
        if current_positions:
            current_risk = self.portfolio_heat
        else:
            current_risk = 0.0
        
        # Available risk budget
        risk_budget = 0.08 - current_risk  # 8% max portfolio heat
        
        if risk_budget <= 0:
            logger.warning("No risk budget available")
            return []
        
        # Adjust signal position sizes
        adjusted_signals = []
        remaining_budget = risk_budget
        
        for signal in signals:
            # Calculate position risk
            position_risk = signal.position_size * (
                abs(signal.stop_loss - signal.take_profit) / signal.take_profit
            )
            
            if position_risk <= remaining_budget:
                adjusted_signals.append(signal)
                remaining_budget -= position_risk
            else:
                # Scale down position
                scale_factor = remaining_budget / position_risk
                if scale_factor > 0.1:  # Minimum 10% of original size
                    signal.position_size *= scale_factor
                    adjusted_signals.append(signal)
                    remaining_budget = 0
                    break
        
        return adjusted_signals
    
    def calculate_dynamic_stops(self, 
                              position: Dict[str, Any],
                              market_data: pd.DataFrame,
                              features: pd.DataFrame) -> Dict[str, float]:
        """Calculate dynamic stop loss and trailing stops"""
        
        current_price = position['current_price']
        entry_price = position['entry_price']
        
        # Get ATR for volatility-based stops
        atr = features.get('atr_14', pd.Series([current_price * 0.02])).iloc[-1]
        
        # Initial stop loss
        if position['side'] == 'long':
            initial_stop = entry_price - (atr * self.config.stop_loss_atr_mult)
            
            # Trailing stop (only moves up)
            if current_price > entry_price:
                trailing_stop = current_price - (atr * self.config.trailing_stop_atr_mult)
                stop_loss = max(initial_stop, trailing_stop, position.get('stop_loss', 0))
            else:
                stop_loss = initial_stop
        else:
            initial_stop = entry_price + (atr * self.config.stop_loss_atr_mult)
            
            # Trailing stop (only moves down)
            if current_price < entry_price:
                trailing_stop = current_price + (atr * self.config.trailing_stop_atr_mult)
                stop_loss = min(initial_stop, trailing_stop, position.get('stop_loss', float('inf')))
            else:
                stop_loss = initial_stop
        
        # Time-based stop
        days_held = (datetime.now() - position['entry_date']).days
        if days_held > self.config.time_stop_days:
            # Tighten stop after holding period
            if position['side'] == 'long':
                time_stop = current_price * 0.98  # 2% stop
                stop_loss = max(stop_loss, time_stop)
            else:
                time_stop = current_price * 1.02
                stop_loss = min(stop_loss, time_stop)
        
        # Volatility-adjusted stop
        current_volatility = features.get('volatility_20d', pd.Series([0.2])).iloc[-1]
        if current_volatility > 0.3:  # High volatility
            # Widen stops in high volatility
            if position['side'] == 'long':
                stop_loss *= 0.95
            else:
                stop_loss *= 1.05
        
        return {
            'stop_loss': stop_loss,
            'trailing_stop': stop_loss,
            'risk_amount': abs(current_price - stop_loss) * position['shares']
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Sample predictions (symbol -> [return, volatility, confidence])
    predictions = {
        'AAPL': np.array([0.03, 0.02, 0.75]),
        'MSFT': np.array([0.025, 0.018, 0.80]),
        'GOOGL': np.array([-0.02, 0.025, 0.65]),
        'JPM': np.array([0.015, 0.015, 0.70]),
    }
    
    # Sample features
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    sample_features = {}
    sample_market_data = {}
    
    for symbol in predictions.keys():
        # Generate fake features
        features = pd.DataFrame(index=dates)
        features['rsi_14'] = np.random.uniform(30, 70, len(dates))
        features['macd_hist_12_26_9'] = np.random.randn(len(dates)) * 0.5
        features['bb_position_20_20'] = np.random.uniform(0, 1, len(dates))
        features['volume_ratio_20'] = np.random.uniform(0.5, 2, len(dates))
        features['atr_14'] = np.random.uniform(1, 3, len(dates))
        features['momentum_aligned_up'] = np.random.choice([0, 1], len(dates))
        features['volatility_20d'] = np.random.uniform(0.1, 0.3, len(dates))
        features['Close'] = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        features['sma_50'] = features['Close'].rolling(50).mean()
        features['sma_200'] = features['Close'].rolling(50).mean() * 0.95
        
        sample_features[symbol] = features
        
        # Generate fake market data
        market_data = pd.DataFrame(index=dates)
        market_data['Close'] = features['Close']
        market_data['Volume'] = np.random.uniform(1e6, 1e7, len(dates))
        
        sample_market_data[symbol] = market_data
    
    # Add market indices
    sample_market_data['SPY'] = pd.DataFrame({
        'Close': 400 + np.cumsum(np.random.randn(len(dates)) * 5)
    }, index=dates)
    
    sample_market_data['VIX'] = pd.DataFrame({
        'Close': np.random.uniform(15, 25, len(dates))
    }, index=dates)
    
    # Initialize signal generator
    signal_generator = SignalGenerator()
    
    # Generate signals
    print("Generating trading signals...")
    signals = signal_generator.generate_signals(
        predictions,
        sample_features,
        sample_market_data,
        sentiment_scores={'AAPL': 0.7, 'MSFT': 0.8, 'GOOGL': 0.3, 'JPM': 0.6}
    )
    
    print(f"\nGenerated {len(signals)} signals:")
    for signal in signals:
        print(f"\n{signal.symbol}:")
        print(f"  Direction: {signal.direction}")
        print(f"  Expected Return: {signal.expected_return:.2%}")
        print(f"  Position Size: {signal.position_size:.2%}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Risk/Reward: {signal.risk_reward_ratio:.2f}")
        print(f"  Total Score: {signal.total_score:.2f}")
        print(f"  Bayesian Score: {signal.bayesian_score:.2f}")
    
    # Test risk manager
    risk_manager = RiskManager()
    
    # Create sample positions
    sample_positions = [
        {
            'symbol': 'AAPL',
            'market_value': 10000,
            'current_price': 150,
            'entry_price': 145,
            'stop_loss': 142,
            'shares': 66.67,
            'side': 'long',
            'entry_date': datetime.now() - timedelta(days=5)
        }
    ]
    
    # Calculate portfolio risk
    print("\nPortfolio Risk Metrics:")
    risk_metrics = risk_manager.calculate_portfolio_risk(sample_positions, sample_market_data)
    for metric, value in risk_metrics.items():
        print(f"  {metric}: {value:.4f}")
