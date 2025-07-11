"""
Professional ML Trading System - Core Architecture
Hedge fund-quality system with GPU acceleration, ensemble ML, and live trading
"""

import os
import sys
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiohttp
from functools import lru_cache, wraps
import time

# GPU Configuration
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use first GPU (3070 Ti)
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    warnings.warn("GPU not available, using CPU")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# System Configuration
@dataclass
class SystemConfig:
    """Central configuration for the trading system"""
    # Data settings
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    
    # Trading parameters
    max_positions: int = 20
    max_position_size: float = 0.10  # 10% max per position
    max_sector_exposure: float = 0.30  # 30% max per sector
    max_portfolio_heat: float = 0.08  # 8% max portfolio risk
    
    # ML parameters
    n_features_select: int = 150
    ensemble_min_agreement: float = 0.60
    retrain_frequency: int = 7  # days
    
    # Execution settings
    slippage_bps: float = 5.0  # basis points
    commission_per_share: float = 0.005
    min_volume_filter: float = 1_000_000  # $1M daily volume
    
    # Risk parameters
    stop_loss_atr_multiple: float = 2.0
    position_size_kelly_fraction: float = 0.25
    
    # API settings
    alpaca_paper: bool = True
    openai_model: str = "gpt-4"
    
    # Performance
    n_jobs: int = -1  # Use all CPU cores
    chunk_size: int = 50  # Symbols per chunk
    
    def __post_init__(self):
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "features").mkdir(exist_ok=True)
        (self.data_dir / "models").mkdir(exist_ok=True)

# Watchlist Management
class WatchlistManager:
    """Manages trading universe with dynamic updates"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.watchlist_file = config.data_dir / "watchlist.json"
        self._watchlist = self._load_watchlist()
        
    def _load_watchlist(self) -> Dict[str, List[str]]:
        """Load watchlist from file or create default"""
        if self.watchlist_file.exists():
            with open(self.watchlist_file, 'r') as f:
                return json.load(f)
        
        # Default watchlist
        default_watchlist = {
            "mega_caps": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ"],
            "large_caps": ["V", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "CVX", "ABBV", "PFE", "TMO", "CSCO", "ACN", "AVGO", "COST"],
            "growth": ["CRM", "ADBE", "NFLX", "PYPL", "SQ", "SHOP", "ROKU", "SNAP", "PINS", "UBER", "ABNB", "DASH"],
            "value": ["WMT", "KO", "PEP", "MCD", "PM", "T", "VZ", "IBM", "INTC", "GE", "F", "GM"],
            "momentum": ["AMD", "MRNA", "ZM", "DOCU", "PTON", "NET", "DDOG", "SNOW", "U", "RBLX"],
            "sector_etfs": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"],
            "international": ["TSM", "BABA", "NVO", "ASML", "SAP", "TM", "SONY", "SHOP", "MELI", "SE"],
            "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "PDBC", "WOOD", "CORN", "WEAT", "SOYB"],
            "bonds_rates": ["TLT", "IEF", "SHY", "AGG", "BND", "HYG", "LQD", "EMB", "TIP", "VTEB"],
            "volatility": ["VXX", "UVXY", "SVXY", "VIX", "VIXY"],
            "crypto_proxies": ["COIN", "MSTR", "SQ", "PYPL", "MARA", "RIOT", "HUT", "BITF"]
        }
        
        # Save default watchlist
        with open(self.watchlist_file, 'w') as f:
            json.dump(default_watchlist, f, indent=2)
            
        return default_watchlist
    
    def get_all_symbols(self) -> List[str]:
        """Get all unique symbols from watchlist"""
        all_symbols = []
        for category, symbols in self._watchlist.items():
            all_symbols.extend(symbols)
        return list(set(all_symbols))
    
    def add_symbol(self, symbol: str, category: str = "custom"):
        """Add symbol to watchlist"""
        if category not in self._watchlist:
            self._watchlist[category] = []
        if symbol not in self._watchlist[category]:
            self._watchlist[category].append(symbol)
            self._save_watchlist()
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from all categories"""
        for category in self._watchlist:
            if symbol in self._watchlist[category]:
                self._watchlist[category].remove(symbol)
        self._save_watchlist()
    
    def _save_watchlist(self):
        """Save watchlist to file"""
        with open(self.watchlist_file, 'w') as f:
            json.dump(self._watchlist, f, indent=2)
    
    def update_liquidity_filter(self, min_volume: float = 1_000_000):
        """Remove symbols that don't meet liquidity requirements"""
        symbols_to_check = self.get_all_symbols()
        liquid_symbols = []
        
        logger.info(f"Checking liquidity for {len(symbols_to_check)} symbols...")
        
        # Check in batches
        for i in range(0, len(symbols_to_check), 50):
            batch = symbols_to_check[i:i+50]
            try:
                data = yf.download(batch, period="5d", progress=False)
                if len(batch) == 1:
                    # Single symbol returns Series
                    avg_dollar_volume = (data['Close'] * data['Volume']).mean()
                    if avg_dollar_volume >= min_volume:
                        liquid_symbols.append(batch[0])
                else:
                    # Multiple symbols
                    for symbol in batch:
                        if symbol in data['Close'].columns:
                            avg_dollar_volume = (data['Close'][symbol] * data['Volume'][symbol]).mean()
                            if avg_dollar_volume >= min_volume:
                                liquid_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error checking liquidity for batch: {e}")
        
        # Update watchlist
        removed_count = 0
        for category in list(self._watchlist.keys()):
            original_len = len(self._watchlist[category])
            self._watchlist[category] = [s for s in self._watchlist[category] if s in liquid_symbols]
            removed_count += original_len - len(self._watchlist[category])
        
        self._save_watchlist()
        logger.info(f"Liquidity filter complete. Removed {removed_count} symbols.")
        
        return liquid_symbols

# Market Data Manager
class MarketDataManager:
    """Handles all market data operations with caching and efficiency"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.cache_dir = config.cache_dir / "market_data"
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}
        
    def _get_cache_path(self, symbol: str, data_type: str = "ohlcv") -> Path:
        """Get cache file path for symbol"""
        return self.cache_dir / f"{symbol}_{data_type}.parquet"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache file is recent enough"""
        if not cache_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.total_seconds() < max_age_hours * 3600
    
    @lru_cache(maxsize=200)
    def get_symbol_data(self, symbol: str, period: str = "2y", 
                       interval: str = "1d", use_cache: bool = True) -> pd.DataFrame:
        """Get market data for symbol with caching"""
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check memory cache first
        if use_cache and cache_key in self._memory_cache:
            return self._memory_cache[cache_key].copy()
        
        # Check disk cache
        cache_path = self._get_cache_path(symbol, f"{period}_{interval}")
        if use_cache and self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                self._memory_cache[cache_key] = df
                return df.copy()
            except Exception as e:
                logger.warning(f"Error reading cache for {symbol}: {e}")
        
        # Fetch from yfinance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean data
            df = self._clean_data(df, symbol)
            
            # Save to cache
            if use_cache:
                df.to_parquet(cache_path)
                self._memory_cache[cache_key] = df
            
            return df.copy()
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate market data"""
        # Remove any duplicate indices
        df = df[~df.index.duplicated(keep='first')]
        
        # Forward fill missing values (up to 2 days)
        df = df.fillna(method='ffill', limit=2)
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        # Add symbol column
        df['Symbol'] = symbol
        
        # Ensure proper dtypes
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_batch_data(self, symbols: List[str], period: str = "2y", 
                      interval: str = "1d", n_jobs: int = 10) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols efficiently"""
        logger.info(f"Fetching data for {len(symbols)} symbols...")
        
        def fetch_symbol(symbol):
            return symbol, self.get_symbol_data(symbol, period, interval)
        
        results = {}
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(fetch_symbol, symbol) for symbol in symbols]
            for future in futures:
                symbol, data = future.result()
                if not data.empty:
                    results[symbol] = data
        
        logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results
    
    def get_real_time_quote(self, symbols: Union[str, List[str]]) -> pd.DataFrame:
        """Get real-time quotes for symbols"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        try:
            tickers = yf.Tickers(' '.join(symbols))
            quotes = []
            
            for symbol in symbols:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                quotes.append({
                    'symbol': symbol,
                    'price': info.get('regularMarketPrice', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'bid': info.get('bid', 0),
                    'ask': info.get('ask', 0),
                    'timestamp': datetime.now()
                })
            
            return pd.DataFrame(quotes)
            
        except Exception as e:
            logger.error(f"Error getting real-time quotes: {e}")
            return pd.DataFrame()
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear old cache files"""
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        
        for cache_file in self.cache_dir.glob("*.parquet"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                cache_file.unlink()
                logger.info(f"Deleted old cache file: {cache_file.name}")
        
        # Clear memory cache
        self._memory_cache.clear()

# Risk Manager
class RiskManager:
    """Manages portfolio risk and position sizing"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.risk_metrics = {}
        
    def calculate_position_size(self, signal_strength: float, volatility: float,
                              portfolio_value: float, existing_positions: int) -> float:
        """Calculate position size using Kelly Criterion with safety adjustments"""
        # Base Kelly fraction
        kelly_fraction = signal_strength * self.config.position_size_kelly_fraction
        
        # Adjust for volatility
        vol_adj = np.exp(-volatility * 10)  # Higher vol = smaller position
        
        # Adjust for number of positions
        position_adj = 1.0 - (existing_positions / self.config.max_positions) * 0.5
        
        # Calculate final position size
        position_pct = kelly_fraction * vol_adj * position_adj
        
        # Apply limits
        position_pct = np.clip(position_pct, 0.01, self.config.max_position_size)
        
        # Calculate dollar amount
        position_size = portfolio_value * position_pct
        
        return position_size
    
    def check_risk_limits(self, portfolio: Dict[str, Any]) -> Dict[str, bool]:
        """Check if portfolio meets risk limits"""
        checks = {
            'max_positions': len(portfolio.get('positions', [])) < self.config.max_positions,
            'portfolio_heat': self._calculate_portfolio_heat(portfolio) < self.config.max_portfolio_heat,
            'sector_concentration': self._check_sector_concentration(portfolio),
            'correlation_limit': self._check_correlation_limit(portfolio)
        }
        
        return checks
    
    def _calculate_portfolio_heat(self, portfolio: Dict[str, Any]) -> float:
        """Calculate total portfolio risk (heat)"""
        total_risk = 0.0
        portfolio_value = portfolio.get('total_value', 1.0)
        
        for position in portfolio.get('positions', []):
            position_value = position.get('market_value', 0)
            stop_loss = position.get('stop_loss', 0)
            current_price = position.get('current_price', 1)
            
            if stop_loss > 0 and current_price > 0:
                risk_pct = (current_price - stop_loss) / current_price
                position_risk = (position_value / portfolio_value) * risk_pct
                total_risk += position_risk
        
        return total_risk
    
    def _check_sector_concentration(self, portfolio: Dict[str, Any]) -> bool:
        """Check sector concentration limits"""
        sector_exposure = {}
        portfolio_value = portfolio.get('total_value', 1.0)
        
        for position in portfolio.get('positions', []):
            sector = position.get('sector', 'Unknown')
            position_value = position.get('market_value', 0)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
        
        for sector, exposure in sector_exposure.items():
            if exposure / portfolio_value > self.config.max_sector_exposure:
                return False
        
        return True
    
    def _check_correlation_limit(self, portfolio: Dict[str, Any]) -> bool:
        """Check if positions are too correlated"""
        # Simplified check - in production, calculate actual correlations
        # For now, just ensure we have some diversification
        unique_sectors = set()
        for position in portfolio.get('positions', []):
            unique_sectors.add(position.get('sector', 'Unknown'))
        
        return len(unique_sectors) >= 3
    
    def calculate_stop_loss(self, entry_price: float, atr: float, 
                           predicted_risk: float) -> float:
        """Calculate dynamic stop loss"""
        # Base stop using ATR
        atr_stop = entry_price - (atr * self.config.stop_loss_atr_multiple)
        
        # Adjust based on predicted risk
        risk_adj = 1.0 + (predicted_risk - 0.5) * 0.5  # Higher risk = tighter stop
        adjusted_stop = entry_price - (entry_price - atr_stop) * risk_adj
        
        # Ensure minimum stop distance (0.5%)
        min_stop = entry_price * 0.995
        
        return max(adjusted_stop, min_stop)

# Performance Tracker
class PerformanceTracker:
    """Tracks and analyzes trading performance"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.trades_file = config.data_dir / "trades_history.parquet"
        self.metrics_file = config.data_dir / "performance_metrics.json"
        self._load_history()
    
    def _load_history(self):
        """Load trade history"""
        if self.trades_file.exists():
            self.trades_df = pd.read_parquet(self.trades_file)
        else:
            self.trades_df = pd.DataFrame()
        
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {}
    
    def record_trade(self, trade: Dict[str, Any]):
        """Record a new trade"""
        trade_df = pd.DataFrame([trade])
        if self.trades_df.empty:
            self.trades_df = trade_df
        else:
            self.trades_df = pd.concat([self.trades_df, trade_df], ignore_index=True)
        
        # Save updated history
        self.trades_df.to_parquet(self.trades_file)
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        if self.trades_df.empty:
            return
        
        # Calculate returns
        self.trades_df['return'] = (self.trades_df['exit_price'] - self.trades_df['entry_price']) / self.trades_df['entry_price']
        self.trades_df['return'] = self.trades_df['return'] * np.where(self.trades_df['side'] == 'buy', 1, -1)
        
        # Basic metrics
        self.metrics['total_trades'] = len(self.trades_df)
        self.metrics['win_rate'] = (self.trades_df['return'] > 0).mean()
        self.metrics['avg_return'] = self.trades_df['return'].mean()
        self.metrics['total_return'] = (1 + self.trades_df['return']).prod() - 1
        
        # Risk metrics
        returns = self.trades_df['return'].values
        if len(returns) > 1:
            self.metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
            self.metrics['sortino_ratio'] = np.sqrt(252) * returns.mean() / returns[returns < 0].std()
            self.metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
        
        # Win/Loss analysis
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) > 0 and len(losses) > 0:
            self.metrics['avg_win'] = wins.mean()
            self.metrics['avg_loss'] = losses.mean()
            self.metrics['profit_factor'] = abs(wins.sum() / losses.sum())
        
        # Save metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return self.metrics.copy()
    
    def get_recent_trades(self, n: int = 10) -> pd.DataFrame:
        """Get recent trades"""
        if self.trades_df.empty:
            return pd.DataFrame()
        return self.trades_df.tail(n)

# Main Trading System
class MLTradingSystem:
    """Main orchestrator for the ML trading system"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # Initialize components
        self.watchlist_manager = WatchlistManager(self.config)
        self.data_manager = MarketDataManager(self.config)
        self.risk_manager = RiskManager(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        
        # Placeholder for other components (to be implemented)
        self.feature_engineer = None  # Will implement EnhancedFeatureEngineer
        self.ensemble_model = None    # Will implement EnsembleModel
        self.signal_generator = None  # Will implement SignalGenerator
        self.execution_engine = None  # Will implement AlpacaExecutionEngine
        self.news_analyzer = None     # Will implement NewsAnalyzer
        
        logger.info("ML Trading System initialized")
    
    def run_daily_pipeline(self):
        """Run the complete daily trading pipeline"""
        logger.info("Starting daily trading pipeline...")
        
        try:
            # 1. Update watchlist
            symbols = self.watchlist_manager.get_all_symbols()
            logger.info(f"Processing {len(symbols)} symbols")
            
            # 2. Fetch latest data
            market_data = self.data_manager.get_batch_data(symbols, period="2y")
            logger.info(f"Fetched data for {len(market_data)} symbols")
            
            # 3. Feature engineering (placeholder)
            # features = self.feature_engineer.engineer_features(market_data)
            
            # 4. Generate predictions (placeholder)
            # predictions = self.ensemble_model.predict(features)
            
            # 5. Generate signals (placeholder)
            # signals = self.signal_generator.generate_signals(predictions, features)
            
            # 6. Risk management (placeholder)
            # sized_signals = self.risk_manager.size_positions(signals)
            
            # 7. Execute trades (placeholder)
            # trades = self.execution_engine.execute_trades(sized_signals)
            
            # 8. Update performance
            # self.performance_tracker.record_trades(trades)
            
            logger.info("Daily pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in daily pipeline: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'symbols_tracked': len(self.watchlist_manager.get_all_symbols()),
            'performance': self.performance_tracker.get_performance_summary(),
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        return status

# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    system = MLTradingSystem()
    
    # Get system status
    status = system.get_system_status()
    print(f"\nSystem Status: {json.dumps(status, indent=2)}")
    
    # Test data fetching
    data_manager = system.data_manager
    test_symbol = "AAPL"
    
    print(f"\nFetching data for {test_symbol}...")
    data = data_manager.get_symbol_data(test_symbol, period="1mo")
    
    if not data.empty:
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"\nLast 5 rows:")
        print(data.tail())
    
    # Test watchlist
    print(f"\nTotal symbols in watchlist: {len(system.watchlist_manager.get_all_symbols())}")
    print(f"Categories: {list(system.watchlist_manager._watchlist.keys())}")
    
    # Test risk calculations
    mock_portfolio = {
        'total_value': 100000,
        'positions': [
            {'symbol': 'AAPL', 'market_value': 10000, 'current_price': 150, 'stop_loss': 145, 'sector': 'Technology'},
            {'symbol': 'JPM', 'market_value': 8000, 'current_price': 140, 'stop_loss': 135, 'sector': 'Finance'}
        ]
    }
    
    risk_checks = system.risk_manager.check_risk_limits(mock_portfolio)
    print(f"\nRisk checks: {json.dumps(risk_checks, indent=2)}")
    
    # Calculate position size
    position_size = system.risk_manager.calculate_position_size(
        signal_strength=0.7,
        volatility=0.02,
        portfolio_value=100000,
        existing_positions=2
    )
    print(f"\nRecommended position size: ${position_size:,.2f}")
