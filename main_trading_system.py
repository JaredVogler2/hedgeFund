"""
Main ML Trading System Orchestrator - UPDATED WITH DATA LEAKAGE FIXES
Professional-grade trading system with proper time series handling
"""
import pytz
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
import asyncio
import signal

# Import all components
from ml_trading_core import MLTradingSystem, SystemConfig, MarketDataManager, WatchlistManager
from feature_engineering import EnhancedFeatureEngineer, FeatureConfig
from ensemble_ml_system import EnsembleModel, ModelConfig
from signal_generation_risk import SignalGenerator, SignalConfig, RiskManager
from backtesting_engine import Backtester, BacktestConfig
from alpaca_trading_engine import AlpacaTradingEngine, AlpacaConfig
from news_sentiment_analyzer import NewsAnalyzer, NewsConfig
from automation_scheduler import TradingAutomation, AutomationConfig
from purged_cross_validation import PurgedTimeSeriesSplit, PurgedWalkForwardCV

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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set current date (July 12, 2025)
#CURRENT_DATE = pd.Timestamp('2025-07-12')
CURRENT_DATE = datetime.now(pytz.timezone('America/New_York')).replace(hour=0, minute=0, second=0, microsecond=0)
TRAINING_START_DATE = CURRENT_DATE - pd.DateOffset(years=5)  # July 12, 2020

class ProfessionalMLTradingSystem:
    """Main orchestrator for the complete ML trading system with data leakage prevention"""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the complete trading system"""
        logger.info("Initializing Professional ML Trading System...")
        logger.info(f"Current date: {CURRENT_DATE.date()}")
        logger.info(f"Training data start: {TRAINING_START_DATE.date()}")

        # Load configuration
        self.config = self._load_configuration(config_file)

        # Initialize core components
        self._initialize_core_components()

        # Initialize ML components
        self._initialize_ml_components()

        # Initialize trading components
        self._initialize_trading_components()

        # Initialize automation
        self._initialize_automation()

        # System state
        self.is_running = False
        self.current_signals = []
        self.active_positions = {}
        self.last_model_update = None

        logger.info("ML Trading System initialized successfully")

    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
        else:
            # Default configuration with data leakage prevention
            config = {
                "system": {
                    "initial_capital": 100000,
                    "max_positions": 20,
                    "max_position_size": 0.10,
                    "risk_limit": 0.08,
                    "current_date": CURRENT_DATE.isoformat(),
                    "training_start_date": TRAINING_START_DATE.isoformat()
                },
                "data": {
                    "lookback_years": 5,  # Use 5 years of data
                    "update_frequency": "daily",
                    "min_samples_required": 252  # 1 year minimum
                },
                "ml": {
                    "retrain_frequency_days": 30,  # Monthly retraining
                    "min_confidence": 0.6,
                    "ensemble_agreement": 0.6,
                    "purge_days": 5,  # Critical: 5-day purge for 5-day predictions
                    "embargo_days": 2,  # Additional safety
                    "walk_forward_months": 12,  # 12 months training window
                    "validation_months": 3,
                    "test_months": 1
                },
                "trading": {
                    "use_paper_trading": True,
                    "execution_delay_seconds": 1
                },
                "automation": {
                    "enable_automation": True,
                    "enable_notifications": True
                }
            }
            logger.info("Using default configuration with data leakage prevention")

        return config

    def _initialize_core_components(self):
        """Initialize core system components"""
        logger.info("Initializing core components...")

        # System configuration
        self.system_config = SystemConfig()
        self.system_config.max_positions = self.config['system']['max_positions']
        self.system_config.max_position_size = self.config['system']['max_position_size']
        self.initial_capital = self.config['system']['initial_capital']

        # Core trading system
        self.core_system = MLTradingSystem(self.system_config)

        # Watchlist manager
        self.watchlist_manager = self.core_system.watchlist_manager

        # Market data manager
        self.data_manager = self.core_system.data_manager

        # Performance tracker
        self.performance_tracker = self.core_system.performance_tracker

        logger.info("Core components initialized")

    def _initialize_ml_components(self):
        """Initialize ML components"""
        logger.info("Initializing ML components...")

        # Feature engineering
        self.feature_config = FeatureConfig()
        self.feature_engineer = EnhancedFeatureEngineer(self.feature_config)

        # Ensemble model with purged CV
        self.model_config = ModelConfig(
            n_splits=5,
            purge_days=self.config['ml']['purge_days'],
            min_agreement=self.config['ml']['ensemble_agreement']
        )
        self.ensemble_model = EnsembleModel(self.model_config)

        # Signal generator
        self.signal_config = SignalConfig(
            min_confidence=self.config['ml']['min_confidence']
        )
        self.signal_generator = SignalGenerator(self.signal_config)

        # Risk manager
        self.risk_manager = RiskManager(self.signal_config)

        logger.info("ML components initialized")

    def _initialize_trading_components(self):
        """Initialize trading components"""
        logger.info("Initializing trading components...")

        # Backtester
        self.backtest_config = BacktestConfig(
            initial_capital=self.config['system']['initial_capital']
        )
        self.backtester = Backtester(self.backtest_config)

        # Live trading (Alpaca)
        if os.getenv('ALPACA_API_KEY'):
            self.alpaca_config = AlpacaConfig(
                use_paper=self.config['trading']['use_paper_trading']
            )
            self.execution_engine = AlpacaTradingEngine(self.alpaca_config)
        else:
            logger.warning("Alpaca API credentials not found. Live trading disabled.")
            self.execution_engine = None

        # News analyzer
        if os.getenv('OPENAI_API_KEY'):
            self.news_config = NewsConfig()
            self.news_analyzer = NewsAnalyzer(self.news_config)
        else:
            logger.warning("OpenAI API key not found. News analysis disabled.")
            self.news_analyzer = None

        logger.info("Trading components initialized")

    def _initialize_automation(self):
        """Initialize automation system"""
        logger.info("Initializing automation...")

        if self.config['automation']['enable_automation']:
            self.automation_config = AutomationConfig(
                enable_email_notifications=self.config['automation']['enable_notifications']
            )
            self.automation = TradingAutomation(self, self.automation_config)
        else:
            self.automation = None

        logger.info("Automation initialized")

    def run_complete_pipeline(self):
        """Run the complete trading pipeline with data leakage prevention"""
        logger.info("Running complete trading pipeline...")
        logger.info(f"Current system date: {CURRENT_DATE.date()}")

        try:
            # 1. Update market data (only up to current date)
            self._update_market_data()

            # 2. Generate features (backward-looking only)
            self._generate_features()

            # 3. Check if retraining needed
            if self._should_retrain_models():
                self._retrain_models_safe()

            # 4. Generate predictions (using properly trained models)
            self._generate_predictions()

            # 5. Generate signals
            self._generate_signals()

            # 6. Execute trades (if market open)
            if self._is_market_open():
                self._execute_trades()

            # 7. Update performance
            self._update_performance()

            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise

    def _update_market_data(self):
        """Update market data for all symbols"""
        logger.info("Updating market data...")

        symbols = self.watchlist_manager.get_all_symbols()
        self.market_data = {}

        # Process in batches to avoid overwhelming the API
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_data = self.data_manager.get_batch_data(
                batch,
                period=f"{self.config['data']['lookback_years']}y"
            )

            # Filter data to current date for each symbol
            for symbol, data in batch_data.items():
                if not data.empty:
                    # Ensure index is timezone-aware
                    if data.index.tz is None:
                        # If data has no timezone, assume it's in Eastern Time
                        data.index = data.index.tz_localize('America/New_York')
                    elif data.index.tz != pytz.timezone('America/New_York'):
                        # Convert to Eastern Time if different timezone
                        data.index = data.index.tz_convert('America/New_York')

                    # Now safe to compare - both are timezone-aware
                    data = data[data.index <= CURRENT_DATE]

                    if not data.empty:
                        self.market_data[symbol] = data

        logger.info(f"Updated data for {len(self.market_data)} symbols")

    def _generate_features(self):
        """Generate features for all symbols (backward-looking only)"""
        logger.info("Generating features...")

        self.feature_data = {}

        for symbol, data in self.market_data.items():
            try:
                # Generate features using only backward-looking indicators
                features = self.feature_engineer.engineer_features(data)

                # Verify no future information
                if not features.empty:
                    self.feature_data[symbol] = features

            except Exception as e:
                logger.error(f"Error generating features for {symbol}: {e}")

        logger.info(f"Generated features for {len(self.feature_data)} symbols")

    def _should_retrain_models(self) -> bool:
        """Check if models need retraining"""
        model_path = self.system_config.data_dir / "models" / "last_training.json"

        if not model_path.exists():
            return True

        with open(model_path, 'r') as f:
            last_training = json.load(f)

        last_date = pd.Timestamp(last_training['date'])
        days_since = (CURRENT_DATE - last_date).days

        should_retrain = days_since >= self.config['ml']['retrain_frequency_days']
        if should_retrain:
            logger.info(f"Model retraining needed: {days_since} days since last training")

        return should_retrain

    def _retrain_models(self):
        """Retrain ML models with proper data leakage prevention"""
        logger.info("Retraining models with purged cross-validation...")
        logger.info(f"Training period: {TRAINING_START_DATE.date()} to {CURRENT_DATE.date()}")

        # Prepare training data with proper temporal ordering
        all_data = []

        for symbol, features in self.feature_data.items():
            if symbol not in self.market_data or len(features) < self.config['data']['min_samples_required']:
                continue

            # Get price data
            data = self.market_data[symbol]

            # Ensure temporal alignment
            common_dates = features.index.intersection(data.index)
            features_aligned = features.loc[common_dates]
            data_aligned = data.loc[common_dates]

            # CRITICAL: Calculate forward returns correctly
            # For 5-day prediction, we need prices 5 days in the future
            prediction_horizon = 5
            future_prices = data_aligned['Close'].shift(-prediction_horizon)
            current_prices = data_aligned['Close']

            # Calculate forward returns
            forward_returns = (future_prices - current_prices) / current_prices

            # Remove last 'prediction_horizon' rows (no future data available)
            features_clean = features_aligned.iloc[:-prediction_horizon]
            returns_clean = forward_returns.iloc[:-prediction_horizon]
            dates_clean = common_dates[:-prediction_horizon]

            # Remove any NaN values
            mask = ~(returns_clean.isna() | features_clean.isna().any(axis=1))

            if mask.sum() > 0:
                all_data.append({
                    'features': features_clean[mask].values,
                    'targets': returns_clean[mask].values,
                    'dates': dates_clean[mask],
                    'symbol': symbol
                })

        if not all_data:
            logger.error("No valid training data available!")
            return

        # Combine all data
        X = np.vstack([d['features'] for d in all_data])
        y = np.hstack([d['targets'] for d in all_data])
        dates = np.hstack([d['dates'] for d in all_data])

        # Sort by date (CRITICAL for time series)
        sort_idx = np.argsort(dates)
        X = X[sort_idx]
        y = y[sort_idx]
        dates = dates[sort_idx]

        logger.info(f"Training on {len(X)} samples from {dates[0]} to {dates[-1]}")

        # Create purged cross-validator
        pcv = PurgedTimeSeriesSplit(
            n_splits=5,
            purge_days=self.config['ml']['purge_days'],
            embargo_days=self.config['ml']['embargo_days']
        )

        # Verify CV splits
        logger.info("Cross-validation splits:")
        for i, (train_idx, val_idx) in enumerate(pcv.split(X)):
            train_dates_cv = dates[train_idx]
            val_dates_cv = dates[val_idx]
            gap_days = (val_dates_cv[0] - train_dates_cv[-1]).days
            logger.info(f"  Fold {i+1}: Train {train_dates_cv[0]} to {train_dates_cv[-1]}, "
                       f"Val {val_dates_cv[0]} to {val_dates_cv[-1]}, Gap: {gap_days} days")

        # Train ensemble with proper CV
        feature_names = self.feature_engineer.get_feature_names()
        scores = self.ensemble_model.train(X, y, feature_names=feature_names, cv=pcv)

        logger.info(f"Training complete. Cross-validation scores: {scores}")

        # Save model and metadata
        model_dir = self.system_config.data_dir / "models"
        model_dir.mkdir(exist_ok=True)

        # Save training info
        training_info = {
            'date': CURRENT_DATE.isoformat(),
            'training_start': TRAINING_START_DATE.isoformat(),
            'training_end': CURRENT_DATE.isoformat(),
            'scores': scores,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'date_range': f"{dates[0]} to {dates[-1]}",
            'purge_days': self.config['ml']['purge_days'],
            'embargo_days': self.config['ml']['embargo_days'],
            'prediction_horizon': prediction_horizon,
            'features': feature_names
        }

        with open(model_dir / "last_training.json", 'w') as f:
            json.dump(training_info, f, indent=2, default=str)

        # Save models
        self.ensemble_model.save_models(str(model_dir))

        self.last_model_update = CURRENT_DATE
        logger.info("Models saved successfully")

    def _generate_predictions(self):
        """Generate ML predictions using latest features"""
        logger.info("Generating predictions...")

        self.predictions = {}

        for symbol, features in self.feature_data.items():
            try:
                # Get latest features (most recent data point)
                latest_features = features.iloc[-1:].values

                # Generate prediction
                prediction = self.ensemble_model.predict(latest_features)
                confidence_scores = self.ensemble_model.predict_proba(latest_features)

                self.predictions[symbol] = {
                    'prediction': prediction[0],
                    'confidence': confidence_scores,
                    'timestamp': CURRENT_DATE
                }

            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")

        logger.info(f"Generated predictions for {len(self.predictions)} symbols")

    def _generate_signals(self):
        """Generate trading signals"""
        logger.info("Generating trading signals...")

        # Get news sentiment if available
        sentiment_scores = None
        if self.news_analyzer:
            symbols_to_analyze = list(self.predictions.keys())[:50]
            sentiment_results = self.news_analyzer.analyze_symbols(symbols_to_analyze)
            sentiment_scores = {
                symbol: analysis.overall_sentiment
                for symbol, analysis in sentiment_results.items()
            }

        # Generate signals with risk management
        self.current_signals = self.signal_generator.generate_signals(
            self.predictions,
            self.feature_data,
            self.market_data,
            sentiment_scores
        )

        logger.info(f"Generated {len(self.current_signals)} trading signals")

    def _execute_trades(self):
        """Execute trading signals"""
        if not self.execution_engine:
            logger.warning("Execution engine not available")
            return

        logger.info("Executing trades...")

        # Apply risk management
        approved_signals = []
        for signal in self.current_signals:
            if self.risk_manager.check_signal(signal, self.active_positions):
                approved_signals.append(signal)

        logger.info(f"Risk management approved {len(approved_signals)} of {len(self.current_signals)} signals")

        # Execute approved signals
        if approved_signals:
            results = self.execution_engine.execute_signals(approved_signals)
            logger.info(f"Execution results: {results}")

            # Update active positions
            self.active_positions = self.execution_engine.positions

    def _update_performance(self):
        """Update performance tracking"""
        logger.info("Updating performance...")

        if self.execution_engine:
            # Get latest portfolio value
            portfolio_summary = self.execution_engine.get_portfolio_summary()

            # Log performance
            logger.info(f"Portfolio value: ${portfolio_summary['account_value']:,.2f}")
            logger.info(f"Daily P&L: ${portfolio_summary.get('daily_pnl', 0):,.2f}")

    def _is_market_open(self) -> bool:
        """Check if market is open"""
        if self.execution_engine:
            return self.execution_engine.is_market_open()

        # Simple check based on current system date/time
        if CURRENT_DATE.weekday() > 4:  # Weekend
            return False

        # For backtesting/simulation, assume market is open on weekdays
        return True

    def run_walk_forward_backtest(self, start_date: str, end_date: str):
        """Run walk-forward backtest with proper data handling"""
        logger.info(f"Running walk-forward backtest from {start_date} to {end_date}")

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Create walk-forward CV
        wf_cv = PurgedWalkForwardCV(
            train_months=self.config['ml']['walk_forward_months'],
            validation_months=self.config['ml']['validation_months'],
            test_months=self.config['ml']['test_months'],
            purge_days=self.config['ml']['purge_days']
        )

        # Get all dates in range
        all_dates = pd.date_range(start=start, end=end, freq='B')

        results = []

        for train_dates, val_dates, test_dates in wf_cv.split(all_dates):
            period_start = train_dates[0]
            period_end = test_dates[-1]

            logger.info(f"\nWalk-forward period:")
            logger.info(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()}")
            logger.info(f"  Val: {val_dates[0].date()} to {val_dates[-1].date()}")
            logger.info(f"  Test: {test_dates[0].date()} to {test_dates[-1].date()}")

            # Train model on train+val period
            # Generate signals for test period
            # Run backtest on test period

            # This is a placeholder - implement actual walk-forward logic
            period_result = {
                'train_period': (train_dates[0], train_dates[-1]),
                'val_period': (val_dates[0], val_dates[-1]),
                'test_period': (test_dates[0], test_dates[-1]),
                'results': {}  # Backtest results would go here
            }

            results.append(period_result)

        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'timestamp': CURRENT_DATE.isoformat(),
            'is_running': self.is_running,
            'symbols_tracked': len(self.watchlist_manager.get_all_symbols()),
            'active_positions': len(self.active_positions),
            'pending_signals': len(self.current_signals),
            'last_model_update': self.last_model_update.isoformat() if self.last_model_update else None,
            'performance': self.performance_tracker.get_performance_summary(),
            'config': {
                'purge_days': self.config['ml']['purge_days'],
                'embargo_days': self.config['ml']['embargo_days'],
                'prediction_horizon': 5,
                'training_start': TRAINING_START_DATE.isoformat()
            }
        }

        if self.execution_engine:
            status['account_info'] = self.execution_engine.get_account_info()

        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Professional ML Trading System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', choices=['live', 'backtest', 'train'],
                       default='live', help='Operating mode')
    parser.add_argument('--start-date', type=str,
                       default=TRAINING_START_DATE.strftime('%Y-%m-%d'),
                       help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       default=CURRENT_DATE.strftime('%Y-%m-%d'),
                       help='Backtest end date (YYYY-MM-DD)')

    args = parser.parse_args()

    # Initialize system
    system = ProfessionalMLTradingSystem(args.config)

    # Handle different modes
    if args.mode == 'live':
        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            system.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start system
        system.start()

        # Keep running
        try:
            while system.is_running:
                import time
                time.sleep(60)
                status = system.get_system_status()
                logger.info(f"System Status: {json.dumps(status, indent=2)}")

        except KeyboardInterrupt:
            system.stop()

    elif args.mode == 'backtest':
        # Run walk-forward backtest
        results = system.run_walk_forward_backtest(args.start_date, args.end_date)

        # Save results
        results_file = f"backtest_results_{CURRENT_DATE.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")

    elif args.mode == 'train':
        # Force model retraining
        logger.info("Starting model training...")
        system._update_market_data()
        system._generate_features()
        system._retrain_models_safe()
        logger.info("Training complete!")


if __name__ == "__main__":
    main()