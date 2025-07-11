"""
Main ML Trading System Orchestrator
Professional-grade trading system with all components integrated
"""

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

class ProfessionalMLTradingSystem:
    """Main orchestrator for the complete ML trading system"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the complete trading system"""
        logger.info("Initializing Professional ML Trading System...")
        
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
        
        logger.info("ML Trading System initialized successfully")
    
    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
        else:
            # Default configuration
            config = {
                "system": {
                    "initial_capital": 100000,
                    "max_positions": 20,
                    "max_position_size": 0.10,
                    "risk_limit": 0.08
                },
                "data": {
                    "lookback_years": 2,
                    "update_frequency": "daily"
                },
                "ml": {
                    "retrain_frequency_days": 7,
                    "min_confidence": 0.6,
                    "ensemble_agreement": 0.6
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
            logger.info("Using default configuration")
        
        return config
    
    def _initialize_core_components(self):
        """Initialize core system components"""
        logger.info("Initializing core components...")

        # System configuration
        self.system_config = SystemConfig()
        # Set values after creation
        self.system_config.max_positions = self.config['system']['max_positions']
        self.system_config.max_position_size = self.config['system']['max_position_size']
        # Store initial capital separately
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
        
        # Ensemble model
        self.model_config = ModelConfig(
            n_splits=5,
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
        """Run the complete trading pipeline"""
        logger.info("Running complete trading pipeline...")
        
        try:
            # 1. Update market data
            self._update_market_data()
            
            # 2. Generate features
            self._generate_features()
            
            # 3. Generate predictions
            self._generate_predictions()
            
            # 4. Generate signals
            self._generate_signals()
            
            # 5. Execute trades (if market open)
            if self._is_market_open():
                self._execute_trades()
            
            # 6. Update performance
            self._update_performance()
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise
    
    def _update_market_data(self):
        """Update market data for all symbols"""
        logger.info("Updating market data...")
        
        symbols = self.watchlist_manager.get_all_symbols()
        self.market_data = self.data_manager.get_batch_data(
            symbols, 
            period=f"{self.config['data']['lookback_years']}y"
        )
        
        logger.info(f"Updated data for {len(self.market_data)} symbols")
    
    def _generate_features(self):
        """Generate features for all symbols"""
        logger.info("Generating features...")
        
        self.feature_data = {}
        
        for symbol, data in self.market_data.items():
            try:
                features = self.feature_engineer.engineer_features(data)
                self.feature_data[symbol] = features
            except Exception as e:
                logger.error(f"Error generating features for {symbol}: {e}")
        
        logger.info(f"Generated features for {len(self.feature_data)} symbols")
    
    def _generate_predictions(self):
        """Generate ML predictions"""
        logger.info("Generating predictions...")
        
        # Check if model needs retraining
        if self._should_retrain_models():
            self._retrain_models()
        
        self.predictions = {}
        
        for symbol, features in self.feature_data.items():
            try:
                # Get latest features
                latest_features = features.iloc[-1:].values
                
                # Generate prediction
                prediction = self.ensemble_model.predict(latest_features)
                self.predictions[symbol] = prediction
                
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
        
        logger.info(f"Generated predictions for {len(self.predictions)} symbols")
    
    def _generate_signals(self):
        """Generate trading signals"""
        logger.info("Generating trading signals...")
        
        # Get news sentiment if available
        sentiment_scores = None
        if self.news_analyzer:
            symbols_to_analyze = list(self.predictions.keys())[:50]  # Top 50
            sentiment_results = self.news_analyzer.analyze_symbols(symbols_to_analyze)
            sentiment_scores = {
                symbol: analysis.overall_sentiment 
                for symbol, analysis in sentiment_results.items()
            }
        
        # Generate signals
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
        
        # Execute signals
        results = self.execution_engine.execute_signals(self.current_signals)
        
        logger.info(f"Execution results: {results}")
        
        # Update active positions
        self.active_positions = self.execution_engine.positions
    
    def _update_performance(self):
        """Update performance tracking"""
        logger.info("Updating performance...")
        
        if self.execution_engine:
            # Get latest portfolio value
            portfolio_summary = self.execution_engine.get_portfolio_summary()
            
            # Update performance tracker
            # This would normally update with actual trade data
            logger.info(f"Portfolio value: ${portfolio_summary['account_value']:,.2f}")
    
    def _should_retrain_models(self) -> bool:
        """Check if models need retraining"""
        # Check last training date
        model_path = self.system_config.data_dir / "models" / "last_training.json"
        
        if not model_path.exists():
            return True
        
        with open(model_path, 'r') as f:
            last_training = json.load(f)
        
        last_date = datetime.fromisoformat(last_training['date'])
        days_since = (datetime.now() - last_date).days
        
        return days_since >= self.config['ml']['retrain_frequency_days']
    
    def _retrain_models(self):
        """Retrain ML models"""
        logger.info("Retraining models...")
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for symbol, features in self.feature_data.items():
            if len(features) < 100:  # Skip if insufficient data
                continue
            
            # Prepare features (excluding last row for prediction)
            X = features.iloc[:-1].values
            
            # Calculate returns for labels
            data = self.market_data[symbol]
            returns = data['Close'].pct_change(5).shift(-5)  # 5-day forward returns
            y = returns.iloc[:-1].values
            
            # Remove NaN
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            X_train.append(X)
            y_train.append(y)
        
        # Combine all data
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        
        # Train ensemble
        logger.info(f"Training on {len(X_train)} samples...")
        scores = self.ensemble_model.train(X_train, y_train)
        logger.info(f"Training scores: {scores}")
        
        # Save training date
        model_path = self.system_config.data_dir / "models" / "last_training.json"
        with open(model_path, 'w') as f:
            json.dump({'date': datetime.now().isoformat(), 'scores': scores}, f)
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        if self.execution_engine:
            return self.execution_engine.is_market_open()
        
        # Simple check based on time
        now = datetime.now()
        if now.weekday() > 4:  # Weekend
            return False
        
        market_open = now.time() >= datetime.strptime("09:30", "%H:%M").time()
        market_close = now.time() <= datetime.strptime("16:00", "%H:%M").time()
        
        return market_open and market_close
    
    def run_backtest(self, start_date: str, end_date: str):
        """Run backtest on historical data"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Convert dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate historical signals
        # This is simplified - in production, properly generate signals for each day
        historical_signals = []
        
        # Run backtest
        results = self.backtester.run_backtest(
            historical_signals,
            self.market_data,
            start,
            end
        )
        
        # Display results
        logger.info("Backtest Results:")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        
        return results
    
    def start(self):
        """Start the trading system"""
        logger.info("Starting ML Trading System...")
        
        self.is_running = True
        
        # Start automation if enabled
        if self.automation:
            self.automation.start()
            logger.info("Automation started")
        
        # Run initial pipeline
        self.run_complete_pipeline()
        
        logger.info("ML Trading System is running")
    
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping ML Trading System...")
        
        self.is_running = False
        
        # Stop automation
        if self.automation:
            self.automation.stop()
        
        # Close all positions if configured
        if self.execution_engine and hasattr(self.execution_engine, 'close_all_positions'):
            logger.info("Closing all positions...")
            self.execution_engine.close_all_positions("system_shutdown")
        
        logger.info("ML Trading System stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'symbols_tracked': len(self.watchlist_manager.get_all_symbols()),
            'active_positions': len(self.active_positions),
            'pending_signals': len(self.current_signals),
            'performance': self.performance_tracker.get_performance_summary()
        }
        
        if self.execution_engine:
            status['account_info'] = self.execution_engine.get_account_info()
        
        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Professional ML Trading System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', choices=['live', 'backtest', 'dashboard'], 
                       default='live', help='Operating mode')
    parser.add_argument('--start-date', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Backtest end date (YYYY-MM-DD)')
    
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
                # Display status every minute
                import time
                time.sleep(60)
                status = system.get_system_status()
                logger.info(f"System Status: {json.dumps(status, indent=2)}")
                
        except KeyboardInterrupt:
            system.stop()
            
    elif args.mode == 'backtest':
        # Run backtest
        if not args.start_date or not args.end_date:
            logger.error("Start and end dates required for backtest")
            sys.exit(1)
        
        results = system.run_backtest(args.start_date, args.end_date)
        
        # Save results
        results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")
        
    elif args.mode == 'dashboard':
        # Launch dashboard
        logger.info("Launching dashboard...")
        
        # Import and run dashboard
        from trading_dashboard import TradingDashboard
        
        dashboard = TradingDashboard(system)
        
        # This would typically launch the Streamlit app
        import subprocess
        subprocess.run(["streamlit", "run", "trading_dashboard.py"])


if __name__ == "__main__":
    main()
