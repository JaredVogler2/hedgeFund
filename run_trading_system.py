#!/usr/bin/env python3
"""
Quick Start Script for ML Trading System
Run ML training, backtesting, and dashboard with simple commands
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import subprocess
import json

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check Python packages
    try:
        import torch
        import xgboost
        import lightgbm
        import catboost
        import alpaca_trade_api
        import streamlit
        print("✅ All required packages installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU not available, will use CPU (slower)")
    
    # Check API keys
    if os.getenv('ALPACA_API_KEY'):
        print("✅ Alpaca API configured")
    else:
        print("⚠️  Alpaca API not configured (set ALPACA_API_KEY)")
    
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API configured")
    else:
        print("⚠️  OpenAI API not configured (news analysis disabled)")
    
    return True

def run_ml_training():
    """Run ML training on full watchlist"""
    print("\n🧠 Starting ML Training...")
    print("This will train on 100+ symbols and may take 10-30 minutes")
    
    from main_trading_system import ProfessionalMLTradingSystem
    
    system = ProfessionalMLTradingSystem('config.json')
    
    # Update market data
    print("📊 Fetching market data...")
    system._update_market_data()
    
    # Generate features
    print("🔧 Engineering features...")
    system._generate_features()
    
    # Train models
    print("🚀 Training ensemble models...")
    system._retrain_models()
    
    print("✅ ML training completed!")

def run_backtest(start_date=None, end_date=None):
    """Run backtest with specified dates"""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📈 Running backtest from {start_date} to {end_date}...")
    
    from main_trading_system import ProfessionalMLTradingSystem
    from backtesting_engine import BacktestAnalyzer
    
    system = ProfessionalMLTradingSystem('config.json')
    
    # Ensure we have data and predictions
    print("📊 Preparing data...")
    system._update_market_data()
    system._generate_features()
    system._generate_predictions()
    
    # Run backtest
    results = system.run_backtest(start_date, end_date)
    
    # Display results
    print("\n📊 Backtest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annual Return: {results['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    
    # Save results
    filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {filename}")

def run_dashboard():
    """Launch the enhanced dashboard"""
    print("\n🚀 Launching dashboard...")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop")
    
    # Check if enhanced dashboard exists
    if os.path.exists('enhanced_dashboard.py'):
        subprocess.run(["streamlit", "run", "enhanced_dashboard.py"])
    else:
        print("⚠️  Enhanced dashboard not found, using simple dashboard")
        subprocess.run(["streamlit", "run", "live_dashboard_simple.py"])

def run_live_trading():
    """Run live trading system"""
    print("\n💼 Starting live trading system...")
    
    from main_trading_system import ProfessionalMLTradingSystem
    
    system = ProfessionalMLTradingSystem('config.json')
    
    # Start the system
    system.start()
    
    print("✅ Live trading system started")
    print("Check the dashboard to monitor activity")
    
    try:
        # Keep running
        while system.is_running:
            import time
            time.sleep(60)
            status = system.get_system_status()
            print(f"\r⏰ {datetime.now().strftime('%H:%M:%S')} - "
                  f"Positions: {status['active_positions']} | "
                  f"Signals: {status['pending_signals']}", end='')
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping system...")
        system.stop()

def quick_analysis():
    """Quick analysis of current market conditions"""
    print("\n🔍 Running quick market analysis...")
    
    from main_trading_system import ProfessionalMLTradingSystem
    
    system = ProfessionalMLTradingSystem('config.json')
    
    # Get top opportunities
    print("📊 Analyzing market...")
    system._update_market_data()
    system._generate_features()
    system._generate_predictions()
    system._generate_signals()
    
    # Display top signals
    if system.current_signals:
        print(f"\n🎯 Top {min(10, len(system.current_signals))} Trading Signals:")
        print("-" * 60)
        
        sorted_signals = sorted(system.current_signals, 
                              key=lambda x: x.ml_score, 
                              reverse=True)[:10]
        
        for i, signal in enumerate(sorted_signals, 1):
            print(f"{i:2d}. {signal.symbol:6s} | "
                  f"Score: {signal.ml_score:.3f} | "
                  f"Direction: {signal.direction:5s} | "
                  f"Position: {signal.position_size:.1%}")
    else:
        print("No signals generated")

def main():
    parser = argparse.ArgumentParser(
        description='ML Trading System - Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_trading_system.py check        # Check environment
  python run_trading_system.py train        # Run ML training
  python run_trading_system.py backtest     # Run backtest (last year)
  python run_trading_system.py dashboard    # Launch dashboard
  python run_trading_system.py live         # Start live trading
  python run_trading_system.py analyze      # Quick market analysis
  
  python run_trading_system.py backtest --start 2023-01-01 --end 2023-12-31
        """
    )
    
    parser.add_argument('command', 
                       choices=['check', 'train', 'backtest', 'dashboard', 'live', 'analyze'],
                       help='Command to run')
    
    parser.add_argument('--start', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Backtest end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Ensure config exists
    if not os.path.exists('config.json'):
        print("⚠️  config.json not found, creating default...")
        default_config = {
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
        with open('config.json', 'w') as f:
            json.dump(default_config, f, indent=2)
    
    # Execute command
    if args.command == 'check':
        check_environment()
    
    elif args.command == 'train':
        if check_environment():
            run_ml_training()
    
    elif args.command == 'backtest':
        if check_environment():
            run_backtest(args.start, args.end)
    
    elif args.command == 'dashboard':
        run_dashboard()
    
    elif args.command == 'live':
        if check_environment():
            run_live_trading()
    
    elif args.command == 'analyze':
        if check_environment():
            quick_analysis()

if __name__ == "__main__":
    main()
