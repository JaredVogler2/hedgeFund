# Professional ML Trading System

A hedge fund-quality machine learning trading system with GPU acceleration, ensemble models, and live trading capabilities.

## 🚀 Features

### Core Capabilities
- **GPU-Accelerated ML**: Leverages NVIDIA GPUs (optimized for 3070 Ti) for fast model training
- **Ensemble Learning**: Multiple models (XGBoost, LightGBM, CatBoost, Deep Learning) with meta-learning
- **30+ Feature Categories**: Comprehensive feature engineering including price, volume, volatility, microstructure
- **Real-time Trading**: Alpaca integration for live paper/real trading
- **News Sentiment**: OpenAI-powered news analysis for sentiment signals
- **Professional Risk Management**: Portfolio heat, position sizing, dynamic stops
- **Walk-Forward Optimization**: Prevents overfitting with proper temporal validation
- **Comprehensive Dashboard**: Real-time monitoring with Streamlit

### Technical Features
- No data leakage with purged cross-validation
- Realistic execution modeling with slippage and market impact
- Multi-timeframe analysis (1min to daily)
- Automated scheduling for 24/7 operation
- Email notifications for trades and alerts
- Detailed performance analytics and reporting

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (3070 Ti recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ for data storage
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Requirements
- Python 3.8+
- CUDA Toolkit 11.0+
- Git

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ml-trading-system.git
cd ml-trading-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 4. Create requirements.txt
```txt
# Core ML Libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0

# ML Models
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0

# Deep Learning (PyTorch installed separately)
# torch>=1.10.0

# Trading Libraries
yfinance>=0.1.70
alpaca-trade-api>=2.0.0
ta-lib>=0.4.24

# Data Processing
pyarrow>=6.0.0
fastparquet>=0.8.0

# Visualization
plotly>=5.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Dashboard
streamlit>=1.20.0

# News & Sentiment
openai>=0.27.0
newsapi-python>=0.2.6
feedparser>=6.0.0
beautifulsoup4>=4.10.0

# Automation
schedule>=1.1.0
python-dotenv>=0.19.0

# Optimization
optuna>=3.0.0

# Utilities
joblib>=1.1.0
tqdm>=4.62.0
requests>=2.26.0
aiohttp>=3.8.0
pytz>=2021.3

# Development
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
```

### 5. Install TA-Lib
TA-Lib requires special installation:

**Windows:**
Download the appropriate .whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```bash
pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl
```

**Mac:**
```bash
brew install ta-lib
pip install ta-lib
```

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

## ⚙️ Configuration

### 1. Create .env File
Create a `.env` file in the project root:

```env
# Alpaca API (get from https://alpaca.markets/)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading first!

# OpenAI API (get from https://platform.openai.com/)
OPENAI_API_KEY=your_openai_api_key_here

# News API (get from https://newsapi.org/)
NEWSAPI_KEY=your_newsapi_key_here

# Email Notifications (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
NOTIFICATION_EMAIL=recipient@email.com
```

### 2. System Configuration
Create `config.json` for system settings:

```json
{
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
    "use_paper_trading": true,
    "execution_delay_seconds": 1
  },
  "automation": {
    "enable_automation": true,
    "enable_notifications": true
  }
}
```

## 🚀 Quick Start

### 1. Test GPU Setup
```python
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

### 2. Initialize System
```python
from main_trading_system import ProfessionalMLTradingSystem

# Initialize system
system = ProfessionalMLTradingSystem('config.json')

# Check status
status = system.get_system_status()
print(status)
```

### 3. Run Backtest
```bash
python main_trading_system.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

### 4. Start Live Trading (Paper)
```bash
python main_trading_system.py --mode live --config config.json
```

### 5. Launch Dashboard
```bash
streamlit run trading_dashboard.py
```

## 📊 Usage Guide

### Daily Workflow

1. **Pre-Market (8:30 AM ET)**
   - System automatically updates data
   - Analyzes overnight news
   - Validates signals

2. **Market Open (9:30 AM ET)**
   - Executes validated signals
   - Sets initial stop losses
   - Sends execution report

3. **Intraday**
   - Monitors positions every 30 minutes
   - Updates trailing stops
   - Checks risk limits

4. **End of Day (3:45 PM ET)**
   - Analyzes daily performance
   - Prepares next day signals
   - Generates reports

### Manual Operations

#### Generate Signals
```python
# Run complete pipeline
system.run_complete_pipeline()

# Get current signals
signals = system.current_signals
for signal in signals:
    print(f"{signal.symbol}: {signal.direction} - Score: {signal.bayesian_score:.2f}")
```

#### Check Positions
```python
# Get portfolio summary
portfolio = system.execution_engine.get_portfolio_summary()
print(f"Portfolio Value: ${portfolio['account_value']:,.2f}")
print(f"Open Positions: {portfolio['positions_count']}")
```

#### Risk Management
```python
# Check risk metrics
risk_metrics = system.risk_manager.calculate_portfolio_risk(
    system.active_positions,
    system.market_data
)
print(f"Portfolio Heat: {risk_metrics['portfolio_heat']:.2%}")
print(f"VaR (95%): ${risk_metrics['var_95']:,.2f}")
```

## 🔧 Advanced Configuration

### Feature Engineering
Modify feature categories in `feature_engineering.py`:
```python
config = FeatureConfig(
    return_periods=[1, 2, 3, 5, 10, 20, 60],
    ma_periods=[5, 10, 20, 50, 100, 200],
    rsi_periods=[7, 14, 21, 28]
)
```

### Model Parameters
Adjust ensemble settings in `ensemble_ml_system.py`:
```python
config = ModelConfig(
    xgb_params={
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01
    }
)
```

### Risk Parameters
Configure risk management in `signal_generation_risk.py`:
```python
config = SignalConfig(
    max_position_size=0.10,
    stop_loss_atr_mult=2.0,
    kelly_fraction=0.25
)
```

## 📈 Performance Monitoring

### Key Metrics to Track
- **Sharpe Ratio**: Target > 1.5
- **Win Rate**: Target > 55%
- **Max Drawdown**: Keep < 20%
- **Portfolio Heat**: Keep < 8%

### Dashboard Pages
1. **Overview**: Portfolio value, P&L, key metrics
2. **Portfolio**: Current positions and allocation
3. **Predictions**: ML signals and confidence scores
4. **ML Analytics**: Feature importance, model performance
5. **Trade History**: Detailed trade analysis
6. **Risk Management**: Real-time risk monitoring

## 🚨 Safety Guidelines

### Before Going Live
1. **Paper Trade First**: Always test with paper trading for at least 1 month
2. **Small Position Sizes**: Start with 1-2% position sizes
3. **Risk Limits**: Set conservative stop losses (2-3%)
4. **Monitor Closely**: Check system multiple times daily initially
5. **Have Kill Switch**: Know how to stop all trading immediately

### Emergency Procedures
```bash
# Stop all trading immediately
python emergency_stop.py

# Close all positions
python close_all_positions.py

# Disable automation
python disable_automation.py
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Reinstall PyTorch with correct CUDA version
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **TA-Lib Import Error**
   - Ensure C++ libraries are installed first
   - Use pre-compiled wheel for Windows

3. **Alpaca Connection Issues**
   - Verify API keys are correct
   - Check if using correct base URL (paper vs live)
   - Ensure account is funded (even paper needs $100k)

4. **Memory Issues**
   - Reduce batch size in data processing
   - Limit number of symbols
   - Use data chunking

## 📚 Additional Resources

### Documentation
- [Alpaca API Docs](https://alpaca.markets/docs/)
- [yfinance Guide](https://pypi.org/project/yfinance/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Strategy Development
- Start with simple strategies
- Gradually increase complexity
- Always validate with backtesting
- Monitor real-world performance

### Community Support
- GitHub Issues for bugs
- Discord community for discussions
- Monthly strategy sharing sessions

## ⚖️ Legal Disclaimer

**IMPORTANT**: This software is for educational purposes only. Trading stocks involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

- Always start with paper trading
- Never trade with money you cannot afford to lose
- Consult with financial advisors
- Ensure compliance with all regulations
- The authors are not responsible for any losses

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Alpaca for the trading API
- OpenAI for sentiment analysis
- The open-source ML community
- All contributors and testers

---

**Happy Trading! 🚀** Remember: Discipline and risk management are more important than any algorithm.