"""
Professional Backtesting Engine
Institutional-grade backtesting with walk-forward optimization and realistic execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Capital and costs
    initial_capital: float = 100000
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0  # basis points
    min_commission: float = 1.0
    
    # Market impact model
    market_impact_model: str = 'square_root'  # 'linear', 'square_root', 'power'
    impact_coefficient: float = 0.1
    
    # Risk limits
    max_position_size: float = 0.10
    max_portfolio_heat: float = 0.08
    max_drawdown_limit: float = 0.20
    
    # Walk-forward parameters
    train_period_months: int = 12
    validation_period_months: int = 3
    test_period_months: int = 1
    
    # Execution parameters
    fill_price_method: str = 'midpoint'  # 'midpoint', 'conservative', 'aggressive'
    partial_fill_threshold: float = 0.5  # 50% of daily volume max
    
    # Analysis parameters
    benchmark_symbol: str = 'SPY'
    risk_free_rate: float = 0.02  # 2% annual

@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_date: datetime
    entry_price: float
    shares: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    commission_paid: float = 0.0
    slippage_paid: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None
    
    @property
    def holding_period(self) -> int:
        if self.exit_date:
            return (self.exit_date - self.entry_date).days
        return 0

@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    cash: float
    positions: Dict[str, float]  # symbol -> shares
    market_values: Dict[str, float]  # symbol -> value
    total_value: float
    returns: float
    drawdown: float
    volatility: float
    sharpe: float
    positions_count: int

class ExecutionSimulator:
    """Simulates realistic order execution"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def execute_order(self, 
                     symbol: str,
                     side: str,
                     shares: float,
                     market_data: pd.Series,
                     volume_data: pd.Series) -> Tuple[float, float, float]:
        """
        Simulate order execution with slippage and market impact
        Returns: (fill_price, commission, slippage)
        """
        
        # Get market prices
        open_price = market_data.get('Open', market_data['Close'])
        high_price = market_data.get('High', market_data['Close'])
        low_price = market_data.get('Low', market_data['Close'])
        close_price = market_data['Close']
        volume = volume_data.get(symbol, 1e6)
        
        # Calculate base fill price
        if self.config.fill_price_method == 'midpoint':
            base_price = (high_price + low_price) / 2
        elif self.config.fill_price_method == 'conservative':
            base_price = high_price if side == 'buy' else low_price
        else:  # aggressive
            base_price = low_price if side == 'buy' else high_price
        
        # Calculate market impact
        trade_size = shares * base_price
        daily_volume = volume * close_price
        participation_rate = trade_size / daily_volume if daily_volume > 0 else 1.0
        
        if self.config.market_impact_model == 'linear':
            impact = self.config.impact_coefficient * participation_rate
        elif self.config.market_impact_model == 'square_root':
            impact = self.config.impact_coefficient * np.sqrt(participation_rate)
        else:  # power
            impact = self.config.impact_coefficient * (participation_rate ** 0.6)
        
        # Apply impact based on side
        if side == 'buy':
            impact_price = base_price * (1 + impact)
        else:
            impact_price = base_price * (1 - impact)
        
        # Ensure fill price is within day's range
        fill_price = np.clip(impact_price, low_price, high_price)
        
        # Calculate slippage (fixed + variable)
        slippage_pct = self.config.slippage_bps / 10000
        slippage_cost = fill_price * shares * slippage_pct
        
        # Calculate commission
        commission = max(shares * self.config.commission_per_share, self.config.min_commission)
        
        return fill_price, commission, slippage_cost
    
    def check_liquidity_constraints(self, 
                                  shares: float,
                                  symbol: str,
                                  volume: float,
                                  price: float) -> float:
        """Check if order size exceeds liquidity constraints"""
        
        # Maximum participation rate
        max_shares = volume * self.config.partial_fill_threshold
        
        if shares > max_shares:
            logger.warning(f"Order size for {symbol} exceeds liquidity. "
                         f"Requested: {shares}, Max: {max_shares}")
            return max_shares
        
        return shares

class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.execution_sim = ExecutionSimulator(self.config)
        self.reset()
        
    def reset(self):
        """Reset backtester state"""
        self.cash = self.config.initial_capital
        self.positions = {}  # symbol -> shares
        self.trades = []
        self.open_trades = {}  # symbol -> Trade
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        
    def run_backtest(self,
                    signals: List[Any],  # List of Signal objects
                    market_data: Dict[str, pd.DataFrame],
                    start_date: datetime,
                    end_date: datetime) -> Dict[str, Any]:
        """Run complete backtest on signals"""
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Create date range
        trading_days = pd.date_range(start_date, end_date, freq='B')
        
        # Group signals by date
        signals_by_date = self._group_signals_by_date(signals)
        
        # Initialize portfolio tracking
        portfolio_values = []
        high_water_mark = self.config.initial_capital
        
        # Main backtest loop
        for date in trading_days:
            # Update portfolio values with current prices
            self._update_portfolio_values(date, market_data)
            
            # Check for exit signals
            self._check_exit_conditions(date, market_data)
            
            # Process new signals for this date
            if date in signals_by_date:
                self._process_signals(signals_by_date[date], date, market_data)
            
            # Calculate portfolio metrics
            portfolio_value = self._calculate_portfolio_value(date, market_data)
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
            
            # Update high water mark and drawdown
            if portfolio_value > high_water_mark:
                high_water_mark = portfolio_value
            
            # Risk management checks
            current_drawdown = (high_water_mark - portfolio_value) / high_water_mark
            if current_drawdown > self.config.max_drawdown_limit:
                logger.warning(f"Max drawdown exceeded on {date}: {current_drawdown:.2%}")
                self._emergency_exit_all(date, market_data)
            
            # Record daily snapshot
            self._record_snapshot(date, market_data)
        
        # Close any remaining positions
        self._close_all_positions(end_date, market_data)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(portfolio_values)
        results['trades'] = self.trade_history
        results['portfolio_history'] = self.portfolio_history
        
        logger.info(f"Backtest completed. Total trades: {len(self.trade_history)}")
        
        return results
    
    def run_walk_forward_optimization(self,
                                    strategy_func: Callable,
                                    market_data: Dict[str, pd.DataFrame],
                                    feature_data: Dict[str, pd.DataFrame],
                                    start_date: datetime,
                                    end_date: datetime) -> Dict[str, Any]:
        """Run walk-forward optimization"""
        
        logger.info("Starting walk-forward optimization...")
        
        results = []
        current_date = start_date
        
        while current_date < end_date:
            # Define periods
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.train_period_months * 30)
            val_start = train_end
            val_end = val_start + timedelta(days=self.config.validation_period_months * 30)
            test_start = val_end
            test_end = min(test_start + timedelta(days=self.config.test_period_months * 30), end_date)
            
            logger.info(f"Walk-forward period: Train: {train_start} to {train_end}, "
                       f"Val: {val_start} to {val_end}, Test: {test_start} to {test_end}")
            
            # Train strategy
            strategy_params = strategy_func(
                market_data, feature_data,
                train_start, train_end,
                val_start, val_end
            )
            
            # Generate signals for test period
            test_signals = strategy_func.generate_signals(
                market_data, feature_data,
                test_start, test_end,
                strategy_params
            )
            
            # Run backtest on test period
            self.reset()
            period_results = self.run_backtest(
                test_signals, market_data,
                test_start, test_end
            )
            
            results.append({
                'period': f"{test_start} to {test_end}",
                'train_period': f"{train_start} to {train_end}",
                'results': period_results
            })
            
            # Move to next period
            current_date = test_start + timedelta(days=self.config.test_period_months * 30)
        
        # Aggregate results
        aggregated_results = self._aggregate_walk_forward_results(results)
        
        return aggregated_results
    
    def _group_signals_by_date(self, signals: List[Any]) -> Dict[datetime, List[Any]]:
        """Group signals by date"""
        signals_by_date = defaultdict(list)
        
        for signal in signals:
            # Extract date from timestamp
            signal_date = signal.timestamp.date()
            signals_by_date[pd.Timestamp(signal_date)] = signal
        
        return dict(signals_by_date)
    
    def _process_signals(self, signals: List[Any], date: datetime, 
                        market_data: Dict[str, pd.DataFrame]):
        """Process trading signals"""
        
        for signal in signals:
            symbol = signal.symbol
            
            # Skip if already have position
            if symbol in self.positions:
                continue
            
            # Get market data
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}")
                continue
            
            try:
                current_data = market_data[symbol].loc[date]
                volume_data = market_data[symbol]['Volume'].loc[date]
            except KeyError:
                continue
            
            # Calculate position size
            portfolio_value = self._calculate_portfolio_value(date, market_data)
            position_value = portfolio_value * signal.position_size
            shares = position_value / current_data['Close']
            
            # Check liquidity
            shares = self.execution_sim.check_liquidity_constraints(
                shares, symbol, volume_data, current_data['Close']
            )
            
            # Execute order
            fill_price, commission, slippage = self.execution_sim.execute_order(
                symbol, 'buy', shares, current_data, {symbol: volume_data}
            )
            
            # Check if we have enough cash
            total_cost = fill_price * shares + commission + slippage
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for {symbol}. "
                             f"Required: ${total_cost:.2f}, Available: ${self.cash:.2f}")
                continue
            
            # Create trade
            trade = Trade(
                trade_id=f"{symbol}_{date.strftime('%Y%m%d')}",
                symbol=symbol,
                side='buy',
                entry_date=date,
                entry_price=fill_price,
                shares=shares,
                commission_paid=commission,
                slippage_paid=slippage
            )
            
            # Update portfolio
            self.cash -= total_cost
            self.positions[symbol] = shares
            self.open_trades[symbol] = trade
            
            logger.info(f"Opened position: {symbol} - {shares:.2f} shares @ ${fill_price:.2f}")
    
    def _check_exit_conditions(self, date: datetime, market_data: Dict[str, pd.DataFrame]):
        """Check exit conditions for open positions"""
        
        for symbol, trade in list(self.open_trades.items()):
            if symbol not in market_data:
                continue
            
            try:
                current_data = market_data[symbol].loc[date]
                current_price = current_data['Close']
            except KeyError:
                continue
            
            # Update MAE/MFE
            price_change = (current_price - trade.entry_price) / trade.entry_price
            if trade.side == 'buy':
                trade.mae = min(trade.mae, price_change)
                trade.mfe = max(trade.mfe, price_change)
            else:
                trade.mae = max(trade.mae, -price_change)
                trade.mfe = min(trade.mfe, -price_change)
            
            # Check stop loss
            # This is simplified - in production, use the stop loss from signal
            stop_loss_pct = -0.05  # 5% stop loss
            if price_change <= stop_loss_pct:
                self._close_position(symbol, date, current_price, 'stop_loss', market_data)
                continue
            
            # Check take profit
            take_profit_pct = 0.10  # 10% take profit
            if price_change >= take_profit_pct:
                self._close_position(symbol, date, current_price, 'take_profit', market_data)
                continue
            
            # Check time stop
            holding_period = (date - trade.entry_date).days
            if holding_period > 20:  # 20 day time stop
                self._close_position(symbol, date, current_price, 'time_stop', market_data)
    
    def _close_position(self, symbol: str, date: datetime, price: float, 
                       reason: str, market_data: Dict[str, pd.DataFrame]):
        """Close a position"""
        
        if symbol not in self.open_trades:
            return
        
        trade = self.open_trades[symbol]
        shares = self.positions[symbol]
        
        # Get volume for execution
        try:
            volume = market_data[symbol]['Volume'].loc[date]
        except:
            volume = 1e6
        
        # Execute sell order
        fill_price, commission, slippage = self.execution_sim.execute_order(
            symbol, 'sell', shares, 
            market_data[symbol].loc[date],
            {symbol: volume}
        )
        
        # Calculate P&L
        gross_pnl = (fill_price - trade.entry_price) * shares
        net_pnl = gross_pnl - commission - slippage - trade.commission_paid - trade.slippage_paid
        pnl_percent = net_pnl / (trade.entry_price * shares)
        
        # Update trade
        trade.exit_date = date
        trade.exit_price = fill_price
        trade.pnl = net_pnl
        trade.pnl_percent = pnl_percent
        trade.exit_reason = reason
        
        # Update portfolio
        self.cash += fill_price * shares - commission - slippage
        del self.positions[symbol]
        del self.open_trades[symbol]
        
        # Record trade
        self.trade_history.append(trade)
        
        logger.info(f"Closed position: {symbol} - P&L: ${net_pnl:.2f} ({pnl_percent:.2%})")
    
    def _emergency_exit_all(self, date: datetime, market_data: Dict[str, pd.DataFrame]):
        """Emergency exit all positions"""
        logger.warning("Emergency exit triggered - closing all positions")
        
        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                try:
                    price = market_data[symbol].loc[date]['Close']
                    self._close_position(symbol, date, price, 'emergency_exit', market_data)
                except:
                    pass
    
    def _close_all_positions(self, date: datetime, market_data: Dict[str, pd.DataFrame]):
        """Close all remaining positions"""
        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                try:
                    price = market_data[symbol].loc[date]['Close']
                    self._close_position(symbol, date, price, 'end_of_backtest', market_data)
                except:
                    pass
    
    def _calculate_portfolio_value(self, date: datetime, 
                                  market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        
        for symbol, shares in self.positions.items():
            if symbol in market_data:
                try:
                    price = market_data[symbol].loc[date]['Close']
                    positions_value += shares * price
                except:
                    pass
        
        return self.cash + positions_value
    
    def _update_portfolio_values(self, date: datetime, market_data: Dict[str, pd.DataFrame]):
        """Update portfolio values with current prices"""
        # This is used for tracking unrealized P&L
        pass
    
    def _record_snapshot(self, date: datetime, market_data: Dict[str, pd.DataFrame]):
        """Record portfolio snapshot"""
        portfolio_value = self._calculate_portfolio_value(date, market_data)
        
        # Calculate returns
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1].total_value
            daily_return = (portfolio_value - prev_value) / prev_value
        else:
            daily_return = 0
        
        self.daily_returns.append(daily_return)
        
        # Calculate metrics
        if len(self.daily_returns) > 20:
            volatility = np.std(self.daily_returns[-252:]) * np.sqrt(252)
            sharpe = (np.mean(self.daily_returns[-252:]) * 252 - self.config.risk_free_rate) / volatility
        else:
            volatility = 0
            sharpe = 0
        
        snapshot = PortfolioSnapshot(
            timestamp=date,
            cash=self.cash,
            positions=self.positions.copy(),
            market_values={s: self.positions[s] * market_data[s].loc[date]['Close'] 
                         for s in self.positions if s in market_data},
            total_value=portfolio_value,
            returns=daily_return,
            drawdown=0,  # Calculate separately
            volatility=volatility,
            sharpe=sharpe,
            positions_count=len(self.positions)
        )
        
        self.portfolio_history.append(snapshot)
    
    def _calculate_performance_metrics(self, portfolio_values: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        # Convert to DataFrame for easier calculation
        pf_df = pd.DataFrame(portfolio_values)
        pf_df['returns'] = pf_df['value'].pct_change()
        
        # Basic metrics
        total_return = (pf_df['value'].iloc[-1] - pf_df['value'].iloc[0]) / pf_df['value'].iloc[0]
        
        # Annualized metrics
        n_days = len(pf_df)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Risk metrics
        daily_returns = pf_df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = annual_return - self.config.risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_returns / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and profit factor
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.pnl > 0]
            losing_trades = [t for t in self.trade_history if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(self.trade_history)
            
            if losing_trades:
                gross_profit = sum(t.pnl for t in winning_trades)
                gross_loss = abs(sum(t.pnl for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            else:
                profit_factor = float('inf') if winning_trades else 0
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            # Trade analysis
            avg_holding_period = np.mean([t.holding_period for t in self.trade_history])
            
            # MAE/MFE analysis
            avg_mae = np.mean([t.mae for t in self.trade_history])
            avg_mfe = np.mean([t.mfe for t in self.trade_history])
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            avg_holding_period = 0
            avg_mae = 0
            avg_mfe = 0
        
        # Value at Risk
        var_95 = np.percentile(daily_returns, 5)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trade_history),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_holding_period': avg_holding_period,
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'final_portfolio_value': pf_df['value'].iloc[-1],
            'portfolio_values': pf_df
        }
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from walk-forward optimization"""
        
        # Collect all metrics
        all_returns = []
        all_sharpes = []
        all_drawdowns = []
        all_trades = []
        
        for period_result in results:
            metrics = period_result['results']
            all_returns.append(metrics['annual_return'])
            all_sharpes.append(metrics['sharpe_ratio'])
            all_drawdowns.append(metrics['max_drawdown'])
            all_trades.extend(metrics['trades'])
        
        # Calculate aggregated metrics
        aggregated = {
            'periods': len(results),
            'avg_annual_return': np.mean(all_returns),
            'std_annual_return': np.std(all_returns),
            'avg_sharpe_ratio': np.mean(all_sharpes),
            'worst_drawdown': min(all_drawdowns),
            'avg_drawdown': np.mean(all_drawdowns),
            'total_trades': len(all_trades),
            'period_results': results,
            'consistency_score': np.mean(all_sharpes) / (np.std(all_sharpes) + 1e-6)
        }
        
        return aggregated

class BacktestAnalyzer:
    """Analyze and visualize backtest results"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        
    def generate_report(self) -> str:
        """Generate comprehensive backtest report"""
        report = []
        report.append("=" * 60)
        report.append("BACKTEST PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Performance Summary
        report.append("\nPERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Return: {self.results['total_return']:.2%}")
        report.append(f"Annual Return: {self.results['annual_return']:.2%}")
        report.append(f"Volatility: {self.results['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {self.results['sortino_ratio']:.2f}")
        report.append(f"Max Drawdown: {self.results['max_drawdown']:.2%}")
        report.append(f"Calmar Ratio: {self.results['calmar_ratio']:.2f}")
        
        # Trade Analysis
        report.append("\nTRADE ANALYSIS")
        report.append("-" * 30)
        report.append(f"Total Trades: {self.results['total_trades']}")
        report.append(f"Win Rate: {self.results['win_rate']:.2%}")
        report.append(f"Profit Factor: {self.results['profit_factor']:.2f}")
        report.append(f"Average Win: ${self.results['avg_win']:.2f}")
        report.append(f"Average Loss: ${self.results['avg_loss']:.2f}")
        report.append(f"Avg Holding Period: {self.results['avg_holding_period']:.1f} days")
        
        # Risk Analysis
        report.append("\nRISK ANALYSIS")
        report.append("-" * 30)
        report.append(f"Value at Risk (95%): {self.results['var_95']:.2%}")
        report.append(f"Conditional VaR (95%): {self.results['cvar_95']:.2%}")
        report.append(f"Average MAE: {self.results['avg_mae']:.2%}")
        report.append(f"Average MFE: {self.results['avg_mfe']:.2%}")
        
        return "\n".join(report)
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Create performance visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Portfolio value over time
        pf_values = self.results['portfolio_values']
        axes[0, 0].plot(pf_values.index, pf_values['value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown chart
        returns = pf_values['returns'].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        axes[0, 1].fill_between(pf_values.index[1:], drawdown * 100, 0, 
                               alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown Chart')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly returns heatmap
        if len(pf_values) > 30:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_pivot = monthly_returns.to_frame('returns').reset_index()
            monthly_pivot['year'] = monthly_pivot['date'].dt.year
            monthly_pivot['month'] = monthly_pivot['date'].dt.month
            
            pivot_table = monthly_pivot.pivot(index='year', columns='month', values='returns')
            
            sns.heatmap(pivot_table * 100, annot=True, fmt='.1f', 
                       cmap='RdYlGn', center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Monthly Returns Heatmap (%)')
        
        # 4. Trade P&L distribution
        if self.results['trades']:
            trade_pnls = [t.pnl for t in self.results['trades']]
            axes[1, 1].hist(trade_pnls, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Trade P&L Distribution')
            axes[1, 1].set_xlabel('P&L ($)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def analyze_trades_by_exit_reason(self) -> pd.DataFrame:
        """Analyze trades grouped by exit reason"""
        if not self.results['trades']:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'holding_period': t.holding_period,
            'exit_reason': t.exit_reason
        } for t in self.results['trades']])
        
        analysis = trades_df.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percent': ['mean', 'std'],
            'holding_period': 'mean'
        }).round(4)
        
        return analysis


# Example usage
if __name__ == "__main__":
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='B')
    
    # Generate sample market data for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'SPY']
    market_data = {}
    
    for symbol in symbols:
        # Generate realistic OHLCV data
        returns = np.random.randn(len(dates)) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        
        high = close * (1 + np.abs(np.random.randn(len(dates)) * 0.01))
        low = close * (1 - np.abs(np.random.randn(len(dates)) * 0.01))
        open_ = close * (1 + np.random.randn(len(dates)) * 0.005)
        volume = np.random.lognormal(16, 0.5, len(dates))
        
        market_data[symbol] = pd.DataFrame({
            'Open': open_,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
    
    # Create sample signals
    from signal_generation_risk import Signal  # Import from previous module
    
    signals = []
    signal_dates = dates[::10]  # Generate signals every 10 days
    
    for date in signal_dates[:20]:  # First 20 signal dates
        # Random signal
        symbol = np.random.choice(symbols[:-1])  # Exclude SPY
        
        signal = Signal(
            symbol=symbol,
            timestamp=date,
            direction='long',
            ml_score=np.random.uniform(0.6, 0.9),
            expected_return=np.random.uniform(0.02, 0.05),
            predicted_volatility=np.random.uniform(0.01, 0.03),
            confidence_interval=(0.01, 0.05),
            technical_score=np.random.uniform(0.5, 0.8),
            feature_quality_score=np.random.uniform(0.6, 0.9),
            pattern_score=np.random.uniform(0.4, 0.7),
            position_size=np.random.uniform(0.02, 0.05),
            stop_loss=95,
            take_profit=110,
            risk_reward_ratio=2.0
        )
        signals.append(signal)
    
    # Initialize backtester
    config = BacktestConfig(initial_capital=100000)
    backtester = Backtester(config)
    
    # Run backtest
    print("Running backtest...")
    results = backtester.run_backtest(
        signals,
        market_data,
        start_date=dates[0],
        end_date=dates[-1]
    )
    
    # Analyze results
    analyzer = BacktestAnalyzer(results)
    
    # Print report
    print("\n" + analyzer.generate_report())
    
    # Analyze trades by exit reason
    print("\nTrade Analysis by Exit Reason:")
    exit_analysis = analyzer.analyze_trades_by_exit_reason()
    if not exit_analysis.empty:
        print(exit_analysis)
    
    # Plot performance
    print("\nGenerating performance plots...")
    analyzer.plot_performance()
