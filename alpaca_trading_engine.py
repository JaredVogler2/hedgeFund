"""
Alpaca Trading Engine - Live Trading Integration
Professional-grade live trading system with Alpaca API
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
import logging
import json
import asyncio
import websocket
from concurrent.futures import ThreadPoolExecutor
import pytz
from enum import Enum
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported by Alpaca"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class TimeInForce(Enum):
    """Time in force options"""
    DAY = "day"
    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At the open
    CLS = "cls"  # At the close

@dataclass
class AlpacaConfig:
    """Configuration for Alpaca trading"""
    # API Configuration
    api_key: str = os.getenv('ALPACA_API_KEY', '')
    secret_key: str = os.getenv('ALPACA_SECRET_KEY', '')
    base_url: str = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Trading Configuration
    use_paper: bool = True
    max_position_size: float = 0.10  # 10% max per position
    max_positions: int = 20
    
    # Order Configuration
    default_order_type: OrderType = OrderType.LIMIT
    default_time_in_force: TimeInForce = TimeInForce.DAY
    limit_price_buffer: float = 0.001  # 0.1% buffer for limit orders
    
    # Risk Configuration
    max_daily_loss: float = 0.02  # 2% max daily loss
    max_order_value: float = 50000  # Max $50k per order
    
    # Execution Configuration
    use_adaptive_orders: bool = True
    partial_fill_timeout: int = 300  # 5 minutes
    order_retry_attempts: int = 3
    
    # Market Hours (ET)
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    pre_market_start: time = time(4, 0)
    after_market_end: time = time(20, 0)

@dataclass
class Position:
    """Enhanced position tracking"""
    symbol: str
    shares: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime
    last_updated: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0

@dataclass
class Order:
    """Order tracking"""
    order_id: str
    symbol: str
    side: str
    qty: float
    order_type: OrderType
    time_in_force: TimeInForce
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_qty: float = 0
    filled_avg_price: Optional[float] = None
    status: str = "new"

class AlpacaTradingEngine:
    """Main Alpaca trading engine"""
    
    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig()
        self._validate_config()
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            self.config.api_key,
            self.config.secret_key,
            self.config.base_url,
            api_version='v2'
        )
        
        # Initialize state
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.executed_orders: List[Order] = []
        self.daily_pnl = 0.0
        self.is_connected = False
        
        # Initialize components
        self.risk_monitor = RiskMonitor(self.config)
        self.execution_engine = ExecutionEngine(self.api, self.config)
        self.position_manager = PositionManager(self.api, self.config)
        
        # Connect and sync
        self._connect()
        
    def _validate_config(self):
        """Validate configuration"""
        if not self.config.api_key or not self.config.secret_key:
            raise ValueError("Alpaca API credentials not configured")
        
        if self.config.use_paper and 'paper' not in self.config.base_url:
            logger.warning("Paper trading enabled but using live URL")
    
    def _connect(self):
        """Connect to Alpaca and sync account state"""
        try:
            # Test connection
            account = self.api.get_account()
            self.is_connected = True
            
            logger.info(f"Connected to Alpaca. Account: {account.account_number}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
            # Sync positions
            self._sync_positions()
            
            # Sync orders
            self._sync_orders()
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self.is_connected = False
            raise
    
    def _sync_positions(self):
        """Sync current positions from Alpaca"""
        try:
            alpaca_positions = self.api.list_positions()
            
            self.positions = {}
            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    shares=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                    entry_time=pos.exchange,  # This might need adjustment
                    last_updated=datetime.now()
                )
                self.positions[pos.symbol] = position
            
            logger.info(f"Synced {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def _sync_orders(self):
        """Sync open orders from Alpaca"""
        try:
            open_orders = self.api.list_orders(status='open')
            
            self.pending_orders = {}
            for order in open_orders:
                tracked_order = Order(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=float(order.qty),
                    order_type=OrderType(order.order_type),
                    time_in_force=TimeInForce(order.time_in_force),
                    limit_price=float(order.limit_price) if order.limit_price else None,
                    stop_price=float(order.stop_price) if order.stop_price else None,
                    submitted_at=order.submitted_at,
                    status=order.status
                )
                self.pending_orders[order.id] = tracked_order
            
            logger.info(f"Synced {len(self.pending_orders)} pending orders")
            
        except Exception as e:
            logger.error(f"Error syncing orders: {e}")
    
    def execute_signals(self, signals: List[Any]) -> Dict[str, Any]:
        """Execute trading signals"""
        if not self.is_connected:
            logger.error("Not connected to Alpaca")
            return {'status': 'error', 'message': 'Not connected'}
        
        # Check market hours
        if not self.is_market_open():
            logger.warning("Market is closed")
            return {'status': 'error', 'message': 'Market closed'}
        
        # Check daily loss limit
        if self.risk_monitor.check_daily_loss_limit(self.daily_pnl):
            logger.warning("Daily loss limit reached")
            return {'status': 'error', 'message': 'Daily loss limit reached'}
        
        results = {
            'executed': [],
            'rejected': [],
            'errors': []
        }
        
        for signal in signals:
            try:
                # Validate signal
                if not self._validate_signal(signal):
                    results['rejected'].append({
                        'symbol': signal.symbol,
                        'reason': 'Invalid signal'
                    })
                    continue
                
                # Check if we already have a position
                if signal.symbol in self.positions:
                    logger.info(f"Already have position in {signal.symbol}")
                    results['rejected'].append({
                        'symbol': signal.symbol,
                        'reason': 'Position exists'
                    })
                    continue
                
                # Execute order
                order_result = self._execute_signal(signal)
                
                if order_result['status'] == 'success':
                    results['executed'].append(order_result)
                else:
                    results['rejected'].append(order_result)
                    
            except Exception as e:
                logger.error(f"Error executing signal for {signal.symbol}: {e}")
                results['errors'].append({
                    'symbol': signal.symbol,
                    'error': str(e)
                })
        
        return results
    
    def _validate_signal(self, signal: Any) -> bool:
        """Validate trading signal"""
        # Check required attributes
        required_attrs = ['symbol', 'direction', 'position_size', 'stop_loss']
        if not all(hasattr(signal, attr) for attr in required_attrs):
            return False
        
        # Check position size
        if signal.position_size <= 0 or signal.position_size > self.config.max_position_size:
            return False
        
        # Check if symbol is tradable
        try:
            asset = self.api.get_asset(signal.symbol)
            if not asset.tradable:
                return False
        except:
            return False
        
        return True
    
    def _execute_signal(self, signal: Any) -> Dict[str, Any]:
        """Execute a single trading signal"""
        try:
            # Get account info
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Calculate order size
            position_value = buying_power * signal.position_size
            
            # Check max order value
            if position_value > self.config.max_order_value:
                position_value = self.config.max_order_value
            
            # Get current price
            quote = self.api.get_latest_quote(signal.symbol)
            current_price = (quote.ask_price + quote.bid_price) / 2
            
            # Calculate shares
            shares = int(position_value / current_price)
            
            if shares == 0:
                return {
                    'status': 'rejected',
                    'symbol': signal.symbol,
                    'reason': 'Order size too small'
                }
            
            # Determine order parameters
            if self.config.default_order_type == OrderType.LIMIT:
                # Place limit order with buffer
                if signal.direction == 'long':
                    limit_price = current_price * (1 + self.config.limit_price_buffer)
                else:
                    limit_price = current_price * (1 - self.config.limit_price_buffer)
            else:
                limit_price = None
            
            # Submit order
            order = self.execution_engine.submit_order(
                symbol=signal.symbol,
                qty=shares,
                side='buy' if signal.direction == 'long' else 'sell',
                order_type=self.config.default_order_type,
                time_in_force=self.config.default_time_in_force,
                limit_price=limit_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit if hasattr(signal, 'take_profit') else None
            )
            
            if order:
                # Track order
                tracked_order = Order(
                    order_id=order.id,
                    symbol=signal.symbol,
                    side='buy' if signal.direction == 'long' else 'sell',
                    qty=shares,
                    order_type=self.config.default_order_type,
                    time_in_force=self.config.default_time_in_force,
                    limit_price=limit_price,
                    submitted_at=datetime.now(),
                    status='pending'
                )
                self.pending_orders[order.id] = tracked_order
                
                logger.info(f"Order submitted: {signal.symbol} - {shares} shares @ "
                          f"${limit_price or 'market':.2f}")
                
                return {
                    'status': 'success',
                    'symbol': signal.symbol,
                    'order_id': order.id,
                    'shares': shares,
                    'order_type': self.config.default_order_type.value
                }
            else:
                return {
                    'status': 'rejected',
                    'symbol': signal.symbol,
                    'reason': 'Order submission failed'
                }
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {
                'status': 'error',
                'symbol': signal.symbol,
                'reason': str(e)
            }
    
    def update_positions(self):
        """Update all position information"""
        self._sync_positions()
        
        # Update stops and targets
        for symbol, position in self.positions.items():
            try:
                # Get latest quote
                quote = self.api.get_latest_quote(symbol)
                current_price = (quote.ask_price + quote.bid_price) / 2
                
                # Update position
                position.current_price = current_price
                position.market_value = position.shares * current_price
                position.unrealized_pnl = (current_price - position.avg_entry_price) * position.shares
                position.unrealized_pnl_pct = position.unrealized_pnl / (position.avg_entry_price * position.shares)
                position.last_updated = datetime.now()
                
                # Check if we need to update stops
                if self.position_manager.should_update_stop(position):
                    new_stop = self.position_manager.calculate_trailing_stop(position)
                    self.position_manager.update_stop_loss(symbol, new_stop)
                    
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
    
    def close_position(self, symbol: str, reason: str = "manual") -> Dict[str, Any]:
        """Close a position"""
        if symbol not in self.positions:
            return {'status': 'error', 'message': f'No position in {symbol}'}
        
        position = self.positions[symbol]
        
        try:
            # Submit close order
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(position.shares),
                side='sell' if position.shares > 0 else 'buy',
                type=OrderType.MARKET.value,
                time_in_force=TimeInForce.DAY.value
            )
            
            logger.info(f"Closing position {symbol}: {position.shares} shares. Reason: {reason}")
            
            return {
                'status': 'success',
                'order_id': order.id,
                'symbol': symbol,
                'shares': position.shares
            }
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def close_all_positions(self, reason: str = "end_of_day"):
        """Close all positions"""
        results = []
        
        for symbol in list(self.positions.keys()):
            result = self.close_position(symbol, reason)
            results.append(result)
        
        return results
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        clock = self.api.get_clock()
        return clock.is_open
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information"""
        try:
            account = self.api.get_account()
            
            return {
                'account_number': account.account_number,
                'status': account.status,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'daily_pnl': float(account.equity) - float(account.last_equity),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        account_info = self.get_account_info()
        
        # Calculate position metrics
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        winning_positions = sum(1 for p in self.positions.values() if p.is_profitable)
        
        return {
            'account_value': account_info.get('portfolio_value', 0),
            'cash': account_info.get('cash', 0),
            'buying_power': account_info.get('buying_power', 0),
            'positions_count': len(self.positions),
            'winning_positions': winning_positions,
            'total_unrealized_pnl': total_unrealized_pnl,
            'daily_pnl': account_info.get('daily_pnl', 0),
            'positions': self.positions
        }

class ExecutionEngine:
    """Handles order execution logic"""
    
    def __init__(self, api: tradeapi.REST, config: AlpacaConfig):
        self.api = api
        self.config = config
        
    def submit_order(self, 
                    symbol: str,
                    qty: int,
                    side: str,
                    order_type: OrderType,
                    time_in_force: TimeInForce,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> Optional[Any]:
        """Submit order with smart routing"""
        
        try:
            # Basic order submission
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type.value,
                time_in_force=time_in_force.value,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            # Submit bracket orders if stops provided
            if stop_loss or take_profit:
                self._submit_bracket_order(
                    symbol, qty, side, order.id,
                    stop_loss, take_profit
                )
            
            return order
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return None
    
    def _submit_bracket_order(self,
                             symbol: str,
                             qty: int,
                             side: str,
                             parent_order_id: str,
                             stop_loss: Optional[float],
                             take_profit: Optional[float]):
        """Submit bracket orders for stop loss and take profit"""
        
        # Opposite side for exits
        exit_side = 'sell' if side == 'buy' else 'buy'
        
        try:
            if stop_loss:
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=exit_side,
                    type='stop',
                    stop_price=stop_loss,
                    time_in_force='gtc',
                    order_class='oto',
                    parent_id=parent_order_id
                )
                
            if take_profit:
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=exit_side,
                    type='limit',
                    limit_price=take_profit,
                    time_in_force='gtc',
                    order_class='oto',
                    parent_id=parent_order_id
                )
                
        except Exception as e:
            logger.error(f"Error submitting bracket orders: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def modify_order(self, 
                    order_id: str,
                    qty: Optional[int] = None,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> bool:
        """Modify an existing order"""
        try:
            self.api.replace_order(
                order_id,
                qty=qty,
                limit_price=limit_price,
                stop_price=stop_price
            )
            return True
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return False

class PositionManager:
    """Manages position lifecycle and updates"""
    
    def __init__(self, api: tradeapi.REST, config: AlpacaConfig):
        self.api = api
        self.config = config
        
    def should_update_stop(self, position: Position) -> bool:
        """Determine if stop loss should be updated"""
        if not position.stop_loss:
            return False
        
        # Only trail stops for profitable positions
        if not position.is_profitable:
            return False
        
        # Check if price has moved enough
        price_change = (position.current_price - position.avg_entry_price) / position.avg_entry_price
        
        # Trail stop if profit > 2%
        return price_change > 0.02
    
    def calculate_trailing_stop(self, position: Position, atr: Optional[float] = None) -> float:
        """Calculate trailing stop loss"""
        if atr is None:
            # Use fixed percentage if ATR not available
            atr = position.current_price * 0.02
        
        # Trail stop at 2 ATR from current price
        trailing_stop = position.current_price - (2 * atr)
        
        # Never lower the stop
        if position.stop_loss:
            return max(trailing_stop, position.stop_loss)
        
        return trailing_stop
    
    def update_stop_loss(self, symbol: str, new_stop: float) -> bool:
        """Update stop loss order"""
        try:
            # Find existing stop order
            orders = self.api.list_orders(
                status='open',
                symbols=symbol
            )
            
            stop_orders = [o for o in orders if o.order_type == 'stop']
            
            if stop_orders:
                # Modify existing stop
                return self.api.replace_order(
                    stop_orders[0].id,
                    stop_price=new_stop
                )
            else:
                # Create new stop order
                position = self.api.get_position(symbol)
                self.api.submit_order(
                    symbol=symbol,
                    qty=abs(float(position.qty)),
                    side='sell' if float(position.qty) > 0 else 'buy',
                    type='stop',
                    stop_price=new_stop,
                    time_in_force='gtc'
                )
                return True
                
        except Exception as e:
            logger.error(f"Error updating stop loss for {symbol}: {e}")
            return False

class RiskMonitor:
    """Monitors portfolio risk in real-time"""
    
    def __init__(self, config: AlpacaConfig):
        self.config = config
        self.daily_loss = 0.0
        self.max_daily_loss_hit = False
        
    def check_daily_loss_limit(self, current_pnl: float) -> bool:
        """Check if daily loss limit is exceeded"""
        self.daily_loss = current_pnl
        
        if current_pnl < 0 and abs(current_pnl) > self.config.max_daily_loss:
            self.max_daily_loss_hit = True
            return True
        
        return False
    
    def check_position_limits(self, positions: Dict[str, Position]) -> List[str]:
        """Check for positions exceeding limits"""
        violations = []
        
        for symbol, position in positions.items():
            # Check position size
            if position.market_value > self.config.max_order_value:
                violations.append(f"{symbol}: Position size exceeds limit")
            
            # Check loss on position
            if position.unrealized_pnl_pct < -0.10:  # 10% loss
                violations.append(f"{symbol}: Position loss exceeds 10%")
        
        return violations
    
    def calculate_portfolio_var(self, positions: Dict[str, Position], confidence: float = 0.95) -> float:
        """Calculate portfolio Value at Risk"""
        if not positions:
            return 0.0
        
        # Simplified VaR calculation
        position_values = [p.market_value for p in positions.values()]
        position_vols = [0.02 for _ in positions]  # Assume 2% daily vol
        
        # Portfolio volatility (simplified - assumes no correlation)
        portfolio_vol = np.sqrt(sum((v * vol)**2 for v, vol in zip(position_values, position_vols)))
        
        # VaR at confidence level
        z_score = stats.norm.ppf(confidence)
        var = portfolio_vol * z_score
        
        return var

class AlpacaDataStream:
    """Real-time data streaming from Alpaca"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.ws = None
        self.subscribed_symbols = set()
        
    async def connect(self):
        """Connect to Alpaca data stream"""
        # Implementation would connect to Alpaca's websocket
        pass
    
    async def subscribe_trades(self, symbols: List[str], handler):
        """Subscribe to trade updates"""
        self.subscribed_symbols.update(symbols)
        # Implementation would subscribe to trade stream
        pass
    
    async def subscribe_quotes(self, symbols: List[str], handler):
        """Subscribe to quote updates"""
        self.subscribed_symbols.update(symbols)
        # Implementation would subscribe to quote stream
        pass


# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = AlpacaConfig()
    
    # Check if credentials are set
    if not config.api_key or not config.secret_key:
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        exit(1)
    
    # Initialize trading engine
    engine = AlpacaTradingEngine(config)
    
    # Get account info
    account_info = engine.get_account_info()
    print(f"\nAccount Information:")
    print(f"Account Number: {account_info.get('account_number')}")
    print(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
    print(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
    print(f"Daily P&L: ${account_info.get('daily_pnl', 0):,.2f}")
    
    # Get portfolio summary
    portfolio = engine.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"Positions: {portfolio['positions_count']}")
    print(f"Unrealized P&L: ${portfolio['total_unrealized_pnl']:,.2f}")
    
    # Display current positions
    if portfolio['positions']:
        print("\nCurrent Positions:")
        for symbol, position in portfolio['positions'].items():
            print(f"\n{symbol}:")
            print(f"  Shares: {position.shares}")
            print(f"  Avg Price: ${position.avg_entry_price:.2f}")
            print(f"  Current Price: ${position.current_price:.2f}")
            print(f"  P&L: ${position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.2%})")
    
    # Example: Create mock signals for testing
    from signal_generation_risk import Signal  # Import from previous module
    
    # Create a test signal (DO NOT USE IN PRODUCTION)
    test_signal = Signal(
        symbol='AAPL',
        timestamp=datetime.now(),
        direction='long',
        ml_score=0.75,
        expected_return=0.03,
        predicted_volatility=0.02,
        confidence_interval=(0.01, 0.05),
        technical_score=0.7,
        feature_quality_score=0.8,
        pattern_score=0.6,
        position_size=0.02,  # 2% position
        stop_loss=150,  # Example stop
        take_profit=165,  # Example target
        risk_reward_ratio=2.0
    )
    
    # Execute signals (commented out for safety)
    # results = engine.execute_signals([test_signal])
    # print(f"\nExecution Results: {results}")
    
    # Check if market is open
    print(f"\nMarket Open: {engine.is_market_open()}")
    
    # Example: Close all positions at end of day
    # if not engine.is_market_open():
    #     results = engine.close_all_positions("end_of_day")
    #     print(f"Closed positions: {results}")
