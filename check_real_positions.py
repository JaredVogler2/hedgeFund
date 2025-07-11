from main_trading_system import ProfessionalMLTradingSystem
import streamlit as st

# Initialize system
system = ProfessionalMLTradingSystem('config.json')

# Get real portfolio data
portfolio = system.execution_engine.get_portfolio_summary()

print("\n📊 Your Real Portfolio:")
print(f"Account Value: ${portfolio['account_value']:,.2f}")
print(f"Buying Power: ${portfolio['buying_power']:,.2f}")
print(f"Positions: {portfolio['positions_count']}")

print("\n📈 Current Positions:")
for symbol, pos in system.execution_engine.positions.items():
    print(f"{symbol}: {pos.shares} shares @ ${pos.current_price:.2f} (P&L: ${pos.unrealized_pnl:+,.2f})")