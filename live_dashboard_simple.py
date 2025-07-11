import streamlit as st
from main_trading_system import ProfessionalMLTradingSystem
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Live Trading Dashboard", page_icon="📈", layout="wide")


# Initialize system
@st.cache_resource
def init_system():
    return ProfessionalMLTradingSystem('config.json')


system = init_system()

st.title("🤖 ML Trading System - Live Dashboard")

# Sidebar
page = st.sidebar.selectbox("Navigation",
                            ["Overview", "Positions", "Account", "Performance"])

if page == "Overview":
    st.header("📊 Live Portfolio Overview")

    # Get real data
    if system.execution_engine:
        portfolio = system.execution_engine.get_portfolio_summary()
        account = system.execution_engine.get_account_info()

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Portfolio Value",
                      f"${portfolio['account_value']:,.2f}",
                      f"${portfolio['daily_pnl']:+,.2f}")

        with col2:
            st.metric("Cash/Buying Power",
                      f"${portfolio['buying_power']:,.2f}")

        with col3:
            st.metric("Positions Count",
                      portfolio['positions_count'])

        with col4:
            daily_return = portfolio['daily_pnl'] / (portfolio['account_value'] - portfolio['daily_pnl'])
            st.metric("Daily Return",
                      f"{daily_return:.2%}")

        # Show some charts
        st.subheader("Account History")
        st.info("Historical data will populate as you trade")

    else:
        st.error("Not connected to Alpaca")

elif page == "Positions":
    st.header("📈 Current Positions")

    if system.execution_engine and system.execution_engine.positions:
        # Create positions table
        positions_data = []

        for symbol, pos in system.execution_engine.positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Shares': pos.shares,
                'Avg Cost': f"${pos.avg_entry_price:.2f}",
                'Current Price': f"${pos.current_price:.2f}",
                'Market Value': f"${pos.market_value:,.2f}",
                'Unrealized P&L': f"${pos.unrealized_pnl:+,.2f}",
                'P&L %': f"{pos.unrealized_pnl_pct:+.2%}",
                'Status': '🟢' if pos.unrealized_pnl >= 0 else '🔴'
            })

        df = pd.DataFrame(positions_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Summary
        total_value = sum(pos.market_value for pos in system.execution_engine.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in system.execution_engine.positions.values())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Position Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Total Unrealized P&L", f"${total_pnl:+,.2f}")
        with col3:
            st.metric("Average P&L", f"{(total_pnl / total_value * 100) if total_value > 0 else 0:+.2f}%")

    else:
        st.info("No open positions")

elif page == "Account":
    st.header("💰 Account Information")

    if system.execution_engine:
        account = system.execution_engine.get_account_info()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Account Details")
            st.write(f"**Account Number:** {account['account_number']}")
            st.write(f"**Status:** {account['status']}")
            st.write(f"**Pattern Day Trader:** {'Yes' if account['pattern_day_trader'] else 'No'}")

        with col2:
            st.subheader("Balances")
            st.metric("Equity", f"${account['equity']:,.2f}")
            st.metric("Cash", f"${account['cash']:,.2f}")
            st.metric("Buying Power", f"${account['buying_power']:,.2f}")

        # Trading Status
        st.subheader("Trading Status")
        if account['trading_blocked']:
            st.error("⛔ Trading is blocked")
        else:
            st.success("✅ Trading is enabled")

elif page == "Performance":
    st.header("📊 Performance Metrics")

    performance = system.performance_tracker.get_performance_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", performance.get('total_trades', 0))
    with col2:
        st.metric("Win Rate", f"{performance.get('win_rate', 0):.1%}")
    with col3:
        st.metric("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
    with col4:
        st.metric("Profit Factor", f"{performance.get('profit_factor', 0):.2f}")

# Auto-refresh every 30 seconds
st.sidebar.button("🔄 Refresh Data")

# Add timestamp
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")