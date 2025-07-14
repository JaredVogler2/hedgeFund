import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import the main system
from main_trading_system import ProfessionalMLTradingSystem
from trading_dashboard import TradingDashboard, DashboardConfig


# Initialize the trading system with your config
@st.cache_resource
def get_trading_system():
    """Initialize and cache the trading system"""
    return ProfessionalMLTradingSystem('config.json')


# Main app
def main():
    st.set_page_config(
        page_title="ML Trading System - Live",
        page_icon="📈",
        layout="wide"
    )

    # Initialize system
    try:
        trading_system = get_trading_system()

        # Verify Alpaca connection
        if trading_system.execution_engine:
            account_info = trading_system.execution_engine.get_account_info()
            st.sidebar.success(f"✅ Connected to Alpaca")
            st.sidebar.info(f"💰 Portfolio: ${account_info['portfolio_value']:,.2f}")
            st.sidebar.info(f"💵 Buying Power: ${account_info['buying_power']:,.2f}")

        # Create and run dashboard
        config = DashboardConfig()
        dashboard = TradingDashboard(trading_system, config)
        dashboard.run()

    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()


if __name__ == "__main__":
    main()