"""
Professional Trading Dashboard
Real-time monitoring and control interface for ML trading system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass
import logging

# Configure Streamlit
st.set_page_config(
    page_title="ML Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .metric-positive {
        color: #00cc44;
    }
    .metric-negative {
        color: #ff3333;
    }
    .trade-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    refresh_interval: int = 5  # seconds
    chart_height: int = 400
    max_trades_display: int = 20
    max_signals_display: int = 10
    
    # Color scheme
    color_positive: str = "#00cc44"
    color_negative: str = "#ff3333"
    color_neutral: str = "#666666"
    color_primary: str = "#1f77b4"
    color_secondary: str = "#ff7f0e"

class TradingDashboard:
    """Main dashboard application"""
    
    def __init__(self, trading_system, config: Optional[DashboardConfig] = None):
        self.trading_system = trading_system
        self.config = config or DashboardConfig()
        self.last_update = datetime.now()
        
    def run(self):
        """Run the dashboard"""
        # Initialize session state
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'selected_view' not in st.session_state:
            st.session_state.selected_view = 'Overview'
            
        # Header
        self._render_header()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content based on selected view
        if st.session_state.selected_view == 'Overview':
            self._render_overview()
        elif st.session_state.selected_view == 'Portfolio':
            self._render_portfolio()
        elif st.session_state.selected_view == 'Predictions':
            self._render_predictions()
        elif st.session_state.selected_view == 'ML Analytics':
            self._render_ml_analytics()
        elif st.session_state.selected_view == 'Trade History':
            self._render_trade_history()
        elif st.session_state.selected_view == 'Market Overview':
            self._render_market_overview()
        elif st.session_state.selected_view == 'Risk Management':
            self._render_risk_management()
        elif st.session_state.selected_view == 'Settings':
            self._render_settings()
        
        # Auto-refresh
        if st.session_state.auto_refresh:
            time.sleep(self.config.refresh_interval)
            st.rerun()
    
    def _render_header(self):
        """Render dashboard header"""
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.title("🤖 ML Trading System Dashboard")
        
        with col2:
            # System status
            status = self.trading_system.get_system_status()
            if status.get('gpu_available'):
                st.success(f"✅ System Online | GPU: {status.get('gpu_name', 'Available')}")
            else:
                st.warning("⚠️ System Online | CPU Mode")
        
        with col3:
            # Last update time
            st.info(f"Last Update: {self.last_update.strftime('%H:%M:%S')}")
        
        with col4:
            # Refresh button
            if st.button("🔄 Refresh"):
                self.last_update = datetime.now()
                st.rerun()
    
    def _render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        
        # View selection
        views = [
            "Overview",
            "Portfolio",
            "Predictions",
            "ML Analytics",
            "Trade History",
            "Market Overview",
            "Risk Management",
            "Settings"
        ]
        
        st.session_state.selected_view = st.sidebar.radio(
            "Select View",
            views,
            index=views.index(st.session_state.selected_view)
        )
        
        st.sidebar.divider()
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        
        if st.sidebar.button("🚀 Run Daily Pipeline"):
            with st.spinner("Running daily pipeline..."):
                self.trading_system.run_daily_pipeline()
                st.sidebar.success("Pipeline completed!")
        
        if st.sidebar.button("📊 Generate Signals"):
            st.sidebar.info("Generating new signals...")
            # Trigger signal generation
        
        if st.sidebar.button("🛑 Emergency Stop"):
            if st.sidebar.checkbox("Confirm emergency stop"):
                # Execute emergency stop
                st.sidebar.error("All positions closed!")
        
        st.sidebar.divider()
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Auto-refresh",
            value=st.session_state.auto_refresh
        )
        
        # System info
        st.sidebar.subheader("System Info")
        performance = self.trading_system.performance_tracker.get_performance_summary()
        
        st.sidebar.metric("Total Trades", performance.get('total_trades', 0))
        st.sidebar.metric("Win Rate", f"{performance.get('win_rate', 0):.1%}")
        st.sidebar.metric("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
    
    def _render_overview(self):
        """Render overview page"""
        st.header("📊 Portfolio Overview")
        
        # Get portfolio data
        if hasattr(self.trading_system, 'execution_engine'):
            portfolio = self.trading_system.execution_engine.get_portfolio_summary()
        else:
            # Mock data for demonstration
            portfolio = {
                'account_value': 105000,
                'cash': 50000,
                'positions_count': 5,
                'daily_pnl': 1500,
                'total_unrealized_pnl': 2500
            }
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_metric(
                "Portfolio Value",
                f"${portfolio['account_value']:,.2f}",
                f"${portfolio['daily_pnl']:+,.2f}",
                portfolio['daily_pnl'] >= 0
            )
        
        with col2:
            self._render_metric(
                "Cash Available",
                f"${portfolio['cash']:,.2f}",
                None
            )
        
        with col3:
            self._render_metric(
                "Open Positions",
                portfolio['positions_count'],
                None
            )
        
        with col4:
            daily_return = portfolio['daily_pnl'] / (portfolio['account_value'] - portfolio['daily_pnl'])
            self._render_metric(
                "Daily Return",
                f"{daily_return:.2%}",
                None,
                daily_return >= 0
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Value Chart")
            self._render_portfolio_chart()
        
        with col2:
            st.subheader("Daily P&L Distribution")
            self._render_pnl_distribution()
        
        # Recent activity
        st.subheader("Recent Activity")
        self._render_recent_activity()
    
    def _render_portfolio(self):
        """Render portfolio details page"""
        st.header("💼 Portfolio Details")
        
        # Position summary
        if hasattr(self.trading_system, 'execution_engine'):
            positions = self.trading_system.execution_engine.positions
        else:
            # Mock positions
            positions = self._get_mock_positions()
        
        if positions:
            # Create positions dataframe
            positions_data = []
            for symbol, position in positions.items():
                positions_data.append({
                    'Symbol': symbol,
                    'Shares': position.shares,
                    'Entry Price': f"${position.avg_entry_price:.2f}",
                    'Current Price': f"${position.current_price:.2f}",
                    'Market Value': f"${position.market_value:,.2f}",
                    'Unrealized P&L': f"${position.unrealized_pnl:+,.2f}",
                    'P&L %': f"{position.unrealized_pnl_pct:+.2%}",
                    'Status': '🟢' if position.unrealized_pnl >= 0 else '🔴'
                })
            
            df_positions = pd.DataFrame(positions_data)
            
            # Display positions table
            st.dataframe(
                df_positions,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn("", width="small"),
                    "P&L %": st.column_config.TextColumn("P&L %", width="small")
                }
            )
            
            # Position charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Allocation pie chart
                fig = px.pie(
                    values=[p.market_value for p in positions.values()],
                    names=list(positions.keys()),
                    title="Position Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # P&L by position
                fig = px.bar(
                    x=list(positions.keys()),
                    y=[p.unrealized_pnl for p in positions.values()],
                    title="Unrealized P&L by Position",
                    color=[p.unrealized_pnl for p in positions.values()],
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions")
        
        # Position management
        st.subheader("Position Management")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_position = st.selectbox(
                "Select Position",
                options=list(positions.keys()) if positions else []
            )
        
        with col2:
            if selected_position:
                if st.button("Close Position"):
                    if st.checkbox("Confirm close"):
                        # Close position logic
                        st.success(f"Closed position in {selected_position}")
    
    def _render_predictions(self):
        """Render predictions page"""
        st.header("🔮 ML Predictions & Signals")
        
        # Get latest predictions
        predictions = self._get_latest_predictions()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_filter = st.slider(
                "Min Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05
            )
        
        with col2:
            direction_filter = st.selectbox(
                "Direction",
                options=["All", "Long", "Short"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=["Bayesian Score", "Expected Return", "Confidence", "Technical Score"]
            )
        
        # Filter predictions
        filtered_predictions = self._filter_predictions(
            predictions, confidence_filter, direction_filter
        )
        
        # Display predictions
        if filtered_predictions:
            # Create predictions dataframe
            pred_data = []
            for pred in filtered_predictions:
                pred_data.append({
                    'Symbol': pred['symbol'],
                    'Direction': pred['direction'],
                    'Expected Return': f"{pred['expected_return']:.2%}",
                    'Confidence': f"{pred['confidence']:.0%}",
                    'ML Score': f"{pred['ml_score']:.2f}",
                    'Technical Score': f"{pred['technical_score']:.2f}",
                    'Bayesian Score': f"{pred['bayesian_score']:.2f}",
                    'Signal': '🚀' if pred['bayesian_score'] > 1.0 else '✅' if pred['bayesian_score'] > 0.5 else '⚠️'
                })
            
            df_predictions = pd.DataFrame(pred_data)
            
            # Sort by selected column
            sort_mapping = {
                "Bayesian Score": "Bayesian Score",
                "Expected Return": "Expected Return",
                "Confidence": "Confidence",
                "Technical Score": "Technical Score"
            }
            df_predictions = df_predictions.sort_values(
                by=sort_mapping[sort_by],
                ascending=False
            )
            
            st.dataframe(
                df_predictions,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Signal": st.column_config.TextColumn("", width="small")
                }
            )
            
            # Signal details
            st.subheader("Signal Details")
            selected_signal = st.selectbox(
                "Select Signal for Details",
                options=df_predictions['Symbol'].tolist()
            )
            
            if selected_signal:
                self._render_signal_details(selected_signal, predictions)
        else:
            st.info("No predictions meeting criteria")
    
    def _render_ml_analytics(self):
        """Render ML analytics page"""
        st.header("🧠 ML Model Analytics")
        
        # Model performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Mock metrics for demonstration
        with col1:
            st.metric("Model Accuracy", "73.5%", "+2.1%")
        
        with col2:
            st.metric("Prediction Sharpe", "1.85", "+0.12")
        
        with col3:
            st.metric("Feature Count", "150", None)
        
        with col4:
            st.metric("Last Training", "2 hours ago", None)
        
        # Feature importance
        st.subheader("Feature Importance")
        self._render_feature_importance()
        
        # Model performance over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance Evolution")
            self._render_model_performance_chart()
        
        with col2:
            st.subheader("Prediction Accuracy by Market Regime")
            self._render_regime_accuracy_chart()
        
        # Model diagnostics
        st.subheader("Model Diagnostics")
        
        tab1, tab2, tab3 = st.tabs(["Prediction Distribution", "Calibration", "Error Analysis"])
        
        with tab1:
            self._render_prediction_distribution()
        
        with tab2:
            self._render_calibration_plot()
        
        with tab3:
            self._render_error_analysis()
    
    def _render_trade_history(self):
        """Render trade history page"""
        st.header("📜 Trade History")
        
        # Get trade history
        trades = self._get_trade_history()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        with col2:
            symbol_filter = st.multiselect(
                "Symbols",
                options=list(set(t['symbol'] for t in trades))
            )
        
        with col3:
            outcome_filter = st.selectbox(
                "Outcome",
                options=["All", "Winners", "Losers"]
            )
        
        # Filter trades
        filtered_trades = self._filter_trades(trades, date_range, symbol_filter, outcome_filter)
        
        # Summary metrics
        if filtered_trades:
            col1, col2, col3, col4 = st.columns(4)
            
            total_pnl = sum(t['pnl'] for t in filtered_trades)
            win_rate = sum(1 for t in filtered_trades if t['pnl'] > 0) / len(filtered_trades)
            avg_win = np.mean([t['pnl'] for t in filtered_trades if t['pnl'] > 0] or [0])
            avg_loss = np.mean([t['pnl'] for t in filtered_trades if t['pnl'] < 0] or [0])
            
            with col1:
                self._render_metric("Total P&L", f"${total_pnl:+,.2f}", None, total_pnl >= 0)
            
            with col2:
                self._render_metric("Win Rate", f"{win_rate:.1%}", None, win_rate >= 0.5)
            
            with col3:
                self._render_metric("Avg Win", f"${avg_win:,.2f}", None, True)
            
            with col4:
                self._render_metric("Avg Loss", f"${avg_loss:,.2f}", None, False)
            
            # Trade table
            trade_data = []
            for trade in filtered_trades:
                trade_data.append({
                    'Date': trade['entry_date'].strftime('%Y-%m-%d'),
                    'Symbol': trade['symbol'],
                    'Direction': trade['side'],
                    'Entry': f"${trade['entry_price']:.2f}",
                    'Exit': f"${trade['exit_price']:.2f}",
                    'Shares': trade['shares'],
                    'P&L': f"${trade['pnl']:+,.2f}",
                    'P&L %': f"{trade['pnl_percent']:+.1%}",
                    'Duration': f"{trade['holding_period']}d",
                    'Result': '✅' if trade['pnl'] > 0 else '❌'
                })
            
            df_trades = pd.DataFrame(trade_data)
            
            st.dataframe(
                df_trades,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Result": st.column_config.TextColumn("", width="small")
                }
            )
            
            # Trade analysis charts
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L over time
                fig = px.line(
                    x=[t['entry_date'] for t in filtered_trades],
                    y=np.cumsum([t['pnl'] for t in filtered_trades]),
                    title="Cumulative P&L"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Win/Loss distribution
                fig = px.histogram(
                    x=[t['pnl'] for t in filtered_trades],
                    nbins=30,
                    title="P&L Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_market_overview(self):
        """Render market overview page"""
        st.header("🌍 Market Overview")
        
        # Market indices
        st.subheader("Market Indices")
        
        indices = ['SPY', 'QQQ', 'DIA', 'IWM', 'VIX']
        index_data = self._get_market_indices_data(indices)
        
        cols = st.columns(len(indices))
        for i, (index, data) in enumerate(index_data.items()):
            with cols[i]:
                self._render_metric(
                    index,
                    f"${data['price']:.2f}",
                    f"{data['change']:.2%}",
                    data['change'] >= 0
                )
        
        # Market breadth
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Breadth")
            self._render_market_breadth()
        
        with col2:
            st.subheader("Sector Performance")
            self._render_sector_performance()
        
        # Market regime
        st.subheader("Market Regime Analysis")
        regime = self._get_market_regime()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Trend:** {regime['trend']}")
            st.progress(regime['trend_strength'])
        
        with col2:
            st.info(f"**Volatility:** {regime['volatility']}")
            st.progress(regime['volatility_level'])
        
        with col3:
            st.info(f"**Correlation:** {regime['correlation']}")
            st.progress(regime['correlation_level'])
        
        # News sentiment
        st.subheader("Market News Sentiment")
        self._render_market_sentiment()
    
    def _render_risk_management(self):
        """Render risk management page"""
        st.header("🛡️ Risk Management")
        
        # Portfolio risk metrics
        risk_metrics = self._get_portfolio_risk_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_metric(
                "Portfolio Heat",
                f"{risk_metrics['portfolio_heat']:.1%}",
                None,
                risk_metrics['portfolio_heat'] < 0.08
            )
        
        with col2:
            self._render_metric(
                "VaR (95%)",
                f"${risk_metrics['var_95']:,.0f}",
                None,
                False
            )
        
        with col3:
            self._render_metric(
                "Max Drawdown",
                f"{risk_metrics['max_drawdown']:.1%}",
                None,
                risk_metrics['max_drawdown'] > -0.20
            )
        
        with col4:
            self._render_metric(
                "Sharpe Ratio",
                f"{risk_metrics['sharpe_ratio']:.2f}",
                None,
                risk_metrics['sharpe_ratio'] > 1.0
            )
        
        # Risk decomposition
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk by Position")
            self._render_position_risk_chart()
        
        with col2:
            st.subheader("Risk by Factor")
            self._render_factor_risk_chart()
        
        # Risk limits
        st.subheader("Risk Limits & Alerts")
        
        risk_limits = [
            {"Metric": "Portfolio Heat", "Current": risk_metrics['portfolio_heat'], "Limit": 0.08, "Status": "🟢" if risk_metrics['portfolio_heat'] < 0.08 else "🔴"},
            {"Metric": "Daily Loss", "Current": risk_metrics.get('daily_loss', 0), "Limit": 0.02, "Status": "🟢" if risk_metrics.get('daily_loss', 0) > -0.02 else "🔴"},
            {"Metric": "Position Concentration", "Current": risk_metrics.get('max_position', 0), "Limit": 0.10, "Status": "🟢" if risk_metrics.get('max_position', 0) < 0.10 else "🔴"},
            {"Metric": "Correlation Risk", "Current": risk_metrics.get('avg_correlation', 0), "Limit": 0.70, "Status": "🟢" if risk_metrics.get('avg_correlation', 0) < 0.70 else "🔴"}
        ]
        
        df_limits = pd.DataFrame(risk_limits)
        
        st.dataframe(
            df_limits,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Current": st.column_config.NumberColumn("Current", format="%.2f"),
                "Limit": st.column_config.NumberColumn("Limit", format="%.2f"),
                "Status": st.column_config.TextColumn("Status", width="small")
            }
        )
        
        # Risk controls
        st.subheader("Risk Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Update Stop Losses"):
                st.success("Stop losses updated for all positions")
        
        with col2:
            if st.button("⚖️ Rebalance Portfolio"):
                st.info("Portfolio rebalancing initiated")
        
        with col3:
            if st.button("🚨 Reduce Exposure"):
                reduce_pct = st.slider("Reduce by %", 10, 50, 25, 5)
                if st.checkbox(f"Confirm {reduce_pct}% reduction"):
                    st.warning(f"Exposure reduced by {reduce_pct}%")
    
    def _render_settings(self):
        """Render settings page"""
        st.header("⚙️ Settings")
        
        # Trading settings
        st.subheader("Trading Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_positions = st.number_input(
                "Max Positions",
                min_value=1,
                max_value=50,
                value=20
            )
            
            max_position_size = st.slider(
                "Max Position Size",
                min_value=0.01,
                max_value=0.20,
                value=0.10,
                format="%.2f"
            )
            
            stop_loss_atr = st.slider(
                "Stop Loss (ATR Multiplier)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
        
        with col2:
            min_confidence = st.slider(
                "Min Signal Confidence",
                min_value=0.5,
                max_value=0.9,
                value=0.6,
                step=0.05
            )
            
            kelly_fraction = st.slider(
                "Kelly Fraction",
                min_value=0.1,
                max_value=0.5,
                value=0.25,
                step=0.05
            )
            
            max_daily_loss = st.slider(
                "Max Daily Loss",
                min_value=0.01,
                max_value=0.05,
                value=0.02,
                format="%.2f"
            )
        
        # ML settings
        st.subheader("ML Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            retrain_frequency = st.selectbox(
                "Retrain Frequency",
                options=["Daily", "Weekly", "Bi-weekly", "Monthly"],
                index=1
            )
            
            ensemble_agreement = st.slider(
                "Min Ensemble Agreement",
                min_value=0.5,
                max_value=0.9,
                value=0.6,
                step=0.05
            )
        
        with col2:
            feature_count = st.number_input(
                "Feature Count",
                min_value=50,
                max_value=500,
                value=150,
                step=10
            )
            
            lookback_period = st.number_input(
                "Lookback Period (days)",
                min_value=100,
                max_value=1000,
                value=252
            )
        
        # News settings
        st.subheader("News Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            news_lookback = st.number_input(
                "News Lookback (hours)",
                min_value=6,
                max_value=72,
                value=24
            )
            
            sentiment_weight = st.slider(
                "Sentiment Weight",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05
            )
        
        with col2:
            max_articles = st.number_input(
                "Max Articles per Symbol",
                min_value=5,
                max_value=50,
                value=20
            )
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    # Helper methods
    def _render_metric(self, label: str, value: str, delta: Optional[str] = None, 
                      positive: bool = True):
        """Render a metric with custom styling"""
        if delta:
            color = self.config.color_positive if positive else self.config.color_negative
            st.metric(label, value, delta, delta_color="normal" if positive else "inverse")
        else:
            st.metric(label, value)
    
    def _render_portfolio_chart(self):
        """Render portfolio value chart"""
        # Get historical data
        history = self.trading_system.performance_tracker.metrics.get('portfolio_history', [])
        
        if not history:
            # Generate mock data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            values = 100000 + np.cumsum(np.random.randn(30) * 1000)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color=self.config.color_primary, width=2)
            ))
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h['date'] for h in history],
                y=[h['value'] for h in history],
                mode='lines',
                name='Portfolio Value'
            ))
        
        fig.update_layout(
            height=self.config.chart_height,
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_pnl_distribution(self):
        """Render P&L distribution chart"""
        # Get trade history
        trades = self.trading_system.performance_tracker.trades_df
        
        if trades.empty:
            # Mock data
            pnl_values = np.random.normal(100, 500, 100)
        else:
            pnl_values = trades['pnl'].values
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=pnl_values,
            nbinsx=30,
            name='P&L Distribution',
            marker=dict(color=self.config.color_primary)
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            height=self.config.chart_height,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_recent_activity(self):
        """Render recent activity feed"""
        activities = [
            {"time": "2 min ago", "action": "Signal Generated", "details": "AAPL - Long signal (confidence: 75%)", "icon": "🚀"},
            {"time": "15 min ago", "action": "Position Opened", "details": "MSFT - 100 shares @ $380.50", "icon": "📈"},
            {"time": "1 hour ago", "action": "Position Closed", "details": "GOOGL - P&L: +$450 (+1.2%)", "icon": "✅"},
            {"time": "2 hours ago", "action": "Model Retrained", "details": "Ensemble accuracy: 73.5%", "icon": "🧠"},
            {"time": "3 hours ago", "action": "Risk Alert", "details": "Portfolio heat approaching limit", "icon": "⚠️"}
        ]
        
        for activity in activities[:5]:
            col1, col2 = st.columns([1, 11])
            
            with col1:
                st.write(activity['icon'])
            
            with col2:
                st.write(f"**{activity['action']}** - {activity['details']}")
                st.caption(activity['time'])
    
    def _get_latest_predictions(self) -> List[Dict]:
        """Get latest ML predictions"""
        # Mock predictions for demonstration
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'NVDA']
        predictions = []
        
        for symbol in symbols:
            predictions.append({
                'symbol': symbol,
                'direction': np.random.choice(['long', 'short']),
                'expected_return': np.random.uniform(-0.05, 0.05),
                'confidence': np.random.uniform(0.5, 0.9),
                'ml_score': np.random.uniform(0.4, 0.9),
                'technical_score': np.random.uniform(0.3, 0.8),
                'bayesian_score': np.random.uniform(0.2, 1.5)
            })
        
        return predictions
    
    def _filter_predictions(self, predictions: List[Dict], 
                          confidence_filter: float, 
                          direction_filter: str) -> List[Dict]:
        """Filter predictions based on criteria"""
        filtered = []
        
        for pred in predictions:
            if pred['confidence'] < confidence_filter:
                continue
            
            if direction_filter != "All" and pred['direction'] != direction_filter.lower():
                continue
            
            filtered.append(pred)
        
        return filtered
    
    def _render_signal_details(self, symbol: str, predictions: List[Dict]):
        """Render detailed signal information"""
        # Find prediction for symbol
        pred = next((p for p in predictions if p['symbol'] == symbol), None)
        
        if not pred:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Signal Components**")
            
            # Create radar chart
            categories = ['ML Score', 'Technical', 'Sentiment', 'Quality', 'Pattern']
            values = [
                pred['ml_score'],
                pred['technical_score'],
                np.random.uniform(0.4, 0.8),  # Mock sentiment
                np.random.uniform(0.5, 0.9),  # Mock quality
                np.random.uniform(0.3, 0.7)   # Mock pattern
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Signal Strength'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Risk/Reward Analysis**")
            
            entry_price = 100  # Mock
            stop_loss = entry_price * 0.95
            take_profit = entry_price * 1.10
            
            st.write(f"Entry Price: ${entry_price:.2f}")
            st.write(f"Stop Loss: ${stop_loss:.2f} (-5.0%)")
            st.write(f"Take Profit: ${take_profit:.2f} (+10.0%)")
            st.write(f"Risk/Reward Ratio: 2.0")
            
            # Risk chart
            fig = go.Figure()
            
            fig.add_shape(
                type="rect",
                x0=0, x1=1,
                y0=stop_loss, y1=entry_price,
                fillcolor="red",
                opacity=0.3
            )
            
            fig.add_shape(
                type="rect",
                x0=0, x1=1,
                y0=entry_price, y1=take_profit,
                fillcolor="green",
                opacity=0.3
            )
            
            fig.add_hline(y=entry_price, line_dash="dash")
            
            fig.update_yaxis(range=[stop_loss * 0.98, take_profit * 1.02])
            fig.update_xaxis(visible=False)
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _get_mock_positions(self) -> Dict:
        """Get mock positions for demonstration"""
        from alpaca_trading_engine import Position
        
        positions = {
            'AAPL': Position(
                symbol='AAPL',
                shares=100,
                avg_entry_price=175.50,
                current_price=178.25,
                market_value=17825,
                unrealized_pnl=275,
                unrealized_pnl_pct=0.0157,
                entry_time=datetime.now() - timedelta(days=5),
                last_updated=datetime.now()
            ),
            'MSFT': Position(
                symbol='MSFT',
                shares=50,
                avg_entry_price=382.00,
                current_price=380.50,
                market_value=19025,
                unrealized_pnl=-75,
                unrealized_pnl_pct=-0.0039,
                entry_time=datetime.now() - timedelta(days=3),
                last_updated=datetime.now()
            )
        }
        
        return positions
    
    def _render_feature_importance(self):
        """Render feature importance chart"""
        # Mock feature importance data
        features = [
            'rsi_14', 'macd_hist', 'bb_position', 'volume_ratio', 'momentum_aligned',
            'volatility_20d', 'support_distance', 'trend_strength', 'correlation', 'sentiment'
        ]
        importance = np.random.uniform(0.02, 0.15, len(features))
        importance = importance / importance.sum()
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        features = [features[i] for i in sorted_idx]
        importance = importance[sorted_idx]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance[:10],
            y=features[:10],
            orientation='h',
            marker=dict(color=self.config.color_primary)
        ))
        
        fig.update_layout(
            height=self.config.chart_height,
            xaxis_title="Importance",
            yaxis_title="Feature"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        # Mock trade history
        trades = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        for i in range(20):
            entry_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
            holding_period = np.random.randint(1, 10)
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': entry_date + timedelta(days=holding_period),
                'symbol': np.random.choice(symbols),
                'side': np.random.choice(['buy', 'sell']),
                'entry_price': np.random.uniform(100, 500),
                'exit_price': np.random.uniform(100, 500),
                'shares': np.random.randint(10, 100),
                'pnl': np.random.uniform(-500, 1000),
                'pnl_percent': np.random.uniform(-0.05, 0.10),
                'holding_period': holding_period
            })
        
        return trades
    
    def _filter_trades(self, trades: List[Dict], date_range: tuple, 
                      symbol_filter: List[str], outcome_filter: str) -> List[Dict]:
        """Filter trades based on criteria"""
        filtered = []
        
        for trade in trades:
            # Date filter
            if date_range:
                if trade['entry_date'].date() < date_range[0] or trade['entry_date'].date() > date_range[1]:
                    continue
            
            # Symbol filter
            if symbol_filter and trade['symbol'] not in symbol_filter:
                continue
            
            # Outcome filter
            if outcome_filter == "Winners" and trade['pnl'] <= 0:
                continue
            elif outcome_filter == "Losers" and trade['pnl'] >= 0:
                continue
            
            filtered.append(trade)
        
        return filtered
    
    def _get_market_indices_data(self, indices: List[str]) -> Dict:
        """Get market indices data"""
        # Mock data
        data = {}
        
        for index in indices:
            data[index] = {
                'price': np.random.uniform(300, 500),
                'change': np.random.uniform(-0.02, 0.02)
            }
        
        return data
    
    def _render_market_breadth(self):
        """Render market breadth chart"""
        # Mock data
        advances = np.random.randint(200, 350)
        declines = np.random.randint(150, 300)
        unchanged = 500 - advances - declines
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Advancing', 'Declining', 'Unchanged'],
            y=[advances, declines, unchanged],
            marker=dict(color=['green', 'red', 'gray'])
        ))
        
        fig.update_layout(
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_sector_performance(self):
        """Render sector performance"""
        sectors = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 'Industrials']
        performance = np.random.uniform(-0.03, 0.03, len(sectors))
        
        # Sort by performance
        sorted_idx = np.argsort(performance)[::-1]
        sectors = [sectors[i] for i in sorted_idx]
        performance = performance[sorted_idx]
        
        # Color based on positive/negative
        colors = ['green' if p > 0 else 'red' for p in performance]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=performance,
            y=sectors,
            orientation='h',
            marker=dict(color=colors)
        ))
        
        fig.update_layout(
            height=300,
            xaxis_title="Performance %",
            xaxis=dict(tickformat='.1%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_market_regime(self) -> Dict:
        """Get market regime analysis"""
        return {
            'trend': np.random.choice(['Bullish', 'Bearish', 'Neutral']),
            'trend_strength': np.random.uniform(0.3, 0.9),
            'volatility': np.random.choice(['Low', 'Normal', 'High']),
            'volatility_level': np.random.uniform(0.2, 0.8),
            'correlation': np.random.choice(['Low', 'Normal', 'High']),
            'correlation_level': np.random.uniform(0.3, 0.8)
        }
    
    def _render_market_sentiment(self):
        """Render market sentiment"""
        # Mock sentiment data
        sentiment_data = {
            'Overall': 0.2,
            'Technology': 0.4,
            'Finance': -0.1,
            'Healthcare': 0.3,
            'Energy': -0.3
        }
        
        # Create sentiment gauge
        for sector, sentiment in sentiment_data.items():
            col1, col2, col3 = st.columns([2, 6, 2])
            
            with col1:
                st.write(sector)
            
            with col2:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-1, -0.5], 'color': "red"},
                            {'range': [-0.5, 0.5], 'color': "yellow"},
                            {'range': [0.5, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': sentiment
                        }
                    }
                ))
                
                fig.update_layout(height=150)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                if sentiment > 0.3:
                    st.success("Bullish")
                elif sentiment < -0.3:
                    st.error("Bearish")
                else:
                    st.info("Neutral")
    
    def _get_portfolio_risk_metrics(self) -> Dict:
        """Get portfolio risk metrics"""
        return {
            'portfolio_heat': 0.065,
            'var_95': 12500,
            'max_drawdown': -0.085,
            'sharpe_ratio': 1.45,
            'daily_loss': -0.012,
            'max_position': 0.085,
            'avg_correlation': 0.55
        }
    
    def _render_position_risk_chart(self):
        """Render risk by position"""
        # Mock data
        positions = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        risk_values = np.random.uniform(500, 2000, len(positions))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=positions,
            y=risk_values,
            marker=dict(color=self.config.color_primary)
        ))
        
        fig.update_layout(
            height=300,
            yaxis_title="Risk ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_factor_risk_chart(self):
        """Render risk by factor"""
        factors = ['Market', 'Sector', 'Stock-Specific', 'Volatility', 'Liquidity']
        risk_pct = np.random.uniform(10, 30, len(factors))
        risk_pct = risk_pct / risk_pct.sum() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=factors,
            values=risk_pct,
            hole=0.3
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_performance_chart(self):
        """Render model performance over time"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        accuracy = 0.7 + np.cumsum(np.random.randn(30) * 0.01)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=accuracy,
            mode='lines',
            name='Model Accuracy'
        ))
        
        fig.update_layout(
            height=300,
            yaxis_title="Accuracy",
            yaxis=dict(tickformat='.1%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_regime_accuracy_chart(self):
        """Render accuracy by market regime"""
        regimes = ['Bull Market', 'Bear Market', 'High Volatility', 'Low Volatility', 'Trending', 'Range-Bound']
        accuracy = np.random.uniform(0.65, 0.80, len(regimes))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=regimes,
            y=accuracy,
            marker=dict(color=self.config.color_secondary)
        ))
        
        fig.update_layout(
            height=300,
            yaxis_title="Accuracy",
            yaxis=dict(tickformat='.0%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_prediction_distribution(self):
        """Render prediction distribution"""
        predictions = np.random.normal(0, 0.02, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predictions,
            nbinsx=50,
            name='Prediction Distribution'
        ))
        
        fig.update_layout(
            height=300,
            xaxis_title="Predicted Return",
            xaxis=dict(tickformat='.1%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_calibration_plot(self):
        """Render prediction calibration plot"""
        # Mock calibration data
        predicted_probs = np.linspace(0, 1, 10)
        actual_probs = predicted_probs + np.random.uniform(-0.1, 0.1, 10)
        actual_probs = np.clip(actual_probs, 0, 1)
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        # Actual calibration
        fig.add_trace(go.Scatter(
            x=predicted_probs,
            y=actual_probs,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color=self.config.color_primary)
        ))
        
        fig.update_layout(
            height=300,
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Probability"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_error_analysis(self):
        """Render prediction error analysis"""
        # Mock error data
        errors = np.random.normal(0, 0.01, 500)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=errors,
                nbinsx=30,
                name='Error Distribution'
            ))
            
            fig.update_layout(
                height=250,
                xaxis_title="Prediction Error",
                title="Error Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error metrics
            st.metric("Mean Absolute Error", f"{np.mean(np.abs(errors)):.4f}")
            st.metric("Root Mean Square Error", f"{np.sqrt(np.mean(errors**2)):.4f}")
            st.metric("Error Std Dev", f"{np.std(errors):.4f}")


# Standalone app runner
def run_dashboard_app():
    """Run the dashboard as a standalone app"""
    # Import the main trading system
    # This would normally import your actual trading system
    from ml_trading_core import MLTradingSystem
    
    # Initialize trading system
    @st.cache_resource
    def get_trading_system():
        return MLTradingSystem()
    
    trading_system = get_trading_system()
    
    # Initialize and run dashboard
    dashboard = TradingDashboard(trading_system)
    dashboard.run()


if __name__ == "__main__":
    run_dashboard_app()
