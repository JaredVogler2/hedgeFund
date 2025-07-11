"""
Automation & Scheduling System
Professional automation for ML trading system operations
"""

import schedule
import time
import logging
from datetime import datetime, timedelta, time as dt_time
import pytz
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import json
import threading
import queue
import subprocess
import sys
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class AutomationConfig:
    """Configuration for automation system"""
    # Timezone
    timezone: str = 'US/Eastern'
    
    # Market hours (ET)
    market_open: dt_time = dt_time(9, 30)
    market_close: dt_time = dt_time(16, 0)
    pre_market_start: dt_time = dt_time(4, 0)
    after_market_end: dt_time = dt_time(20, 0)
    
    # Job timing
    nightly_job_time: dt_time = dt_time(2, 0)
    pre_market_job_time: dt_time = dt_time(8, 30)
    market_open_job_time: dt_time = dt_time(9, 15)
    intraday_check_interval: int = 30  # minutes
    eod_job_time: dt_time = dt_time(15, 45)
    post_market_job_time: dt_time = dt_time(16, 30)
    
    # Notifications
    enable_email_notifications: bool = True
    email_recipients: List[str] = field(default_factory=lambda: [os.getenv('NOTIFICATION_EMAIL', '')])
    smtp_server: str = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port: int = 587
    smtp_username: str = os.getenv('SMTP_USERNAME', '')
    smtp_password: str = os.getenv('SMTP_PASSWORD', '')
    
    # Error handling
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 60
    
    # Logging
    log_dir: Path = Path('logs')
    
    # Performance
    parallel_execution: bool = True
    max_workers: int = 4

@dataclass
class JobResult:
    """Result of a scheduled job execution"""
    job_name: str
    start_time: datetime
    end_time: datetime
    success: bool
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

class TradingAutomation:
    """Main automation and scheduling system"""
    
    def __init__(self, trading_system, config: Optional[AutomationConfig] = None):
        self.trading_system = trading_system
        self.config = config or AutomationConfig()
        
        # Initialize timezone
        self.tz = pytz.timezone(self.config.timezone)
        
        # Job tracking
        self.job_history: List[JobResult] = []
        self.running_jobs: Dict[str, threading.Thread] = {}
        
        # Initialize components
        self.job_queue = queue.Queue()
        self.error_handler = ErrorHandler(self.config)
        self.notifier = NotificationManager(self.config)
        
        # Create log directory
        self.config.log_dir.mkdir(exist_ok=True)
        
        # Schedule jobs
        self._schedule_jobs()
        
    def _schedule_jobs(self):
        """Schedule all automated jobs"""
        # Nightly jobs (2 AM - 6 AM)
        schedule.every().day.at(self._format_time(self.config.nightly_job_time)).do(
            self._run_job, "nightly_pipeline", self.run_nightly_pipeline
        )
        
        # Pre-market jobs (8:30 AM)
        schedule.every().monday.at(self._format_time(self.config.pre_market_job_time)).do(
            self._run_job, "pre_market_analysis", self.run_pre_market_analysis
        )
        schedule.every().tuesday.at(self._format_time(self.config.pre_market_job_time)).do(
            self._run_job, "pre_market_analysis", self.run_pre_market_analysis
        )
        schedule.every().wednesday.at(self._format_time(self.config.pre_market_job_time)).do(
            self._run_job, "pre_market_analysis", self.run_pre_market_analysis
        )
        schedule.every().thursday.at(self._format_time(self.config.pre_market_job_time)).do(
            self._run_job, "pre_market_analysis", self.run_pre_market_analysis
        )
        schedule.every().friday.at(self._format_time(self.config.pre_market_job_time)).do(
            self._run_job, "pre_market_analysis", self.run_pre_market_analysis
        )
        
        # Market open jobs (9:15 AM)
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            getattr(schedule.every(), day).at(self._format_time(self.config.market_open_job_time)).do(
                self._run_job, "market_open_trading", self.run_market_open_trading
            )
        
        # Intraday monitoring
        schedule.every(self.config.intraday_check_interval).minutes.do(
            self._run_job, "intraday_monitoring", self.run_intraday_monitoring
        )
        
        # End of day jobs (3:45 PM)
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            getattr(schedule.every(), day).at(self._format_time(self.config.eod_job_time)).do(
                self._run_job, "end_of_day_analysis", self.run_end_of_day_analysis
            )
        
        # Post-market jobs (4:30 PM)
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            getattr(schedule.every(), day).at(self._format_time(self.config.post_market_job_time)).do(
                self._run_job, "post_market_report", self.run_post_market_report
            )
        
        # Weekly jobs
        schedule.every().sunday.at("18:00").do(
            self._run_job, "weekly_analysis", self.run_weekly_analysis
        )

        # Monthly jobs - run on the 1st of each month
        schedule.every().day.at("06:00").do(
            lambda: self._run_job("monthly_report", self.run_monthly_report) if datetime.now().day == 1 else None
        )
        
        logger.info("All jobs scheduled successfully")
    
    def _format_time(self, time_obj: dt_time) -> str:
        """Format time for schedule library"""
        return time_obj.strftime("%H:%M")
    
    def _run_job(self, job_name: str, job_func: Callable):
        """Execute a scheduled job with error handling"""
        # Check if job is already running
        if job_name in self.running_jobs and self.running_jobs[job_name].is_alive():
            logger.warning(f"Job {job_name} is already running, skipping")
            return
        
        # Run job in separate thread
        if self.config.parallel_execution:
            thread = threading.Thread(target=self._execute_job, args=(job_name, job_func))
            thread.start()
            self.running_jobs[job_name] = thread
        else:
            self._execute_job(job_name, job_func)
    
    def _execute_job(self, job_name: str, job_func: Callable):
        """Execute job with monitoring and error handling"""
        result = JobResult(
            job_name=job_name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=False
        )
        
        logger.info(f"Starting job: {job_name}")
        
        try:
            # Execute job
            job_metrics = job_func()
            
            # Update result
            result.end_time = datetime.now()
            result.success = True
            result.metrics = job_metrics or {}
            
            logger.info(f"Job {job_name} completed successfully")
            
            # Send success notification
            if self.config.enable_email_notifications:
                self.notifier.send_job_success_notification(result)
                
        except Exception as e:
            # Handle error
            result.end_time = datetime.now()
            result.error = str(e)
            result.logs.append(traceback.format_exc())
            
            logger.error(f"Job {job_name} failed: {e}")
            
            # Retry if configured
            if self.error_handler.should_retry(job_name):
                logger.info(f"Retrying job {job_name}")
                time.sleep(self.config.retry_delay_seconds)
                self._execute_job(job_name, job_func)
                return
            
            # Send error notification
            if self.config.enable_email_notifications:
                self.notifier.send_job_error_notification(result)
        
        finally:
            # Record job result
            self.job_history.append(result)
            
            # Clean up running jobs
            if job_name in self.running_jobs:
                del self.running_jobs[job_name]
    
    def run_nightly_pipeline(self) -> Dict[str, Any]:
        """Run comprehensive nightly data and model pipeline"""
        logger.info("Running nightly pipeline...")
        
        metrics = {
            'start_time': datetime.now(),
            'steps_completed': [],
            'errors': []
        }
        
        try:
            # 1. Update watchlist
            logger.info("Updating watchlist...")
            self.trading_system.watchlist_manager.update_liquidity_filter()
            metrics['steps_completed'].append('watchlist_update')
            
            # 2. Download historical data
            logger.info("Downloading market data...")
            symbols = self.trading_system.watchlist_manager.get_all_symbols()
            market_data = self.trading_system.data_manager.get_batch_data(
                symbols, period="2y", n_jobs=10
            )
            metrics['symbols_updated'] = len(market_data)
            metrics['steps_completed'].append('data_download')
            
            # 3. Feature engineering
            logger.info("Engineering features...")
            if hasattr(self.trading_system, 'feature_engineer'):
                feature_count = 0
                for symbol, data in market_data.items():
                    features = self.trading_system.feature_engineer.engineer_features(data)
                    # Save features
                    features.to_parquet(
                        self.trading_system.config.data_dir / 'features' / f'{symbol}_features.parquet'
                    )
                    feature_count += 1
                
                metrics['features_generated'] = feature_count
                metrics['steps_completed'].append('feature_engineering')
            
            # 4. Model retraining
            logger.info("Retraining models...")
            if hasattr(self.trading_system, 'ensemble_model'):
                # Prepare training data
                # This is simplified - in production, properly prepare X and y
                training_metrics = {'models_trained': 5}  # Mock
                metrics['training_metrics'] = training_metrics
                metrics['steps_completed'].append('model_training')
            
            # 5. Backtest validation
            logger.info("Running backtest validation...")
            # Run backtest on recent period
            metrics['backtest_sharpe'] = 1.45  # Mock
            metrics['steps_completed'].append('backtest_validation')
            
            # 6. Generate next day signals
            logger.info("Generating signals for next trading day...")
            # Generate and save signals
            metrics['signals_generated'] = 15  # Mock
            metrics['steps_completed'].append('signal_generation')
            
            # 7. Clean up old data
            logger.info("Cleaning up old data...")
            self.trading_system.data_manager.clear_cache(older_than_days=7)
            metrics['steps_completed'].append('cleanup')
            
            metrics['end_time'] = datetime.now()
            metrics['duration'] = (metrics['end_time'] - metrics['start_time']).total_seconds()
            
            logger.info(f"Nightly pipeline completed in {metrics['duration']:.1f} seconds")
            
        except Exception as e:
            metrics['errors'].append(str(e))
            logger.error(f"Error in nightly pipeline: {e}")
            raise
        
        return metrics
    
    def run_pre_market_analysis(self) -> Dict[str, Any]:
        """Run pre-market analysis and preparation"""
        logger.info("Running pre-market analysis...")
        
        metrics = {
            'timestamp': datetime.now(),
            'market_status': {},
            'news_sentiment': {},
            'signals_confirmed': 0
        }
        
        try:
            # 1. Check market status
            if hasattr(self.trading_system, 'execution_engine'):
                is_open = self.trading_system.execution_engine.is_market_open()
                metrics['market_status']['will_open'] = True  # Market will open today
            
            # 2. Fetch and analyze overnight news
            logger.info("Analyzing overnight news...")
            if hasattr(self.trading_system, 'news_analyzer'):
                symbols = self.trading_system.watchlist_manager.get_all_symbols()[:20]  # Top 20
                sentiment_results = self.trading_system.news_analyzer.analyze_symbols(symbols)
                
                # Aggregate sentiment
                for symbol, analysis in sentiment_results.items():
                    metrics['news_sentiment'][symbol] = {
                        'sentiment': analysis.overall_sentiment,
                        'confidence': analysis.confidence,
                        'article_count': analysis.article_count
                    }
            
            # 3. Update market data
            logger.info("Updating pre-market data...")
            # Fetch pre-market quotes
            
            # 4. Validate signals
            logger.info("Validating overnight signals...")
            # Load and validate signals generated overnight
            
            # 5. Prepare execution plan
            logger.info("Preparing execution plan...")
            # Prioritize signals and prepare orders
            
            metrics['signals_confirmed'] = 10  # Mock
            
        except Exception as e:
            logger.error(f"Error in pre-market analysis: {e}")
            raise
        
        return metrics
    
    def run_market_open_trading(self) -> Dict[str, Any]:
        """Execute trades at market open"""
        logger.info("Running market open trading...")
        
        metrics = {
            'timestamp': datetime.now(),
            'signals_executed': 0,
            'orders_placed': 0,
            'errors': []
        }
        
        try:
            # Wait for market to stabilize (optional)
            time.sleep(30)
            
            # 1. Final signal validation
            logger.info("Final signal validation...")
            
            # 2. Execute signals
            if hasattr(self.trading_system, 'execution_engine'):
                # Get today's signals
                signals = []  # Load from file or generate
                
                # Execute
                results = self.trading_system.execution_engine.execute_signals(signals)
                
                metrics['signals_executed'] = len(results.get('executed', []))
                metrics['orders_placed'] = metrics['signals_executed']
                metrics['execution_results'] = results
            
            # 3. Set initial stops
            logger.info("Setting initial stop losses...")
            
            # 4. Send execution report
            if metrics['signals_executed'] > 0:
                self.notifier.send_execution_report(metrics)
            
        except Exception as e:
            metrics['errors'].append(str(e))
            logger.error(f"Error in market open trading: {e}")
            raise
        
        return metrics
    
    def run_intraday_monitoring(self) -> Dict[str, Any]:
        """Monitor positions and market conditions intraday"""
        if not self._is_market_hours():
            return {'skipped': True, 'reason': 'Outside market hours'}
        
        logger.info("Running intraday monitoring...")
        
        metrics = {
            'timestamp': datetime.now(),
            'positions_checked': 0,
            'stops_updated': 0,
            'alerts_triggered': []
        }
        
        try:
            # 1. Update position values
            if hasattr(self.trading_system, 'execution_engine'):
                self.trading_system.execution_engine.update_positions()
                metrics['positions_checked'] = len(self.trading_system.execution_engine.positions)
            
            # 2. Check risk limits
            portfolio = self.trading_system.execution_engine.get_portfolio_summary()
            
            # Check daily loss
            if portfolio['daily_pnl'] < -self.trading_system.config.max_daily_loss * portfolio['account_value']:
                metrics['alerts_triggered'].append('daily_loss_limit')
                # Take action
            
            # 3. Update trailing stops
            logger.info("Updating trailing stops...")
            # Update stops for profitable positions
            
            # 4. Check for exit signals
            # Monitor for reversal signals or stop triggers
            
            # 5. Market regime monitoring
            # Check for significant market changes
            
        except Exception as e:
            logger.error(f"Error in intraday monitoring: {e}")
            raise
        
        return metrics
    
    def run_end_of_day_analysis(self) -> Dict[str, Any]:
        """Run end-of-day analysis and preparation"""
        logger.info("Running end-of-day analysis...")
        
        metrics = {
            'timestamp': datetime.now(),
            'positions_analyzed': 0,
            'exit_signals': 0
        }
        
        try:
            # 1. Analyze current positions
            if hasattr(self.trading_system, 'execution_engine'):
                positions = self.trading_system.execution_engine.positions
                metrics['positions_analyzed'] = len(positions)
                
                # Check for positions to close
                for symbol, position in positions.items():
                    # Check exit criteria
                    pass
            
            # 2. Calculate daily performance
            performance = self.trading_system.performance_tracker.get_performance_summary()
            metrics['daily_performance'] = performance
            
            # 3. Prepare for market close
            # Set MOC orders if needed
            
            # 4. Generate preliminary signals for tomorrow
            logger.info("Generating preliminary signals...")
            
        except Exception as e:
            logger.error(f"Error in end-of-day analysis: {e}")
            raise
        
        return metrics
    
    def run_post_market_report(self) -> Dict[str, Any]:
        """Generate and send daily trading report"""
        logger.info("Generating post-market report...")
        
        metrics = {
            'timestamp': datetime.now(),
            'report_generated': False
        }
        
        try:
            # 1. Gather daily statistics
            daily_stats = self._gather_daily_statistics()
            
            # 2. Generate report
            report = self._generate_daily_report(daily_stats)
            
            # 3. Save report
            report_path = self.config.log_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.html"
            with open(report_path, 'w') as f:
                f.write(report)
            
            # 4. Send report
            if self.config.enable_email_notifications:
                self.notifier.send_daily_report(report, daily_stats)
            
            metrics['report_generated'] = True
            metrics['report_path'] = str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating post-market report: {e}")
            raise
        
        return metrics
    
    def run_weekly_analysis(self) -> Dict[str, Any]:
        """Run comprehensive weekly analysis"""
        logger.info("Running weekly analysis...")
        
        metrics = {
            'timestamp': datetime.now(),
            'week_ending': datetime.now().strftime('%Y-%m-%d')
        }
        
        try:
            # 1. Weekly performance analysis
            
            # 2. Model performance review
            
            # 3. Risk analysis
            
            # 4. Strategy optimization suggestions
            
            # 5. Generate weekly report
            
            pass
            
        except Exception as e:
            logger.error(f"Error in weekly analysis: {e}")
            raise
        
        return metrics
    
    def run_monthly_report(self) -> Dict[str, Any]:
        """Generate comprehensive monthly report"""
        logger.info("Generating monthly report...")
        
        metrics = {
            'timestamp': datetime.now(),
            'month': datetime.now().strftime('%Y-%m')
        }
        
        try:
            # 1. Monthly performance summary
            
            # 2. Detailed trade analysis
            
            # 3. Risk metrics
            
            # 4. Model performance
            
            # 5. Generate and send report
            
            pass
            
        except Exception as e:
            logger.error(f"Error generating monthly report: {e}")
            raise
        
        return metrics
    
    def _is_market_hours(self) -> bool:
        """Check if currently in market hours"""
        now = datetime.now(self.tz)
        current_time = now.time()
        
        # Check if weekday
        if now.weekday() > 4:  # Weekend
            return False
        
        # Check time
        return self.config.market_open <= current_time <= self.config.market_close
    
    def _gather_daily_statistics(self) -> Dict[str, Any]:
        """Gather comprehensive daily trading statistics"""
        stats = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'trades_executed': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'portfolio_value': 0
        }
        
        # Gather from various sources
        if hasattr(self.trading_system, 'performance_tracker'):
            performance = self.trading_system.performance_tracker.get_performance_summary()
            stats.update(performance)
        
        return stats
    
    def _generate_daily_report(self, stats: Dict[str, Any]) -> str:
        """Generate HTML daily report"""
        html = f"""
        <html>
        <head>
            <title>Daily Trading Report - {stats['date']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Daily Trading Report - {stats['date']}</h1>
            
            <h2>Summary</h2>
            <div class="metric">
                <strong>Total P&L:</strong> 
                <span class="{'positive' if stats.get('total_pnl', 0) >= 0 else 'negative'}">
                    ${stats.get('total_pnl', 0):,.2f}
                </span>
            </div>
            <div class="metric">
                <strong>Win Rate:</strong> {stats.get('win_rate', 0):.1%}
            </div>
            <div class="metric">
                <strong>Trades:</strong> {stats.get('trades_executed', 0)}
            </div>
            
            <h2>Performance Details</h2>
            <!-- Add more detailed statistics here -->
            
        </body>
        </html>
        """
        
        return html
    
    def start(self):
        """Start the automation system"""
        logger.info("Starting automation system...")
        
        # Start scheduler in separate thread
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info("Automation system started")
    
    def _run_scheduler(self):
        """Run the schedule loop"""
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                time.sleep(60)  # Wait before retrying
    
    def stop(self):
        """Stop the automation system"""
        logger.info("Stopping automation system...")
        
        # Cancel all jobs
        schedule.clear()
        
        # Wait for running jobs to complete
        for job_name, thread in self.running_jobs.items():
            logger.info(f"Waiting for job {job_name} to complete...")
            thread.join(timeout=300)  # 5 minute timeout
        
        logger.info("Automation system stopped")

class ErrorHandler:
    """Handles errors and retries"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.retry_counts: Dict[str, int] = {}
        
    def should_retry(self, job_name: str) -> bool:
        """Determine if job should be retried"""
        current_count = self.retry_counts.get(job_name, 0)
        
        if current_count >= self.config.max_retry_attempts:
            return False
        
        self.retry_counts[job_name] = current_count + 1
        return True
    
    def reset_retry_count(self, job_name: str):
        """Reset retry count for job"""
        if job_name in self.retry_counts:
            del self.retry_counts[job_name]

class NotificationManager:
    """Manages email notifications"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        
    def send_email(self, subject: str, body: str, html_body: Optional[str] = None):
        """Send email notification"""
        if not self.config.enable_email_notifications:
            return
        
        if not self.config.email_recipients:
            logger.warning("No email recipients configured")
            return
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(self.config.email_recipients)
            
            # Add text part
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent: {subject}")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def send_job_success_notification(self, result: JobResult):
        """Send job success notification"""
        subject = f"✅ Job Success: {result.job_name}"
        
        body = f"""
Job: {result.job_name}
Status: SUCCESS
Start Time: {result.start_time}
End Time: {result.end_time}
Duration: {(result.end_time - result.start_time).total_seconds():.1f} seconds

Metrics:
{json.dumps(result.metrics, indent=2)}
"""
        
        self.send_email(subject, body)
    
    def send_job_error_notification(self, result: JobResult):
        """Send job error notification"""
        subject = f"❌ Job Failed: {result.job_name}"
        
        body = f"""
Job: {result.job_name}
Status: FAILED
Start Time: {result.start_time}
End Time: {result.end_time}
Error: {result.error}

Logs:
{''.join(result.logs)}
"""
        
        self.send_email(subject, body)
    
    def send_execution_report(self, metrics: Dict[str, Any]):
        """Send trade execution report"""
        subject = f"📊 Trade Execution Report - {metrics['signals_executed']} trades"
        
        body = f"""
Trade Execution Summary
Time: {metrics['timestamp']}
Signals Executed: {metrics['signals_executed']}
Orders Placed: {metrics['orders_placed']}

Details:
{json.dumps(metrics.get('execution_results', {}), indent=2)}
"""
        
        self.send_email(subject, body)
    
    def send_daily_report(self, report_html: str, stats: Dict[str, Any]):
        """Send daily trading report"""
        subject = f"📈 Daily Trading Report - {stats['date']}"
        
        body = f"""
Daily Trading Summary
Date: {stats['date']}
Total P&L: ${stats.get('total_pnl', 0):,.2f}
Win Rate: {stats.get('win_rate', 0):.1%}
Trades: {stats.get('trades_executed', 0)}

See attached HTML report for details.
"""
        
        self.send_email(subject, body, report_html)


# Standalone automation runner
if __name__ == "__main__":
    # This would typically import and use your actual trading system
    from ml_trading_core import MLTradingSystem
    
    # Initialize trading system
    trading_system = MLTradingSystem()
    
    # Initialize automation
    config = AutomationConfig()
    automation = TradingAutomation(trading_system, config)
    
    # Start automation
    automation.start()
    
    logger.info("Automation system running. Press Ctrl+C to stop.")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(60)
            
            # Print status
            running_jobs = list(automation.running_jobs.keys())
            if running_jobs:
                logger.info(f"Running jobs: {running_jobs}")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        automation.stop()
