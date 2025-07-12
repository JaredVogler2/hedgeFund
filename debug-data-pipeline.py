"""
Debug script to find where your data is being lost
Run this to diagnose the data pipeline issue
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set dates
CURRENT_DATE = pd.Timestamp('2025-07-12')
TRAINING_START = pd.Timestamp('2020-07-12')


def debug_data_pipeline():
    """Step-by-step debugging of the data pipeline"""
    print("=" * 80)
    print("DATA PIPELINE DEBUGGING")
    print("=" * 80)
    
    # Step 1: Check raw data download
    print("\nSTEP 1: Checking raw data download...")
    print("-" * 40)
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=TRAINING_START, end=CURRENT_DATE)
        
        print(f"\n{symbol}:")
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        print(f"  Number of days: {len(data)}")
        print(f"  Expected days: ~{5 * 252} (5 years × 252 trading days)")
        print(f"  Data shape: {data.shape}")
        
        if len(data) < 1000:
            print(f"  ⚠️ WARNING: Much less data than expected!")
    
    # Step 2: Check feature engineering
    print("\n\nSTEP 2: Checking feature engineering...")
    print("-" * 40)
    
    from feature_engineering import EnhancedFeatureEngineer, FeatureConfig
    
    # Get data for one symbol
    symbol = 'AAPL'
    data = yf.download(symbol, start=TRAINING_START, end=CURRENT_DATE, progress=False)
    print(f"\nRaw data shape for {symbol}: {data.shape}")
    
    # Generate features
    feature_engineer = EnhancedFeatureEngineer(FeatureConfig())
    features = feature_engineer.engineer_features(data)
    
    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {len(features.columns)}")
    print(f"Non-NaN rows: {features.dropna().shape[0]}")
    
    # Step 3: Check data preparation for ML
    print("\n\nSTEP 3: Checking ML data preparation...")
    print("-" * 40)
    
    # Simulate what happens in prepare_training_data
    prediction_horizon = 5
    
    # Calculate forward returns
    forward_returns = data['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
    
    print(f"Original data length: {len(data)}")
    print(f"After removing last {prediction_horizon} days: {len(data) - prediction_horizon}")
    
    # Check alignment
    features_clean = features.iloc[:-prediction_horizon]
    returns_clean = forward_returns.iloc[:-prediction_horizon]
    
    # Remove NaN
    mask = ~(returns_clean.isna() | features_clean.isna().any(axis=1))
    
    print(f"After removing NaN: {mask.sum()} samples")
    print(f"Percentage kept: {mask.sum() / len(mask) * 100:.1f}%")
    
    # Step 4: Check multiple symbols combined
    print("\n\nSTEP 4: Checking multiple symbols combined...")
    print("-" * 40)
    
    all_samples = 0
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=TRAINING_START, end=CURRENT_DATE, progress=False)
            features = feature_engineer.engineer_features(data)
            
            # Calculate valid samples after cleaning
            forward_returns = data['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            features_clean = features.iloc[:-prediction_horizon]
            returns_clean = forward_returns.iloc[:-prediction_horizon]
            mask = ~(returns_clean.isna() | features_clean.isna().any(axis=1))
            
            valid_samples = mask.sum()
            all_samples += valid_samples
            
            print(f"{symbol}: {valid_samples} valid samples")
            
        except Exception as e:
            print(f"{symbol}: ERROR - {e}")
    
    print(f"\nTotal samples across {len(symbols)} symbols: {all_samples}")
    print(f"Average per symbol: {all_samples / len(symbols):.0f}")
    
    # Step 5: Check cross-validation splits
    print("\n\nSTEP 5: Checking cross-validation splits...")
    print("-" * 40)
    
    from purged_cross_validation import PurgedTimeSeriesSplit
    
    n_samples = all_samples
    pcv = PurgedTimeSeriesSplit(n_splits=5, purge_days=5, embargo_days=2)
    
    # Simulate CV splits
    X_dummy = np.zeros((n_samples, 10))
    
    for i, (train_idx, val_idx) in enumerate(pcv.split(X_dummy)):
        print(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        if len(train_idx) < 100:
            print(f"  ⚠️ WARNING: Very small training set!")
    
    # Step 6: Common issues
    print("\n\nSTEP 6: Common Issues Found...")
    print("-" * 40)
    
    print("\nPossible issues:")
    print("1. Too many NaN values after feature engineering")
    print("   - Technical indicators need warm-up period")
    print("   - Check if first ~50-100 rows are all NaN")
    print("\n2. Data filtering too aggressive")
    print("   - Check min_volume_filter in your code")
    print("   - Check date range filters")
    print("\n3. Cross-validation splits reducing data")
    print("   - With 5 splits and purging, each fold gets much less data")
    print("\n4. Feature calculation errors")
    print("   - Some features might be creating NaN for entire columns")


def check_actual_training_data(X, y, feature_names=None):
    """
    Add this function to your training code to debug
    Call it right before model.train()
    """
    print("\n" + "=" * 60)
    print("ACTUAL TRAINING DATA CHECK")
    print("=" * 60)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X size in MB: {X.nbytes / 1024 / 1024:.2f}")
    
    # Check for NaN/Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"\nNaN values in X: {nan_count}")
    print(f"Inf values in X: {inf_count}")
    
    # Check variance
    if X.shape[0] > 0:
        feature_vars = np.var(X, axis=0)
        zero_var_features = np.sum(feature_vars < 1e-10)
        print(f"Zero variance features: {zero_var_features}")
        
        if feature_names and zero_var_features > 0:
            zero_var_indices = np.where(feature_vars < 1e-10)[0]
            print("Zero variance feature names:")
            for idx in zero_var_indices[:5]:  # Show first 5
                print(f"  - {feature_names[idx]}")
    
    # Check target distribution
    print(f"\nTarget (y) statistics:")
    print(f"  Mean: {np.mean(y):.6f}")
    print(f"  Std: {np.std(y):.6f}")
    print(f"  Min: {np.min(y):.6f}")
    print(f"  Max: {np.max(y):.6f}")
    print(f"  % zeros: {(np.abs(y) < 1e-10).sum() / len(y) * 100:.1f}%")
    
    # Show sample of data
    print(f"\nFirst 5 samples of X:")
    print(X[:5, :5])  # First 5 rows, first 5 columns
    
    print(f"\nFirst 10 target values:")
    print(y[:10])
    
    return X.shape[0]  # Return number of samples


def fix_data_pipeline(market_data, feature_data):
    """
    Fix common issues in the data pipeline
    """
    print("\nAttempting to fix data pipeline issues...")
    
    fixed_data = {}
    
    for symbol, features in feature_data.items():
        if symbol not in market_data:
            continue
            
        # Get price data
        prices = market_data[symbol]['Close']
        
        # Remove rows where most features are NaN
        nan_threshold = 0.5  # If more than 50% features are NaN, drop row
        nan_mask = features.isna().sum(axis=1) / features.shape[1] < nan_threshold
        
        # Keep only rows with valid data
        features_clean = features[nan_mask]
        
        # Also need to align prices
        prices_clean = prices[features_clean.index]
        
        print(f"{symbol}: {len(features)} -> {len(features_clean)} rows after cleaning")
        
        if len(features_clean) > 100:  # Only keep if we have enough data
            fixed_data[symbol] = {
                'features': features_clean,
                'prices': prices_clean
            }
    
    return fixed_data


# Add this to your main_trading_system.py in the _prepare_training_data method
def debug_prepare_training_data(self):
    """Debug version of prepare_training_data"""
    logger.info("\nDEBUG: Preparing training data...")
    
    all_X = []
    all_y = []
    all_dates = []
    
    prediction_horizon = 5
    
    # Debug counters
    total_features = 0
    total_after_returns = 0
    total_after_nan = 0
    
    for symbol in self.feature_data:
        features = self.feature_data[symbol]
        prices = self.market_data[symbol]['Close']
        
        # Debug: Original size
        original_size = len(features)
        total_features += original_size
        
        # Align
        common_dates = features.index.intersection(prices.index)
        features_aligned = features.loc[common_dates]
        prices_aligned = prices.loc[common_dates]
        
        # Calculate forward returns
        forward_returns = prices_aligned.pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Remove last 5 days
        features_clean = features_aligned.iloc[:-prediction_horizon]
        returns_clean = forward_returns.iloc[:-prediction_horizon]
        dates_clean = common_dates[:-prediction_horizon]
        
        after_returns_size = len(features_clean)
        total_after_returns += after_returns_size
        
        # Remove NaN
        mask = ~(returns_clean.isna() | features_clean.isna().any(axis=1))
        
        final_size = mask.sum()
        total_after_nan += final_size
        
        logger.info(f"{symbol}: {original_size} -> {after_returns_size} -> {final_size} samples")
        
        if mask.sum() > 0:
            all_X.append(features_clean[mask].values)
            all_y.append(returns_clean[mask].values)
            all_dates.append(dates_clean[mask])
    
    logger.info(f"\nData reduction summary:")
    logger.info(f"  Total original: {total_features}")
    logger.info(f"  After removing future: {total_after_returns}")
    logger.info(f"  After removing NaN: {total_after_nan}")
    logger.info(f"  Percentage kept: {total_after_nan / total_features * 100:.1f}%")
    
    if all_X:
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        dates = np.hstack(all_dates)
        
        logger.info(f"\nFinal training data: {X.shape}")
        
        # Check the actual data
        check_actual_training_data(X, y, self.feature_engineer.get_feature_names())
    else:
        logger.error("No valid training data!")
        X, y, dates = np.array([]), np.array([]), np.array([])
    
    return X, y, dates


if __name__ == "__main__":
    # Run the debugging
    debug_data_pipeline()
    
    # If you want to test with your actual data loading code:
    print("\n\nTo debug your specific case, add this to your code:")
    print("1. In prepare_training_data(), add logging for each step")
    print("2. Before training, call: check_actual_training_data(X, y)")
    print("3. Check if features have too many NaN values")
    print("4. Verify your date filtering isn't too restrictive")
