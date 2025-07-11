"""
Purged Cross-Validation for Time Series
Prevents data leakage in financial ML models
"""

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from typing import Iterator, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PurgedTimeSeriesSplit(_BaseKFold):
    """
    Time series cross-validator with purging and embargo

    Prevents data leakage by:
    1. Using expanding windows (train always starts from beginning)
    2. Adding gap (purge) between train and validation to prevent leakage
    3. Optionally embargoing samples after validation
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 5,
                 embargo_days: int = 0, samples_per_day: int = 1):
        """
        Parameters:
        -----------
        n_splits : int
            Number of splits
        purge_days : int
            Number of days to skip between train and validation
            Should match your prediction horizon (5 for 5-day returns)
        embargo_days : int
            Days to skip after validation set (for avoiding leakage in next fold)
        samples_per_day : int
            If you have multiple samples per day (e.g., multiple stocks)
        """
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.samples_per_day = samples_per_day

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and validation set."""
        n_samples = len(X)
        n_splits = self.n_splits

        # Calculate samples to purge/embargo
        purge_samples = self.purge_days * self.samples_per_day
        embargo_samples = self.embargo_days * self.samples_per_day

        # Calculate test size for each fold
        # Account for gaps between folds
        total_gaps = purge_samples * n_splits + embargo_samples * (n_splits - 1)
        available_samples = n_samples - total_gaps
        test_size = available_samples // (n_splits + 1)

        if test_size <= 0:
            raise ValueError(f"Not enough samples ({n_samples}) for {n_splits} splits "
                             f"with {self.purge_days} purge days and {self.embargo_days} embargo days")

        indices = np.arange(n_samples)

        for i in range(n_splits):
            # Expanding window for train
            train_end_idx = (i + 1) * test_size

            # Add purge gap
            test_start_idx = train_end_idx + purge_samples
            test_end_idx = test_start_idx + test_size

            # Add embargo at the end (except for last fold)
            if i < n_splits - 1:
                test_end_idx = min(test_end_idx, test_end_idx + embargo_samples)

            # Ensure we don't exceed array bounds
            test_end_idx = min(test_end_idx, n_samples)

            # Ensure we have enough test samples
            if test_end_idx - test_start_idx < 10:
                logger.warning(f"Fold {i + 1}: Not enough test samples, skipping")
                continue

            train_indices = indices[:train_end_idx]
            test_indices = indices[test_start_idx:test_end_idx]

            yield train_indices, test_indices


class PurgedWalkForwardCV:
    """
    Walk-forward cross-validation with purging
    Simulates real-world trading where you retrain periodically
    """

    def __init__(self, train_months: int = 12, validation_months: int = 3,
                 test_months: int = 1, purge_days: int = 5,
                 retrain_frequency: str = 'monthly'):
        """
        Parameters:
        -----------
        train_months : int
            Length of training period in months
        validation_months : int
            Length of validation period in months
        test_months : int
            Length of test period in months
        purge_days : int
            Days to purge between periods
        retrain_frequency : str
            How often to retrain ('monthly', 'quarterly')
        """
        self.train_months = train_months
        self.validation_months = validation_months
        self.test_months = test_months
        self.purge_days = purge_days
        self.retrain_frequency = retrain_frequency

    def split(self, dates: pd.DatetimeIndex) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/validation/test splits

        Yields:
        -------
        train_dates, val_dates, test_dates
        """
        start_date = dates[0]
        end_date = dates[-1]

        current_date = start_date

        while current_date < end_date:
            # Define train period
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.train_months)

            # Add purge gap
            val_start = train_end + pd.DateOffset(days=self.purge_days)
            val_end = val_start + pd.DateOffset(months=self.validation_months)

            # Add another purge gap
            test_start = val_end + pd.DateOffset(days=self.purge_days)
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Get indices for each period
            train_mask = (dates >= train_start) & (dates <= train_end)
            val_mask = (dates >= val_start) & (dates <= val_end)
            test_mask = (dates >= test_start) & (dates <= test_end)

            train_dates = dates[train_mask]
            val_dates = dates[val_mask]
            test_dates = dates[test_mask]

            # Ensure we have enough samples
            if len(train_dates) < 100 or len(val_dates) < 20 or len(test_dates) < 20:
                logger.warning(f"Not enough samples in period starting {train_start}, skipping")
                current_date = current_date + pd.DateOffset(months=1)
                continue

            yield train_dates, val_dates, test_dates

            # Move to next period based on retrain frequency
            if self.retrain_frequency == 'monthly':
                current_date = test_start
            elif self.retrain_frequency == 'quarterly':
                current_date = test_start + pd.DateOffset(months=2)
            else:
                current_date = test_start


def create_purged_k_fold(n_splits: int = 5, purge_pct: float = 0.02):
    """
    Create a purged K-fold splitter for time series

    Parameters:
    -----------
    n_splits : int
        Number of folds
    purge_pct : float
        Percentage of data to purge between train and test
    """
    return PurgedTimeSeriesSplit(
        n_splits=n_splits,
        purge_days=int(252 * purge_pct),  # Convert percentage to trading days
        embargo_days=int(252 * purge_pct / 2)
    )


def verify_no_leakage(train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                      min_gap_days: int = 5) -> bool:
    """
    Verify there's no data leakage between train and test sets

    Returns:
    --------
    bool : True if no leakage detected
    """
    if len(train_dates) == 0 or len(test_dates) == 0:
        return True

    latest_train = train_dates.max()
    earliest_test = test_dates.min()

    gap = (earliest_test - latest_train).days

    if gap < min_gap_days:
        logger.error(f"Data leakage detected! Gap is only {gap} days, need at least {min_gap_days}")
        return False

    return True


# Example usage and testing
if __name__ == "__main__":
    print("Testing Purged Cross-Validation Implementation")
    print("=" * 60)

    # Test basic purged split
    n_samples = 1000
    X = np.random.randn(n_samples, 10)

    print("\n1. Testing PurgedTimeSeriesSplit:")
    print("-" * 40)

    pcv = PurgedTimeSeriesSplit(n_splits=5, purge_days=5, embargo_days=2)

    for i, (train_idx, test_idx) in enumerate(pcv.split(X)):
        gap = test_idx[0] - train_idx[-1] - 1
        print(f"Fold {i + 1}: Train size={len(train_idx)}, Test size={len(test_idx)}, Gap={gap} samples")

        # Verify no overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train and test sets overlap!"
        assert gap >= 5, f"Gap too small: {gap} < 5"

    print("\n2. Testing Walk-Forward CV:")
    print("-" * 40)

    # Create date range from 5 years ago to today
    today = pd.Timestamp('2025-07-12')
    start_date = today - pd.DateOffset(years=5)
    dates = pd.date_range(start=start_date, end=today, freq='B')  # Business days

    wf_cv = PurgedWalkForwardCV(
        train_months=12,
        validation_months=3,
        test_months=1,
        purge_days=5
    )

    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"Total trading days: {len(dates)}")
    print("\nWalk-forward periods:")

    for i, (train_dates, val_dates, test_dates) in enumerate(wf_cv.split(dates)):
        print(f"\nPeriod {i + 1}:")
        print(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
        print(f"  Val:   {val_dates[0].date()} to {val_dates[-1].date()} ({len(val_dates)} days)")
        print(f"  Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

        # Verify gaps
        gap1 = (val_dates[0] - train_dates[-1]).days
        gap2 = (test_dates[0] - val_dates[-1]).days
        print(f"  Gaps: Train->Val={gap1} days, Val->Test={gap2} days")

        # Verify no leakage
        assert verify_no_leakage(train_dates, val_dates), "Leakage between train and val!"
        assert verify_no_leakage(val_dates, test_dates), "Leakage between val and test!"

    print("\n✅ All tests passed! No data leakage detected.")