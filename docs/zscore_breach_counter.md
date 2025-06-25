# Z-Score Breach Counter for Statistical Arbitrage

This module provides functionality to analyze historical price data and calculate how many times 
a spread's Z-score has breached certain thresholds, which can be useful for statistical 
arbitrage strategy development and analysis.

## Features

- Counts the number of times a pair's spread Z-score exceeds upper threshold (2.0)
- Counts the number of times a pair's spread Z-score falls below lower threshold (-2.0)
- Uses historical price data from PostgreSQL database
- Updates results in the `pair_statistics` table
- Fully configurable thresholds and analysis windows

## Installation

1. Ensure the required packages are installed:

```
pip install -r requirements.txt
```

2. Create a `.env` file with your database credentials (see `.env.example` for reference)

3. Run the database schema update script to add the necessary columns:

```bash
python scripts/add_zscore_breach_columns.py
```

## Usage

Here's a basic example of how to use the Z-Score breach counter:

```python
from zscore_breach_counter import run_zscore_breach_analysis

# Run analysis with default parameters (365-day lookback, 252-day window)
results = run_zscore_breach_analysis()

# Run with custom parameters
results = run_zscore_breach_analysis(lookback_days=730, window=125)

# Process results
for pair, counts in results.items():
    print(f"Pair: {pair}")
    print(f"  Breaches above 2.0: {counts.get('zscore_breach_count_above_2.0', 0)}")
    print(f"  Breaches below -2.0: {counts.get('zscore_breach_count_below_-2.0', 0)}")
```

See the `examples/use_zscore_breach_counter.py` file for a more detailed example.

## Database Structure

The module adds two columns to the existing `pair_statistics` table:

- `zscore_breach_count_above_2`: Integer count of times the Z-score exceeded 2.0
- `zscore_breach_count_below_neg2`: Integer count of times the Z-score fell below -2.0

## Testing

Run the unit tests with pytest:

```bash
pytest tests/test_zscore_breach_counter.py -v
```

## How It Works

1. The module fetches historical price data for each symbol pair from the PostgreSQL database.
2. It calculates the spread between the two price series.
3. Rolling Z-scores are calculated using a configurable window (default: 252 days).
4. The module counts how many times the Z-score breaches the upper and lower thresholds.
5. Results are stored in the `pair_statistics` table for each symbol pair.

## Interpretation

Pairs with a high number of Z-score breaches may indicate good candidates for statistical arbitrage strategies, as they show a tendency to revert to the mean after significant deviations.

- More breaches = more trading opportunities
- Fewer breaches = potentially more stable pairs

## Configuration

Key parameters:

- `lookback_days`: Historical data period to analyze (default: 365 days)
- `window`: Rolling window size for Z-score calculation (default: 252 days)
- Upper threshold: Default is 2.0 (customizable)
- Lower threshold: Default is -2.0 (customizable)
