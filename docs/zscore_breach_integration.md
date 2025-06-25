# Z-Score Breach Counter Integration

This document explains the integration of Z-Score breach counting functionality into the main Statistical Arbitrage Analysis system.

## Overview

The Z-Score breach counter enhancement adds functionality to track how many times a spread's Z-score crosses above 2.0 or below -2.0 over a given time period. This provides valuable insight into the historical behavior of pairs and helps identify more reliable trading opportunities.

## Implementation Details

The following changes were made to the main statistical arbitrage analysis system:

1. Added a new `count_zscore_breaches` method to the `StatisticalArbitrageAnalytics` class that:
   - Takes a spread series as input
   - Calculates rolling Z-scores using a configurable window
   - Counts breaches above 2.0 and below -2.0
   - Returns a dictionary with breach counts

2. Modified the `analyze_pair` method to include Z-Score breach calculations

3. Updated the `save_results_to_database` method to store Z-Score breach counts in the database

4. Modified the `display_top_signals` method to show Z-Score breach counts in the output

## Database Changes

Two new columns were added to the `pair_statistics` table:
- `zscore_breach_count_above_2`: Integer count of breaches above 2.0
- `zscore_breach_count_below_neg2`: Integer count of breaches below -2.0

## Usage

The Z-Score breach functionality is now fully integrated into the main statistical arbitrage analysis workflow. No additional steps are required to use it.

When running the main analysis using `run_statistical_arbitrage_analysis.py`, the Z-Score breach counts will automatically be calculated, stored in the database, and displayed in the output.

## Interpreting Results

Pairs with a high number of Z-Score breaches may indicate good candidates for statistical arbitrage strategies, as they show a tendency to revert to the mean after significant deviations.

- More breaches = more trading opportunities
- Fewer breaches = potentially more stable pairs

The Z-Score breach counts can be used alongside other metrics like correlation, cointegration p-value, and half-life to form a more comprehensive view of pair stability and trading potential.
