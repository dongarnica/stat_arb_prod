# Hourly and Daily Statistical Analysis System

## Overview

This system performs comprehensive statistical arbitrage analysis with two main components:

1. **Daily Cointegration Analysis**: Identifies and stores cointegrated pairs using statistical tests
2. **Hourly Statistical Analysis**: Performs fast calculations on pre-identified pairs for trading signals

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Application                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Scheduler     │  │ Daily Analysis  │  │   Hourly    │ │
│  │                 │  │                 │  │  Analysis   │ │
│  │ - Task Mgmt     │  │ - Cointegration │  │ - Z-scores  │ │
│  │ - Error Retry   │  │ - Pair Storage  │  │ - Signals   │ │
│  │ - Monitoring    │  │ - Cleanup       │  │ - MA calc   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Data Management Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Configuration   │  │  Database       │  │    Data     │ │
│  │   Manager       │  │   Manager       │  │   Manager   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      Database Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ cointegrated_   │  │ hourly_         │  │ price_data  │ │
│  │    pairs        │  │ statistics      │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### Daily Process
- **Comprehensive Cointegration Testing**: Uses Engle-Granger and Johansen tests
- **Efficient Pair Generation**: Tests all possible symbol combinations
- **Statistical Validation**: Configurable significance levels and minimum observations
- **Result Storage**: Stores cointegrated pairs with metadata for hourly use
- **Automatic Cleanup**: Removes old results to maintain performance

### Hourly Process
- **Pre-filtered Analysis**: Only analyzes proven cointegrated pairs
- **Fast Calculations**: Z-scores, moving averages, momentum indicators
- **Signal Generation**: Trading signals based on statistical thresholds
- **Real-time Monitoring**: Current spread analysis and alerts
- **Batch Processing**: Efficient handling of multiple pairs

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your database credentials and settings
   ```

3. **Initialize Database**:
   ```bash
   python main.py status --create-tables
   ```

## Configuration

Edit the `.env` file with your settings:

```env
# Database
DB_HOST=localhost
DB_NAME=statistical_arbitrage
DB_USER=your_username
DB_PASSWORD=your_password

# Analysis Parameters
COINTEGRATION_SIGNIFICANCE_LEVEL=0.05
ZSCORE_THRESHOLD=2.0
ZSCORE_WINDOW=252

# Scheduling
DAILY_ANALYSIS_TIME=06:00
HOURLY_ANALYSIS_ENABLED=true
TRADING_HOURS=9,10,11,12,13,14,15,16
```

## Usage

### Run the Full Scheduler
```bash
python main.py scheduler
```
This starts the complete system with automated daily and hourly analysis.

### Run Daily Analysis Once
```bash
# Analyze all available symbols
python main.py daily

# Analyze specific symbols
python main.py daily --symbols SPY IVV QQQ

# Using dedicated script
python run_daily_analysis.py --symbols SPY IVV
```

### Run Hourly Analysis Once
```bash
# Analyze using today's cointegrated pairs
python main.py hourly

# Analyze using specific date's pairs
python main.py hourly --date 2024-01-15

# Using dedicated script
python run_hourly_analysis.py --show-signals
```

### Check System Status
```bash
python main.py status
```

## Database Schema

### cointegrated_pairs
Stores results from daily cointegration analysis:

```sql
CREATE TABLE cointegrated_pairs (
    id SERIAL PRIMARY KEY,
    symbol1 VARCHAR(20) NOT NULL,
    symbol2 VARCHAR(20) NOT NULL,
    test_statistic DOUBLE PRECISION NOT NULL,
    p_value DOUBLE PRECISION NOT NULL,
    critical_value DOUBLE PRECISION NOT NULL,
    hedge_ratio DOUBLE PRECISION NOT NULL,
    test_method VARCHAR(50) NOT NULL,
    analysis_date DATE NOT NULL,
    metadata JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol1, symbol2, analysis_date)
);
```

### hourly_statistics
Stores results from hourly statistical analysis:

```sql
CREATE TABLE hourly_statistics (
    id SERIAL PRIMARY KEY,
    symbol1 VARCHAR(20) NOT NULL,
    symbol2 VARCHAR(20) NOT NULL,
    current_price1 DOUBLE PRECISION NOT NULL,
    current_price2 DOUBLE PRECISION NOT NULL,
    hedge_ratio DOUBLE PRECISION NOT NULL,
    spread_value DOUBLE PRECISION NOT NULL,
    current_z_score DOUBLE PRECISION NOT NULL,
    spread_mean DOUBLE PRECISION NOT NULL,
    spread_std DOUBLE PRECISION NOT NULL,
    ma_short DOUBLE PRECISION,
    ma_long DOUBLE PRECISION,
    signal_type VARCHAR(20) NOT NULL,
    signal_strength DOUBLE PRECISION NOT NULL,
    has_signal BOOLEAN DEFAULT false,
    analysis_timestamp TIMESTAMP NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Usage

### Programmatic Access

```python
from analytics.stats.configuration_manager import ConfigurationManager
from analytics.stats.cointegration_manager import CointegrationManager
from analytics.stats.hourly_statistics_manager import HourlyStatisticsManager

# Initialize
config = ConfigurationManager()
coint_manager = CointegrationManager(config)
hourly_manager = HourlyStatisticsManager(config)

# Run daily analysis
results = coint_manager.run_daily_cointegration_analysis(['SPY', 'IVV'])

# Get cointegrated pairs
pairs = coint_manager.get_cointegrated_pairs()

# Run hourly analysis
hourly_results = hourly_manager.run_hourly_analysis()

# Get current signals
signals = hourly_manager.get_current_signals(min_z_score=2.0)
```

## Performance Optimization

### Daily Process Efficiency
- **Batch Processing**: Tests multiple pairs simultaneously
- **Vectorized Calculations**: Uses pandas/numpy for fast computations
- **Connection Pooling**: Efficient database operations
- **Smart Cleanup**: Removes old data automatically

### Hourly Process Speed
- **Pre-filtered Pairs**: Only analyzes cointegrated pairs
- **Minimal Data Fetching**: Gets only necessary recent data
- **Cached Calculations**: Reuses common computations
- **Parallel Processing**: Configurable multi-threading

## Monitoring and Alerts

### Logging
- Comprehensive logging at all levels
- Separate log files for different components
- Configurable log levels
- Audit trail for all operations

### Error Handling
- Automatic retry logic with exponential backoff
- Graceful degradation on failures
- Detailed error reporting
- Recovery mechanisms

### Performance Monitoring
- Execution time tracking
- Success rate monitoring
- Database performance metrics
- Resource usage tracking

## Scheduling Details

### Default Schedule
- **Daily Analysis**: 6:00 AM (configurable)
- **Hourly Analysis**: Every hour during trading hours
- **Cleanup**: Automatic old data removal
- **Retry Logic**: Up to 3 attempts with delays

### Customization
```env
DAILY_ANALYSIS_TIME=06:00
TRADING_HOURS=9,10,11,12,13,14,15,16
MAX_RETRIES=3
RETRY_DELAY_MINUTES=30
```

## Statistical Methods

### Cointegration Tests
1. **Engle-Granger Test**: Two-step procedure for bivariate cointegration
2. **Johansen Test**: Multivariate approach with trace statistics
3. **Augmented Dickey-Fuller**: Stationarity testing of residuals

### Hourly Calculations
1. **Z-Score**: Standardized spread deviation
2. **Moving Averages**: Short and long-term trend indicators
3. **Momentum**: Rate of change in spread
4. **Correlation**: Current price relationship strength

## Signal Generation

### Signal Types
- **long_spread**: Long symbol1, Short symbol2 (Z-score < -threshold)
- **short_spread**: Short symbol1, Long symbol2 (Z-score > threshold)
- **hold**: No action (|Z-score| < threshold)

### Signal Strength
- **1.0**: Very strong (|Z-score| >= 3.0)
- **0.8**: Strong (|Z-score| >= 2.5)
- **0.6**: Moderate (|Z-score| >= 2.0)
- **0.4**: Weak (|Z-score| >= 1.5)

## Troubleshooting

### Common Issues

1. **No Cointegrated Pairs Found**
   - Check significance level (try 0.10 instead of 0.05)
   - Verify sufficient historical data
   - Ensure price data quality

2. **Database Connection Errors**
   - Verify database credentials in .env
   - Check network connectivity
   - Ensure database exists and is accessible

3. **Performance Issues**
   - Reduce the number of symbols
   - Increase connection pool size
   - Consider parallel processing

4. **Missing Price Data**
   - Verify data sources are available
   - Check date ranges in queries
   - Ensure data pipeline is running

### Log Analysis
```bash
# Check recent errors
tail -f statistical_arbitrage_main.log | grep ERROR

# Monitor daily analysis
tail -f daily_cointegration_analysis.log

# Watch hourly analysis
tail -f hourly_statistics_analysis.log
```

## Best Practices

1. **Data Quality**: Ensure clean, consistent price data
2. **Parameter Tuning**: Adjust thresholds based on market conditions
3. **Regular Monitoring**: Check system status and performance
4. **Backup Strategy**: Regular database backups
5. **Testing**: Validate with paper trading before live deployment

## Future Enhancements

1. **Real-time Data Integration**: Live price feeds
2. **Advanced Signal Filtering**: Machine learning models
3. **Risk Management**: Position sizing and stop-losses
4. **Web Dashboard**: Real-time monitoring interface
5. **API Endpoints**: RESTful API for external access
