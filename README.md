# Statistical Arbitrage Analytics System

A production-ready, modular statistical arbitrage system featuring advanced cointegration analysis, real-time hourly monitoring, and sophisticated trading signal generation with comprehensive risk management.

## 🎯 **Latest Updates** (June 2025)

**✅ Advanced Metrics Implementation Completed:**
- **Half-Life of Mean Reversion**: AR(1) regression-based calculation for spread analysis
- **Sharpe Ratio**: Real-time spread returns performance monitoring
- **Z-Score Breach Counting**: Statistical threshold violation tracking (|Z| > 2)
- **Dedicated Database Columns**: Optimized storage for enhanced query performance
- **Production Integration**: Fully integrated with hourly analysis scheduler

**✅ Complete System Architecture:**
- **Daily Cointegration Analysis**: Automated pair identification and validation
- **Hourly Statistical Monitoring**: Real-time spread analysis and signal generation
- **Advanced Risk Metrics**: Half-life, Sharpe ratios, and breach detection
- **Production Scheduler**: Automated daily (06:00) and hourly analysis execution
- **PostgreSQL Backend**: Remote cloud database with 2.6M+ historical records

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Configure your environment variables
# Edit .env with your database credentials and parameters
# All configuration is handled via the .env file
```

### 2. Database Initialization
```bash
# Create required database tables
python setup.py
```

### 3. Run the System

#### **Full Scheduler (Recommended)**
```bash
# Start the complete system with daily and hourly analysis
python main.py scheduler
```

#### **Individual Analysis Commands**
```bash
# Run daily cointegration analysis once
python main.py daily

# Run hourly statistics analysis once  
python main.py hourly

# Check system status
python main.py status

# Run analysis for specific symbols
python main.py daily --symbols SPY IVV QQQ

# Run with different log levels
python main.py scheduler --log-level DEBUG
```

#### **Direct Script Execution**
```bash
# Run individual analysis modules
python run_daily_analysis.py
python run_hourly_analysis.py

# Concatenate and process data files
python concatenate_files.py
```

### 4. Monitor Results
```bash
# Check system status and recent results
python main.py status

# View scheduler logs
tail -f statistical_arbitrage_main.log

# Check audit logs
tail -f statistics_audit.log
```

## 📁 System Architecture

```
stat_arb_prod/
├── .env                              # Main configuration file
├── main.py                          # Primary system entry point
├── setup.py                         # Database table creation
├── analytics/stats/                 # Core analytics modules
│   ├── configuration_manager.py     # Environment configuration
│   ├── database_manager.py          # PostgreSQL operations
│   ├── data_manager.py              # Historical data management
│   ├── cointegration_manager.py     # Daily pair analysis
│   ├── hourly_statistics_manager.py # Real-time monitoring
│   ├── metrics.py                   # 🆕 Advanced metrics calculation
│   ├── scheduler.py                 # Automated execution
│   └── audit_logger.py              # Compliance logging
├── run_daily_analysis.py            # Daily analysis runner
├── run_hourly_analysis.py           # Hourly analysis runner
├── docker-compose.yml               # Production deployment
└── requirements.txt                 # Python dependencies
```

## ⚙️ Core Features

### **Daily Cointegration Analysis**
- **Pair Identification**: Automated discovery of cointegrated asset pairs
- **Statistical Testing**: Johansen and Engle-Granger cointegration tests
- **Hedge Ratio Calculation**: Optimal pair trading ratios
- **Result Storage**: PostgreSQL database with comprehensive metadata

### **Hourly Statistical Monitoring**  
- **Real-time Analysis**: Continuous monitoring of cointegrated pairs
- **Z-Score Calculation**: Dynamic spread standardization and signal detection
- **Moving Averages**: Short/long-term trend analysis (configurable windows)
- **Advanced Metrics**: Half-life, Sharpe ratios, and breach counting
- **Signal Generation**: Automated trading signal creation with strength scoring

### **Advanced Metrics System** 🆕
- **Half-Life of Mean Reversion**: `-log(2) / log(beta)` calculation via AR(1) regression
- **Sharpe Ratio**: `mean_return / std_return` for spread performance evaluation  
- **Z-Score Breach Counts**: Statistical significance tracking:
  - `zscore_over_2`: Count of Z-scores > +2
  - `zscore_under_minus_2`: Count of Z-scores < -2
- **Database Integration**: Dedicated columns for optimized querying

### **Production Scheduler**
- **Daily Schedule**: 06:00 AM cointegration analysis
- **Hourly Monitoring**: Continuous statistical analysis during trading hours
- **Error Handling**: Automatic retry logic with configurable delays
- **Audit Logging**: Comprehensive compliance and debugging logs

## 🔧 Configuration

All system parameters are configured via the `.env` file:

### **Database Configuration**
```properties
DB_HOST=your-database-host
DB_PORT=25060
DB_NAME=stats
DB_USER=your-username
DB_PASSWORD=your-password
```

### **Analysis Parameters**
```properties
# Asset universe
ASSET_SYMBOLS=AGG,BND,EWA,EWC,GLD,IAU,IVV,IWM,QQQ,SPY,VDE,VFH,VOO,VTWO,XLE,XLF,XLK

# Cointegration settings
CORRELATION_THRESHOLD=0.7
COINTEGRATION_SIGNIFICANCE_LEVEL=0.05
COINTEGRATION_LOOKBACK_DAYS=365

# Hourly analysis settings
ZSCORE_WINDOW=120
MA_SHORT_WINDOW=10
MA_LONG_WINDOW=20
HOURLY_LOOKBACK_HOURS=480
ZSCORE_THRESHOLD=2.0
```

### **Scheduler Configuration**
```properties
DAILY_ANALYSIS_TIME=06:00
HOURLY_ANALYSIS_ENABLED=true
TRADING_HOURS=9,10,11,12,13,14,15,16
MAX_RETRIES=3
RETRY_DELAY_MINUTES=30
```

## 📊 System Performance

### **Recent Production Results**
- **Daily Analysis**: 117 symbols → 6,786 pairs tested → **495 cointegrated pairs**
- **Hourly Analysis**: 495 pairs monitored → **33 active signals** generated
- **Database Performance**: Sub-second query times with optimized indexing
- **Advanced Metrics**: Real-time calculation of half-life, Sharpe ratios, and breach counts

### **Statistical Metrics**
- **Half-Life Range**: 0.5 - 50+ hours (mean reversion speed)
- **Sharpe Ratios**: -0.5 to +0.5 (spread return quality)
- **Z-Score Breaches**: 3-8% of observations exceed ±2 threshold
- **Signal Generation**: 5-10% of pairs generate actionable signals hourly

## 🛠️ Advanced Usage

### **Custom Analysis Windows**
```bash
# Analyze specific date range
python main.py daily --date 2025-06-20

# Custom symbol analysis
python main.py daily --symbols SPY IVV QQQ VTI

# Enable table creation
python main.py scheduler --create-tables
```

### **Database Operations**
```bash
# Check table structures
python check_table_structure.py

# Verify data ranges
python check_date_range.py

# Debug cointegration queries
python debug_cointegration_query.py
```

### **Development & Testing**
```bash
# Test configuration
python test_config.py

# Test hourly configuration
python test_hourly_config.py

# Test advanced metrics
python test_advanced_metrics.py

# Verify metrics in database
python verify_advanced_metrics.py
```

## 📈 Monitoring & Operations

### **Log Files**
- `statistical_arbitrage_main.log`: Primary system logs
- `statistics_audit.log`: Compliance and audit trail
- `hourly_statistics_monitor.log`: Hourly analysis details
- `statistics_calculator.log`: Calculation debugging

### **Database Tables**
- `historical_bars_1_day`: Daily price data (117 symbols)
- `historical_bars_1_hour`: Hourly price data  
- `cointegrated_pairs`: Daily cointegration results
- `hourly_statistics`: Real-time analysis with advanced metrics

### **Key Metrics to Monitor**
- **Cointegration Success Rate**: % of pairs passing statistical tests
- **Signal Generation Rate**: Active signals per hour
- **Half-Life Distribution**: Mean reversion speed across pairs
- **Z-Score Breach Frequency**: Statistical significance events
- **System Uptime**: Scheduler reliability and error rates

## 🔧 Technical Implementation

### **Dependencies**
- **Python 3.12+**: Modern async/await support
- **PostgreSQL**: Production database backend
- **Pandas/NumPy**: High-performance data analysis
- **Statsmodels**: Advanced statistical modeling
- **Psycopg2**: PostgreSQL connectivity
- **Schedule**: Task automation

### **Database Schema**
```sql
-- Advanced metrics columns added
ALTER TABLE hourly_statistics 
ADD COLUMN half_life DOUBLE PRECISION,
ADD COLUMN sharpe_ratio DOUBLE PRECISION,
ADD COLUMN zscore_over_2 INTEGER DEFAULT 0,
ADD COLUMN zscore_under_minus_2 INTEGER DEFAULT 0;
```

### **Performance Optimizations**
- **Connection Pooling**: 1-10 concurrent database connections
- **Batch Processing**: Efficient bulk data operations
- **Indexed Queries**: Optimized database access patterns
- **Memory Management**: Pandas DataFrame optimization

## 🚨 Troubleshooting

### **Common Issues**
```bash
# Permission errors
chmod +x main.py run_*.py

# Missing dependencies
pip install -r requirements.txt

# Database connection issues
python -c "from analytics.stats.database_manager import DatabaseManager; DatabaseManager()"

# Configuration problems
python test_config.py
```

### **Error Recovery**
- **Failed Analysis**: System automatically retries with exponential backoff
- **Database Disconnects**: Connection pool handles reconnection
- **Data Quality Issues**: Robust error handling with detailed logging
- **Scheduler Interrupts**: Graceful shutdown and restart capabilities

## � Development Roadmap

- **Machine Learning Integration**: Enhanced signal generation with ML models
- **Real-time Trading**: Direct broker integration for automated execution
- **Portfolio Optimization**: Multi-pair capital allocation algorithms
- **Alternative Data**: Integration of sentiment and news data sources
- **Performance Attribution**: Advanced analytics for strategy decomposition

---

**🎯 Production Status**: Fully operational with 495 pairs monitored hourly and advanced metrics calculation integrated.

**📞 Support**: Check logs and run diagnostic scripts for troubleshooting.
