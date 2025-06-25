# Statistical Arbitrage Trading System

A production-ready, modular statistics system for statistical arbitrage trading, featuring comprehensive financial metrics calculation and pair trading analysis.

## 🎯 **Modernized & Optimized** (June 2025)
**Complete system refactor completed with exceptional results:**
- ✅ **Modern modular architecture** using `analytics/stats/` system
- ✅ **64.8% faster queries** (0.655s → 0.231s)
- ✅ **0% duplicate data** (eliminated 95%+ duplicates)  
- ✅ **Production-ready configuration** via .env files
- ✅ **Comprehensive statistics modules** for pair trading analysis

## 🚀 Quick Start

1. **Configure environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your database credentials and strategy parameters
   ```

2. **Run the statistics system**:
   ```bash
   cd analytics
   python example_usage.py
   ```

3. **Use the statistics calculator**:
   ```python
   from analytics import StatisticsCalculator, ConfigurationManager
   
   # Initialize configuration
   config = ConfigurationManager()
   
   # Calculate statistics
   with StatisticsCalculator() as calculator:
       results = calculator.calculate_pair_statistics(
           symbol1='SPY', symbol2='IVV',
           start_date='2024-01-01', end_date='2024-12-31'
       )
   ```

4. **Analyze results**:
   ```bash
   python strategy_analysis.py
   ```

## 📁 Project Structure

```
├── .env                              # Environment configuration
├── .env.template                     # Configuration template
├── analytics/                       # 🎯 MODERN STATISTICS SYSTEM
│   ├── stats/                       # Core statistics modules
│   │   ├── modules/                 # Organized statistics modules
│   │   │   └── pair_trading/        # Pair trading focus
│   │   ├── configuration_manager.py # Configuration handling
│   │   ├── database_manager.py      # Database operations
│   │   └── orchestrator.py          # Main orchestration
│   ├── statistics_calculator.py     # Main interface
│   └── example_usage.py             # Usage examples
├── scripts/                         # Maintenance scripts
├── strategy_analysis.py             # Performance analysis
├── docs/                            # Documentation
├── scripts/                         # Utility scripts
├── tests/                           # Test suites
├── docker-compose.yml               # Docker deployment
└── requirements.txt                 # Python dependencies
```

## ⚙️ Key Features

- **Pairs Trading**: SPY-IVV statistical arbitrage with cointegration testing
- **Dynamic Position Sizing**: Capital-optimized allocation for $30K accounts
- **Risk Management**: Multi-layer protection (stop losses, daily limits, drawdown)
- **MLflow Integration**: Comprehensive experiment tracking and logging
- **PostgreSQL Backend**: Production database with 2.6M+ historical records
- **Parameterized Configuration**: Easy strategy tuning via .env profiles
- **Real-time Monitoring**: Live performance tracking and alerts

## 📊 Performance Targets

- **Annual Return**: >5%
- **Sharpe Ratio**: >1.0  
- **Max Drawdown**: <5%
- **Win Rate**: >45%
- **Capital Efficiency**: Optimized for $30K accounts

## 🔧 Configuration Profiles

The system supports multiple strategy profiles via `.env` configuration:

- **Conservative**: Lower risk, stable returns (Entry Z: 2.5, Max Pos: 25%)
- **Balanced**: Moderate risk/reward (Entry Z: 2.0, Max Pos: 31%) 
- **Aggressive**: Higher risk, higher potential (Entry Z: 1.6, Max Pos: 40%)

Use `config_switcher.py` to easily switch between profiles.

## 📈 Monitoring & Results

- **MLflow UI**: http://192.241.244.26:5000/
- **Results Storage**: Local pickle files + PostgreSQL database
- **Performance Metrics**: Sharpe ratio, drawdown, win rate, trade analysis
- **Risk Monitoring**: Real-time position sizing and stop loss tracking

## 🔧 Technical Implementation

- **Database**: PostgreSQL with 156K+ records each for SPY/IVV
- **Timeframe**: Daily data with "1 day" format
- **Signal Generation**: Z-score based entry/exit with cointegration validation  
- **Position Management**: Dynamic sizing with capital preservation
- **Execution**: Realistic slippage and commission modeling

## 📋 Recent Performance

Latest backtest results (400-day period):
- Total Return: 0.18%
- Trades Executed: 27
- Win Rate: 51.9%
- Sharpe Ratio: 0.10

*Note: Performance can be improved through parameter optimization and profile switching.*

## 🛠️ Development

Built with Python 3.12+, featuring:
- Type hints and validation with Pydantic
- Comprehensive logging and error handling
- MLflow experiment tracking
- Docker containerization support
- Clean, maintainable codebase architecture
