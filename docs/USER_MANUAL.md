# Forex Trading Platform User Manual

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Installation and Setup](#2-installation-and-setup)
3. [Starting and Stopping the System](#3-starting-and-stopping-the-system)
4. [Configuration](#4-configuration)
5. [Trading Modes](#5-trading-modes)
6. [Monitoring Dashboard](#6-monitoring-dashboard)
7. [Risk Management](#7-risk-management)
8. [Troubleshooting](#8-troubleshooting)

## 1. System Overview

The Forex Trading Platform is a comprehensive AI-powered trading system that supports both live and paper trading of forex currency pairs. The platform includes:

- Multi-agent architecture for market data, technical analysis, fundamental analysis, sentiment analysis, and portfolio management
- Advanced risk management capabilities
- Backtesting and strategy optimization tools
- Real-time monitoring dashboard

## 2. Installation and Setup

### Prerequisites

- Python 3.8 or higher
- An OANDA trading account (practice or live)
- OpenAI API key (for AI-powered trading strategies)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/forex-trading.git
   cd forex-trading
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.template .env
   ```

4. Edit the `.env` file with your API credentials:
   - OANDA_API_KEY
   - OANDA_ACCOUNT_ID
   - OANDA_API_URL
   - OPENAI_API_KEY (if using AI strategies)

## 3. Starting and Stopping the System

### Starting the System

The platform can be started with the main.py script using different trading modes:

```bash
# Start in paper trading mode (default)
python main.py start --mode paper

# Start in live trading mode
python main.py start --mode live

# Start with specific currency pairs
python main.py start --mode paper --instruments EUR_USD,GBP_USD,USD_JPY
```

### Stopping the System

To safely stop the trading platform:

```bash
python main.py stop
```

This performs a graceful shutdown, closing positions if configured, and saving the system state.

### Checking System Status

```bash
python main.py status
```

This shows the current state of the trading system, including active modes, open positions, and connected agents.

## 4. Configuration

### Main Configuration

The system is configured through several files:

1. **Environment variables** (`.env` file):
   - API credentials
   - Risk parameters
   - Logging configuration

2. **System configuration** (`config/config.yaml`):
   - Trading strategies
   - Market data sources
   - Technical indicators
   - Risk management parameters

### Trading Strategy Configuration

Edit `config/config.yaml` to configure trading strategies:

```yaml
technical_analysis:
  strategies:
    active_strategies:
      - rsi_reversal
      - bollinger_bands
      - macd
      - moving_average_crossover
    
    # Strategy-specific parameters
    rsi_reversal:
      rsi_period: 14
      oversold: 25
      overbought: 75
      timeframes: ["1h", "4h"]
      symbols: ["EUR/USD", "GBP/USD", "USD/JPY"]
```

### Risk Management Configuration

Risk parameters can be adjusted in both `.env` and `config/config.yaml`:

```yaml
risk_management:
  max_risk_per_trade: 0.02  # 2% of account
  max_open_positions: 5
  max_risk_per_currency: 0.05  # 5% of account
  max_daily_drawdown: 0.05  # 5% of account
  stop_loss_atr_multiplier: 2.0
```

## 5. Trading Modes

### Paper Trading Mode

Paper trading allows simulation without risking real funds:

```bash
python main.py start --mode paper
```

Use this mode to test strategies and familiarize yourself with the platform.

### Live Trading Mode

Live trading executes real trades using your OANDA account:

```bash
python main.py start --mode live
```

**Important**: Ensure you've properly configured risk parameters before using live mode.

### Backtesting Mode

Backtest strategies against historical data:

```bash
python main.py backtest --start-date 2023-01-01 --end-date 2023-12-31 --instruments EUR_USD,GBP_USD
```

### Strategy Optimization

Optimize strategy parameters based on historical performance:

```bash
python main.py optimize --parameters "rsi_period,oversold,overbought" --start-date 2023-01-01 --end-date 2023-12-31
```

## 6. Monitoring Dashboard

### Starting the Dashboard

```bash
streamlit run monitoring.py
```

This launches the web-based monitoring dashboard on http://localhost:8501

### Dashboard Sections

1. **Overview**: Summary of system status, account balance, and active trades
2. **System Status**: Server health, agent status, and API connections
3. **Trading**: Open positions, pending orders, and trade history
4. **Market**: Real-time price charts, volatility, and spread analysis
5. **Performance**: P&L metrics, drawdown analysis, and performance charts
6. **Alerts**: System notifications and custom alerts

### Interpreting Dashboard Metrics

- **Agent Status**: Green indicates normal operation, yellow indicates warnings, red indicates failures
- **Open Positions**: Current trades with entry price, current price, P&L, and risk metrics
- **Account Balance**: Timeline of account equity changes
- **Performance Metrics**:
  - Win Rate: Percentage of winning trades
  - Profit Factor: Gross profit divided by gross loss
  - Sharpe Ratio: Risk-adjusted return metric
  - Maximum Drawdown: Largest peak-to-trough decline
  - Expectancy: Average profit/loss per trade

### Alert Configuration

Configure monitoring alerts in `config/monitoring_config.json`:

```json
"alerts": {
    "email": {
        "enabled": true,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your_email@gmail.com",
        "password": "your_app_password",
        "recipients": ["alerts@example.com"]
    }
}
```

## 7. Risk Management

### Setting Risk Parameters

1. **Per-Trade Risk**: Percentage of account risked per trade
   ```
   RISK_PER_TRADE=1.0  # in .env file (1% of account)
   ```

2. **Maximum Open Positions**: Limit concurrent exposure
   ```
   MAX_OPEN_TRADES=5  # in .env file
   ```

3. **Stop Loss and Take Profit**: Default values in pips
   ```
   DEFAULT_STOP_LOSS=50
   DEFAULT_TAKE_PROFIT=100
   ```

4. **Strategy-Specific Risk**: Different risk profiles per strategy
   ```yaml
   risk_per_strategy:
     rsi_reversal: 0.025    # 2.5% risk for best performer
     bollinger_bands: 0.015 # 1.5% risk
   ```

### Monitoring Risk Metrics

The dashboard provides real-time risk metrics:
- Current exposure per currency
- Portfolio heat map
- Correlation matrix
- Value at Risk (VaR)
- Expected Shortfall (ES)
- Margin utilization

## 8. Troubleshooting

### Common Issues

1. **API Connection Failures**:
   - Verify API credentials in .env file
   - Check internet connection
   - Ensure OANDA service is operational

2. **Strategy Not Trading**:
   - Check if the strategy is active in config.yaml
   - Verify market conditions meet strategy criteria
   - Check logs for trade decision details

3. **Dashboard Not Showing Data**:
   - Ensure the main system is running
   - Check log files for errors
   - Verify data connections in monitoring_config.json

### Log Files

Logs provide detailed information for troubleshooting:
- `forex_trading.log`: Main system log
- `monitoring.log`: Dashboard log
- `wallet_manager.log`: Trade execution log

### Getting Support

For additional support, contact the development team or refer to the official documentation. 