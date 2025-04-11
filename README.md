# Forex Trading Platform

A comprehensive AI-powered forex trading platform that supports live trading, paper trading, backtesting, and strategy optimization.

## Overview

The Forex Trading Platform is a sophisticated trading system that integrates multiple AI agents to analyze markets, execute trades, and manage risk. The platform includes:

- Multi-agent architecture for market data, technical analysis, fundamental analysis, sentiment analysis, and portfolio management
- Advanced risk management capabilities
- Backtesting and strategy optimization tools
- Real-time monitoring dashboard
- Support for both paper trading and live trading via OANDA

## Features

- **AI-Powered Analysis**: Combines technical, fundamental, and sentiment analysis to generate trading signals
- **Risk Management**: Sophisticated risk controls with per-trade, per-currency, and portfolio-level risk parameters
- **Multiple Trading Modes**: Live trading, paper trading, backtesting, and strategy optimization
- **Comprehensive Monitoring**: Real-time dashboard for system status, trading performance, and market conditions
- **Configurable Strategies**: Easily customize and deploy trading strategies with optimized parameters

## Quick Start

```bash
# Clone the repository
git clone https://github.com/imashishkh/forex-trading.git
cd forex-trading

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.template .env
# Edit .env with your API credentials

# Start in paper trading mode
python main.py start --mode paper
```

## Documentation

For full documentation on installation, configuration, and usage, see:

- [User Manual](docs/USER_MANUAL.md): Comprehensive guide to using the platform
- [API Documentation](docs/API.md): API reference for developers
- [Strategy Guide](docs/STRATEGIES.md): Guide to implementing custom strategies

## System Architecture

The platform consists of several integrated agents:

1. **Market Data Agent**: Fetches and processes market data
2. **Technical Analysis Agent**: Analyzes price patterns and generates signals
3. **Fundamental Analysis Agent**: Analyzes economic indicators and news
4. **Sentiment Analysis Agent**: Analyzes market sentiment from various sources
5. **Risk Manager Agent**: Manages risk parameters and exposure
6. **Portfolio Manager Agent**: Executes trades and manages positions

## Portfolio Manager Agent

The Portfolio Manager Agent is responsible for trade execution and portfolio tracking in the Forex Trading Platform. It handles all aspects of managing a trading portfolio, including:

- Executing trades based on signals from other agents
- Managing risk and position sizing
- Tracking and updating portfolio positions
- Calculating performance metrics

### Portfolio Manager Features

- Smart trade execution based on multiple signals
- Position management with stop-loss, take-profit, and trailing stops
- Dynamic risk management with adjustable risk per trade
- Session-aware trading to focus on optimal market hours
- Comprehensive performance tracking and metrics calculation
- Command-line interface for managing the portfolio

### Portfolio Manager Usage

```bash
python -m portfolio_manager_agent run \
  --config config/portfolio_manager.json \
  --market-data data/market_data.json \
  --technical data/technical_signals.json \
  --fundamental data/fundamental_insights.json \
  --sentiment data/sentiment_insights.json \
  --risk data/risk_assessment.json \
  --output results/
```

## Performance Metrics

The platform calculates and tracks numerous performance metrics:

- Win rate
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Expectancy
- Risk-reward ratio
- Volatility
- Value at Risk (VaR)

## License

MIT

## Contact

For support or inquiries, please contact the development team 
