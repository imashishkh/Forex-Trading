# Portfolio Manager Agent Module for Forex Trading Platform

The Portfolio Manager Agent is responsible for trade execution and portfolio tracking in the Forex Trading Platform. It handles all aspects of managing a trading portfolio, including:

- Executing trades based on signals from other agents
- Managing risk and position sizing
- Tracking and updating portfolio positions
- Calculating performance metrics

## Features

- Smart trade execution based on multiple signals (technical, fundamental, sentiment)
- Position management with stop-loss, take-profit, and trailing stops
- Dynamic risk management with adjustable risk per trade
- Session-aware trading to focus on optimal market hours
- Comprehensive performance tracking and metrics calculation
- Command-line interface for managing the portfolio

## Installation

```bash
git clone https://github.com/username/forex-trading.git
cd forex-trading
pip install -r requirements.txt
```

## Usage

### Initialize Configuration

```bash
python -m portfolio_manager_agent init --output config/portfolio_manager.json
```

### Run the Portfolio Manager

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

### Check Portfolio Status

```bash
python -m portfolio_manager_agent status --config config/portfolio_manager.json
```

### Modify Configuration

```bash
python -m portfolio_manager_agent config \
  --config config/portfolio_manager.json \
  --set risk.max_risk_per_trade_pct 1.5
```

## Configuration

The configuration file allows for extensive customization of the portfolio manager:

```json
{
  "account_size": 10000,
  "leverage": 30,
  "max_open_positions": 5,
  "position_sizing_method": "risk_based",
  "risk": {
    "max_drawdown_pct": 20,
    "max_risk_per_trade_pct": 2,
    "min_risk_reward_ratio": 1.5,
    "min_margin_level": 200,
    "stop_loss_method": "atr",
    "take_profit_method": "risk_reward",
    "trailing_stop_activation": 1.0,
    "trailing_stop_distance": 0.5
  },
  "trading_sessions": {
    "enabled": true,
    "preferred_sessions": ["London-New York"],
    "session_volume_threshold": 0.7
  },
  ...
}
```

## Module Structure

- `agent.py`: Main agent implementation
- `config.py`: Configuration management
- `cli.py`: Command-line interface
- `__main__.py`: Entry point for running as a module

## Integration with Other Agents

The Portfolio Manager Agent integrates with other agents in the Forex Trading Platform:

1. **Technical Analysis Agent**: Provides technical signals for trade decisions
2. **Fundamental Analysis Agent**: Provides fundamental insights for trade decisions
3. **Sentiment Analysis Agent**: Provides sentiment insights for trade decisions
4. **Risk Manager Agent**: Provides risk assessment for position sizing and management

## Performance Metrics

The agent calculates and tracks numerous performance metrics:

- Win rate
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Expectancy
- Risk-reward ratio
- Volatility
- Value at Risk (VaR) 