#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtesting Script for Forex Trading Strategies

This script backtests multiple technical analysis strategies on real historical EUR/USD data
from the OANDA API and compares their performance metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import project modules
from market_data_agent.agent import MarketDataAgent
from technical_analyst_agent.agent import TechnicalAnalystAgent
from utils.config_manager import ConfigManager

# Constants
INSTRUMENT = "EUR_USD"
TIMEFRAME = "H1"
LOOKBACK_DAYS = 365  # 1 year of data
START_BALANCE = 10000  # Initial account balance for backtesting

# List of strategies to test
STRATEGIES = [
    "moving_average_crossover",
    "rsi_reversal",
    "bollinger_bands",
    "macd",
    "breakout"
]

# Strategy parameters for testing
STRATEGY_PARAMS = {
    "moving_average_crossover": [
        {"fast_period": 10, "slow_period": 50},
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 20, "slow_period": 100}
    ],
    "rsi_reversal": [
        {"rsi_period": 14, "oversold": 30, "overbought": 70},
        {"rsi_period": 7, "oversold": 25, "overbought": 75},
        {"rsi_period": 21, "oversold": 35, "overbought": 65}
    ],
    "bollinger_bands": [
        {"period": 20, "std_dev": 2.0},
        {"period": 10, "std_dev": 1.5},
        {"period": 30, "std_dev": 2.5}
    ],
    "macd": [
        {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        {"fast_period": 8, "slow_period": 17, "signal_period": 9},
        {"fast_period": 16, "slow_period": 34, "signal_period": 9}
    ],
    "breakout": [
        {"period": 20, "threshold": 0.01},
        {"period": 10, "threshold": 0.005},
        {"period": 40, "threshold": 0.02}
    ]
}


def load_historical_data() -> pd.DataFrame:
    """
    Load historical EUR/USD data from OANDA API using MarketDataAgent.
    
    Returns:
        pd.DataFrame: Historical price data
    """
    print(f"Fetching historical {INSTRUMENT} data from OANDA API...")
    
    try:
        # Initialize ConfigManager to load credentials from .env
        config_manager = ConfigManager()
        
        # Create MarketDataAgent
        market_data_agent = MarketDataAgent()
        
        # Initialize agent to connect to OANDA
        success = market_data_agent.initialize()
        if not success:
            raise Exception("Failed to initialize MarketDataAgent")
        
        # Calculate date range for historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=LOOKBACK_DAYS)
        
        # Get historical data
        df = market_data_agent.get_historical_data(
            instrument=INSTRUMENT,
            timeframe=TIMEFRAME,
            count=5000,  # Maximum allowed by OANDA
            from_time=start_time,
            to_time=end_time
        )
        
        if df.empty:
            raise Exception(f"No data returned from OANDA for {INSTRUMENT}")
            
        print(f"Successfully retrieved {len(df)} candles from OANDA API")
        return df
    
    except Exception as e:
        print(f"Error fetching data from OANDA API: {str(e)}")
        raise


def backtest_strategy(data: pd.DataFrame, strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtest a single strategy with specified parameters.
    
    Args:
        data: Historical price data
        strategy: Strategy name
        params: Strategy parameters
        
    Returns:
        Dict containing backtest results and performance metrics
    """
    # Create a TechnicalAnalystAgent
    technical_analyst = TechnicalAnalystAgent()
    
    # Run the strategy backtest
    backtest_results = technical_analyst.backtest_strategy(data, strategy, params)
    
    # Create equity curve and calculate additional metrics
    positions = backtest_results['positions']
    equity_curve = pd.Series(START_BALANCE, index=data.index, dtype=float)
    returns = backtest_results['strategy_returns']
    
    # Calculate cumulative equity
    for i in range(1, len(data)):
        equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + returns.iloc[i])
    
    # Extract key metrics
    metrics = {
        'strategy': strategy,
        'parameters': params,
        'total_trades': backtest_results['total_trades'],
        'win_rate': backtest_results['win_rate'],
        'profit_factor': backtest_results.get('profit_factor', 0),
        'sharpe_ratio': backtest_results['sharpe_ratio'],
        'max_drawdown': backtest_results['max_drawdown'],
        'cagr': backtest_results['cagr'],
        'final_return': backtest_results['final_return'],
        'final_equity': equity_curve.iloc[-1]
    }
    
    # Add the equity curve for plotting
    backtest_results['equity_curve'] = equity_curve
    
    return {
        'metrics': metrics,
        'backtest_results': backtest_results
    }


def run_backtests(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Run backtests for all strategies and parameter combinations.
    
    Args:
        data: Historical price data
        
    Returns:
        List of backtest results
    """
    results = []
    
    # For each strategy and parameter combination
    for strategy in STRATEGIES:
        print(f"\nTesting strategy: {strategy}")
        
        for params in STRATEGY_PARAMS[strategy]:
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            print(f"  Parameters: {param_str}")
            
            # Run backtest
            result = backtest_strategy(data, strategy, params)
            results.append(result)
            
            # Print key metrics
            metrics = result['metrics']
            print(f"    Win Rate: {metrics['win_rate']:.2%}")
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"    Final Return: {metrics['final_return']:.2%}")
            print(f"    Final Equity: ${metrics['final_equity']:.2f}")
    
    return results


def plot_equity_curves(results: List[Dict[str, Any]], data: pd.DataFrame):
    """
    Plot equity curves for all strategies.
    
    Args:
        results: List of backtest results
        data: Historical price data
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the buy and hold strategy for comparison
    buy_hold_returns = data['close'] / data['close'].iloc[0]
    buy_hold_equity = START_BALANCE * buy_hold_returns
    plt.plot(data.index, buy_hold_equity, 'k--', label=f'Buy & Hold ({buy_hold_returns.iloc[-1]:.2f}x)')
    
    # Plot each strategy
    for result in results:
        metrics = result['metrics']
        equity_curve = result['backtest_results']['equity_curve']
        
        strategy_name = metrics['strategy']
        params_str = ", ".join([f"{k}={v}" for k, v in metrics['parameters'].items()])
        label = f"{strategy_name} ({params_str}) ({metrics['final_return']:.2%})"
        
        plt.plot(data.index, equity_curve, label=label)
    
    plt.title(f'Strategy Equity Curves - {INSTRUMENT} {TIMEFRAME}')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize='small')
    
    # Save the plot
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'backtest_strategies_{INSTRUMENT}_{TIMEFRAME}.png')
    print(f"\nEquity curve plot saved to plots/backtest_strategies_{INSTRUMENT}_{TIMEFRAME}.png")
    
    # Close the plot to prevent display
    plt.close()


def rank_strategies(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Rank strategies based on various performance metrics.
    
    Args:
        results: List of backtest results
        
    Returns:
        DataFrame with strategy rankings
    """
    # Extract metrics
    metrics_list = [r['metrics'] for r in results]
    
    # Create DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Convert parameters to string for better display
    df['parameters_str'] = df['parameters'].apply(lambda x: ", ".join([f"{k}={v}" for k, v in x.items()]))
    
    # Create strategy name with parameters
    df['strategy_name'] = df['strategy'] + " (" + df['parameters_str'] + ")"
    
    # Select columns for display and sort by Sharpe ratio
    display_cols = ['strategy_name', 'total_trades', 'win_rate', 'sharpe_ratio', 
                    'max_drawdown', 'final_return', 'final_equity']
    
    # Return sorted DataFrame
    return df[display_cols].sort_values('sharpe_ratio', ascending=False)


def main():
    """
    Main function to run the backtest.
    """
    print(f"Starting backtesting for {INSTRUMENT} on {TIMEFRAME} timeframe...")
    
    try:
        # Load historical data from OANDA
        data = load_historical_data()
        if data is None or len(data) == 0:
            print("Error: Could not load historical data")
            return
        
        print(f"Loaded {len(data)} periods of historical data")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Run backtests
        results = run_backtests(data)
        
        # Plot results
        plot_equity_curves(results, data)
        
        # Rank strategies
        rankings = rank_strategies(results)
        
        # Print rankings
        print("\nStrategy Rankings (by Sharpe Ratio):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(rankings)
        
        # Print best strategy
        best_strategy = rankings.iloc[0]
        print("\nBest Performing Strategy:")
        print(f"  {best_strategy['strategy_name']}")
        print(f"  Win Rate: {best_strategy['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {best_strategy['max_drawdown']:.2%}")
        print(f"  Final Return: {best_strategy['final_return']:.2%}")
        print(f"  Final Equity: ${best_strategy['final_equity']:.2f}")
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")


if __name__ == "__main__":
    main() 