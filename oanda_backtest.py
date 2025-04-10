#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone OANDA Backtesting Script for Forex Trading Strategies

This script backtests multiple technical analysis strategies on real historical EUR/USD data
from the OANDA API and compares their performance metrics.
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
INSTRUMENT = "EUR_USD"
TIMEFRAME = "H1"
LOOKBACK_DAYS = 365  # 1 year of data
START_BALANCE = 10000  # Initial account balance for backtesting

# OANDA API credentials
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_URL = os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")

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


class OandaClient:
    """
    Simple OANDA API client for fetching historical data
    """
    
    def __init__(self):
        """
        Initialize OANDA API client
        """
        self.api_key = OANDA_API_KEY
        self.account_id = OANDA_ACCOUNT_ID
        self.api_url = OANDA_API_URL
        
        # Validate credentials
        if not all([self.api_key, self.account_id, self.api_url]):
            raise ValueError("Missing OANDA API credentials in .env file")
        
        # Set up request headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_historical_candles(self, 
                             instrument: str, 
                             granularity: str = "H1",
                             count: Optional[int] = None,
                             from_date: Optional[datetime] = None,
                             to_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical candle data for an instrument.
        
        Args:
            instrument: The instrument name (e.g., "EUR_USD")
            granularity: The candle granularity (e.g., "M1", "H1", "D")
            count: Number of candles to retrieve (max 5000)
            from_date: Start date for data retrieval
            to_date: End date for data retrieval
            
        Returns:
            DataFrame with OHLC price data
        """
        url = f"{self.api_url}/v3/instruments/{instrument}/candles"
        
        # Build query parameters
        params = {
            "price": "M",  # Midpoint candles
            "granularity": granularity
        }
        
        # Add time range parameters
        if count is not None:
            params["count"] = min(count, 5000)  # OANDA limit is 5000
        elif from_date is not None and to_date is not None:
            # Format datetime objects to RFC3339 format as required by OANDA API
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif from_date is not None:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif to_date is not None:
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            # Default to 500 candles if no range specified
            params["count"] = 500
        
        # Make the API request
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get historical data: {response.text}")
        
        # Parse response JSON
        data = response.json()
        candles = data.get("candles", [])
        
        if not candles:
            return pd.DataFrame()
        
        # Build DataFrame
        df_data = []
        
        for candle in candles:
            # Skip incomplete candles
            if candle["complete"] is False:
                continue
                
            timestamp = datetime.fromisoformat(candle["time"].replace("Z", "+00:00"))
            mid = candle["mid"]
            
            df_data.append({
                "timestamp": timestamp,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(candle["volume"])
            })
        
        # Create DataFrame and set timestamp as index
        df = pd.DataFrame(df_data)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
        
        return df


class TechnicalAnalysis:
    """
    Technical analysis indicators and strategies
    """
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        fast_ema = TechnicalAnalysis.calculate_ema(data, fast_period)
        slow_ema = TechnicalAnalysis.calculate_ema(data, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = TechnicalAnalysis.calculate_sma(data, period)
        std = data['close'].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band


class Backtester:
    """
    Strategy backtester
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize backtester with historical data
        
        Args:
            data: Historical price data
        """
        self.data = data
        self.ta = TechnicalAnalysis()
    
    def ma_crossover_strategy(self, fast_period: int = 10, slow_period: int = 50) -> Dict[str, pd.Series]:
        """
        Implement Moving Average Crossover strategy.
        
        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate moving averages
        fast_ma = TechnicalAnalysis.calculate_ema(self.data, fast_period)
        slow_ma = TechnicalAnalysis.calculate_ema(self.data, slow_period)
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        positions = pd.Series(0, index=self.data.index)
        entry_prices = pd.Series(np.nan, index=self.data.index)
        exit_prices = pd.Series(np.nan, index=self.data.index)
        
        current_position = 0
        
        for i in range(1, len(self.data)):
            # Buy signal: fast MA crosses above slow MA
            if fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: fast MA crosses below slow MA
            elif fast_ma.iloc[i-1] >= slow_ma.iloc[i-1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma
        }
    
    def rsi_reversal_strategy(self, rsi_period: int = 14, 
                             oversold: int = 30, overbought: int = 70) -> Dict[str, pd.Series]:
        """
        Implement RSI Reversal strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            oversold: RSI threshold for oversold condition
            overbought: RSI threshold for overbought condition
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate RSI
        rsi = TechnicalAnalysis.calculate_rsi(self.data, rsi_period)
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        positions = pd.Series(0, index=self.data.index)
        entry_prices = pd.Series(np.nan, index=self.data.index)
        exit_prices = pd.Series(np.nan, index=self.data.index)
        
        current_position = 0
        
        for i in range(1, len(self.data)):
            # Buy signal: RSI crosses above oversold threshold
            if rsi.iloc[i-1] <= oversold and rsi.iloc[i] > oversold:
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: RSI crosses below overbought threshold
            elif rsi.iloc[i-1] >= overbought and rsi.iloc[i] < overbought:
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'rsi': rsi
        }
    
    def bollinger_bands_strategy(self, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Implement Bollinger Bands strategy.
        
        Args:
            period: Period for moving average calculation
            std_dev: Number of standard deviations for the bands
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate Bollinger Bands
        upper, middle, lower = TechnicalAnalysis.calculate_bollinger_bands(self.data, period, std_dev)
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        positions = pd.Series(0, index=self.data.index)
        entry_prices = pd.Series(np.nan, index=self.data.index)
        exit_prices = pd.Series(np.nan, index=self.data.index)
        
        current_position = 0
        
        for i in range(1, len(self.data)):
            # Buy signal: price crosses above lower band
            if self.data['close'].iloc[i-1] <= lower.iloc[i-1] and self.data['close'].iloc[i] > lower.iloc[i]:
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: price crosses below upper band
            elif self.data['close'].iloc[i-1] >= upper.iloc[i-1] and self.data['close'].iloc[i] < upper.iloc[i]:
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'upper_band': upper,
            'middle_band': middle,
            'lower_band': lower
        }
    
    def macd_strategy(self, fast_period: int = 12, slow_period: int = 26, 
                     signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Implement MACD strategy.
        
        Args:
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalAnalysis.calculate_macd(
            self.data, fast_period, slow_period, signal_period
        )
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        positions = pd.Series(0, index=self.data.index)
        entry_prices = pd.Series(np.nan, index=self.data.index)
        exit_prices = pd.Series(np.nan, index=self.data.index)
        
        current_position = 0
        
        for i in range(1, len(self.data)):
            # Buy signal: MACD line crosses above signal line
            if macd_line.iloc[i-1] <= signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: MACD line crosses below signal line
            elif macd_line.iloc[i-1] >= signal_line.iloc[i-1] and macd_line.iloc[i] < signal_line.iloc[i]:
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    def breakout_strategy(self, period: int = 20, threshold: float = 0.01) -> Dict[str, pd.Series]:
        """
        Implement Breakout strategy.
        
        Args:
            period: Period for identifying support/resistance
            threshold: Threshold for confirming breakout
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate resistance and support levels
        highs = self.data['high'].rolling(window=period).max()
        lows = self.data['low'].rolling(window=period).min()
        
        # Generate signals
        signals = pd.Series(0, index=self.data.index)
        positions = pd.Series(0, index=self.data.index)
        entry_prices = pd.Series(np.nan, index=self.data.index)
        exit_prices = pd.Series(np.nan, index=self.data.index)
        
        current_position = 0
        
        for i in range(period, len(self.data)):
            # Buy signal: price breaks above resistance
            if self.data['close'].iloc[i] > highs.iloc[i-1] * (1 + threshold):
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: price breaks below support
            elif self.data['close'].iloc[i] < lows.iloc[i-1] * (1 - threshold):
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = self.data['close'].iloc[i]
                    current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'highs': highs,
            'lows': lows
        }
    
    def implement_strategy(self, strategy: str, params: Dict[str, Any] = None) -> Dict[str, pd.Series]:
        """
        Apply a predefined strategy.
        
        Args:
            strategy: Name of the strategy to implement
            params: Parameters for the strategy (default: None, use strategy defaults)
            
        Returns:
            Dictionary with 'signals', 'positions', 'entry_prices', and 'exit_prices' Series
        """
        # Define available strategies
        strategies = {
            'moving_average_crossover': self.ma_crossover_strategy,
            'rsi_reversal': self.rsi_reversal_strategy,
            'bollinger_bands': self.bollinger_bands_strategy,
            'macd': self.macd_strategy,
            'breakout': self.breakout_strategy
        }
        
        # Check if strategy exists
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available strategies: {list(strategies.keys())}")
        
        # Get strategy parameters (use defaults if not provided)
        strategy_params = params or {}
        
        # Execute the strategy
        strategy_function = strategies[strategy]
        return strategy_function(**strategy_params)
    
    def backtest_strategy(self, strategy: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Test a strategy on historical data.
        
        Args:
            strategy: Name of the strategy to backtest
            params: Strategy parameters (default: None)
            
        Returns:
            Dictionary with backtest results
        """
        # Implement the strategy
        results = self.implement_strategy(strategy, params)
        
        # Extract signals and positions
        signals = results['signals']
        positions = results['positions']
        entry_prices = results['entry_prices']
        exit_prices = results['exit_prices']
        
        # Calculate returns
        strategy_returns = pd.Series(0.0, index=self.data.index)
        cumulative_returns = pd.Series(1.0, index=self.data.index)
        
        # Loop through the positions and calculate returns
        for i in range(1, len(self.data)):
            if positions.iloc[i-1] != 0:
                # Calculate returns based on position
                if positions.iloc[i-1] > 0:  # Long position
                    strategy_returns.iloc[i] = (self.data['close'].iloc[i] / self.data['close'].iloc[i-1]) - 1
                else:  # Short position
                    strategy_returns.iloc[i] = 1 - (self.data['close'].iloc[i] / self.data['close'].iloc[i-1])
                
                # Apply position size
                strategy_returns.iloc[i] *= positions.iloc[i-1]
            
            # Calculate cumulative returns
            cumulative_returns.iloc[i] = cumulative_returns.iloc[i-1] * (1 + strategy_returns.iloc[i])
        
        # Calculate buy & hold returns for comparison
        buy_hold_returns = (self.data['close'] / self.data['close'].iloc[0])
        
        # Calculate performance metrics
        total_trades = sum(abs(signals) > 0)
        
        # Count profitable trades
        profitable_trades = 0
        if total_trades > 0:
            for i in range(len(signals)):
                if signals.iloc[i] != 0:
                    # Find the price delta for this trade
                    if signals.iloc[i] > 0:  # Buy signal
                        entry = self.data['close'].iloc[i]
                        # Find next sell signal or use last price
                        for j in range(i+1, len(signals)):
                            if signals.iloc[j] < 0:
                                exit = self.data['close'].iloc[j]
                                if exit > entry:  # Profitable long trade
                                    profitable_trades += 1
                                break
                    elif signals.iloc[i] < 0:  # Sell signal
                        entry = self.data['close'].iloc[i]
                        # Find next buy signal or use last price
                        for j in range(i+1, len(signals)):
                            if signals.iloc[j] > 0:
                                exit = self.data['close'].iloc[j]
                                if exit < entry:  # Profitable short trade
                                    profitable_trades += 1
                                break
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate maximum drawdown
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (annualized)
        risk_free_rate = 0.0  # Assume zero risk-free rate for simplicity
        sharpe_ratio = np.sqrt(252) * (strategy_returns.mean() - risk_free_rate) / strategy_returns.std() if strategy_returns.std() > 0 else 0
        
        # Calculate CAGR (Compound Annual Growth Rate)
        days = (self.data.index[-1] - self.data.index[0]).days
        cagr = (cumulative_returns.iloc[-1] ** (365 / days)) - 1 if days > 0 else 0
        
        # Prepare results
        backtest_results = {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'strategy_returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'buy_hold_returns': buy_hold_returns,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'cagr': cagr,
            'final_return': cumulative_returns.iloc[-1] - 1
        }
        
        # Add strategy-specific results
        for key, value in results.items():
            if key not in backtest_results:
                backtest_results[key] = value
        
        return backtest_results


def load_historical_data() -> pd.DataFrame:
    """
    Load historical EUR/USD data from OANDA API.
    
    Returns:
        pd.DataFrame: Historical price data
    """
    print(f"Fetching historical {INSTRUMENT} data from OANDA API...")
    
    # Initialize OANDA client
    oanda_client = OandaClient()
    
    # Calculate date range for historical data
    # Use current time but set to the start of the current hour to avoid future date issues
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=LOOKBACK_DAYS)
    
    # Get historical data
    df = oanda_client.get_historical_candles(
        instrument=INSTRUMENT,
        granularity=TIMEFRAME,
        from_date=start_time,
        to_date=end_time
    )
    
    if df.empty:
        raise Exception(f"No data returned from OANDA for {INSTRUMENT}")
        
    print(f"Successfully retrieved {len(df)} candles from OANDA API")
    return df


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
    # Create a backtester
    backtester = Backtester(data)
    
    # Run the strategy backtest
    backtest_results = backtester.backtest_strategy(strategy, params)
    
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
        'profit_factor': 0,  # Not calculated here
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