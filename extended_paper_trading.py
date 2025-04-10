#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended Paper Trading Simulation

This script simulates paper trading for a week using historical data,
applying the best strategies from the configuration file.
It generates a comprehensive performance report at the end.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables for API access
load_dotenv()

# Constants
INITIAL_BALANCE = 10000  # Initial trading balance
RISK_PER_TRADE = 0.02    # Risk 2% per trade
SIMULATION_DAYS = 7      # Simulate a week of trading
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_URL = os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")

class OandaClient:
    """
    Simple OANDA API client for fetching historical data
    """
    
    def __init__(self):
        """Initialize OANDA API client"""
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
    """Technical analysis indicators and strategies"""
    
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
        
class PaperTrader:
    """
    Paper trading simulator using historical data.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml', initial_balance: float = INITIAL_BALANCE):
        """
        Initialize the paper trader.
        
        Args:
            config_path: Path to configuration file
            initial_balance: Initial account balance
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Store strategy settings
        self.strategy_config = self.config['technical_analysis']['strategies']
        self.active_strategies = self.strategy_config.get('active_strategies', [])
        
        # Initialize portfolio
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # Symbol -> {position_size, entry_price, direction}
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.trade_id = 1
        
        # Technical analysis tools
        self.ta = TechnicalAnalysis()
        
        # Data fetcher
        self.client = OandaClient()
        
        # Ensure plots directory exists
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        
        print(f"Paper trader initialized with {len(self.active_strategies)} strategies")
        print(f"Active strategies: {', '.join(self.active_strategies)}")
        
    def get_position_size(self, symbol: str, price: float, risk_amount: float) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            symbol: Trading instrument
            price: Current price
            risk_amount: Amount to risk in account currency
            
        Returns:
            Position size in units
        """
        # For simplicity, let's assume a 1% stop loss distance
        stop_distance = price * 0.01
        
        # Calculate position size
        position_size = risk_amount / stop_distance
        
        return position_size
    
    def apply_strategy(self, strategy: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a trading strategy to historical data.
        
        Args:
            strategy: Strategy name
            data: Historical price data
            params: Strategy parameters
            
        Returns:
            Dictionary with trading signals
        """
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        current_position = 0
        
        if strategy == 'moving_average_crossover':
            # Extract parameters
            fast_period = params.get('fast_period', 10)
            slow_period = params.get('slow_period', 50)
            
            # Calculate indicators
            fast_ma = self.ta.calculate_ema(data, fast_period)
            slow_ma = self.ta.calculate_ema(data, slow_period)
            
            # Generate signals
            for i in range(1, len(data)):
                # Buy signal: fast MA crosses above slow MA
                if fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
                    signals.iloc[i] = 1
                    current_position = 1
                
                # Sell signal: fast MA crosses below slow MA
                elif fast_ma.iloc[i-1] >= slow_ma.iloc[i-1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
                    signals.iloc[i] = -1
                    current_position = -1
                
                positions.iloc[i] = current_position
                
        elif strategy == 'rsi_reversal':
            # Extract parameters
            rsi_period = params.get('rsi_period', 14)
            oversold = params.get('oversold', 30)
            overbought = params.get('overbought', 70)
            
            # Calculate indicators
            rsi = self.ta.calculate_rsi(data, rsi_period)
            
            # Generate signals
            for i in range(1, len(data)):
                # Buy signal: RSI crosses above oversold threshold
                if rsi.iloc[i-1] <= oversold and rsi.iloc[i] > oversold:
                    signals.iloc[i] = 1
                    current_position = 1
                
                # Sell signal: RSI crosses below overbought threshold
                elif rsi.iloc[i-1] >= overbought and rsi.iloc[i] < overbought:
                    signals.iloc[i] = -1
                    current_position = -1
                
                positions.iloc[i] = current_position
                
        elif strategy == 'bollinger_bands':
            # Extract parameters
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            
            # Calculate indicators
            upper, middle, lower = self.ta.calculate_bollinger_bands(data, period, std_dev)
            
            # Generate signals
            for i in range(1, len(data)):
                # Buy signal: price crosses above lower band
                if data['close'].iloc[i-1] <= lower.iloc[i-1] and data['close'].iloc[i] > lower.iloc[i]:
                    signals.iloc[i] = 1
                    current_position = 1
                
                # Sell signal: price crosses below upper band
                elif data['close'].iloc[i-1] >= upper.iloc[i-1] and data['close'].iloc[i] < upper.iloc[i]:
                    signals.iloc[i] = -1
                    current_position = -1
                
                positions.iloc[i] = current_position
                
        elif strategy == 'macd':
            # Extract parameters
            fast_period = params.get('fast_period', 12)
            slow_period = params.get('slow_period', 26)
            signal_period = params.get('signal_period', 9)
            
            # Calculate indicators
            macd_line, signal_line, histogram = self.ta.calculate_macd(
                data, fast_period, slow_period, signal_period
            )
            
            # Generate signals
            for i in range(1, len(data)):
                # Buy signal: MACD line crosses above signal line
                if macd_line.iloc[i-1] <= signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
                    signals.iloc[i] = 1
                    current_position = 1
                
                # Sell signal: MACD line crosses below signal line
                elif macd_line.iloc[i-1] >= signal_line.iloc[i-1] and macd_line.iloc[i] < signal_line.iloc[i]:
                    signals.iloc[i] = -1
                    current_position = -1
                
                positions.iloc[i] = current_position
                
        elif strategy == 'breakout':
            # Extract parameters
            period = params.get('period', 20)
            threshold = params.get('threshold', 0.01)
            
            # Calculate resistance and support levels
            highs = data['high'].rolling(window=period).max()
            lows = data['low'].rolling(window=period).min()
            
            # Generate signals
            for i in range(period, len(data)):
                # Buy signal: price breaks above resistance
                if data['close'].iloc[i] > highs.iloc[i-1] * (1 + threshold):
                    signals.iloc[i] = 1
                    current_position = 1
                
                # Sell signal: price breaks below support
                elif data['close'].iloc[i] < lows.iloc[i-1] * (1 - threshold):
                    signals.iloc[i] = -1
                    current_position = -1
                
                positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions
        }
    
    def process_market_data(self, data: Dict[str, pd.DataFrame], simulation_date: datetime) -> None:
        """
        Process market data for the current simulation step.
        
        Args:
            data: Dictionary of market data by symbol
            simulation_date: Current simulation date
        """
        print(f"\nProcessing market data for {simulation_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Apply each active strategy to relevant symbols and timeframes
        for strategy in self.active_strategies:
            # Skip if strategy configuration is missing
            if strategy not in self.strategy_config:
                continue
                
            strategy_params = self.strategy_config[strategy]
            strategy_symbols = strategy_params.get('symbols', [])
            strategy_timeframes = strategy_params.get('timeframes', [])
            
            # Convert to OANDA format (replace / with _)
            oanda_symbols = [symbol.replace('/', '_') for symbol in strategy_symbols]
            
            print(f"Applying {strategy} to {', '.join(strategy_symbols)}")
            
            # Apply the strategy to each symbol and timeframe
            for symbol in oanda_symbols:
                if symbol not in data:
                    continue
                    
                symbol_data = data[symbol]
                
                # Apply strategy and get signals
                signals_dict = self.apply_strategy(strategy, symbol_data, strategy_params)
                signals = signals_dict['signals']
                
                # Get the signal for the current timestamp if it exists
                current_signal = 0
                if simulation_date in signals.index:
                    current_signal = signals.loc[simulation_date]
                
                if current_signal != 0:
                    self.execute_trade(symbol, current_signal, symbol_data.loc[simulation_date, 'close'], strategy)
    
    def execute_trade(self, symbol: str, signal: int, price: float, strategy: str) -> None:
        """
        Execute a trade based on the signal.
        
        Args:
            symbol: Trading instrument
            signal: Trade signal (1 for buy, -1 for sell)
            price: Current price
            strategy: Strategy that generated the signal
        """
        # Close any existing position in the opposite direction
        if symbol in self.positions and self.positions[symbol]['direction'] != signal:
            self.close_position(symbol, price)
        
        # Open a new position if we don't have one already
        if symbol not in self.positions:
            # Calculate position size and risk amount
            risk_amount = self.balance * RISK_PER_TRADE
            position_size = self.get_position_size(symbol, price, risk_amount)
            
            # Record the trade
            trade = {
                'id': self.trade_id,
                'symbol': symbol,
                'direction': "BUY" if signal > 0 else "SELL",
                'entry_price': price,
                'position_size': position_size,
                'entry_time': datetime.now(),
                'exit_price': None,
                'exit_time': None,
                'profit_loss': None,
                'strategy': strategy
            }
            
            self.trades.append(trade)
            self.trade_id += 1
            
            # Update positions
            self.positions[symbol] = {
                'position_size': position_size,
                'entry_price': price,
                'direction': signal,
                'trade_id': trade['id']
            }
            
            print(f"Opened {trade['direction']} position in {symbol} at {price:.5f}, size: {position_size:.2f}")
    
    def close_position(self, symbol: str, price: float) -> None:
        """
        Close an open position.
        
        Args:
            symbol: Trading instrument
            price: Current price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Find the trade record
        trade = next((t for t in self.trades if t['id'] == position['trade_id']), None)
        
        if trade:
            # Calculate profit/loss
            if position['direction'] > 0:  # Long position
                profit_loss = (price - position['entry_price']) * position['position_size']
            else:  # Short position
                profit_loss = (position['entry_price'] - price) * position['position_size']
            
            # Update trade record
            trade['exit_price'] = price
            trade['exit_time'] = datetime.now()
            trade['profit_loss'] = profit_loss
            
            # Update account balance
            self.balance += profit_loss
            
            print(f"Closed {trade['direction']} position in {symbol} at {price:.5f}, P/L: {profit_loss:.2f}")
        
        # Remove the position
        del self.positions[symbol]
    
    def update_equity(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> None:
        """
        Update the equity curve with the current balance plus unrealized P/L.
        
        Args:
            data: Current market data
            timestamp: Current simulation timestamp
        """
        equity = self.balance
        
        # Add unrealized P/L from open positions
        for symbol, position in self.positions.items():
            if symbol in data and timestamp in data[symbol].index:
                current_price = data[symbol].loc[timestamp, 'close']
                
                if position['direction'] > 0:  # Long position
                    unrealized_pl = (current_price - position['entry_price']) * position['position_size']
                else:  # Short position
                    unrealized_pl = (position['entry_price'] - current_price) * position['position_size']
                    
                equity += unrealized_pl
        
        # Record to equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
    
    def simulate(self, start_date: datetime, end_date: datetime) -> None:
        """
        Run the paper trading simulation.
        
        Args:
            start_date: Simulation start date
            end_date: Simulation end date
        """
        print(f"Starting paper trading simulation from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get unique symbols from all strategies
        all_symbols = set()
        for strategy in self.active_strategies:
            if strategy in self.strategy_config:
                strategy_symbols = self.strategy_config[strategy].get('symbols', [])
                all_symbols.update([symbol.replace('/', '_') for symbol in strategy_symbols])
        
        # Get historical data for each symbol
        data = {}
        for symbol in all_symbols:
            try:
                # Get data with some lookback for indicator calculations
                lookback_start = start_date - timedelta(days=30)  # 30 days lookback for indicator calculation
                symbol_data = self.client.get_historical_candles(
                    instrument=symbol,
                    granularity="H1",  # Use hourly data
                    from_date=lookback_start,
                    to_date=end_date
                )
                
                if not symbol_data.empty:
                    data[symbol] = symbol_data
                    print(f"Downloaded {len(symbol_data)} candles for {symbol}")
                else:
                    print(f"No data available for {symbol}")
            except Exception as e:
                print(f"Error getting data for {symbol}: {str(e)}")
        
        if not data:
            print("No data available for simulation. Exiting.")
            return
        
        # Create a list of unique timestamps across all data
        all_timestamps = set()
        for symbol_data in data.values():
            all_timestamps.update(symbol_data.index)
        
        # Filter to simulation period and sort
        simulation_timestamps = sorted([ts for ts in all_timestamps if start_date <= ts <= end_date])
        
        if not simulation_timestamps:
            print("No timestamps in simulation period. Exiting.")
            return
        
        print(f"Running simulation with {len(simulation_timestamps)} time steps")
        
        # Run the simulation time step by time step
        for timestamp in simulation_timestamps:
            # Process market data and execute trades
            self.process_market_data(data, timestamp)
            
            # Update equity curve
            self.update_equity(data, timestamp)
        
        # Close any remaining positions at the end
        for symbol in list(self.positions.keys()):
            if symbol in data and simulation_timestamps[-1] in data[symbol].index:
                self.close_position(symbol, data[symbol].loc[simulation_timestamps[-1], 'close'])
        
        # Generate the performance report
        self.generate_report()
    
    def generate_report(self) -> None:
        """
        Generate a performance report for the simulation.
        """
        print("\n===== PAPER TRADING SIMULATION REPORT =====")
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty:
            print("No equity data to report.")
            return
            
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate performance metrics
        initial_equity = self.initial_balance
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / initial_equity) - 1
        
        # Calculate daily returns
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        # Calculate metrics
        sharpe_ratio = 0
        if not equity_df['daily_return'].iloc[1:].empty:
            avg_daily_return = equity_df['daily_return'].iloc[1:].mean()
            daily_std = equity_df['daily_return'].iloc[1:].std()
            if daily_std > 0:
                sharpe_ratio = (avg_daily_return / daily_std) * np.sqrt(252)  # Annualized Sharpe ratio
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak']) - 1
        max_drawdown = equity_df['drawdown'].min()
        
        # Trade statistics
        num_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['profit_loss'] and trade['profit_loss'] > 0)
        losing_trades = sum(1 for trade in self.trades if trade['profit_loss'] and trade['profit_loss'] < 0)
        
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        total_profit = sum(trade['profit_loss'] for trade in self.trades if trade['profit_loss'] and trade['profit_loss'] > 0)
        total_loss = sum(trade['profit_loss'] for trade in self.trades if trade['profit_loss'] and trade['profit_loss'] < 0)
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Print report
        print(f"\nInitial Balance: ${initial_equity:.2f}")
        print(f"Final Balance: ${final_equity:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"\nTotal Trades: {num_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.2%})")
        print(f"Losing Trades: {losing_trades}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Print trades summary by strategy
        strategy_performance = {}
        for trade in self.trades:
            if trade['strategy'] not in strategy_performance:
                strategy_performance[trade['strategy']] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0
                }
            
            perf = strategy_performance[trade['strategy']]
            perf['trades'] += 1
            
            if trade['profit_loss']:
                if trade['profit_loss'] > 0:
                    perf['wins'] += 1
                else:
                    perf['losses'] += 1
                    
                perf['profit'] += trade['profit_loss']
        
        print("\nStrategy Performance:")
        for strategy, perf in strategy_performance.items():
            win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            print(f"  {strategy}:")
            print(f"    Trades: {perf['trades']}")
            print(f"    Win Rate: {win_rate:.2%}")
            print(f"    Net Profit: ${perf['profit']:.2f}")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('plots/equity_curve_paper_trading.png')
        print("\nEquity curve saved to plots/equity_curve_paper_trading.png")
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        plt.fill_between(equity_df.index, equity_df['drawdown'] * 100, 0, color='red', alpha=0.3)
        plt.title('Drawdown (%)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('plots/drawdown_paper_trading.png')
        print("Drawdown chart saved to plots/drawdown_paper_trading.png")
        
        # Save trade history to CSV
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv('plots/trade_history.csv', index=False)
            print("Trade history saved to plots/trade_history.csv")

def main():
    """
    Main function to run the extended paper trading simulation.
    """
    print("=== Extended Paper Trading Simulation ===")
    
    try:
        # Use a fixed historical date range to ensure data availability
        # Using a 7-day period from the recent past
        end_time = datetime(2023, 12, 31, 0, 0, 0, tzinfo=timezone.utc)  # Use Dec 31, 2023 as end date with UTC timezone
        start_time = end_time - timedelta(days=SIMULATION_DAYS)
        
        print(f"Using historical data period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # Initialize paper trader
        trader = PaperTrader()
        
        # Run simulation
        trader.simulate(start_time, end_time)
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 