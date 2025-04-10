#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Paper Trading Simulation

This script implements enhanced trading strategies with improved risk management
based on paper trading results. It includes:
- RSI strategy with MA confirmation
- Bollinger Bands with RSI filter
- MACD with histogram and trend filters
- MA Crossover with volume confirmation
- Enhanced risk management with strategy-specific position sizing
- Take profit, stop loss, and trailing stops
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
import copy

# Import enhanced technical analysis
from enhanced_technical_analysis import EnhancedTechnicalAnalysis, EnhancedStrategies

# Load environment variables for API access
load_dotenv()

# Constants
INITIAL_BALANCE = 10000  # Initial trading balance
SIMULATION_DAYS = 14     # Extended to 2 weeks for better results
MAX_OPEN_POSITIONS = 3   # Limit concurrent positions
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_URL = os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")


class OandaClient:
    """Simple OANDA API client for fetching historical data"""
    
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


class ImprovedPaperTrader:
    """
    Improved paper trading simulator with enhanced strategies and risk management.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml', initial_balance: float = INITIAL_BALANCE):
        """
        Initialize the improved paper trader.
        
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
        self.risk_per_strategy = self.strategy_config.get('risk_per_strategy', {})
        self.risk_management = self.strategy_config.get('risk_management', {})
        
        # Initialize portfolio
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # Symbol -> {position_size, entry_price, direction, etc.}
        self.open_position_count = 0
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.trade_id = 1
        
        # Technical analysis tools
        self.ta = EnhancedTechnicalAnalysis()
        self.strategies = EnhancedStrategies()
        
        # Data fetcher
        self.client = OandaClient()
        
        # Correlation tracking
        self.correlation_threshold = self.risk_management.get('correlation_threshold', 0.7)
        self.price_data = {}  # Store price data for correlation calculation
        
        # Ensure plots directory exists
        plots_dir = Path('plots/improved')
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Improved paper trader initialized with {len(self.active_strategies)} strategies")
        print(f"Active strategies: {', '.join(self.active_strategies)}")
        print(f"Risk management settings: {self.risk_management}")
        
    def get_position_size(self, symbol: str, price: float, risk_amount: float, 
                         atr_value: float) -> float:
        """
        Calculate position size based on risk and ATR.
        
        Args:
            symbol: Trading instrument
            price: Current price
            risk_amount: Amount to risk in account currency
            atr_value: Current ATR value
            
        Returns:
            Position size in units
        """
        # Use ATR for stop loss distance
        atr_multiple = self.risk_management.get('atr_period', 2.0)
        stop_distance = atr_value * atr_multiple
        
        # Ensure minimum stop distance
        min_stop_distance = price * 0.005  # 0.5% minimum stop
        stop_distance = max(stop_distance, min_stop_distance)
        
        # Calculate position size
        position_size = risk_amount / stop_distance
        
        return position_size
    
    def get_risk_amount(self, strategy: str) -> float:
        """
        Get risk amount for a specific strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Risk amount in account currency
        """
        # Get risk percentage for this strategy
        risk_percentage = self.risk_per_strategy.get(strategy, 0.01)  # Default to 1%
        
        # Calculate risk amount
        risk_amount = self.balance * risk_percentage
        
        return risk_amount
    
    def is_correlated_with_open_positions(self, symbol: str, 
                                        direction: int, timestamp: datetime) -> bool:
        """
        Check if a potential trade is correlated with existing positions.
        
        Args:
            symbol: Symbol to check
            direction: Trade direction (1 for long, -1 for short)
            timestamp: Current timestamp
            
        Returns:
            True if correlated above threshold, False otherwise
        """
        if not self.positions:
            return False
            
        # Convert the current symbol to the correlation format
        base_currency = symbol.split('_')[0]
        quote_currency = symbol.split('_')[1]
        
        for pos_symbol, position in self.positions.items():
            # Convert position symbol to components
            pos_base = pos_symbol.split('_')[0]
            pos_quote = pos_symbol.split('_')[1]
            
            # Check for shared currencies
            if base_currency in [pos_base, pos_quote] or quote_currency in [pos_base, pos_quote]:
                # Calculate correlation only if we have enough data
                if (symbol in self.price_data and pos_symbol in self.price_data and
                    len(self.price_data[symbol]) > 20 and len(self.price_data[pos_symbol]) > 20):
                    
                    # Calculate correlation
                    period = 20  # Use 20 periods for correlation
                    corr = self.price_data[symbol]['close'].tail(period).corr(
                        self.price_data[pos_symbol]['close'].tail(period)
                    )
                    
                    # Check if correlation exceeds threshold and direction is the same
                    if abs(corr) > self.correlation_threshold and direction == position['direction']:
                        print(f"Skipping {symbol} trade due to correlation {corr:.2f} with {pos_symbol}")
                        return True
        
        return False
    
    def apply_strategy(self, strategy: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the appropriate enhanced trading strategy.
        
        Args:
            strategy: Strategy name
            data: Historical price data
            params: Strategy parameters
            
        Returns:
            Dictionary with trading signals
        """
        if strategy == 'rsi_reversal':
            return self.strategies.enhanced_rsi_reversal(data, params)
        elif strategy == 'bollinger_bands':
            return self.strategies.enhanced_bollinger_bands(data, params)
        elif strategy == 'macd':
            return self.strategies.enhanced_macd(data, params)
        elif strategy == 'moving_average_crossover':
            return self.strategies.enhanced_ma_crossover(data, params)
        else:
            # Fallback to empty signals
            return {
                'signals': pd.Series(0, index=data.index),
                'positions': pd.Series(0, index=data.index)
            }
    
    def process_market_data(self, data: Dict[str, pd.DataFrame], simulation_date: datetime) -> None:
        """
        Process market data for the current simulation step.
        
        Args:
            data: Dictionary of market data by symbol
            simulation_date: Current simulation date
        """
        print(f"\nProcessing market data for {simulation_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Update price data for correlation calculation
        for symbol, df in data.items():
            if simulation_date in df.index:
                if symbol not in self.price_data:
                    self.price_data[symbol] = pd.DataFrame()
                
                # Append current price data
                current_data = df.loc[simulation_date].to_dict()
                current_data['timestamp'] = simulation_date
                
                new_row = pd.DataFrame([current_data])
                new_row.set_index('timestamp', inplace=True)
                
                self.price_data[symbol] = pd.concat([self.price_data[symbol], new_row])
                
                # Keep only the last 50 data points to save memory
                if len(self.price_data[symbol]) > 50:
                    self.price_data[symbol] = self.price_data[symbol].tail(50)
        
        # Check for time-based exits on existing positions
        self.check_time_based_exits(simulation_date)
        
        # Apply each active strategy to relevant symbols and timeframes
        for strategy in self.active_strategies:
            # Skip if strategy configuration is missing or if we've reached max positions
            if strategy not in self.strategy_config or self.open_position_count >= MAX_OPEN_POSITIONS:
                continue
                
            strategy_params = self.strategy_config[strategy]
            strategy_symbols = strategy_params.get('symbols', [])
            strategy_timeframes = strategy_params.get('timeframes', [])
            
            # Convert to OANDA format (replace / with _)
            oanda_symbols = [symbol.replace('/', '_') for symbol in strategy_symbols]
            
            print(f"Applying {strategy} to {', '.join(strategy_symbols)}")
            
            # Apply the strategy to each symbol
            for symbol in oanda_symbols:
                if symbol not in data:
                    continue
                    
                symbol_data = data[symbol]
                
                # Apply strategy and get signals
                signals_dict = self.apply_strategy(strategy, symbol_data, strategy_params)
                signals = signals_dict.get('signals', pd.Series(0, index=symbol_data.index))
                
                # Get the signal for the current timestamp if it exists
                current_signal = 0
                if simulation_date in signals.index:
                    current_signal = signals.loc[simulation_date]
                
                if current_signal != 0:
                    # Calculate ATR for position sizing
                    atr = self.ta.calculate_atr(symbol_data)
                    current_atr = atr.loc[simulation_date] if simulation_date in atr.index else atr.iloc[-1]
                    
                    self.execute_trade(
                        symbol, 
                        current_signal, 
                        symbol_data.loc[simulation_date, 'close'], 
                        strategy, 
                        current_atr,
                        simulation_date,
                        symbol_data
                    )
    
    def execute_trade(self, symbol: str, signal: int, price: float, strategy: str, 
                     atr_value: float, timestamp: datetime, data: pd.DataFrame) -> None:
        """
        Execute a trade based on the signal.
        
        Args:
            symbol: Trading instrument
            signal: Trade signal (1 for buy, -1 for sell)
            price: Current price
            strategy: Strategy that generated the signal
            atr_value: Current ATR value
            timestamp: Current timestamp
            data: Historical data for this symbol
        """
        # Check if we have too many open positions
        if self.open_position_count >= MAX_OPEN_POSITIONS and symbol not in self.positions:
            print(f"Skipping {symbol} trade due to maximum open positions ({MAX_OPEN_POSITIONS})")
            return
        
        # Check for correlation with existing positions
        if self.is_correlated_with_open_positions(symbol, signal, timestamp):
            return
            
        # Close any existing position in the opposite direction
        if symbol in self.positions and self.positions[symbol]['direction'] != signal:
            self.close_position(symbol, price, timestamp)
        
        # Open a new position if we don't have one already
        if symbol not in self.positions:
            # Calculate position size and risk amount
            risk_amount = self.get_risk_amount(strategy)
            position_size = self.get_position_size(symbol, price, risk_amount, atr_value)
            
            # Calculate stop loss and take profit levels
            reward_risk_ratio = self.risk_management.get('reward_risk_ratio', 2.0)
            atr_multiple = self.risk_management.get('trailing_stop_atr', 2.5)
            
            stop_loss, take_profit = self.ta.calculate_reward_risk_ratio(
                data, price, signal, atr_multiple, reward_risk_ratio
            )
            
            # Record the trade
            trade = {
                'id': self.trade_id,
                'symbol': symbol,
                'direction': "BUY" if signal > 0 else "SELL",
                'entry_price': price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': None,  # Will be updated when price moves favorably
                'entry_time': timestamp,
                'exit_time': None,
                'exit_price': None,
                'profit_loss': None,
                'strategy': strategy,
                'max_favorable_excursion': 0.0,  # Track maximum favorable price movement
                'max_adverse_excursion': 0.0,    # Track maximum unfavorable price movement
            }
            
            self.trades.append(trade)
            self.trade_id += 1
            
            # Update positions
            self.positions[symbol] = {
                'position_size': position_size,
                'entry_price': price,
                'direction': signal,
                'trade_id': trade['id'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': None,
                'entry_time': timestamp,
                'strategy': strategy
            }
            
            # Increment open position count
            self.open_position_count += 1
            
            print(f"Opened {trade['direction']} position in {symbol} at {price:.5f}, size: {position_size:.2f}")
            print(f"  Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}")
    
    def close_position(self, symbol: str, price: float, timestamp: datetime, reason: str = "signal") -> None:
        """
        Close an open position.
        
        Args:
            symbol: Trading instrument
            price: Current price
            timestamp: Current timestamp
            reason: Reason for closing (signal, stop_loss, take_profit, trailing_stop, time)
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
            trade['exit_time'] = timestamp
            trade['profit_loss'] = profit_loss
            trade['exit_reason'] = reason
            
            # Update account balance
            self.balance += profit_loss
            
            print(f"Closed {trade['direction']} position in {symbol} at {price:.5f}, P/L: {profit_loss:.2f} ({reason})")
        
        # Remove the position
        del self.positions[symbol]
        
        # Decrement open position count
        self.open_position_count -= 1
    
    def check_time_based_exits(self, timestamp: datetime) -> None:
        """
        Check for time-based exits on existing positions.
        
        Args:
            timestamp: Current timestamp
        """
        max_trade_duration = self.risk_management.get('max_trade_duration', 48)  # hours
        
        for symbol, position in list(self.positions.items()):
            # Calculate how long position has been open
            duration = timestamp - position['entry_time']
            hours_open = duration.total_seconds() / 3600
            
            if hours_open >= max_trade_duration:
                # Close position due to time exit
                if symbol in self.price_data and timestamp in self.price_data[symbol].index:
                    current_price = self.price_data[symbol].loc[timestamp, 'close']
                    self.close_position(symbol, current_price, timestamp, "time_exit")
    
    def check_stops_and_targets(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> None:
        """
        Check stop loss, trailing stop, and take profit levels.
        
        Args:
            data: Dictionary of market data by symbol
            timestamp: Current timestamp
        """
        for symbol, position in list(self.positions.items()):
            if symbol not in data or timestamp not in data[symbol].index:
                continue
            
            # Get current price data
            current_price = data[symbol].loc[timestamp, 'close']
            current_high = data[symbol].loc[timestamp, 'high']
            current_low = data[symbol].loc[timestamp, 'low']
            
            # Find the trade record
            trade = next((t for t in self.trades if t['id'] == position['trade_id']), None)
            
            if not trade:
                continue
            
            # Update max favorable/adverse excursion
            if position['direction'] > 0:  # Long position
                favorable_excursion = current_high - position['entry_price']
                adverse_excursion = position['entry_price'] - current_low
            else:  # Short position
                favorable_excursion = position['entry_price'] - current_low
                adverse_excursion = current_high - position['entry_price']
            
            # Update max excursions
            if trade['max_favorable_excursion'] < favorable_excursion:
                trade['max_favorable_excursion'] = favorable_excursion
                
                # Update trailing stop if needed
                if favorable_excursion > 0:
                    # Calculate new trailing stop based on ATR
                    atr = self.ta.calculate_atr(data[symbol])
                    current_atr = atr.loc[timestamp] if timestamp in atr.index else atr.iloc[-1]
                    atr_multiple = self.risk_management.get('trailing_stop_atr', 2.5)
                    
                    if position['direction'] > 0:  # Long position
                        trailing_stop = current_high - (current_atr * atr_multiple)
                        # Only update if better than current stop
                        if position['trailing_stop'] is None or trailing_stop > position['trailing_stop']:
                            position['trailing_stop'] = trailing_stop
                    else:  # Short position
                        trailing_stop = current_low + (current_atr * atr_multiple)
                        # Only update if better than current stop
                        if position['trailing_stop'] is None or trailing_stop < position['trailing_stop']:
                            position['trailing_stop'] = trailing_stop
            
            if trade['max_adverse_excursion'] < adverse_excursion:
                trade['max_adverse_excursion'] = adverse_excursion
            
            # Check stop loss hit
            if position['direction'] > 0:  # Long position
                if current_low <= position['stop_loss']:
                    self.close_position(symbol, position['stop_loss'], timestamp, "stop_loss")
                    continue
            else:  # Short position
                if current_high >= position['stop_loss']:
                    self.close_position(symbol, position['stop_loss'], timestamp, "stop_loss")
                    continue
            
            # Check take profit hit
            if position['direction'] > 0:  # Long position
                if current_high >= position['take_profit']:
                    self.close_position(symbol, position['take_profit'], timestamp, "take_profit")
                    continue
            else:  # Short position
                if current_low <= position['take_profit']:
                    self.close_position(symbol, position['take_profit'], timestamp, "take_profit")
                    continue
            
            # Check trailing stop hit
            if position['trailing_stop'] is not None:
                if position['direction'] > 0:  # Long position
                    if current_low <= position['trailing_stop']:
                        self.close_position(symbol, position['trailing_stop'], timestamp, "trailing_stop")
                else:  # Short position
                    if current_high >= position['trailing_stop']:
                        self.close_position(symbol, position['trailing_stop'], timestamp, "trailing_stop")
    
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
        print(f"Starting improved paper trading simulation from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
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
            # Check stops and targets
            self.check_stops_and_targets(data, timestamp)
            
            # Process market data and execute trades
            self.process_market_data(data, timestamp)
            
            # Update equity curve
            self.update_equity(data, timestamp)
        
        # Close any remaining positions at the end
        for symbol in list(self.positions.keys()):
            if symbol in data and simulation_timestamps[-1] in data[symbol].index:
                self.close_position(symbol, data[symbol].loc[simulation_timestamps[-1], 'close'], 
                                   simulation_timestamps[-1], "end_of_simulation")
        
        # Generate the performance report
        self.generate_report()
    
    def generate_report(self) -> None:
        """
        Generate a performance report for the simulation.
        """
        print("\n===== IMPROVED PAPER TRADING SIMULATION REPORT =====")
        
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
        
        # Exit reason statistics
        exit_reasons = {}
        for trade in self.trades:
            if 'exit_reason' in trade:
                reason = trade['exit_reason']
                if reason not in exit_reasons:
                    exit_reasons[reason] = {'count': 0, 'profit': 0}
                
                exit_reasons[reason]['count'] += 1
                if trade['profit_loss']:
                    exit_reasons[reason]['profit'] += trade['profit_loss']
        
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
        
        # Print exit reason statistics
        print("\nExit Reason Statistics:")
        for reason, stats in exit_reasons.items():
            print(f"  {reason}: {stats['count']} trades, P/L: ${stats['profit']:.2f}")
        
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
        plt.title('Improved Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('plots/improved/equity_curve.png')
        print("\nEquity curve saved to plots/improved/equity_curve.png")
        
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
        plt.savefig('plots/improved/drawdown.png')
        print("Drawdown chart saved to plots/improved/drawdown.png")
        
        # Save trade history to CSV
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv('plots/improved/trade_history.csv', index=False)
            print("Trade history saved to plots/improved/trade_history.csv")


def main():
    """
    Main function to run the improved paper trading simulation.
    """
    print("=== Improved Paper Trading Simulation ===")
    
    try:
        # Use a fixed historical date range to ensure data availability
        # Using a 14-day period (2 weeks) from the recent past
        end_time = datetime(2023, 11, 30, 0, 0, 0, tzinfo=timezone.utc)  # Use Nov 30, 2023 as end date
        start_time = end_time - timedelta(days=SIMULATION_DAYS)
        
        print(f"Using historical data period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        # Initialize paper trader
        trader = ImprovedPaperTrader()
        
        # Run simulation
        trader.simulate(start_time, end_time)
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 