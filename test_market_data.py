#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Market Data functionality

This script demonstrates basic forex market data retrieval functionality
without requiring API keys or actual API calls.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Basic mock of market data functionality
class SimpleMarketDataTest:
    """
    A simplified test class that demonstrates market data functionality
    without requiring actual API connections.
    """
    
    def __init__(self):
        """Initialize the test class with mock data."""
        print("Initializing Market Data Test...")
        self.instruments = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CAD', 
                           'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY']
        
    def get_current_price(self, instrument):
        """Get current mock price for an instrument."""
        if instrument not in self.instruments:
            raise ValueError(f"Invalid instrument: {instrument}")
            
        # Generate realistic mock price data
        base_prices = {
            'EUR_USD': 1.12,
            'USD_JPY': 148.5,
            'GBP_USD': 1.31,
            'AUD_USD': 0.67,
            'USD_CAD': 1.35,
            'NZD_USD': 0.61,
            'EUR_GBP': 0.85,
            'EUR_JPY': 166.3,
            'GBP_JPY': 195.2,
            'AUD_JPY': 99.5
        }
        
        # Use base price or default to 1.0 with small random variation
        base_price = base_prices.get(instrument, 1.0)
        noise = np.random.normal(0, 0.0002)
        
        return {
            'instrument': instrument,
            'time': datetime.now().isoformat(),
            'bid': round(base_price - 0.0001 + noise, 5),
            'ask': round(base_price + 0.0001 + noise, 5),
            'spread': 0.0002,
            'status': 'tradeable'
        }
        
    def get_historical_data(self, instrument, timeframe='H1', count=100):
        """Get mock historical data for an instrument."""
        if instrument not in self.instruments:
            raise ValueError(f"Invalid instrument: {instrument}")
            
        # Generate a DataFrame with mock OHLC data
        end_time = datetime.now()
        
        # Determine time interval based on timeframe
        if timeframe.startswith('M'):
            # Minutes
            minutes = int(timeframe[1:])
            interval = timedelta(minutes=minutes)
        elif timeframe.startswith('H'):
            # Hours
            hours = int(timeframe[1:])
            interval = timedelta(hours=hours)
        elif timeframe == 'D':
            # Days
            interval = timedelta(days=1)
        else:
            # Default to 1 hour
            interval = timedelta(hours=1)
            
        # Generate timestamps
        timestamps = [end_time - interval * i for i in range(count)]
        timestamps.reverse()  # Oldest first
        
        # Base price similar to current price
        base_prices = {
            'EUR_USD': 1.12,
            'USD_JPY': 148.5,
            'GBP_USD': 1.31,
            'AUD_USD': 0.67,
            'USD_CAD': 1.35,
            'NZD_USD': 0.61,
            'EUR_GBP': 0.85,
            'EUR_JPY': 166.3,
            'GBP_JPY': 195.2,
            'AUD_JPY': 99.5
        }
        base_price = base_prices.get(instrument, 1.0)
        
        # Generate random walk
        np.random.seed(42)  # For reproducibility
        random_walk = np.random.normal(0, 0.0002, count).cumsum()
        
        # Generate OHLC data
        prices = base_price + random_walk
        opens = prices
        highs = prices + np.random.uniform(0, 0.001, count)
        lows = prices - np.random.uniform(0, 0.001, count)
        closes = prices + np.random.normal(0, 0.0001, count)
        volumes = np.random.randint(100, 1000, count)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'complete': [True] * count
        }, index=pd.DatetimeIndex(timestamps, name='datetime'))
        
        # Calculate additional fields
        df['returns'] = df['close'].pct_change()
        df['range'] = df['high'] - df['low']
        
        return df
    
    def get_instruments(self):
        """Get list of available instruments."""
        return self.instruments
    
    def calculate_derived_data(self, data):
        """Calculate additional metrics from price data."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate returns
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate price range
        if all(col in df.columns for col in ['high', 'low']):
            df['range'] = df['high'] - df['low']
            df['range_pct'] = df['range'] / df['close'].shift(1) * 100
        
        # Calculate moving averages
        if 'close' in df.columns:
            df['ma_10'] = df['close'].rolling(window=10).mean()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate volatility (standard deviation of returns)
        if 'returns' in df.columns:
            df['volatility_10'] = df['returns'].rolling(window=10).std() * np.sqrt(10)
            df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(20)
        
        return df
        
def main():
    """Main test function to demonstrate market data functionality."""
    
    # Create test instance
    market_data = SimpleMarketDataTest()
    
    # Get and display current price for EUR_USD
    print("\nFetching current price for EUR_USD...")
    price_data = market_data.get_current_price("EUR_USD")
    
    print("\nCurrent price for EUR_USD:")
    print(f"Time: {price_data.get('time')}")
    print(f"Bid: {price_data.get('bid')}")
    print(f"Ask: {price_data.get('ask')}")
    print(f"Spread: {price_data.get('spread', 'N/A')}")
    print(f"Status: {price_data.get('status', 'N/A')}")
    
    # Get and display available instruments
    print("\nAvailable instruments:")
    instruments = market_data.get_instruments()
    print(", ".join(instruments))
    
    # Get and display historical data
    print("\nFetching historical data for EUR_USD (H1 timeframe, last 5 points)...")
    historical_data = market_data.get_historical_data("EUR_USD", "H1", 20)
    print(historical_data.tail(5))
    
    # Calculate and display derived data
    print("\nCalculating derived metrics...")
    derived_data = market_data.calculate_derived_data(historical_data)
    print("\nDerived data (last 5 points):")
    # Select a subset of columns to display
    display_columns = ['close', 'returns', 'ma_10', 'volatility_10']
    print(derived_data[display_columns].tail(5))
    
    print("\nNote: This is a demonstration with mock data.")
    print("In a real scenario, you would need valid OANDA API credentials.")
    print("\nTo use the actual MarketDataAgent:")
    print("1. Set OANDA_API_KEY, OANDA_ACCOUNT_ID, and OANDA_API_URL environment variables")
    print("2. Import and use the MarketDataAgent class instead of this test class")
    
    print("\nTest completed at", datetime.now())

if __name__ == "__main__":
    main() 