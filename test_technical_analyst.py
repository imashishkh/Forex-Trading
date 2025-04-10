#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Technical Analysis functionality

This script demonstrates the calculation of technical indicators (RSI and MACD)
for EUR/USD using historical data generated in the script.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test technical analysis calculations"""
    
    # Define the instrument to analyze
    instrument = "EUR_USD"
    count = 200  # Get 200 candles
    
    logger.info(f"Creating mock data for {instrument}...")
    
    try:
        # Create mock historical data
        historical_data = create_mock_data(instrument, count)
        
        # Display a summary of the data
        logger.info(f"Created {len(historical_data)} candles of {instrument} data")
        logger.info(f"Data range: {historical_data.index[0]} to {historical_data.index[-1]}")
        
        # Calculate RSI
        logger.info("Calculating RSI...")
        rsi = calculate_rsi(historical_data, period=14)
        
        # Calculate MACD
        logger.info("Calculating MACD...")
        macd_line, signal_line, histogram = calculate_macd(
            historical_data, 
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        # Print last 5 values of each indicator
        logger.info("\nRSI (last 5 values):")
        for date, value in rsi.tail(5).items():
            logger.info(f"{date}: {value:.2f}")
        
        logger.info("\nMACD (last 5 values):")
        for date, value in macd_line.tail(5).items():
            signal = signal_line.loc[date]
            hist = histogram.loc[date]
            logger.info(f"{date}: MACD Line: {value:.5f}, Signal: {signal:.5f}, Histogram: {hist:.5f}")
        
        # Generate a basic plot
        logger.info("\nGenerating plots...")
        plot_indicators(historical_data, rsi, macd_line, signal_line, histogram)
        
        logger.info("\nTechnical analysis completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during technical analysis: {e}", exc_info=True)

def create_mock_data(instrument, count):
    """Create mock market data for testing"""
    
    end_time = datetime.now()
    interval = timedelta(hours=1)
    
    # Generate timestamps
    timestamps = [end_time - interval * i for i in range(count)]
    timestamps.reverse()  # Oldest first
    
    # Base prices for common instruments
    base_prices = {
        'EUR_USD': 1.12,
        'USD_JPY': 148.5,
        'GBP_USD': 1.31,
        'AUD_USD': 0.67,
        'USD_CAD': 1.35,
        'NZD_USD': 0.61,
    }
    base_price = base_prices.get(instrument, 1.0)
    
    # Generate random walk
    np.random.seed(42)  # For reproducibility
    random_walk = np.random.normal(0, 0.0005, count).cumsum()
    
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
    
    return df

def calculate_rsi(data, period=14, price_column='close'):
    """
    Calculate the Relative Strength Index (RSI)
    
    Args:
        data: DataFrame with price data
        period: RSI period (default 14)
        price_column: Column to use for calculations (default 'close')
        
    Returns:
        pandas.Series: RSI values
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate price changes
    delta = df[price_column].diff()
    
    # Separate positive and negative changes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9, price_column='close'):
    """
    Calculate the Moving Average Convergence Divergence (MACD)
    
    Args:
        data: DataFrame with price data
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
        price_column: Column to use for calculations (default 'close')
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    # Calculate EMAs
    ema_fast = data[price_column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data[price_column].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def plot_indicators(data, rsi, macd_line, signal_line, histogram):
    """Create a simple plot of price data and calculated indicators"""
    
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create a figure with 3 subplots: price, RSI, and MACD
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot price
    ax1.set_title('EUR/USD Price')
    ax1.plot(data.index, data['close'], label='Close Price')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot RSI
    ax2.set_title('RSI (14)')
    ax2.plot(rsi.index, rsi, label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    # Plot MACD
    ax3.set_title('MACD (12,26,9)')
    ax3.plot(macd_line.index, macd_line, label='MACD Line', color='blue')
    ax3.plot(signal_line.index, signal_line, label='Signal Line', color='red')
    ax3.bar(histogram.index, histogram, label='Histogram', alpha=0.5, width=0.0008)
    ax3.set_ylabel('MACD')
    ax3.grid(True)
    ax3.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('plots/technical_analysis_test.png')
    logger.info("Plot saved to 'plots/technical_analysis_test.png'")

if __name__ == "__main__":
    main() 