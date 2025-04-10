#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Technical Analysis module for Forex Trading.

This module provides improved technical indicators and analysis functions
with additional confirmation filters based on paper trading results.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional


class EnhancedTechnicalAnalysis:
    """Enhanced technical analysis with improved indicator calculations and filters."""
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int, price_col: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average."""
        return data[price_col].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int, price_col: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data[price_col].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data[price_col].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9, 
                       price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Returns:
            Tuple containing (macd_line, signal_line, histogram)
        """
        fast_ema = EnhancedTechnicalAnalysis.calculate_ema(data, fast_period, price_col)
        slow_ema = EnhancedTechnicalAnalysis.calculate_ema(data, slow_period, price_col)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, 
                                  std_dev: float = 2.0, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Returns:
            Tuple containing (upper_band, middle_band, lower_band)
        """
        middle_band = EnhancedTechnicalAnalysis.calculate_sma(data, period, price_col)
        std = data[price_col].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with OHLC data
            period: Period for ATR calculation
            
        Returns:
            Series containing ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_relative_volume(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate relative volume compared to average.
        
        Args:
            data: DataFrame with volume data
            period: Period for average volume calculation
            
        Returns:
            Series containing relative volume ratio
        """
        avg_volume = data['volume'].rolling(window=period).mean()
        relative_volume = data['volume'] / avg_volume
        
        return relative_volume
    
    @staticmethod
    def is_price_above_ma(data: pd.DataFrame, ma_period: int, 
                         price_col: str = 'close') -> pd.Series:
        """
        Check if price is above moving average.
        
        Returns:
            Boolean Series where True means price above MA
        """
        ma = EnhancedTechnicalAnalysis.calculate_sma(data, ma_period, price_col)
        return data[price_col] > ma
    
    @staticmethod
    def calculate_reward_risk_ratio(data: pd.DataFrame, entry_price: float, 
                                  direction: int, atr_multiple: float = 1.0,
                                  reward_risk_ratio: float = 2.0) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels based on ATR.
        
        Args:
            data: DataFrame with OHLC data
            entry_price: Trade entry price
            direction: Trade direction (1 for long, -1 for short)
            atr_multiple: Multiple of ATR to use for stop loss
            reward_risk_ratio: Target reward to risk ratio
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Use most recent ATR value
        atr = EnhancedTechnicalAnalysis.calculate_atr(data)
        latest_atr = atr.iloc[-1]
        
        # Calculate stop distance
        stop_distance = latest_atr * atr_multiple
        
        # Calculate stop loss and take profit prices
        if direction > 0:  # Long position
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * reward_risk_ratio)
        else:  # Short position
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * reward_risk_ratio)
            
        return stop_loss, take_profit
    
    @staticmethod
    def calculate_correlation(df1: pd.Series, df2: pd.Series, 
                            period: int = 20) -> pd.Series:
        """
        Calculate rolling correlation between two price series.
        
        Args:
            df1: First price series
            df2: Second price series
            period: Correlation calculation period
            
        Returns:
            Series of correlation values
        """
        return df1.rolling(window=period).corr(df2)


# Enhanced Strategy Implementations
class EnhancedStrategies:
    """Improved trading strategies with confirmation filters."""
    
    @staticmethod
    def enhanced_rsi_reversal(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Enhanced RSI Reversal strategy with moving average confirmation.
        
        Args:
            data: DataFrame with OHLC data
            params: Strategy parameters
                - rsi_period: Period for RSI calculation
                - oversold: RSI threshold for oversold condition
                - overbought: RSI threshold for overbought condition
                - confirmation_ma_period: Period for confirmation MA
                
        Returns:
            Dictionary with signals, positions, etc.
        """
        # Extract parameters
        rsi_period = params.get('rsi_period', 14)
        oversold = params.get('oversold', 25)
        overbought = params.get('overbought', 75)
        confirmation_ma_period = params.get('confirmation_ma_period', 20)
        
        # Calculate indicators
        ta = EnhancedTechnicalAnalysis()
        rsi = ta.calculate_rsi(data, rsi_period)
        confirmation_ma = ta.calculate_sma(data, confirmation_ma_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        current_position = 0
        
        for i in range(1, len(data)):
            # Buy signal: RSI crosses above oversold threshold AND price above MA
            if (rsi.iloc[i-1] <= oversold and rsi.iloc[i] > oversold and 
                data['close'].iloc[i] > confirmation_ma.iloc[i]):
                signals.iloc[i] = 1
                current_position = 1
            
            # Sell signal: RSI crosses below overbought threshold AND price below MA
            elif (rsi.iloc[i-1] >= overbought and rsi.iloc[i] < overbought and 
                  data['close'].iloc[i] < confirmation_ma.iloc[i]):
                signals.iloc[i] = -1
                current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'rsi': rsi,
            'confirmation_ma': confirmation_ma
        }
    
    @staticmethod
    def enhanced_bollinger_bands(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Enhanced Bollinger Bands strategy with RSI filter.
        
        Args:
            data: DataFrame with OHLC data
            params: Strategy parameters
                - period: Period for Bollinger Bands calculation
                - std_dev: Number of standard deviations
                - rsi_filter_period: Period for RSI filter
                - rsi_lower_threshold: RSI threshold for buy signals
                - rsi_upper_threshold: RSI threshold for sell signals
                
        Returns:
            Dictionary with signals, positions, etc.
        """
        # Extract parameters
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2.5)
        rsi_period = params.get('rsi_filter_period', 14)
        rsi_lower = params.get('rsi_lower_threshold', 40)
        rsi_upper = params.get('rsi_upper_threshold', 60)
        
        # Calculate indicators
        ta = EnhancedTechnicalAnalysis()
        upper, middle, lower = ta.calculate_bollinger_bands(data, period, std_dev)
        rsi = ta.calculate_rsi(data, rsi_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        current_position = 0
        
        for i in range(1, len(data)):
            # Buy signal: price crosses above lower band AND RSI < lower threshold
            if (data['close'].iloc[i-1] <= lower.iloc[i-1] and 
                data['close'].iloc[i] > lower.iloc[i] and 
                rsi.iloc[i] < rsi_lower):
                signals.iloc[i] = 1
                current_position = 1
            
            # Sell signal: price crosses below upper band AND RSI > upper threshold
            elif (data['close'].iloc[i-1] >= upper.iloc[i-1] and 
                  data['close'].iloc[i] < upper.iloc[i] and 
                  rsi.iloc[i] > rsi_upper):
                signals.iloc[i] = -1
                current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'upper_band': upper,
            'middle_band': middle,
            'lower_band': lower,
            'rsi': rsi
        }
    
    @staticmethod
    def enhanced_macd(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Enhanced MACD strategy with histogram threshold and trend filter.
        
        Args:
            data: DataFrame with OHLC data
            params: Strategy parameters
                - fast_period: Fast EMA period
                - slow_period: Slow EMA period
                - signal_period: Signal line period
                - histogram_threshold: Minimum histogram value for signal
                - trend_ma_period: Period for trend MA filter
                
        Returns:
            Dictionary with signals, positions, etc.
        """
        # Extract parameters
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        histogram_threshold = params.get('histogram_threshold', 0.0005)
        trend_ma_period = params.get('trend_ma_period', 50)
        
        # Calculate indicators
        ta = EnhancedTechnicalAnalysis()
        macd_line, signal_line, histogram = ta.calculate_macd(
            data, fast_period, slow_period, signal_period
        )
        trend_ma = ta.calculate_ema(data, trend_ma_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        current_position = 0
        
        for i in range(1, len(data)):
            # Buy signal: MACD crosses above signal AND histogram > threshold AND price > trend MA
            if (macd_line.iloc[i-1] <= signal_line.iloc[i-1] and 
                macd_line.iloc[i] > signal_line.iloc[i] and 
                abs(histogram.iloc[i]) > histogram_threshold and
                data['close'].iloc[i] > trend_ma.iloc[i]):
                signals.iloc[i] = 1
                current_position = 1
            
            # Sell signal: MACD crosses below signal AND histogram > threshold AND price < trend MA
            elif (macd_line.iloc[i-1] >= signal_line.iloc[i-1] and 
                  macd_line.iloc[i] < signal_line.iloc[i] and 
                  abs(histogram.iloc[i]) > histogram_threshold and
                  data['close'].iloc[i] < trend_ma.iloc[i]):
                signals.iloc[i] = -1
                current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'trend_ma': trend_ma
        }
    
    @staticmethod
    def enhanced_ma_crossover(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Enhanced Moving Average Crossover with volume filter.
        
        Args:
            data: DataFrame with OHLC and volume data
            params: Strategy parameters
                - fast_period: Fast MA period
                - slow_period: Slow MA period
                - volume_threshold: Volume multiple threshold
                - volume_period: Period for volume average
                
        Returns:
            Dictionary with signals, positions, etc.
        """
        # Extract parameters
        fast_period = params.get('fast_period', 5)
        slow_period = params.get('slow_period', 20)
        volume_threshold = params.get('volume_threshold', 1.5)
        volume_period = params.get('volume_period', 20)
        
        # Calculate indicators
        ta = EnhancedTechnicalAnalysis()
        fast_ma = ta.calculate_ema(data, fast_period)
        slow_ma = ta.calculate_ema(data, slow_period)
        relative_volume = ta.calculate_relative_volume(data, volume_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        current_position = 0
        
        for i in range(1, len(data)):
            # Check volume condition
            volume_condition = relative_volume.iloc[i] > volume_threshold
            
            # Buy signal: fast MA crosses above slow MA AND volume > threshold
            if (fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and 
                fast_ma.iloc[i] > slow_ma.iloc[i] and 
                volume_condition):
                signals.iloc[i] = 1
                current_position = 1
            
            # Sell signal: fast MA crosses below slow MA AND volume > threshold
            elif (fast_ma.iloc[i-1] >= slow_ma.iloc[i-1] and 
                  fast_ma.iloc[i] < slow_ma.iloc[i] and 
                  volume_condition):
                signals.iloc[i] = -1
                current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'relative_volume': relative_volume
        } 