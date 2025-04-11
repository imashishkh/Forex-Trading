#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical Analyst Agent for the Forex Trading Platform

This module provides a comprehensive implementation of a technical analysis agent
that calculates indicators, identifies patterns, generates signals, and implements
trading strategies for forex market data.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
from datetime import datetime, timedelta
from pathlib import Path
import time
import json

# Data analysis imports
import numpy as np
import pandas as pd
from scipy import stats

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Base Agent class
from utils.base_agent import BaseAgent, AgentState, AgentMessage

# Import the MarketDataAgent for data retrieval
from market_data_agent import MarketDataAgent

# LangGraph imports
import langgraph.graph as lg
from langgraph.checkpoint.memory import MemorySaver


class TechnicalAnalystAgent(BaseAgent):
    """
    Technical Analyst Agent for performing technical analysis on forex market data.
    
    This agent is responsible for calculating technical indicators, identifying chart
    patterns, generating trading signals, and implementing trading strategies based on
    technical analysis methodologies.
    """
    
    def __init__(
        self,
        agent_name: str = "technical_analyst_agent",
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        market_data_agent: Optional[MarketDataAgent] = None
    ):
        """
        Initialize the Technical Analyst Agent.
        
        Args:
            agent_name: Name identifier for the agent
            llm: Language model to use (defaults to OpenAI if None)
            config: Configuration parameters for the agent
            logger: Logger instance for the agent
            market_data_agent: Instance of MarketDataAgent for data retrieval
        """
        # Initialize BaseAgent
        super().__init__(agent_name, llm, config, logger)
        
        # Store reference to MarketDataAgent
        self.market_data_agent = market_data_agent
        
        # Default indicator settings
        self.indicator_settings = {
            'moving_averages': {
                'sma_periods': [5, 10, 20, 50, 100, 200],
                'ema_periods': [5, 10, 20, 50, 100, 200],
            },
            'oscillators': {
                'rsi_period': 14,
                'stochastic_k_period': 14,
                'stochastic_d_period': 3,
                'macd_fast_period': 12,
                'macd_slow_period': 26,
                'macd_signal_period': 9,
            },
            'trend': {
                'adx_period': 14,
                'supertrend_period': 10,
                'supertrend_multiplier': 3,
            },
            'volatility': {
                'bollinger_period': 20,
                'bollinger_std_dev': 2,
                'atr_period': 14,
            },
            'volume': {
                'mfi_period': 14,
            }
        }
        
        # Override default settings with config if provided
        if config and 'indicators' in config:
            self._update_nested_dict(self.indicator_settings, config['indicators'])
        
        # Setup visualization settings
        self.visualization_settings = {
            'dpi': 100,
            'figsize': (12, 8),
            'style': 'seaborn',
            'save_plots': True,
            'plot_dir': 'plots/technical_analysis',
        }
        
        # Override visualization settings if provided
        if config and 'visualization' in config:
            self.visualization_settings.update(config['visualization'])
            
        # Ensure plot directory exists if save_plots is enabled
        if self.visualization_settings['save_plots']:
            os.makedirs(self.visualization_settings['plot_dir'], exist_ok=True)
            
        # Strategy settings
        self.strategy_settings = {
            'default_strategy': 'moving_average_crossover',
            'default_parameters': {
                'fast_period': 10,
                'slow_period': 50,
            },
            'risk_per_trade': 0.01,  # 1% risk per trade
        }
        
        # Override strategy settings if provided
        if config and 'strategies' in config:
            self.strategy_settings.update(config['strategies'])
            
        self.log_action("init", f"Technical Analyst Agent initialized with {len(self.indicator_settings)} indicator groups")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Update a nested dictionary with another dictionary.
        
        Args:
            d: Original dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def analyze(self, market_data):
        """
        Analyze market data using technical indicators
        
        Args:
            market_data (dict or pd.DataFrame): Market data to analyze.
                If dict, keys are symbols and values are DataFrames.
                If DataFrame, a single symbol's data.
        
        Returns:
            dict: Analysis results by symbol and indicator
        """
        self.logger.info("Analyzing market data with technical indicators")
        
        # Handle if market_data is a dict (multiple symbols) or DataFrame (single symbol)
        if isinstance(market_data, dict):
            results = {}
            for symbol, df in market_data.items():
                results[symbol] = self._analyze_single(df)
            return results
        elif isinstance(market_data, pd.DataFrame):
            return self._analyze_single(market_data)
        else:
            self.logger.error(f"Invalid market data type: {type(market_data)}")
            return None
    
    def _analyze_single(self, df):
        """
        Analyze a single symbol's market data
        
        Args:
            df (pd.DataFrame): Market data for a single symbol
        
        Returns:
            dict: Analysis results by indicator type
        """
        if df is None or len(df) == 0:
            self.logger.warning("Empty market data provided for analysis")
            return {}
        
        results = {}
        
        # Calculate moving averages
        if 'moving_averages' in self.indicator_settings:
            results['moving_averages'] = self._calculate_moving_averages(df)
        
        # Calculate oscillators
        if 'oscillators' in self.indicator_settings:
            results['oscillators'] = self._calculate_oscillators(df)
        
        # Calculate volatility indicators
        if 'volatility' in self.indicator_settings:
            results['volatility'] = self._calculate_volatility_indicators(df)
        
        # Calculate trend indicators
        if 'trend' in self.indicator_settings:
            results['trend'] = self._calculate_trend_indicators(df)
        
        # Calculate signals
        results['signals'] = self._generate_signals(results)
        
        return results
    
    def _calculate_moving_averages(self, df):
        """
        Calculate moving averages for the market data
        
        Args:
            df (pd.DataFrame): Market data
        
        Returns:
            dict: Moving average results
        """
        results = {}
        
        # Simple Moving Average (SMA)
        if 'SMA' in self.indicator_settings.get('moving_averages', []):
            for period_name, period in self.indicator_settings['moving_averages'].items():
                sma_key = f"SMA_{period}"
                results[sma_key] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Average (EMA)
        if 'EMA' in self.indicator_settings.get('moving_averages', []):
            for period_name, period in self.indicator_settings['moving_averages'].items():
                ema_key = f"EMA_{period}"
                results[ema_key] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Weighted Moving Average (WMA)
        if 'WMA' in self.indicator_settings.get('moving_averages', []):
            for period_name, period in self.indicator_settings['moving_averages'].items():
                wma_key = f"WMA_{period}"
                weights = np.arange(1, period + 1)
                results[wma_key] = df['close'].rolling(period).apply(
                    lambda x: np.sum(weights * x) / weights.sum(), raw=True
                )
        
        return results
    
    def _calculate_oscillators(self, df):
        """
        Calculate oscillator indicators for the market data
        
        Args:
            df (pd.DataFrame): Market data
        
        Returns:
            dict: Oscillator results
        """
        results = {}
        
        # Relative Strength Index (RSI)
        if 'RSI' in self.indicator_settings.get('oscillators', []):
            period = self.indicator_settings['oscillators']['rsi_period']
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            results['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        if 'MACD' in self.indicator_settings.get('oscillators', []):
            fast_period = self.indicator_settings['oscillators']['macd_fast_period']
            slow_period = self.indicator_settings['oscillators']['macd_slow_period']
            signal_period = self.indicator_settings['oscillators']['macd_signal_period']
            
            fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
            
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            results['MACD_line'] = macd_line
            results['MACD_signal'] = signal_line
            results['MACD_histogram'] = histogram
        
        # Stochastic Oscillator
        if 'Stochastic' in self.indicator_settings.get('oscillators', []):
            k_period = self.indicator_settings['oscillators']['stochastic_k_period']
            d_period = self.indicator_settings['oscillators']['stochastic_d_period']
            
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=d_period).mean()
            
            results['Stochastic_K'] = k
            results['Stochastic_D'] = d
        
        return results
    
    def _calculate_volatility_indicators(self, df):
        """
        Calculate volatility indicators for the market data
        
        Args:
            df (pd.DataFrame): Market data
        
        Returns:
            dict: Volatility indicator results
        """
        results = {}
        
        # Bollinger Bands
        if 'Bollinger Bands' in self.indicator_settings.get('volatility', []):
            period = self.indicator_settings['volatility']['bollinger_period']
            std_dev = self.indicator_settings['volatility']['bollinger_std_dev']
            
            middle_band = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            results['BB_upper'] = upper_band
            results['BB_middle'] = middle_band
            results['BB_lower'] = lower_band
            results['BB_width'] = (upper_band - lower_band) / middle_band
        
        # Average True Range (ATR)
        if 'ATR' in self.indicator_settings.get('volatility', []):
            period = self.indicator_settings['volatility']['atr_period']
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            results['ATR'] = atr
        
        return results
    
    def _calculate_trend_indicators(self, df):
        """
        Calculate trend indicators for the market data
        
        Args:
            df (pd.DataFrame): Market data
        
        Returns:
            dict: Trend indicator results
        """
        results = {}
        
        # Average Directional Index (ADX)
        if 'ADX' in self.indicator_settings.get('trend', []):
            period = self.indicator_settings['trend']['adx_period']
            
            # Calculate +DM and -DM
            high_diff = df['high'].diff()
            low_diff = df['low'].diff().multiply(-1)
            
            plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
            minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.ewm(span=period, adjust=False).mean()
            
            # Calculate +DI and -DI
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
            
            # Calculate DX and ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean()
            
            results['ADX'] = adx
            results['plus_DI'] = plus_di
            results['minus_DI'] = minus_di
        
        # Ichimoku Cloud
        if 'Ichimoku' in self.indicator_settings.get('trend', []):
            # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
            tenkan_period = 9
            high_tenkan = df['high'].rolling(window=tenkan_period).max()
            low_tenkan = df['low'].rolling(window=tenkan_period).min()
            tenkan_sen = (high_tenkan + low_tenkan) / 2
            
            # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
            kijun_period = 26
            high_kijun = df['high'].rolling(window=kijun_period).max()
            low_kijun = df['low'].rolling(window=kijun_period).min()
            kijun_sen = (high_kijun + low_kijun) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted forward by 26 periods
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
            
            # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, shifted forward by 26 periods
            senkou_period = 52
            high_senkou = df['high'].rolling(window=senkou_period).max()
            low_senkou = df['low'].rolling(window=senkou_period).min()
            senkou_span_b = ((high_senkou + low_senkou) / 2).shift(kijun_period)
            
            # Chikou Span (Lagging Span): Close price, shifted backwards by 26 periods
            chikou_span = df['close'].shift(-kijun_period)
            
            results['Ichimoku_Tenkan'] = tenkan_sen
            results['Ichimoku_Kijun'] = kijun_sen
            results['Ichimoku_SenkouA'] = senkou_span_a
            results['Ichimoku_SenkouB'] = senkou_span_b
            results['Ichimoku_Chikou'] = chikou_span
        
        return results
    
    def _generate_signals(self, indicator_results):
        """
        Generate trading signals based on technical indicators
        
        Args:
            indicator_results (dict): Results from technical indicators
        
        Returns:
            dict: Trading signals
        """
        signals = {'buy': [], 'sell': [], 'neutral': []}
        
        # Check moving average signals
        if 'moving_averages' in indicator_results:
            ma_results = indicator_results['moving_averages']
            
            # Check for golden cross (short-term MA crosses above long-term MA)
            if all(k in ma_results for k in ['SMA_50', 'SMA_200']):
                sma_50 = ma_results['SMA_50']
                sma_200 = ma_results['SMA_200']
                
                # Current values
                current_sma_50 = sma_50.iloc[-1]
                current_sma_200 = sma_200.iloc[-1]
                
                # Previous values
                prev_sma_50 = sma_50.iloc[-2] if len(sma_50) > 1 else None
                prev_sma_200 = sma_200.iloc[-2] if len(sma_200) > 1 else None
                
                # Golden cross (bullish)
                if (prev_sma_50 is not None and prev_sma_200 is not None and
                    prev_sma_50 <= prev_sma_200 and current_sma_50 > current_sma_200):
                    signals['buy'].append('Golden Cross: SMA 50 crossed above SMA 200')
                
                # Death cross (bearish)
                if (prev_sma_50 is not None and prev_sma_200 is not None and
                    prev_sma_50 >= prev_sma_200 and current_sma_50 < current_sma_200):
                    signals['sell'].append('Death Cross: SMA 50 crossed below SMA 200')
        
        # Check oscillator signals
        if 'oscillators' in indicator_results:
            osc_results = indicator_results['oscillators']
            
            # RSI signals
            if 'RSI' in osc_results:
                rsi = osc_results['RSI']
                current_rsi = rsi.iloc[-1]
                
                if current_rsi < 30:
                    signals['buy'].append(f'RSI oversold: {current_rsi:.2f}')
                elif current_rsi > 70:
                    signals['sell'].append(f'RSI overbought: {current_rsi:.2f}')
            
            # MACD signals
            if all(k in osc_results for k in ['MACD_line', 'MACD_signal']):
                macd_line = osc_results['MACD_line']
                signal_line = osc_results['MACD_signal']
                
                # Current values
                current_macd = macd_line.iloc[-1]
                current_signal = signal_line.iloc[-1]
                
                # Previous values
                prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else None
                prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else None
                
                # Bullish crossover
                if (prev_macd is not None and prev_signal is not None and
                    prev_macd <= prev_signal and current_macd > current_signal):
                    signals['buy'].append('MACD: Bullish crossover')
                
                # Bearish crossover
                if (prev_macd is not None and prev_signal is not None and
                    prev_macd >= prev_signal and current_macd < current_signal):
                    signals['sell'].append('MACD: Bearish crossover')
        
        # Check volatility signals
        if 'volatility' in indicator_results:
            vol_results = indicator_results['volatility']
            
            # Bollinger Bands signals
            if all(k in vol_results for k in ['BB_upper', 'BB_middle', 'BB_lower']):
                bb_upper = vol_results['BB_upper']
                bb_middle = vol_results['BB_middle']
                bb_lower = vol_results['BB_lower']
        
        # Combine signals
        if len(signals['buy']) > len(signals['sell']):
            signals['overall'] = 'BUY'
        elif len(signals['sell']) > len(signals['buy']):
            signals['overall'] = 'SELL'
        else:
            signals['overall'] = 'NEUTRAL'
        
        return signals
    
    def initialize(self) -> bool:
        """
        Initialize the Technical Analyst Agent.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.log_action("initialize", "Initializing Technical Analyst Agent")
        
        try:
            # Check if MarketDataAgent is connected
            if not self.market_data_agent:
                self.log_action("initialize", "Warning: No MarketDataAgent connected")
            
            # Update status
            self.status = "ready"
            self.state["status"] = "ready"
            
            return True
            
        except Exception as e:
            self.handle_error(e)
            self.status = "error"
            self.state["status"] = "error"
            return False
            
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the technical analysis task.
        
        This method implements the abstract method from BaseAgent and serves
        as the primary entry point for technical analysis operations.
        
        Args:
            task: Task description and parameters including:
                - type: Type of analysis to perform (e.g., "analyze", "strategy", "backtest")
                - symbol: The currency pair to analyze
                - timeframe: The timeframe for analysis (e.g., "1h", "4h", "1d")
                - indicators: Optional specific indicators to calculate
                - strategy: Optional strategy to implement

        Returns:
            Dict[str, Any]: Task execution results including:
                - status: "success" or "error"
                - data: Analysis results (for "analyze" and "strategy" tasks)
                - signals: Generated trading signals
                - metrics: Performance metrics (for "backtest" tasks)
                - chart_paths: Paths to generated charts (if any)
        """
        self.log_action("run_task", f"Running task: {task.get('type', 'Unknown')}")
        
        # Update state
        self.state["tasks"].append(task)
        self.last_active = datetime.now()
        self.state["last_active"] = self.last_active
        
        try:
            task_type = task.get("type", "analyze")
            symbol = task.get("symbol", "EUR_USD")
            timeframe = task.get("timeframe", "1h")
            
            # Get market data if needed and not provided
            if "data" not in task and self.market_data_agent:
                market_data = self.market_data_agent.get_historical_data(
                    instrument=symbol, 
                    timeframe=timeframe, 
                    count=task.get("count", 500)
                )
            else:
                market_data = task.get("data")
                
            if market_data is None or (isinstance(market_data, pd.DataFrame) and market_data.empty):
                return {
                    "status": "error",
                    "message": f"No market data available for {symbol} ({timeframe})"
                }
            
            # Execute appropriate task based on type
            if task_type == "analyze":
                # Perform technical analysis
                indicators = task.get("indicators", None)
                results = self.analyze(market_data)
                
                # Generate visualization if requested
                chart_paths = []
                if task.get("visualize", False) and self.visualization_settings["save_plots"]:
                    # Generate charts for analysis results
                    chart_paths = self._generate_analysis_charts(symbol, timeframe, market_data, results)
                
                return {
                    "status": "success",
                    "data": results,
                    "signals": results.get("signals", {}),
                    "chart_paths": chart_paths,
                    "timestamp": datetime.now().isoformat()
                }
                
            elif task_type == "strategy":
                # Implement trading strategy
                strategy_name = task.get("strategy", self.strategy_settings["default_strategy"])
                params = task.get("parameters", self.strategy_settings["default_parameters"])
                
                strategy_results = self.implement_strategy(market_data, strategy_name, params)
                
                return {
                    "status": "success",
                    "data": strategy_results,
                    "signals": strategy_results.get("signals", pd.Series()),
                    "timestamp": datetime.now().isoformat()
                }
                
            elif task_type == "backtest":
                # Backtest a strategy
                strategy_name = task.get("strategy", self.strategy_settings["default_strategy"])
                params = task.get("parameters", self.strategy_settings["default_parameters"])
                
                backtest_results = self.backtest_strategy(market_data, strategy_name, params)
                
                return {
                    "status": "success",
                    "data": backtest_results,
                    "metrics": backtest_results.get("metrics", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
            elif task_type == "optimize":
                # Optimize strategy parameters
                strategy_name = task.get("strategy", self.strategy_settings["default_strategy"])
                param_grid = task.get("param_grid", {})
                
                if not param_grid:
                    return {
                        "status": "error",
                        "message": "Parameter grid required for optimization"
                    }
                
                optimization_results = self.optimize_parameters(market_data, strategy_name, param_grid)
                
                return {
                    "status": "success",
                    "data": optimization_results,
                    "best_parameters": optimization_results.get("best_parameters", {}),
                    "metrics": optimization_results.get("metrics", {}),
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            self.handle_error(e)
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_analysis_charts(self, symbol: str, timeframe: str, data: pd.DataFrame, 
                             results: Dict[str, Any]) -> List[str]:
        """
        Generate charts for analysis results
        
        Args:
            symbol: Currency pair symbol
            timeframe: Timeframe of the data
            data: Market data
            results: Analysis results
            
        Returns:
            List[str]: Paths to saved chart files
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.gridspec import GridSpec
        
        chart_paths = []
        try:
            # Use the visualization settings
            plt.style.use(self.visualization_settings.get('style', 'seaborn'))
            dpi = self.visualization_settings.get('dpi', 100)
            figsize = self.visualization_settings.get('figsize', (12, 8))
            
            # Create main price chart with indicators
            fig = plt.figure(figsize=figsize, dpi=dpi)
            
            # Prepare filename and path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}.png"
            filepath = os.path.join(self.visualization_settings['plot_dir'], filename)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close(fig)
            
            chart_paths.append(filepath)
            
            # Create separate indicator charts if needed
            # ...
            
            return chart_paths
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {str(e)}")
            return chart_paths
    
    # Moving Average Indicators
    
    def calculate_sma(self, data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: DataFrame containing price data
            period: Period for SMA calculation
            price_column: Column name for price data (default: 'close')
            
        Returns:
            Series containing SMA values
        """
        self.log_action("calculate_indicator", f"Calculating SMA with period {period}")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
            
        return data[price_column].rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: DataFrame containing price data
            period: Period for EMA calculation
            price_column: Column name for price data (default: 'close')
            
        Returns:
            Series containing EMA values
        """
        self.log_action("calculate_indicator", f"Calculating EMA with period {period}")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
            
        return data[price_column].ewm(span=period, adjust=False).mean()
    
    def calculate_wma(self, data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Weighted Moving Average (WMA).
        
        Args:
            data: DataFrame containing price data
            period: Period for WMA calculation
            price_column: Column name for price data (default: 'close')
            
        Returns:
            Series containing WMA values
        """
        self.log_action("calculate_indicator", f"Calculating WMA with period {period}")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
            
        weights = np.arange(1, period + 1)
        return data[price_column].rolling(period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )
    
    def calculate_hull_ma(self, data: pd.DataFrame, period: int, price_column: str = 'close') -> pd.Series:
        """
        Calculate Hull Moving Average (HMA).
        HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        
        Args:
            data: DataFrame containing price data
            period: Period for HMA calculation
            price_column: Column name for price data (default: 'close')
            
        Returns:
            Series containing HMA values
        """
        self.log_action("calculate_indicator", f"Calculating Hull MA with period {period}")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
            
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self.calculate_wma(data, half_period, price_column)
        wma_full = self.calculate_wma(data, period, price_column)
        
        # 2 * WMA(n/2) - WMA(n)
        inner_value = 2 * wma_half - wma_full
        
        # Create temporary DataFrame for the final WMA calculation
        temp_df = pd.DataFrame({price_column: inner_value})
        
        # WMA of the above with period sqrt(n)
        return self.calculate_wma(temp_df, sqrt_period, price_column)
    
    def calculate_vwap(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data: DataFrame containing price data with 'high', 'low', 'close', and 'volume' columns
            period: Period for VWAP calculation (None for entire dataset)
            
        Returns:
            Series containing VWAP values
        """
        self.log_action("calculate_indicator", f"Calculating VWAP with period {period if period else 'full'}")
        
        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate VWAP
        if period is None:
            # Cumulative VWAP for the entire dataset
            cumulative_tp_vol = (typical_price * data['volume']).cumsum()
            cumulative_vol = data['volume'].cumsum()
            vwap = cumulative_tp_vol / cumulative_vol
        else:
            # Rolling VWAP for the specified period
            tp_vol = typical_price * data['volume']
            vwap = tp_vol.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        
        return vwap
    
    # Oscillator Indicators
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14, price_column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame containing price data
            period: Period for RSI calculation (default: 14)
            price_column: Column name for price data (default: 'close')
            
        Returns:
            Series containing RSI values (0-100)
        """
        self.log_action("calculate_indicator", f"Calculating RSI with period {period}")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
            
        # Calculate price changes
        delta = data[price_column].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        Args:
            data: DataFrame containing price data with 'high', 'low', and 'close' columns
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D calculation (default: 3)
            
        Returns:
            Tuple of Series (k_line, d_line) containing Stochastic values (0-100)
        """
        self.log_action("calculate_indicator", f"Calculating Stochastic with K period {k_period}, D period {d_period}")
        
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate %K
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        k_line = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (SMA of %K)
        d_line = k_line.rolling(window=d_period).mean()
        
        return k_line, d_line
    
    def calculate_macd(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                      signal_period: int = 9, price_column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: DataFrame containing price data
            fast_period: Period for fast EMA (default: 12)
            slow_period: Period for slow EMA (default: 26)
            signal_period: Period for signal line (default: 9)
            price_column: Column name for price data (default: 'close')
            
        Returns:
            Tuple of Series (macd_line, signal_line, histogram)
        """
        self.log_action("calculate_indicator", 
                       f"Calculating MACD with fast period {fast_period}, slow period {slow_period}, signal period {signal_period}")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
            
        # Calculate fast and slow EMAs
        fast_ema = self.calculate_ema(data, fast_period, price_column)
        slow_ema = self.calculate_ema(data, slow_period, price_column)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        # Create a temporary DataFrame to use with calculate_ema
        temp_df = pd.DataFrame({price_column: macd_line})
        signal_line = self.calculate_ema(temp_df, signal_period, price_column)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            data: DataFrame containing price data with 'high', 'low', and 'close' columns
            period: Period for CCI calculation (default: 20)
            
        Returns:
            Series containing CCI values
        """
        self.log_action("calculate_indicator", f"Calculating CCI with period {period}")
        
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate the simple moving average of the typical price
        tp_sma = typical_price.rolling(window=period).mean()
        
        # Calculate the mean deviation
        # We use mean absolute deviation rather than standard deviation as per CCI definition
        mean_deviation = pd.Series(
            [abs(typical_price.iloc[i - period + 1:i + 1] - tp_sma.iloc[i]).mean() 
             for i in range(period - 1, len(typical_price))],
            index=typical_price.index[period - 1:])
        
        # Backfill the mean_deviation Series to match the length of the original data
        mean_deviation = mean_deviation.reindex(typical_price.index, fill_value=np.nan)
        
        # Calculate CCI
        cci = (typical_price - tp_sma) / (0.015 * mean_deviation)
        
        return cci
    
    # Trend Indicators
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX) with +DI and -DI.
        
        Args:
            data: DataFrame containing price data with 'high', 'low', and 'close' columns
            period: Period for ADX calculation (default: 14)
            
        Returns:
            Tuple of Series (adx, plus_di, minus_di)
        """
        self.log_action("calculate_indicator", f"Calculating ADX with period {period}")
        
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Calculate directional movement
        plus_dm = data['high'].diff()
        minus_dm = data['low'].diff(-1).abs()
        
        # Positive directional movement
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        
        # Negative directional movement
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        # Smooth the TR, +DM, and -DM with the Wilder's smoothing technique
        smoothed_tr = true_range.rolling(window=period).sum()
        smoothed_plus_dm = plus_dm.rolling(window=period).sum()
        smoothed_minus_dm = minus_dm.rolling(window=period).sum()
        
        # Calculate +DI and -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # Calculate directional index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX as the smoothed average of DX
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def calculate_aroon(self, data: pd.DataFrame, period: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Aroon Indicator (Aroon Up, Aroon Down, and Aroon Oscillator).
        
        Args:
            data: DataFrame containing price data with 'high' and 'low' columns
            period: Period for Aroon calculation (default: 25)
            
        Returns:
            Tuple of Series (aroon_up, aroon_down, aroon_oscillator)
        """
        self.log_action("calculate_indicator", f"Calculating Aroon with period {period}")
        
        required_columns = ['high', 'low']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate Aroon Up
        rolling_high = data['high'].rolling(window=period)
        aroon_up = 100 * rolling_high.apply(lambda x: x.argmax() / (period - 1), raw=False)
        
        # Calculate Aroon Down
        rolling_low = data['low'].rolling(window=period)
        aroon_down = 100 * rolling_low.apply(lambda x: x.argmin() / (period - 1), raw=False)
        
        # Calculate Aroon Oscillator
        aroon_oscillator = aroon_up - aroon_down
        
        return aroon_up, aroon_down, aroon_oscillator
    
    def calculate_ichimoku(self, data: pd.DataFrame, 
                          tenkan_period: int = 9, 
                          kijun_period: int = 26, 
                          senkou_b_period: int = 52, 
                          displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud indicator components.
        
        Args:
            data: DataFrame containing price data with 'high' and 'low' columns
            tenkan_period: Period for Tenkan-sen (Conversion Line) (default: 9)
            kijun_period: Period for Kijun-sen (Base Line) (default: 26)
            senkou_b_period: Period for Senkou Span B (default: 52)
            displacement: Displacement period for Senkou Span A and B (default: 26)
            
        Returns:
            Dictionary containing Ichimoku components:
            - tenkan_sen (Conversion Line)
            - kijun_sen (Base Line)
            - senkou_span_a (Leading Span A)
            - senkou_span_b (Leading Span B)
            - chikou_span (Lagging Span)
        """
        self.log_action("calculate_indicator", f"Calculating Ichimoku Cloud")
        
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_sen = (data['high'].rolling(window=tenkan_period).max() + 
                      data['low'].rolling(window=tenkan_period).min()) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_sen = (data['high'].rolling(window=kijun_period).max() + 
                    data['low'].rolling(window=kijun_period).min()) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_span_b = ((data['high'].rolling(window=senkou_b_period).max() + 
                         data['low'].rolling(window=senkou_b_period).min()) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = data['close'].shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    # Volatility Indicators
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, 
                                 price_column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame containing price data
            period: Period for moving average calculation (default: 20)
            std_dev: Number of standard deviations for the bands (default: 2.0)
            price_column: Column name for price data (default: 'close')
            
        Returns:
            Tuple of Series (upper_band, middle_band, lower_band)
        """
        self.log_action("calculate_indicator", f"Calculating Bollinger Bands with period {period}, std_dev {std_dev}")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        # Calculate middle band (SMA)
        middle_band = self.calculate_sma(data, period, price_column)
        
        # Calculate standard deviation
        std = data[price_column].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame containing price data with 'high', 'low', and 'close' columns
            period: Period for ATR calculation (default: 14)
            
        Returns:
            Series containing ATR values
        """
        self.log_action("calculate_indicator", f"Calculating ATR with period {period}")
        
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Calculate ATR using an Exponential Moving Average
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def calculate_keltner_channel(self, data: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, 
                                 atr_multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels.
        
        Args:
            data: DataFrame containing price data with 'high', 'low', and 'close' columns
            ema_period: Period for EMA calculation (default: 20)
            atr_period: Period for ATR calculation (default: 10)
            atr_multiplier: Multiplier for ATR (default: 2.0)
            
        Returns:
            Tuple of Series (upper_channel, middle_line, lower_channel)
        """
        self.log_action("calculate_indicator", f"Calculating Keltner Channels with EMA period {ema_period}, ATR period {atr_period}")
        
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate middle line (EMA)
        middle_line = self.calculate_ema(data, ema_period, 'close')
        
        # Calculate ATR
        atr = self.calculate_atr(data, atr_period)
        
        # Calculate upper and lower channels
        upper_channel = middle_line + (atr_multiplier * atr)
        lower_channel = middle_line - (atr_multiplier * atr)
        
        return upper_channel, middle_line, lower_channel
    
    def calculate_donchian_channel(self, data: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Channels.
        
        Args:
            data: DataFrame containing price data with 'high' and 'low' columns
            period: Period for the channel calculation (default: 20)
            
        Returns:
            Tuple of Series (upper_channel, middle_channel, lower_channel)
        """
        self.log_action("calculate_indicator", f"Calculating Donchian Channels with period {period}")
        
        required_columns = ['high', 'low']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate upper channel (highest high)
        upper_channel = data['high'].rolling(window=period).max()
        
        # Calculate lower channel (lowest low)
        lower_channel = data['low'].rolling(window=period).min()
        
        # Calculate middle channel
        middle_channel = (upper_channel + lower_channel) / 2
        
        return upper_channel, middle_channel, lower_channel
    
    # Volume Indicators
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            data: DataFrame containing price data with 'close' and 'volume' columns
            
        Returns:
            Series containing OBV values
        """
        self.log_action("calculate_indicator", f"Calculating On-Balance Volume")
        
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Initialize OBV with the first volume value
        obv = pd.Series(0, index=data.index)
        
        # Calculate OBV based on price movements
        price_changes = data['close'].diff()
        
        # When price increases, add volume
        obv.loc[price_changes > 0] = data.loc[price_changes > 0, 'volume']
        
        # When price decreases, subtract volume
        obv.loc[price_changes < 0] = -data.loc[price_changes < 0, 'volume']
        
        # When price is unchanged, OBV is unchanged (volume = 0)
        obv.loc[price_changes == 0] = 0
        
        # Calculate cumulative sum
        obv = obv.cumsum()
        
        return obv
    
    def calculate_money_flow_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            data: DataFrame containing price data with 'high', 'low', 'close', and 'volume' columns
            period: Period for MFI calculation (default: 14)
            
        Returns:
            Series containing MFI values (0-100)
        """
        self.log_action("calculate_indicator", f"Calculating Money Flow Index with period {period}")
        
        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * data['volume']
        
        # Get positive and negative money flow
        tp_diff = typical_price.diff()
        positive_flow = pd.Series(np.where(tp_diff > 0, raw_money_flow, 0), index=data.index)
        negative_flow = pd.Series(np.where(tp_diff < 0, raw_money_flow, 0), index=data.index)
        
        # Sum positive and negative money flows
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Calculate money flow ratio
        money_flow_ratio = positive_mf / negative_mf
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    def calculate_volume_rate_of_change(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Volume Rate of Change.
        
        Args:
            data: DataFrame containing price data with 'volume' column
            period: Period for calculation (default: 14)
            
        Returns:
            Series containing Volume Rate of Change values
        """
        self.log_action("calculate_indicator", f"Calculating Volume Rate of Change with period {period}")
        
        if 'volume' not in data.columns:
            raise ValueError("'volume' column not found in data")
        
        # Calculate Volume Rate of Change
        v_roc = ((data['volume'] - data['volume'].shift(period)) / data['volume'].shift(period)) * 100
        
        return v_roc
    
    def calculate_accumulation_distribution(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line (A/D Line).
        
        Args:
            data: DataFrame containing price data with 'high', 'low', 'close', and 'volume' columns
            
        Returns:
            Series containing A/D Line values
        """
        self.log_action("calculate_indicator", f"Calculating Accumulation/Distribution Line")
        
        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate Close Location Value (CLV)
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        
        # Handle cases where high equals low (prevent division by zero)
        clv = clv.replace([np.inf, -np.inf], 0)
        clv = clv.fillna(0)
        
        # Calculate Money Flow Volume
        money_flow_volume = clv * data['volume']
        
        # Calculate Accumulation/Distribution Line
        ad_line = money_flow_volume.cumsum()
        
        return ad_line
    
    # Pattern Recognition Methods
    
    def identify_support_resistance(self, data: pd.DataFrame, window: int = 10, threshold: float = 0.03) -> Dict[str, pd.Series]:
        """
        Identify support and resistance levels.
        
        Args:
            data: DataFrame containing price data with 'high' and 'low' columns
            window: Window size for peak/trough detection (default: 10)
            threshold: Price movement threshold to qualify as support/resistance (default: 0.03 or 3%)
            
        Returns:
            Dictionary with 'supports' and 'resistances' Series
        """
        self.log_action("pattern_recognition", f"Identifying support and resistance levels with window {window}")
        
        required_columns = ['high', 'low']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Find resistance levels (peaks in high prices)
        highs = data['high'].copy()
        resistance_points = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(highs) - window):
            if all(highs.iloc[i] > highs.iloc[i-window:i]) and all(highs.iloc[i] > highs.iloc[i+1:i+window+1]):
                # This is a peak
                resistance_points.iloc[i] = highs.iloc[i]
        
        # Find support levels (troughs in low prices)
        lows = data['low'].copy()
        support_points = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(lows) - window):
            if all(lows.iloc[i] < lows.iloc[i-window:i]) and all(lows.iloc[i] < lows.iloc[i+1:i+window+1]):
                # This is a trough
                support_points.iloc[i] = lows.iloc[i]
        
        # Clean up by removing insignificant levels
        if 'close' in data.columns:
            avg_price = data['close'].mean()
            min_move = avg_price * threshold
            
            # Filter out resistance points too close to each other
            real_resistances = []
            last_resistance = None
            resistance_indices = resistance_points.dropna().index
            
            for idx in resistance_indices:
                if last_resistance is None or abs(resistance_points[idx] - last_resistance) > min_move:
                    real_resistances.append((idx, resistance_points[idx]))
                    last_resistance = resistance_points[idx]
            
            # Filter out support points too close to each other
            real_supports = []
            last_support = None
            support_indices = support_points.dropna().index
            
            for idx in support_indices:
                if last_support is None or abs(support_points[idx] - last_support) > min_move:
                    real_supports.append((idx, support_points[idx]))
                    last_support = support_points[idx]
            
            # Create series with only the significant points
            resistance_points = pd.Series({idx: val for idx, val in real_resistances}, dtype=float)
            support_points = pd.Series({idx: val for idx, val in real_supports}, dtype=float)
        
        return {
            'supports': support_points,
            'resistances': resistance_points
        }
    
    def detect_chart_patterns(self, data: pd.DataFrame, pattern_types: List[str] = None) -> Dict[str, pd.Series]:
        """
        Detect common chart patterns.
        
        Args:
            data: DataFrame containing price data with OHLC columns
            pattern_types: List of patterns to detect (if None, detect all available patterns)
            
        Returns:
            Dictionary with pattern names as keys and Series of pattern locations as values
        """
        self.log_action("pattern_recognition", f"Detecting chart patterns")
        
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Default to detecting all patterns if none specified
        all_patterns = ['head_and_shoulders', 'inverse_head_and_shoulders', 'double_top', 
                       'double_bottom', 'triangle', 'flag', 'wedge']
        
        if pattern_types is None:
            pattern_types = all_patterns
        
        # Validate pattern types
        for pattern in pattern_types:
            if pattern not in all_patterns:
                raise ValueError(f"Unsupported pattern type: {pattern}")
        
        patterns = {}
        
        # Helper function for smoother price data
        def smooth_prices(prices, window=5):
            return prices.rolling(window=window).mean()
        
        # Detect Head and Shoulders pattern
        if 'head_and_shoulders' in pattern_types:
            patterns['head_and_shoulders'] = self._detect_head_shoulders(data, inverse=False)
        
        # Detect Inverse Head and Shoulders pattern
        if 'inverse_head_and_shoulders' in pattern_types:
            patterns['inverse_head_and_shoulders'] = self._detect_head_shoulders(data, inverse=True)
        
        # Detect Double Top pattern
        if 'double_top' in pattern_types:
            patterns['double_top'] = self._detect_double_formation(data, formation_type='top')
        
        # Detect Double Bottom pattern
        if 'double_bottom' in pattern_types:
            patterns['double_bottom'] = self._detect_double_formation(data, formation_type='bottom')
        
        # Detect Triangle patterns
        if 'triangle' in pattern_types:
            patterns['triangle'] = self._detect_triangle(data)
        
        # Detect Flag patterns
        if 'flag' in pattern_types:
            patterns['bull_flag'] = self._detect_flag(data, bull=True)
            patterns['bear_flag'] = self._detect_flag(data, bull=False)
        
        # Detect Wedge patterns
        if 'wedge' in pattern_types:
            patterns['rising_wedge'] = self._detect_wedge(data, rising=True)
            patterns['falling_wedge'] = self._detect_wedge(data, rising=False)
        
        return patterns
    
    def _detect_head_shoulders(self, data: pd.DataFrame, inverse: bool = False) -> pd.Series:
        """
        Helper method to detect head and shoulders or inverse head and shoulders patterns.
        
        Args:
            data: DataFrame containing price data
            inverse: If True, detect inverse head and shoulders (bottom formation)
            
        Returns:
            Series with pattern locations
        """
        # This is a simplified detection algorithm
        # A full implementation would use more sophisticated peak/trough detection
        window = 5
        price = data['low'] if inverse else data['high']
        pattern_locations = pd.Series(0, index=data.index)
        
        # Calculate rate of change to identify peaks and troughs
        roc = price.pct_change(periods=window)
        
        # Find potential peaks/troughs
        for i in range(window * 3, len(price) - window * 3):
            # For head and shoulders (3 peaks with middle one higher)
            if not inverse:
                # Check if we have a left shoulder, head, right shoulder formation
                if (roc.iloc[i-window*2] > 0 and roc.iloc[i-window] < 0 and  # Left shoulder
                    roc.iloc[i-window] < 0 and roc.iloc[i] > 0 and           # Head
                    roc.iloc[i] > 0 and roc.iloc[i+window] < 0 and           # Head to right shoulder
                    price.iloc[i] > price.iloc[i-window*2] and               # Head higher than left shoulder
                    price.iloc[i] > price.iloc[i+window*2]):                 # Head higher than right shoulder
                    pattern_locations.iloc[i] = 1
            
            # For inverse head and shoulders (3 troughs with middle one lower)
            else:
                # Check if we have a left shoulder, head, right shoulder formation (inverted)
                if (roc.iloc[i-window*2] < 0 and roc.iloc[i-window] > 0 and  # Left shoulder
                    roc.iloc[i-window] > 0 and roc.iloc[i] < 0 and           # Head
                    roc.iloc[i] < 0 and roc.iloc[i+window] > 0 and           # Head to right shoulder
                    price.iloc[i] < price.iloc[i-window*2] and               # Head lower than left shoulder
                    price.iloc[i] < price.iloc[i+window*2]):                 # Head lower than right shoulder
                    pattern_locations.iloc[i] = 1
        
        return pattern_locations
    
    def _detect_double_formation(self, data: pd.DataFrame, formation_type: str = 'top') -> pd.Series:
        """
        Helper method to detect double top or double bottom patterns.
        
        Args:
            data: DataFrame containing price data
            formation_type: 'top' for double top, 'bottom' for double bottom
            
        Returns:
            Series with pattern locations
        """
        window = 10
        price = data['high'] if formation_type == 'top' else data['low']
        pattern_locations = pd.Series(0, index=data.index)
        
        # Find peaks or troughs
        for i in range(window, len(price) - window):
            # For double top (two similar peaks)
            if formation_type == 'top':
                if i >= 2*window:
                    # Find first peak
                    if all(price.iloc[i-window] > price.iloc[i-window-window:i-window]) and \
                       all(price.iloc[i-window] > price.iloc[i-window+1:i]):
                        first_peak = price.iloc[i-window]
                        
                        # Find second peak
                        if all(price.iloc[i] > price.iloc[i-window:i]) and \
                           all(price.iloc[i] > price.iloc[i+1:i+window+1]):
                            second_peak = price.iloc[i]
                            
                            # Verify similar heights
                            if abs(first_peak - second_peak) / first_peak < 0.03:  # Within 3%
                                pattern_locations.iloc[i] = 1
            
            # For double bottom (two similar troughs)
            else:
                if i >= 2*window:
                    # Find first trough
                    if all(price.iloc[i-window] < price.iloc[i-window-window:i-window]) and \
                       all(price.iloc[i-window] < price.iloc[i-window+1:i]):
                        first_trough = price.iloc[i-window]
                        
                        # Find second trough
                        if all(price.iloc[i] < price.iloc[i-window:i]) and \
                           all(price.iloc[i] < price.iloc[i+1:i+window+1]):
                            second_trough = price.iloc[i]
                            
                            # Verify similar depths
                            if abs(first_trough - second_trough) / first_trough < 0.03:  # Within 3%
                                pattern_locations.iloc[i] = 1
        
        return pattern_locations
    
    def _detect_triangle(self, data: pd.DataFrame) -> pd.Series:
        """
        Helper method to detect triangle patterns.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Series with pattern locations
        """
        # Placeholder for triangle pattern detection
        # A proper implementation would use linear regression to detect converging trendlines
        pattern_locations = pd.Series(0, index=data.index)
        
        # Simplified triangle detection (conceptual example)
        highs = data['high'].rolling(window=3).max()
        lows = data['low'].rolling(window=3).min()
        
        high_slope = highs.diff(20) / 20
        low_slope = lows.diff(20) / 20
        
        # Check for converging slopes (simplified)
        for i in range(40, len(data) - 5):
            if (high_slope.iloc[i] < -0.0001 and low_slope.iloc[i] > 0.0001):
                # Potential symmetrical triangle
                pattern_locations.iloc[i] = 1
        
        return pattern_locations
    
    def _detect_flag(self, data: pd.DataFrame, bull: bool = True) -> pd.Series:
        """
        Helper method to detect flag patterns.
        
        Args:
            data: DataFrame containing price data
            bull: If True, detect bullish flags, otherwise bearish flags
            
        Returns:
            Series with pattern locations
        """
        # Simplified flag pattern detection
        pattern_locations = pd.Series(0, index=data.index)
        
        # Will be implemented in a more comprehensive version
        
        return pattern_locations
    
    def _detect_wedge(self, data: pd.DataFrame, rising: bool = True) -> pd.Series:
        """
        Helper method to detect wedge patterns.
        
        Args:
            data: DataFrame containing price data
            rising: If True, detect rising wedge, otherwise falling wedge
            
        Returns:
            Series with pattern locations
        """
        # Simplified wedge pattern detection
        pattern_locations = pd.Series(0, index=data.index)
        
        # Will be implemented in a more comprehensive version
        
        return pattern_locations
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detect common candlestick patterns.
        
        Args:
            data: DataFrame containing price data with OHLC columns
            
        Returns:
            Dictionary with pattern names as keys and Series of pattern locations as values
        """
        self.log_action("pattern_recognition", f"Detecting candlestick patterns")
        
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        patterns = {}
        
        # Doji pattern (open and close are very close)
        doji = pd.Series(0, index=data.index)
        body_size = abs(data['close'] - data['open'])
        wick_size = data['high'] - data['low']
        doji_condition = body_size < (wick_size * 0.1)  # Body is less than 10% of the range
        doji[doji_condition] = 1
        patterns['doji'] = doji
        
        # Hammer pattern (small body at the top with a long lower wick)
        hammer = pd.Series(0, index=data.index)
        body_top = data[['open', 'close']].max(axis=1)
        body_bottom = data[['open', 'close']].min(axis=1)
        lower_wick = body_bottom - data['low']
        upper_wick = data['high'] - body_top
        
        hammer_condition = (
            (lower_wick > (2 * body_size)) &  # Lower wick at least 2x the body
            (upper_wick < (0.1 * body_size))  # Very small or no upper wick
        )
        hammer[hammer_condition] = 1
        patterns['hammer'] = hammer
        
        # Shooting Star pattern (small body at the bottom with a long upper wick)
        shooting_star = pd.Series(0, index=data.index)
        
        shooting_star_condition = (
            (upper_wick > (2 * body_size)) &  # Upper wick at least 2x the body
            (lower_wick < (0.1 * body_size))  # Very small or no lower wick
        )
        shooting_star[shooting_star_condition] = 1
        patterns['shooting_star'] = shooting_star
        
        # Bullish Engulfing pattern
        bullish_engulfing = pd.Series(0, index=data.index)
        
        for i in range(1, len(data)):
            # Current candle completely engulfs previous candle
            if (data['open'].iloc[i] <= data['close'].iloc[i-1] and  # Open below previous close
                data['close'].iloc[i] >= data['open'].iloc[i-1] and  # Close above previous open
                data['close'].iloc[i] > data['open'].iloc[i] and     # Current is bullish (green)
                data['close'].iloc[i-1] < data['open'].iloc[i-1]):   # Previous is bearish (red)
                bullish_engulfing.iloc[i] = 1
        
        patterns['bullish_engulfing'] = bullish_engulfing
        
        # Bearish Engulfing pattern
        bearish_engulfing = pd.Series(0, index=data.index)
        
        for i in range(1, len(data)):
            # Current candle completely engulfs previous candle
            if (data['open'].iloc[i] >= data['close'].iloc[i-1] and  # Open above previous close
                data['close'].iloc[i] <= data['open'].iloc[i-1] and  # Close below previous open
                data['close'].iloc[i] < data['open'].iloc[i] and     # Current is bearish (red)
                data['close'].iloc[i-1] > data['open'].iloc[i-1]):   # Previous is bullish (green)
                bearish_engulfing.iloc[i] = 1
        
        patterns['bearish_engulfing'] = bearish_engulfing
        
        return patterns
    
    def detect_divergence(self, data: pd.DataFrame, indicator: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """
        Detect divergence between price and an indicator.
        
        Args:
            data: DataFrame containing price data with 'close' column
            indicator: Series containing indicator values
            window: Window size for peak/trough detection (default: 20)
            
        Returns:
            Dictionary with 'bullish_divergence' and 'bearish_divergence' Series
        """
        self.log_action("pattern_recognition", f"Detecting divergence with window {window}")
        
        if 'close' not in data.columns:
            raise ValueError("'close' column not found in data")
        
        if len(indicator) != len(data):
            raise ValueError("Indicator must have the same length as price data")
        
        # Ensure indicator has the same index as data
        indicator = pd.Series(indicator.values, index=data.index)
        
        # Initialize results
        bullish_divergence = pd.Series(0, index=data.index)
        bearish_divergence = pd.Series(0, index=data.index)
        
        # Find highs and lows in price and indicator
        for i in range(window, len(data) - window):
            # Check for higher highs in price
            if (data['close'].iloc[i] > data['close'].iloc[i-1:i+1].max() and 
                data['close'].iloc[i] > data['close'].iloc[i+1:i+window+1].max()):
                price_high = True
            else:
                price_high = False
                
            # Check for lower lows in price
            if (data['close'].iloc[i] < data['close'].iloc[i-1:i+1].min() and 
                data['close'].iloc[i] < data['close'].iloc[i+1:i+window+1].min()):
                price_low = True
            else:
                price_low = False
                
            # Check for higher highs in indicator
            if (indicator.iloc[i] > indicator.iloc[i-1:i+1].max() and 
                indicator.iloc[i] > indicator.iloc[i+1:i+window+1].max()):
                indicator_high = True
            else:
                indicator_high = False
                
            # Check for lower lows in indicator
            if (indicator.iloc[i] < indicator.iloc[i-1:i+1].min() and 
                indicator.iloc[i] < indicator.iloc[i+1:i+window+1].min()):
                indicator_low = True
            else:
                indicator_low = False
            
            # Bearish divergence: Higher high in price but lower high in indicator
            if price_high and not indicator_high:
                bearish_divergence.iloc[i] = 1
                
            # Bullish divergence: Lower low in price but higher low in indicator
            if price_low and not indicator_low:
                bullish_divergence.iloc[i] = 1
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    # Signal Generation Methods
    
    def generate_trend_signals(self, data: pd.DataFrame, ma_short: int = 20, ma_long: int = 50) -> pd.Series:
        """
        Generate signals based on trend analysis.
        
        Args:
            data: DataFrame containing price data with 'close' column
            ma_short: Period for short-term moving average (default: 20)
            ma_long: Period for long-term moving average (default: 50)
            
        Returns:
            Series with signal values (1 for buy, -1 for sell, 0 for no signal)
        """
        self.log_action("signal_generation", f"Generating trend signals")
        
        if 'close' not in data.columns:
            raise ValueError("'close' column not found in data")
            
        # Calculate moving averages
        ma_short_values = self.calculate_ema(data, ma_short)
        ma_long_values = self.calculate_ema(data, ma_long)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on moving average crossover
        for i in range(1, len(data)):
            # Golden Cross (short MA crosses above long MA)
            if ma_short_values.iloc[i-1] <= ma_long_values.iloc[i-1] and ma_short_values.iloc[i] > ma_long_values.iloc[i]:
                signals.iloc[i] = 1
            # Death Cross (short MA crosses below long MA)
            elif ma_short_values.iloc[i-1] >= ma_long_values.iloc[i-1] and ma_short_values.iloc[i] < ma_long_values.iloc[i]:
                signals.iloc[i] = -1
        
        return signals
    
    def generate_reversal_signals(self, data: pd.DataFrame, rsi_period: int = 14, 
                                 oversold: int = 30, overbought: int = 70) -> pd.Series:
        """
        Generate signals for potential reversals.
        
        Args:
            data: DataFrame containing price data
            rsi_period: Period for RSI calculation (default: 14)
            oversold: RSI threshold for oversold condition (default: 30)
            overbought: RSI threshold for overbought condition (default: 70)
            
        Returns:
            Series with signal values (1 for buy, -1 for sell, 0 for no signal)
        """
        self.log_action("signal_generation", f"Generating reversal signals")
        
        # Calculate RSI
        rsi = self.calculate_rsi(data, rsi_period)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on RSI
        for i in range(1, len(data)):
            # Buy signal: RSI crosses above oversold threshold
            if rsi.iloc[i-1] <= oversold and rsi.iloc[i] > oversold:
                signals.iloc[i] = 1
            # Sell signal: RSI crosses below overbought threshold
            elif rsi.iloc[i-1] >= overbought and rsi.iloc[i] < overbought:
                signals.iloc[i] = -1
        
        return signals
    
    def generate_breakout_signals(self, data: pd.DataFrame, period: int = 20, threshold: float = 0.01) -> pd.Series:
        """
        Generate signals for breakouts.
        
        Args:
            data: DataFrame containing price data with 'high', 'low', and 'close' columns
            period: Period for identifying support/resistance (default: 20)
            threshold: Threshold for confirming breakout (default: 0.01 or 1%)
            
        Returns:
            Series with signal values (1 for buy, -1 for sell, 0 for no signal)
        """
        self.log_action("signal_generation", f"Generating breakout signals")
        
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate resistance and support levels
        highs = data['high'].rolling(window=period).max()
        lows = data['low'].rolling(window=period).min()
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on breakouts
        for i in range(period, len(data)):
            # Bullish breakout: Close breaks above previous high
            if data['close'].iloc[i] > highs.iloc[i-1] * (1 + threshold):
                signals.iloc[i] = 1
            # Bearish breakout: Close breaks below previous low
            elif data['close'].iloc[i] < lows.iloc[i-1] * (1 - threshold):
                signals.iloc[i] = -1
        
        return signals
    
    def combine_signals(self, signals: Dict[str, pd.Series], weights: Dict[str, float] = None) -> pd.Series:
        """
        Combine multiple signals into a weighted recommendation.
        
        Args:
            signals: Dictionary of named signals
            weights: Dictionary of weights for each signal (default: equal weights)
            
        Returns:
            Series with combined signal values
        """
        self.log_action("signal_generation", f"Combining {len(signals)} signals")
        
        if not signals:
            return pd.Series()
        
        # Use equal weights if not provided
        if weights is None:
            weights = {name: 1.0 / len(signals) for name in signals}
        else:
            # Validate weights
            missing_signals = set(signals.keys()) - set(weights.keys())
            if missing_signals:
                raise ValueError(f"Missing weights for signals: {missing_signals}")
        
        # Check that all signals have the same index
        index = next(iter(signals.values())).index
        for name, signal in signals.items():
            if not signal.index.equals(index):
                raise ValueError(f"Signal '{name}' has a different index")
        
        # Initialize combined signal
        combined = pd.Series(0, index=index)
        
        # Add weighted signals
        for name, signal in signals.items():
            combined += signal * weights[name]
        
        return combined
    
    # Strategy Implementation Methods
    
    def implement_strategy(self, data: pd.DataFrame, strategy_name: str, 
                          params: Dict[str, Any] = None) -> Dict[str, pd.Series]:
        """
        Apply a predefined strategy.
        
        Args:
            data: DataFrame containing price data
            strategy_name: Name of the strategy to implement
            params: Parameters for the strategy (default: None, use strategy defaults)
            
        Returns:
            Dictionary with 'signals', 'positions', 'entry_prices', and 'exit_prices' Series
        """
        self.log_action("strategy", f"Implementing {strategy_name} strategy")
        
        # Define available strategies
        strategies = {
            'moving_average_crossover': self._ma_crossover_strategy,
            'rsi_reversal': self._rsi_reversal_strategy,
            'bollinger_bands': self._bollinger_bands_strategy,
            'macd': self._macd_strategy,
            'breakout': self._breakout_strategy
        }
        
        # Check if strategy exists
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {list(strategies.keys())}")
        
        # Get strategy parameters (use defaults from settings if not provided)
        strategy_params = {}
        if strategy_name in self.strategy_settings:
            strategy_params.update(self.strategy_settings[strategy_name])
        if params:
            strategy_params.update(params)
        
        # Execute the strategy
        strategy_function = strategies[strategy_name]
        return strategy_function(data, **strategy_params)
    
    def _ma_crossover_strategy(self, data: pd.DataFrame, fast_period: int = 10, 
                              slow_period: int = 50) -> Dict[str, pd.Series]:
        """
        Implement Moving Average Crossover strategy.
        
        Args:
            data: DataFrame containing price data
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate moving averages
        fast_ma = self.calculate_ema(data, fast_period)
        slow_ma = self.calculate_ema(data, slow_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        entry_prices = pd.Series(np.nan, index=data.index)
        exit_prices = pd.Series(np.nan, index=data.index)
        
        current_position = 0
        
        for i in range(1, len(data)):
            # Buy signal: fast MA crosses above slow MA
            if fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: fast MA crosses below slow MA
            elif fast_ma.iloc[i-1] >= slow_ma.iloc[i-1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = data['close'].iloc[i]
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
    
    def _rsi_reversal_strategy(self, data: pd.DataFrame, rsi_period: int = 14, 
                              oversold: int = 30, overbought: int = 70) -> Dict[str, pd.Series]:
        """
        Implement RSI Reversal strategy.
        
        Args:
            data: DataFrame containing price data
            rsi_period: Period for RSI calculation
            oversold: RSI threshold for oversold condition
            overbought: RSI threshold for overbought condition
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate RSI
        rsi = self.calculate_rsi(data, rsi_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        entry_prices = pd.Series(np.nan, index=data.index)
        exit_prices = pd.Series(np.nan, index=data.index)
        
        current_position = 0
        
        for i in range(1, len(data)):
            # Buy signal: RSI crosses above oversold threshold
            if rsi.iloc[i-1] <= oversold and rsi.iloc[i] > oversold:
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: RSI crosses below overbought threshold
            elif rsi.iloc[i-1] >= overbought and rsi.iloc[i] < overbought:
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = data['close'].iloc[i]
                    current_position = -1
            
            positions.iloc[i] = current_position
        
        return {
            'signals': signals,
            'positions': positions,
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'rsi': rsi
        }
    
    def _bollinger_bands_strategy(self, data: pd.DataFrame, period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Implement Bollinger Bands strategy.
        
        Args:
            data: DataFrame containing price data
            period: Period for moving average calculation
            std_dev: Number of standard deviations for the bands
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(data, period, std_dev)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        entry_prices = pd.Series(np.nan, index=data.index)
        exit_prices = pd.Series(np.nan, index=data.index)
        
        current_position = 0
        
        for i in range(1, len(data)):
            # Buy signal: price crosses above lower band
            if data['close'].iloc[i-1] <= lower.iloc[i-1] and data['close'].iloc[i] > lower.iloc[i]:
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: price crosses below upper band
            elif data['close'].iloc[i-1] >= upper.iloc[i-1] and data['close'].iloc[i] < upper.iloc[i]:
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = data['close'].iloc[i]
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
    
    def _macd_strategy(self, data: pd.DataFrame, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Implement MACD strategy.
        
        Args:
            data: DataFrame containing price data
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate MACD
        macd_line, signal_line, histogram = self.calculate_macd(data, fast_period, slow_period, signal_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        entry_prices = pd.Series(np.nan, index=data.index)
        exit_prices = pd.Series(np.nan, index=data.index)
        
        current_position = 0
        
        for i in range(1, len(data)):
            # Buy signal: MACD line crosses above signal line
            if macd_line.iloc[i-1] <= signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: MACD line crosses below signal line
            elif macd_line.iloc[i-1] >= signal_line.iloc[i-1] and macd_line.iloc[i] < signal_line.iloc[i]:
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = data['close'].iloc[i]
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
    
    def _breakout_strategy(self, data: pd.DataFrame, period: int = 20, 
                          threshold: float = 0.01) -> Dict[str, pd.Series]:
        """
        Implement Breakout strategy.
        
        Args:
            data: DataFrame containing price data
            period: Period for identifying support/resistance
            threshold: Threshold for confirming breakout
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate resistance and support levels
        highs = data['high'].rolling(window=period).max()
        lows = data['low'].rolling(window=period).min()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        entry_prices = pd.Series(np.nan, index=data.index)
        exit_prices = pd.Series(np.nan, index=data.index)
        
        current_position = 0
        
        for i in range(period, len(data)):
            # Buy signal: price breaks above resistance
            if data['close'].iloc[i] > highs.iloc[i-1] * (1 + threshold):
                signals.iloc[i] = 1
                if current_position <= 0:  # If not already long
                    entry_prices.iloc[i] = data['close'].iloc[i]
                    current_position = 1
            
            # Sell signal: price breaks below support
            elif data['close'].iloc[i] < lows.iloc[i-1] * (1 - threshold):
                signals.iloc[i] = -1
                if current_position >= 0:  # If not already short
                    exit_prices.iloc[i] = data['close'].iloc[i]
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
    
    def backtest_strategy(self, data: pd.DataFrame, strategy: str, 
                         params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Test a strategy on historical data.
        
        Args:
            data: DataFrame containing price data
            strategy: Name of the strategy to backtest
            params: Strategy parameters (default: None)
            
        Returns:
            Dictionary with backtest results
        """
        self.log_action("backtest", f"Backtesting {strategy} strategy")
        
        # Implement the strategy
        results = self.implement_strategy(data, strategy, params)
        
        # Extract signals and positions
        signals = results['signals']
        positions = results['positions']
        entry_prices = results['entry_prices']
        exit_prices = results['exit_prices']
        
        # Calculate returns
        strategy_returns = pd.Series(0.0, index=data.index)
        cumulative_returns = pd.Series(1.0, index=data.index)
        
        # Loop through the positions and calculate returns
        for i in range(1, len(data)):
            if positions.iloc[i-1] != 0:
                # Calculate returns based on position
                if positions.iloc[i-1] > 0:  # Long position
                    strategy_returns.iloc[i] = (data['close'].iloc[i] / data['close'].iloc[i-1]) - 1
                else:  # Short position
                    strategy_returns.iloc[i] = 1 - (data['close'].iloc[i] / data['close'].iloc[i-1])
                
                # Apply position size
                strategy_returns.iloc[i] *= positions.iloc[i-1]
            
            # Calculate cumulative returns
            cumulative_returns.iloc[i] = cumulative_returns.iloc[i-1] * (1 + strategy_returns.iloc[i])
        
        # Calculate buy & hold returns for comparison
        buy_hold_returns = (data['close'] / data['close'].iloc[0])
        
        # Calculate performance metrics
        total_trades = sum(abs(signals) > 0)
        profitable_trades = sum((signals * strategy_returns) > 0)
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate maximum drawdown
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (annualized)
        risk_free_rate = 0.0  # Assume zero risk-free rate for simplicity
        sharpe_ratio = np.sqrt(252) * (strategy_returns.mean() - risk_free_rate) / strategy_returns.std() if strategy_returns.std() > 0 else 0
        
        # Calculate CAGR (Compound Annual Growth Rate)
        days = (data.index[-1] - data.index[0]).days
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
    
    def optimize_parameters(self, data: pd.DataFrame, strategy: str, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Find optimal parameters for a strategy.
        
        Args:
            data: DataFrame containing price data
            strategy: Name of the strategy to optimize
            param_grid: Dictionary with parameter names as keys and lists of values to test
            
        Returns:
            Dictionary with optimal parameters and performance metrics
        """
        self.log_action("optimize", f"Optimizing parameters for {strategy} strategy")
        
        # Track best parameters and performance
        best_params = None
        best_performance = -float('inf')  # Initialize with negative infinity
        all_results = []
        
        # Determine the performance metric to optimize (Sharpe ratio by default)
        performance_metric = 'sharpe_ratio'
        
        # Generate all combinations of parameters
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Test each combination
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            
            # Run backtest with these parameters
            backtest_result = self.backtest_strategy(data, strategy, params)
            
            # Extract performance metric
            performance = backtest_result[performance_metric]
            
            # Store results
            result = {
                'params': params,
                'performance': backtest_result[performance_metric],
                'sharpe_ratio': backtest_result['sharpe_ratio'],
                'final_return': backtest_result['final_return'],
                'max_drawdown': backtest_result['max_drawdown'],
        } 