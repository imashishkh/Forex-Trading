#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Market Data Agent implementation for the Forex Trading Platform

This module provides a comprehensive implementation of a market data agent
that fetches, processes, and stores forex market data from the OANDA API.
"""

import os
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
import time
import requests
from requests.exceptions import RequestException, Timeout, HTTPError
import pandas as pd
import numpy as np
from functools import lru_cache

# OANDA API
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts

# Base Agent class
from utils.base_agent import BaseAgent, AgentState, AgentMessage

# Config Manager
from utils.config_manager import ConfigManager

# LangGraph imports
import langgraph.graph as lg
from langgraph.checkpoint.memory import MemorySaver


class MarketDataAgent(BaseAgent):
    """
    Market Data Agent for retrieving and processing forex market data.
    
    This agent is responsible for connecting to the OANDA API, fetching
    various types of market data, processing it, and storing it for use
    by other agents in the forex trading platform.
    """
    
    def __init__(
        self,
        agent_name: str = "market_data_agent",
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Market Data Agent.
        
        Args:
            agent_name: Name identifier for the agent
            llm: Language model to use (defaults to OpenAI if None)
            config: Configuration parameters for the agent
            logger: Logger instance for the agent
        """
        # Initialize BaseAgent
        super().__init__(agent_name, llm, config, logger)
        
        # If config not provided, use ConfigManager
        if config is None:
            config_manager = ConfigManager()
            self.config = config_manager.as_dict()
        
        # OANDA API configuration
        self.oanda_config = self.config.get('api_credentials', {}).get('oanda', {})
        self.api_key = self.oanda_config.get('api_key')
        self.account_id = self.oanda_config.get('account_id')
        self.api_url = self.oanda_config.get('api_url')
        
        # Determine environment (practice or live)
        self.is_practice = 'practice' in self.api_url.lower() if self.api_url else True
        
        # OANDA API client
        self.api_client = None
        
        # Data storage paths
        self.data_dir = Path(self.config.get('system', {}).get('data_storage_path', 'data'))
        self.market_data_dir = self.data_dir / 'market_data'
        
        # Cache settings
        self.cache_enabled = self.config.get('system', {}).get('cache_enabled', True)
        self.cache_expiry_days = self.config.get('system', {}).get('cache_expiry_days', 7)
        
        # Internal cache for frequently accessed data
        self._data_cache = {}
        
        # Set of available instruments
        self._available_instruments = set()
        
        # Max retry attempts for API connection
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # Initialize data directories
        self._ensure_data_dirs_exist()
        
        self.log_action("init", f"Market Data Agent initialized. Practice mode: {self.is_practice}")
    
    def _ensure_data_dirs_exist(self) -> None:
        """Create data directories if they don't exist."""
        self.market_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different types of data
        (self.market_data_dir / 'price_data').mkdir(exist_ok=True)
        (self.market_data_dir / 'order_book').mkdir(exist_ok=True)
        (self.market_data_dir / 'streaming').mkdir(exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Initialize the Market Data Agent and connect to the OANDA API.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.log_action("initialize", "Initializing Market Data Agent")
        
        try:
            # Validate required configuration
            if not all([self.api_key, self.account_id, self.api_url]):
                self.log_action("initialize", "Missing OANDA API credentials")
                self.handle_error(ValueError("Missing required OANDA API credentials"))
                return False
            
            # Initialize OANDA API client
            self.api_client = API(access_token=self.api_key, environment=self.api_url)
            
            # Test connection by fetching account details
            self._test_api_connection()
            
            # Fetch available instruments
            self._fetch_available_instruments()
            
            # Update status
            self.status = "ready"
            self.state["status"] = "ready"
            
            self.log_action("initialize", "Market Data Agent initialized successfully")
            return True
            
        except Exception as e:
            self.handle_error(e)
            self.status = "error"
            self.state["status"] = "error"
            return False
    
    def _test_api_connection(self) -> bool:
        """
        Test connection to the OANDA API.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Request account details to verify connection
            r = accounts.AccountSummary(accountID=self.account_id)
            self.api_client.request(r)
            
            return True
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to connect to OANDA API: {str(e)}")
    
    def _fetch_available_instruments(self) -> None:
        """
        Fetch the list of available trading instruments from OANDA.
        """
        try:
            # Request account instruments
            r = accounts.AccountInstruments(accountID=self.account_id)
            response = self.api_client.request(r)
            
            # Extract instruments
            instruments_list = response.get('instruments', [])
            self._available_instruments = {inst.get('name') for inst in instruments_list}
            
            self.log_action("fetch_instruments", f"Fetched {len(self._available_instruments)} available instruments")
            
        except V20Error as e:
            self.handle_error(e)
            self.log_action("fetch_instruments", f"Failed to fetch instruments: {str(e)}")
    
    def _with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic for handling API connection issues.
        
        Args:
            func: The function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
            
        Raises:
            Exception: If all retry attempts fail
        """
        attempts = 0
        last_error = None
        
        while attempts < self.max_retries:
            try:
                return func(*args, **kwargs)
            except (V20Error, RequestException, ConnectionError, Timeout) as e:
                attempts += 1
                last_error = e
                
                if attempts < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempts - 1))  # Exponential backoff
                    self.log_action("retry", f"API request failed, retrying in {wait_time}s ({attempts}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    self.log_action("retry", f"Maximum retry attempts reached ({self.max_retries})")
        
        # If we get here, all retries failed
        raise last_error
    
    # Market Data Retrieval Methods
    
    def get_current_price(self, instrument: str) -> Dict[str, Any]:
        """
        Get the current price for a currency pair.
        
        Args:
            instrument: The currency pair (e.g., "EUR_USD")
            
        Returns:
            Dict containing price information including bid, ask, and spread
            
        Raises:
            ValueError: If the instrument is invalid
            ConnectionError: If there's an issue with the API connection
        """
        self.log_action("get_current_price", f"Fetching current price for {instrument}")
        
        # Validate instrument
        if not self._validate_instrument(instrument):
            raise ValueError(f"Invalid instrument: {instrument}")
        
        try:
            # Define the request
            params = {"instruments": instrument}
            request = pricing.PricingInfo(accountID=self.account_id, params=params)
            
            # Execute the request with retry
            response = self._with_retry(self.api_client.request, request)
            
            # Extract and format the price data
            if 'prices' in response and len(response['prices']) > 0:
                price_data = response['prices'][0]
                
                result = {
                    'instrument': price_data.get('instrument'),
                    'time': price_data.get('time'),
                    'bid': float(price_data.get('bids', [{}])[0].get('price', 0)) if price_data.get('bids') else None,
                    'ask': float(price_data.get('asks', [{}])[0].get('price', 0)) if price_data.get('asks') else None,
                    'status': price_data.get('status', 'unknown')
                }
                
                # Calculate spread if bid and ask are available
                if result['bid'] is not None and result['ask'] is not None:
                    result['spread'] = result['ask'] - result['bid']
                
                return result
            else:
                raise ValueError(f"No price data received for {instrument}")
                
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to fetch current price: {str(e)}")
    
    def get_historical_data(
        self, 
        instrument: str, 
        timeframe: str = 'H1', 
        count: int = 500,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical candle data for a currency pair.
        
        Args:
            instrument: The currency pair (e.g., "EUR_USD")
            timeframe: The timeframe/granularity (e.g., "M1", "H1", "D")
            count: Number of candles to retrieve (max 5000)
            from_time: Start time for historical data (optional)
            to_time: End time for historical data (optional)
            
        Returns:
            DataFrame containing historical price data
            
        Raises:
            ValueError: If the instrument or parameters are invalid
            ConnectionError: If there's an issue with the API connection
        """
        self.log_action("get_historical_data", 
                        f"Fetching historical data for {instrument}, timeframe {timeframe}, count {count}")
        
        # Validate instrument
        if not self._validate_instrument(instrument):
            raise ValueError(f"Invalid instrument: {instrument}")
        
        # Validate timeframe
        if not self._validate_timeframe(timeframe):
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Limit count to 5000 (OANDA API limit)
        count = min(count, 5000)
        
        try:
            # Prepare parameters
            params = {
                "granularity": timeframe,
                "price": "M"  # Midpoint candles
            }
            
            # Add time parameters if provided
            if from_time and to_time:
                params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
                params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
            else:
                params["count"] = count
            
            # Create request
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            
            # Execute the request with retry
            response = self._with_retry(self.api_client.request, request)
            
            # Process the response
            if 'candles' in response:
                candles = response['candles']
                
                # Extract data into lists
                times = []
                opens = []
                highs = []
                lows = []
                closes = []
                volumes = []
                complete = []
                
                for candle in candles:
                    times.append(candle['time'])
                    opens.append(float(candle['mid']['o']))
                    highs.append(float(candle['mid']['h']))
                    lows.append(float(candle['mid']['l']))
                    closes.append(float(candle['mid']['c']))
                    volumes.append(int(candle['volume']))
                    complete.append(candle['complete'])
                
                # Create DataFrame
                df = pd.DataFrame({
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes,
                    'complete': complete
                }, index=pd.DatetimeIndex(times, name='datetime'))
                
                # Calculate additional metrics
                df['returns'] = df['close'].pct_change()
                df['range'] = df['high'] - df['low']
                
                # Cache the data if caching is enabled
                if self.cache_enabled:
                    cache_key = f"{instrument}_{timeframe}_{count}"
                    self._data_cache[cache_key] = {
                        'data': df,
                        'timestamp': datetime.now()
                    }
                
                return df
            else:
                raise ValueError(f"No candle data received for {instrument}")
                
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to fetch historical data: {str(e)}")
    
    def get_order_book(self, instrument: str) -> Dict[str, Any]:
        """
        Get order book data for a currency pair.
        
        Args:
            instrument: The currency pair (e.g., "EUR_USD")
            
        Returns:
            Dict containing order book data including bids and asks
            
        Raises:
            ValueError: If the instrument is invalid
            ConnectionError: If there's an issue with the API connection
        """
        self.log_action("get_order_book", f"Fetching order book for {instrument}")
        
        # Validate instrument
        if not self._validate_instrument(instrument):
            raise ValueError(f"Invalid instrument: {instrument}")
        
        try:
            # Define the request
            params = {"bucketWidth": "0.0005"}  # Default bucket width
            request = instruments.InstrumentsOrderBook(instrument=instrument, params=params)
            
            # Execute the request with retry
            response = self._with_retry(self.api_client.request, request)
            
            # Process the response
            if 'orderBook' in response:
                order_book = response['orderBook']
                
                result = {
                    'instrument': response.get('instrument'),
                    'time': response.get('time'),
                    'bids': order_book.get('buckets', []),
                    'asks': order_book.get('buckets', []),
                    'price': float(response.get('price', 0))
                }
                
                # Save to cache if enabled
                if self.cache_enabled:
                    cache_key = f"orderbook_{instrument}"
                    self._data_cache[cache_key] = {
                        'data': result,
                        'timestamp': datetime.now()
                    }
                
                return result
            else:
                raise ValueError(f"No order book data received for {instrument}")
                
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to fetch order book: {str(e)}")
    
    def stream_prices(self, instruments: List[str]) -> None:
        """
        Set up a real-time price stream for specified instruments.
        
        Note: This method is not meant to be called directly but through
        an asynchronous process or a separate thread.
        
        Args:
            instruments: List of currency pairs to stream
            
        Raises:
            ValueError: If any instrument is invalid
            ConnectionError: If there's an issue with the API connection
        """
        self.log_action("stream_prices", f"Setting up price stream for {instruments}")
        
        # Validate all instruments
        invalid_instruments = [i for i in instruments if not self._validate_instrument(i)]
        if invalid_instruments:
            raise ValueError(f"Invalid instruments: {invalid_instruments}")
        
        # Convert list to comma-separated string as required by OANDA API
        instruments_str = ",".join(instruments)
        
        self.log_action("stream_prices", 
                       "Note: Streaming implementation is a placeholder. In production, this would connect to the OANDA streaming API")
        
        # In a production implementation, this would set up a websocket connection
        # and process incoming price updates in real-time.
        # For now, we'll just log that streaming would be started.
        
        # Example of how the implementation would be structured:
        """
        from oandapyV20.endpoints.pricing import PricingStream
        
        # Create stream request
        params = {"instruments": instruments_str}
        request = PricingStream(accountID=self.account_id, params=params)
        
        try:
            # Open stream connection
            for msg in self.api_client.request(request):
                if msg["type"] == "PRICE":
                    # Process price update
                    instrument = msg["instrument"]
                    time = msg["time"]
                    bid = msg["bids"][0]["price"]
                    ask = msg["asks"][0]["price"]
                    
                    # Emit signal, update cache, or process as needed
                    self._process_price_update(instrument, time, bid, ask)
                    
        except Exception as e:
            self.handle_error(e)
            # Reconnect logic would go here
        """
        
    def get_instruments(self) -> List[str]:
        """
        Get a list of available trading instruments.
        
        Returns:
            List of instrument names (e.g., ["EUR_USD", "USD_JPY"])
        """
        # If we haven't fetched instruments yet, do it now
        if not self._available_instruments:
            self._fetch_available_instruments()
            
        return list(self._available_instruments)
    
    # Data Processing Methods
    
    def normalize_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Convert API data to a standardized format.
        
        Args:
            data: Raw data from the API (DataFrame or Dict)
            
        Returns:
            Normalized data in a consistent format
        """
        self.log_action("normalize_data", "Normalizing data")
        
        if isinstance(data, pd.DataFrame):
            # For DataFrame (historical candle data)
            
            # Ensure column names are standardized
            column_mapping = {
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
                'openMid': 'open', 'highMid': 'high', 'lowMid': 'low', 'closeMid': 'close'
            }
            
            # Rename columns if needed
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data = data.rename(columns={old_col: new_col})
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'time' in data.columns:
                    data['datetime'] = pd.to_datetime(data['time'])
                    data = data.set_index('datetime')
                elif 'datetime' in data.columns:
                    data['datetime'] = pd.to_datetime(data['datetime'])
                    data = data.set_index('datetime')
            
            # Ensure numeric types for price columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data
            
        elif isinstance(data, dict):
            # For dictionary data (current prices, order book)
            
            # Create a new normalized dictionary
            normalized = {}
            
            # Common fields
            if 'instrument' in data:
                normalized['instrument'] = data['instrument']
            
            if 'time' in data:
                normalized['time'] = data['time']
                normalized['datetime'] = pd.to_datetime(data['time'])
            
            # Price data
            if 'bids' in data and isinstance(data['bids'], list) and len(data['bids']) > 0:
                if isinstance(data['bids'][0], dict) and 'price' in data['bids'][0]:
                    normalized['bid'] = float(data['bids'][0]['price'])
                else:
                    normalized['bids'] = data['bids']
            
            if 'asks' in data and isinstance(data['asks'], list) and len(data['asks']) > 0:
                if isinstance(data['asks'][0], dict) and 'price' in data['asks'][0]:
                    normalized['ask'] = float(data['asks'][0]['price'])
                else:
                    normalized['asks'] = data['asks']
            
            if 'bid' in normalized and 'ask' in normalized:
                normalized['mid'] = (normalized['bid'] + normalized['ask']) / 2
                normalized['spread'] = normalized['ask'] - normalized['bid']
            
            return normalized
            
        else:
            # Unsupported data type
            raise ValueError(f"Unsupported data type for normalization: {type(data)}")
    
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Change the timeframe of historical data.
        
        Args:
            data: DataFrame containing historical data with a datetime index
            timeframe: Target timeframe (e.g., "1H", "4H", "1D")
            
        Returns:
            Resampled DataFrame
            
        Raises:
            ValueError: If the data is not properly formatted or timeframe is invalid
        """
        self.log_action("resample_data", f"Resampling data to timeframe {timeframe}")
        
        # Validate that data is a DataFrame with datetime index
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
        # Validate and parse timeframe
        valid_units = {'M': 'min', 'H': 'H', 'D': 'D', 'W': 'W'}
        
        if len(timeframe) < 2:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
        
        # Extract numeric value and unit
        unit = timeframe[-1]
        try:
            value = int(timeframe[:-1])
        except ValueError:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
        
        if unit not in valid_units:
            raise ValueError(f"Invalid timeframe unit: {unit}")
        
        # Create pandas-compatible frequency string
        freq = f"{value}{valid_units[unit]}"
        
        # Perform resampling
        resampled = pd.DataFrame()
        
        # OHLC (using pandas resample with appropriate aggregation methods)
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in data.columns else None
        }).dropna()
        
        # Calculate additional fields if they were in the original data
        if 'returns' in data.columns:
            resampled['returns'] = resampled['close'].pct_change()
        
        if 'range' in data.columns:
            resampled['range'] = resampled['high'] - resampled['low']
        
        return resampled
    
    def calculate_derived_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional fields like returns, volatility, etc.
        
        Args:
            data: DataFrame containing historical price data
            
        Returns:
            DataFrame with additional calculated fields
        """
        self.log_action("calculate_derived_data", "Calculating derived data fields")
        
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
    
    def detect_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and handle gaps in data.
        
        Args:
            data: DataFrame containing historical data with a datetime index
            
        Returns:
            DataFrame with additional gap information
        """
        self.log_action("detect_gaps", "Detecting gaps in data")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Sort by datetime index
        df = df.sort_index()
        
        # Calculate time differences between consecutive rows
        df['time_diff'] = df.index.to_series().diff()
        
        # Get the most common time difference (expected interval)
        expected_interval = df['time_diff'].mode()[0]
        
        # Identify gaps (where time difference is greater than expected)
        df['gap'] = df['time_diff'] > expected_interval * 1.5
        
        # Calculate gap size as multiple of expected interval
        df['gap_size'] = df['time_diff'] / expected_interval
        
        # Mark weekend gaps (might not be real gaps)
        if 'gap' in df.columns:
            df['weekend_gap'] = df['gap'] & (
                (df.index.dayofweek == 0) |  # Monday
                ((df.index.dayofweek == 6) | (df.index.dayofweek == 5))  # Weekend
            )
        
        # Identify price gaps (significant price difference after a time gap)
        if 'gap' in df.columns and 'open' in df.columns and 'close' in df.columns:
            df['price_gap'] = np.nan
            
            # Calculate price difference between current open and previous close
            price_diff = df['open'] - df['close'].shift(1)
            
            # Mark price gaps where there's also a time gap
            df.loc[df['gap'], 'price_gap'] = price_diff[df['gap']]
        
        return df
    
    # Data Storage Methods
    
    def save_data(self, data: Union[pd.DataFrame, Dict], filepath: str) -> bool:
        """
        Save data to disk.
        
        Args:
            data: Data to save (DataFrame or Dict)
            filepath: Path to save the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                # Save DataFrame to CSV
                data.to_csv(filepath)
                self.log_action("save_data", f"Saved DataFrame to {filepath}")
            elif isinstance(data, dict):
                # Save dict to JSON
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                self.log_action("save_data", f"Saved dict to {filepath}")
            else:
                raise ValueError(f"Unsupported data type for saving: {type(data)}")
            
            return True
        except Exception as e:
            self.handle_error(e)
            self.log_action("save_data", f"Failed to save data: {str(e)}")
            return False
    
    def load_data(self, filepath: str) -> Union[pd.DataFrame, Dict, None]:
        """
        Load data from disk.
        
        Args:
            filepath: Path to load the data from
            
        Returns:
            Loaded data (DataFrame or Dict) or None if failed
        """
        try:
            if not os.path.exists(filepath):
                self.log_action("load_data", f"File not found: {filepath}")
                return None
            
            # Check file extension to determine how to load
            if filepath.endswith('.csv'):
                # Load as DataFrame
                df = pd.read_csv(filepath)
                
                # Convert datetime column to index if it exists
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.set_index('datetime')
                
                self.log_action("load_data", f"Loaded DataFrame from {filepath}")
                return df
            elif filepath.endswith('.json'):
                # Load as dict
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.log_action("load_data", f"Loaded dict from {filepath}")
                return data
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
        except Exception as e:
            self.handle_error(e)
            self.log_action("load_data", f"Failed to load data: {str(e)}")
            return None
    
    def maintain_data_cache(self) -> None:
        """
        Manage a cache of frequently used data, clearing old entries.
        """
        if not self.cache_enabled:
            return
        
        current_time = datetime.now()
        keys_to_remove = []
        
        # Identify expired cache entries
        for key, cache_entry in self._data_cache.items():
            cache_time = cache_entry.get('timestamp')
            if cache_time:
                # Check if entry is older than cache expiry setting
                age = (current_time - cache_time).total_seconds() / (3600 * 24)  # Age in days
                if age > self.cache_expiry_days:
                    keys_to_remove.append(key)
        
        # Remove expired entries
        for key in keys_to_remove:
            del self._data_cache[key]
        
        self.log_action("maintain_cache", f"Removed {len(keys_to_remove)} expired cache entries")
    
    # Utility Methods
    
    def _validate_instrument(self, instrument: str) -> bool:
        """
        Validate that an instrument is available for trading.
        
        Args:
            instrument: The instrument name to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # If we haven't fetched instruments yet, do it now
        if not self._available_instruments:
            self._fetch_available_instruments()
            
        return instrument in self._available_instruments
    
    def _validate_timeframe(self, timeframe: str) -> bool:
        """
        Validate that a timeframe is supported.
        
        Args:
            timeframe: The timeframe to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        valid_timeframes = {
            'S5', 'S10', 'S15', 'S30',  # Seconds
            'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30',  # Minutes
            'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12',  # Hours
            'D', 'W', 'M'  # Day, Week, Month
        }
        
        return timeframe in valid_timeframes
    
    # LangGraph Integration
    
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a market data task based on the task description.
        
        This method implements the abstract method from BaseAgent and
        serves as the primary entry point for task execution.
        
        Args:
            task: Task description and parameters
            
        Returns:
            Dict[str, Any]: Task execution results
        """
        self.log_action("run_task", f"Running market data task: {task.get('action', 'unknown')}")
        
        # Update state
        self.state["tasks"].append(task)
        self.last_active = datetime.now()
        self.state["last_active"] = self.last_active
        
        try:
            # Extract task parameters
            action = task.get('action', '')
            instrument = task.get('instrument')
            timeframe = task.get('timeframe')
            
            # Dispatch to appropriate method based on action
            if action == 'get_current_price' and instrument:
                result = self.get_current_price(instrument)
                
            elif action == 'get_historical_data' and instrument:
                count = task.get('count', 500)
                from_time = task.get('from_time')
                to_time = task.get('to_time')
                result = self.get_historical_data(
                    instrument, timeframe, count, from_time, to_time
                )
                
                # Convert DataFrame to dict for JSON serialization
                if isinstance(result, pd.DataFrame):
                    result = {
                        'data': result.reset_index().to_dict(orient='records'),
                        'count': len(result),
                        'instrument': instrument,
                        'timeframe': timeframe
                    }
                
            elif action == 'get_order_book' and instrument:
                result = self.get_order_book(instrument)
                
            elif action == 'get_instruments':
                result = {'instruments': self.get_instruments()}
                
            elif action == 'stream_prices':
                instruments = task.get('instruments', [])
                # This would be handled differently in production
                # to return a streaming resource identifier
                self.stream_prices(instruments)
                result = {'status': 'streaming_initiated', 'instruments': instruments}
                
            else:
                result = {'error': f"Invalid action: {action}"}
            
            return {
                'status': 'success',
                'task': task,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.handle_error(e)
            return {
                'status': 'error',
                'task': task,
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat()
            } 