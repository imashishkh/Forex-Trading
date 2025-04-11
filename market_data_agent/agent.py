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
        
        # Added for the new last_update_time attribute
        self.last_update_time: Optional[datetime] = None
        
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
        
        if self.use_mock_data:
            self.log_action("initialize", "Setting up mock data for paper trading")
            # Initialize mock data structure
            self.mock_data = self._generate_initial_mock_data()
            self.status = "ready" # Ready with mock data
            self.last_update_time = datetime.now()
            self.log_action("initialize", "Market Data Agent initialized in MOCK mode")
            return True
        else:
            # --- FIX: Require OANDA credentials for live mode --- 
            self.account_id = os.getenv("OANDA_ACCOUNT_ID") or self.config.get('api_credentials', {}).get('oanda', {}).get('account_id')
            self.access_token = os.getenv("OANDA_API_KEY") or self.config.get('api_credentials', {}).get('oanda', {}).get('api_key')
            self.environment = "live" if (os.getenv("OANDA_ENVIRONMENT", "practice").lower() == "live") else "practice"
            api_url_config = self.config.get('api_credentials', {}).get('oanda', {}).get('api_url')
            self.api_url = os.getenv("OANDA_API_URL", api_url_config if api_url_config else (
                "https://api-fxtrade.oanda.com" if self.environment == "live" else "https://api-fxpractice.oanda.com"
            ))

            if not self.account_id or not self.access_token:
                self.log_action("initialize", "CRITICAL: OANDA credentials (Account ID or API Key) missing for LIVE mode. Cannot initialize.")
                self.status = "error"
                return False # Fail initialization if credentials missing in live mode
                
            self.headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Test connection
            if self._test_oanda_connection():
                self.status = "ready"
                self.last_update_time = datetime.now()
                self.log_action("initialize", f"Market Data Agent initialized in LIVE mode ({self.environment})")
                # Optionally start streaming if configured
                # if self.config.get("streaming_enabled", False):
                #     self.start_streaming()
                return True
            else:
                self.log_action("initialize", "Failed to connect to OANDA API for LIVE mode.")
                self.status = "error"
                return False
            # --- END FIX ---
    
    def _setup_mock_data(self) -> None:
        """
        Set up mock data for paper trading mode when API credentials are missing.
        """
        self.log_action("setup_mock", "Setting up mock data for paper trading")
        
        # Define common forex pairs as available instruments
        self._available_instruments = {
            "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", 
            "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"
        }
        
        # Create a dictionary to store mock price data
        self._mock_prices = {
            "EUR_USD": {"bid": 1.1852, "ask": 1.1854},
            "USD_JPY": {"bid": 109.75, "ask": 109.78},
            "GBP_USD": {"bid": 1.3862, "ask": 1.3865},
            "USD_CHF": {"bid": 0.9142, "ask": 0.9145},
            "AUD_USD": {"bid": 0.7352, "ask": 0.7355},
            "USD_CAD": {"bid": 1.2512, "ask": 1.2515},
            "NZD_USD": {"bid": 0.7002, "ask": 0.7005},
            "EUR_GBP": {"bid": 0.8551, "ask": 0.8554}
        }
        
        # Store a timestamp for last update
        self._mock_last_update = datetime.now()
        
        self.log_action("setup_mock", f"Mock data set up with {len(self._available_instruments)} instruments")
            
    def _test_api_connection(self) -> bool:
        """
        Test connection to the OANDA API.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Request account details to verify connection
            r = accounts.AccountSummary(accountID=self.account_id)
            self.api_client = API(access_token=self.access_token, environment=self.api_url)
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
        
        # Use mock data if in mock mode
        if self.status == "ready_mock":
            return self._get_mock_price(instrument)
            
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
                
                # Update last_update_time
                self.last_update_time = datetime.now()
                
                return result
            else:
                raise ValueError(f"No price data received for {instrument}")
                
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to fetch current price: {str(e)}")
    
    def _get_mock_price(self, instrument: str) -> Dict[str, Any]:
        """
        Get a mock price for paper trading mode.
        
        Args:
            instrument: The currency pair (e.g., "EUR_USD")
            
        Returns:
            Dict containing simulated price information
        """
        # Check if we have this instrument in our mock data
        if instrument not in self._mock_prices:
            # Create some realistic mock data for this pair
            base_price = 1.0
            if "JPY" in instrument:
                base_price = 100.0
            
            self._mock_prices[instrument] = {
                "bid": base_price * 0.9995,
                "ask": base_price * 1.0005
            }
            
        # Get the base mock price
        mock_price = self._mock_prices[instrument]
        
        # Add some random movement to simulate market changes (±0.1%)
        import random
        movement = random.uniform(-0.001, 0.001)
        mock_price["bid"] *= (1 + movement)
        mock_price["ask"] *= (1 + movement)
        
        # Store the updated prices
        self._mock_prices[instrument] = mock_price
        
        # Create the result dictionary
        result = {
            'instrument': instrument,
            'time': datetime.now().isoformat(),
            'bid': mock_price["bid"],
            'ask': mock_price["ask"],
            'spread': mock_price["ask"] - mock_price["bid"],
            'status': 'tradeable',
            'is_mock': True
        }
        
        # Update last_update_time
        self.last_update_time = datetime.now()
        
        return result
    
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
        
        # Use mock data if in mock mode
        if self.status == "ready_mock":
            return self._get_mock_historical_data(instrument, timeframe, count, from_time, to_time)
            
        # Limit count to 5000 (OANDA API limit)
        if count > 5000:
            self.log_action("get_historical_data", "Limiting count to 5000 (OANDA API limit)")
            count = 5000
            
        try:
            # Set up params based on inputs
            params = {
                "granularity": timeframe,
                "price": "M"  # Midpoint (average of bid and ask)
            }
            
            # Add count or time parameters
            if from_time and to_time:
                params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
                params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
            else:
                params["count"] = count
            
            # Execute the request
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            response = self._with_retry(self.api_client.request, request)
            
            # Process the response into a DataFrame
            if 'candles' in response and response['candles']:
                data = []
                for candle in response['candles']:
                    if candle['complete']:  # Only use complete candles
                        mid = candle['mid']
                        row = {
                            'time': pd.to_datetime(candle['time']),
                            'open': float(mid['o']),
                            'high': float(mid['h']),
                            'low': float(mid['l']),
                            'close': float(mid['c']),
                            'volume': int(candle['volume'])
                        }
                        data.append(row)
                
                if data:
                    df = pd.DataFrame(data)
                    df.set_index('time', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Calculate returns
                    if not df.empty and 'close' in df.columns and len(df) > 1:
                        # Ensure 'close' is numeric
                        df['close'] = pd.to_numeric(df['close'], errors='coerce')
                        # Calculate percentage returns
                        df['returns'] = df['close'].pct_change()
                    else:
                        df['returns'] = pd.Series(dtype='float64') # Add empty returns column if needed
                    
                    # Update last_update_time
                    self.last_update_time = datetime.now()
                    
                    return df
                else:
                    return pd.DataFrame()
            else:
                self.log_action("get_historical_data", f"No candle data received for {instrument}")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.log_action("get_historical_data", f"Error fetching historical data: {str(e)}")
            
            # Return empty DataFrame instead of raising exception
            # This allows the system to continue even if historical data fetch fails
            return pd.DataFrame()
            
    def _get_mock_historical_data(
        self, 
        instrument: str, 
        timeframe: str, 
        count: int,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate mock historical data for paper trading mode.
        
        Args:
            instrument: The currency pair (e.g., "EUR_USD")
            timeframe: The timeframe/granularity (e.g., "M1", "H1", "D")
            count: Number of candles to retrieve
            from_time: Start time for historical data (optional)
            to_time: End time for historical data (optional)
            
        Returns:
            DataFrame containing simulated historical price data
        """
        import numpy as np
        
        self.log_action("mock_data", f"Generating mock historical data for {instrument}")
        
        # Determine time interval from timeframe string
        interval_map = {
            'M1': pd.Timedelta(minutes=1),
            'M5': pd.Timedelta(minutes=5),
            'M15': pd.Timedelta(minutes=15),
            'M30': pd.Timedelta(minutes=30),
            'H1': pd.Timedelta(hours=1),
            'H4': pd.Timedelta(hours=4),
            'H8': pd.Timedelta(hours=8),
            'D': pd.Timedelta(days=1),
            'W': pd.Timedelta(weeks=1),
            'M': pd.Timedelta(days=30)
        }
        
        interval = interval_map.get(timeframe, pd.Timedelta(hours=1))
        
        # Set up time range
        if to_time is None:
            to_time = datetime.now()
        
        if from_time is None:
            from_time = to_time - (interval * count)
        
        # Create a date range for our data
        dates = pd.date_range(start=from_time, end=to_time, periods=count)
        
        # Get base price from mock prices or create one
        if instrument in self._mock_prices:
            base_price = (self._mock_prices[instrument]['bid'] + self._mock_prices[instrument]['ask']) / 2
        else:
            base_price = 1.0
            if "JPY" in instrument:
                base_price = 100.0
        
        # Generate mock price data with random walk
        np.random.seed(hash(instrument) % 10000)  # Consistent randomness per instrument
        
        # Parameters for price simulation
        volatility = 0.0015  # Daily volatility
        if timeframe in ['M1', 'M5', 'M15', 'M30']:
            volatility *= 0.2  # Lower volatility for shorter timeframes
        elif timeframe in ['H1', 'H4', 'H8']:
            volatility *= 0.5  # Medium volatility for hourly timeframes
        
        # Generate log returns with drift (slightly upward trend)
        drift = 0.00005  # Small upward drift
        returns = np.random.normal(drift, volatility, count)
        
        # Create price series
        price_series = np.exp(np.cumsum(returns))
        prices = base_price * price_series
        
        # Generate OHLC data with some randomness
        data = []
        for i, date in enumerate(dates):
            close_price = prices[i]
            
            # Create realistic OHLC with some randomness
            high_price = close_price * (1 + np.random.uniform(0, volatility * 2))
            low_price = close_price * (1 - np.random.uniform(0, volatility * 2))
            open_price = low_price + np.random.uniform(0, high_price - low_price)
            
            # Ensure OHLC relationships hold
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate reasonable volume
            volume = int(np.random.uniform(1000, 10000))
            
            data.append({
                'time': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Calculate returns
        if not df.empty and 'close' in df.columns and len(df) > 1:
            # Ensure 'close' is numeric
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            # Calculate percentage returns
            df['returns'] = df['close'].pct_change()
        else:
            df['returns'] = pd.Series(dtype='float64') # Add empty returns column if needed
        
        # Update last_update_time
        self.last_update_time = datetime.now()
        
        return df
    
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
                
                # Update last_update_time
                self.last_update_time = datetime.now()
                
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

    def _generate_mock_order_book(self, instrument: str) -> Dict[str, Any]:
        """Generates a plausible, but empty or minimal, mock order book structure."""
        # Return a structure that looks like OANDA's but might be empty
        return {
            "orderBook": {
                "instrument": instrument,
                "time": datetime.now().isoformat() + "Z",
                "price": str(self.mock_data.get(instrument, {}).get('price', 1.1)), # Use mock price
                "bucketWidth": "0.0005",
                "buckets": [
                    # Add a couple of dummy buckets if needed for testing downstream logic
                    # {"price": "1.0995", "longCountPercent": "50.0", "shortCountPercent": "0.0"},
                    # {"price": "1.1005", "longCountPercent": "0.0", "shortCountPercent": "50.0"}
                ]
            }
        }
        
    def _generate_initial_mock_data(self) -> Dict:
        """Generates initial base prices for mock data."""
        base_prices = {
            "EUR_USD": 1.1000,
            "GBP_USD": 1.2500,
            "USD_JPY": 140.00,
            "AUD_USD": 0.6700,
            "USD_CAD": 1.3500,
            "EUR_GBP": 0.8800,
            "NZD_USD": 0.6200,
            "USD_CHF": 0.9000
        }
        mock_data = {}
        for instrument, price in base_prices.items():
            mock_data[instrument] = {
                'price': price,
                'bid': price - 0.0001, # Small spread
                'ask': price + 0.0001,
                'last_updated': datetime.now()
            }
        return mock_data
        
    def _test_oanda_connection(self) -> bool:
        """
        Test connection to the OANDA API.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Request account details to verify connection
            r = accounts.AccountSummary(accountID=self.account_id)
            self.api_client = API(access_token=self.access_token, environment=self.api_url)
            self.api_client.request(r)
            
            return True
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to connect to OANDA API: {str(e)}")
        
    def _fetch_oanda_order_book(self, instrument: str) -> Dict[str, Any]:
        """
        Fetch the order book from the OANDA API.
        
        Args:
            instrument: The currency pair (e.g., "EUR_USD")
            
        Returns:
            Dict containing order book data including bids and asks
        """
        self.log_action("get_order_book", f"Fetching order book for {instrument}")
        
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
                
                # Update last_update_time
                self.last_update_time = datetime.now()
                
                return result
            else:
                raise ValueError(f"No order book data received for {instrument}")
                
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to fetch order book: {str(e)}") 