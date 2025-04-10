#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fundamentals Agent implementation for the Forex Trading Platform

This module provides a comprehensive implementation of a fundamentals analyst agent
that retrieves and analyzes economic data, news, and other fundamental factors
affecting forex markets.
"""

import os
import json
import logging
import requests
import traceback
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
from datetime import datetime, timedelta
from pathlib import Path
import time
import abc

# Data analysis imports
import pandas as pd
import numpy as np

# Web scraping and API
import requests
from bs4 import BeautifulSoup

# Base Agent class
from utils.base_agent import BaseAgent, AgentState, AgentMessage

# LangGraph imports
import langgraph.graph as lg
from langgraph.checkpoint.memory import MemorySaver


class FundamentalsAgent(BaseAgent):
    """
    Fundamentals Agent for analyzing economic data and news for forex trading.
    
    This agent is responsible for retrieving and analyzing economic indicators,
    central bank announcements, news, and other fundamental factors that affect
    the forex market. It provides insights on economic strength, potential currency
    movements, and trading signals based on fundamental analysis.
    """
    
    def __init__(
        self,
        agent_name: str = "fundamentals_agent",
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Fundamentals Agent.
        
        Args:
            agent_name: Name identifier for the agent
            llm: Language model to use (defaults to OpenAI if None)
            config: Configuration parameters for the agent
            logger: Logger instance for the agent
        """
        # Initialize BaseAgent
        super().__init__(agent_name, llm, config, logger)
        
        # Data storage paths
        self.data_dir = Path(self.config.get('system', {}).get('data_storage_path', 'data'))
        self.fundamentals_data_dir = self.data_dir / 'fundamentals'
        
        # Ensure data directories exist
        self.fundamentals_data_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.api_keys = self.config.get('api_credentials', {})
        
        # Cache settings
        self.cache_enabled = self.config.get('system', {}).get('cache_enabled', True)
        self.cache_expiry_hours = self.config.get('system', {}).get('cache_expiry_hours', 24)
        
        # Internal cache for frequently accessed data
        self._data_cache = {}
        
        # Define common economic indicators
        self.economic_indicators = {
            'GDP': 'Gross Domestic Product',
            'CPI': 'Consumer Price Index',
            'NFP': 'Non-Farm Payrolls',
            'FOMC': 'Federal Open Market Committee',
            'PMI': 'Purchasing Managers Index',
            'Retail Sales': 'Retail Sales',
            'Interest Rate': 'Central Bank Interest Rate',
            'Trade Balance': 'Trade Balance',
            'Unemployment': 'Unemployment Rate',
            'Industrial Production': 'Industrial Production',
            'PPI': 'Producer Price Index',
            'Housing Starts': 'Housing Starts',
            'Consumer Sentiment': 'Consumer Sentiment'
        }
        
        # Map of currencies to countries/regions
        self.countries = {
            'USD': 'United States',
            'EUR': 'Euro Zone',
            'GBP': 'United Kingdom',
            'JPY': 'Japan',
            'AUD': 'Australia',
            'CAD': 'Canada',
            'CHF': 'Switzerland',
            'NZD': 'New Zealand',
            'CNY': 'China',
            'SGD': 'Singapore',
            'MXN': 'Mexico',
            'SEK': 'Sweden',
            'NOK': 'Norway'
        }
        
        # Central bank mapping
        self.central_banks = {
            'USD': {'name': 'Federal Reserve', 'short': 'Fed'},
            'EUR': {'name': 'European Central Bank', 'short': 'ECB'},
            'GBP': {'name': 'Bank of England', 'short': 'BoE'},
            'JPY': {'name': 'Bank of Japan', 'short': 'BoJ'},
            'AUD': {'name': 'Reserve Bank of Australia', 'short': 'RBA'},
            'CAD': {'name': 'Bank of Canada', 'short': 'BoC'},
            'CHF': {'name': 'Swiss National Bank', 'short': 'SNB'},
            'NZD': {'name': 'Reserve Bank of New Zealand', 'short': 'RBNZ'},
            'CNY': {'name': 'People\'s Bank of China', 'short': 'PBoC'}
        }
        
        # Default data providers
        self.default_providers = {
            'economic_calendar': 'forex_factory',
            'news': 'investing_com',
            'economic_data': 'trading_economics'
        }
        
        # Override provider settings if provided
        if config and 'data_providers' in config:
            self.default_providers.update(config['data_providers'])
            
        self.log_action("init", f"Fundamentals Agent initialized")
    
    def initialize(self) -> bool:
        """
        Set up the Fundamentals Agent and its resources.
        
        This method handles resource allocation, connection to external
        services, and any other initialization tasks.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.log_action("initialize", "Fundamentals Agent initialization started")
        
        try:
            # Test connections to data providers
            test_results = self._test_data_providers()
            
            if not all(test_results.values()):
                failing_providers = [p for p, status in test_results.items() if not status]
                self.logger.warning(f"Some data providers are unavailable: {failing_providers}")
                # Continue initialization despite some failing providers
            
            # Initialize the data cache
            self._initialize_cache()
            
            # Update status
            self.status = "ready"
            self.state["status"] = "ready"
            
            self.log_action("initialize", "Fundamentals Agent initialized successfully")
            return True
            
        except Exception as e:
            self.handle_error(e)
            self.status = "error"
            self.state["status"] = "error"
            return False
    
    def _test_data_providers(self) -> Dict[str, bool]:
        """
        Test connections to data providers.
        
        Returns:
            Dict[str, bool]: Status of each data provider connection
        """
        results = {}
        
        # This is just a placeholder for actual implementation
        # In a real scenario, this would test API connectivity
        
        # For now, all tests pass to enable the agent to continue
        for provider in self.default_providers.values():
            results[provider] = True
            
        return results
    
    def _initialize_cache(self) -> None:
        """Initialize the local data cache."""
        self.log_action("cache", "Initializing data cache")
        
        # Create subdirectories for different types of data
        (self.fundamentals_data_dir / 'economic_calendar').mkdir(exist_ok=True)
        (self.fundamentals_data_dir / 'news').mkdir(exist_ok=True)
        (self.fundamentals_data_dir / 'indicators').mkdir(exist_ok=True)
        (self.fundamentals_data_dir / 'central_banks').mkdir(exist_ok=True)
    
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the fundamentals agent's primary functionality.
        
        This method handles various tasks related to fundamental analysis,
        including retrieving economic data, analyzing news, and generating
        trading signals.
        
        Args:
            task: Task description and parameters

        Returns:
            Dict[str, Any]: Task execution results
        """
        self.log_action("run_task", f"Running task: {task.get('type', 'Unknown')}")
        
        # Update state
        self.state["tasks"].append(task)
        self.last_active = datetime.now()
        self.state["last_active"] = self.last_active
        
        # Task dispatcher
        task_type = task.get('type', '')
        
        if task_type == 'get_economic_calendar':
            start_date = task.get('start_date')
            end_date = task.get('end_date')
            countries = task.get('countries')
            result = self.get_economic_calendar(start_date, end_date, countries)
            return {"status": "success", "result": result}
            
        elif task_type == 'get_economic_indicators':
            country = task.get('country')
            indicator = task.get('indicator')
            result = self.get_economic_indicators(country, indicator)
            return {"status": "success", "result": result}
            
        elif task_type == 'get_central_bank_rates':
            result = self.get_central_bank_rates()
            return {"status": "success", "result": result}
            
        elif task_type == 'get_gdp_data':
            countries = task.get('countries')
            result = self.get_gdp_data(countries)
            return {"status": "success", "result": result}
            
        elif task_type == 'get_inflation_data':
            countries = task.get('countries')
            result = self.get_inflation_data(countries)
            return {"status": "success", "result": result}
            
        elif task_type == 'fetch_forex_news':
            currencies = task.get('currencies')
            result = self.fetch_forex_news(currencies)
            return {"status": "success", "result": result}
            
        elif task_type == 'analyze_news_impact':
            news = task.get('news')
            currency = task.get('currency')
            result = self.analyze_news_impact(news, currency)
            return {"status": "success", "result": result}
            
        elif task_type == 'calculate_interest_rate_differential':
            currency_pair = task.get('currency_pair')
            result = self.calculate_interest_rate_differential(currency_pair)
            return {"status": "success", "result": result}
            
        elif task_type == 'generate_fundamental_signals':
            currency_pair = task.get('currency_pair')
            result = self.generate_fundamental_signals(currency_pair)
            return {"status": "success", "result": result}
            
        elif task_type == 'rate_fundamental_strength':
            currency = task.get('currency')
            result = self.rate_fundamental_strength(currency)
            return {"status": "success", "result": result}
            
        elif task_type == 'identify_divergence':
            currency = task.get('currency')
            result = self.identify_divergence(currency)
            return {"status": "success", "result": result}
            
        else:
            error = f"Unknown task type: {task_type}"
            self.logger.error(error)
            return {"status": "error", "error": error}
    
    # === Economic Data Methods ===
    
    def get_economic_calendar(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch upcoming economic events from the economic calendar.
        
        Args:
            start_date: Start date for the calendar (default: today)
            end_date: End date for the calendar (default: 7 days from start)
            countries: List of country codes to filter events
            
        Returns:
            pd.DataFrame: Economic calendar events
        """
        self.log_action("get_economic_calendar", f"Fetching economic calendar {start_date} to {end_date}")
        
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d')
            
        # Check cache
        cache_key = f"economic_calendar_{start_date}_{end_date}"
        if countries:
            countries_key = "_".join(sorted(countries))
            cache_key += f"_{countries_key}"
            
        cached_data = self._get_from_cache(cache_key, 'economic_calendar')
        if cached_data is not None:
            self.logger.info(f"Using cached economic calendar data")
            return cached_data
        
        # Use the default economic calendar provider or fall back to forex_factory
        provider = self.default_providers.get('economic_calendar', 'forex_factory')
        
        calendar_data = None
        
        # Ensure we have a connection to the data provider
        if not self._ensure_provider_connection(provider):
            self.logger.warning(f"Could not connect to {provider}, falling back to mock data")
            calendar_data = self._generate_mock_economic_calendar(start_date, end_date, countries)
        else:
            try:
                # Fetch from the appropriate data provider
                if provider == "forex_factory":
                    calendar_data = self._fetch_forex_factory_calendar(start_date, end_date, countries)
                elif provider == "investing_com":
                    calendar_data = self._fetch_investing_calendar(start_date, end_date, countries)
                elif provider == "trading_economics":
                    calendar_data = self._fetch_trading_economics_calendar(start_date, end_date, countries)
                else:
                    self.logger.warning(f"Unknown provider {provider}, falling back to mock data")
                    calendar_data = self._generate_mock_economic_calendar(start_date, end_date, countries)
            except Exception as e:
                self.handle_error(e)
                self.logger.warning("Error fetching calendar data, falling back to mock data")
                calendar_data = self._generate_mock_economic_calendar(start_date, end_date, countries)
        
        # Cache the data
        if calendar_data is not None and not calendar_data.empty:
            self._save_to_cache(calendar_data, cache_key, 'economic_calendar')
        
        return calendar_data
    
    def _ensure_provider_connection(self, provider: str) -> bool:
        """Ensure we have a connection to the data provider."""
        if not hasattr(self, '_data_providers') or not self._data_providers or provider not in self._data_providers:
            return self.connect_to_data_provider(provider)
        return self._data_providers[provider].get('connected', False)
    
    def _fetch_forex_factory_calendar(
        self, 
        start_date: str, 
        end_date: str, 
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch economic calendar data from Forex Factory."""
        self.log_action("fetch_data", f"Fetching calendar from Forex Factory: {start_date} to {end_date}")
        
        try:
            # Get the session
            session = self._data_providers['forex_factory']['session']
            
            # Format dates for Forex Factory's URL structure
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Forex Factory uses week-based URLs
            # We'll need to fetch data week by week
            calendar_data = []
            current_dt = start_dt
            
            while current_dt <= end_dt:
                # Format date for the URL (e.g., "week=2023-06-04")
                week_param = current_dt.strftime('%Y-%m-%d')
                url = f"{self._data_providers['forex_factory']['base_url']}/calendar?week={week_param}"
                
                response = session.get(url)
                if response.status_code != 200:
                    self.logger.warning(f"Failed to fetch Forex Factory calendar: {response.status_code}")
                    break
                
                # Parse the HTML - this requires a HTML parser like BeautifulSoup
                # For demonstration, we'll use a simplified approach - in production, use proper HTML parsing
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, 'html.parser')
                calendar_rows = soup.select('table.calendar__table tr.calendar__row')
                
                for row in calendar_rows:
                    # Skip header rows
                    if 'calendar__row--header' in row.get('class', []):
                        continue
                        
                    # Extract data
                    try:
                        currency_elem = row.select_one('.calendar__currency')
                        currency = currency_elem.text.strip() if currency_elem else ""
                        
                        # If countries filter is applied, skip non-matching rows
                        if countries and currency not in countries:
                            continue
                            
                        date_elem = row.select_one('.calendar__date')
                        date_str = date_elem.text.strip() if date_elem else ""
                        
                        time_elem = row.select_one('.calendar__time')
                        time_str = time_elem.text.strip() if time_elem else ""
                        
                        event_elem = row.select_one('.calendar__event')
                        event_text = event_elem.text.strip() if event_elem else ""
                        
                        importance_elem = row.select_one('.calendar__importance')
                        importance_class = importance_elem.get('class', []) if importance_elem else []
                        importance = "low"
                        if 'high' in ' '.join(importance_class):
                            importance = "high"
                        elif 'medium' in ' '.join(importance_class):
                            importance = "medium"
                            
                        previous_elem = row.select_one('.calendar__previous')
                        previous = previous_elem.text.strip() if previous_elem else ""
                        
                        forecast_elem = row.select_one('.calendar__forecast')
                        forecast = forecast_elem.text.strip() if forecast_elem else ""
                        
                        actual_elem = row.select_one('.calendar__actual')
                        actual = actual_elem.text.strip() if actual_elem else None
                        
                        # Format date properly
                        calendar_date = None
                        if date_str:
                            # Parse the date from Forex Factory format
                            # This might need adjustment based on their format
                            try:
                                # Try to parse the date using the current week's year/month
                                week_year = current_dt.year
                                week_month = current_dt.month
                                
                                # Forex Factory might show dates like "Mon" or "4" (day of month)
                                if len(date_str) <= 3:  # It's a day abbreviation
                                    # Find the day in the current week
                                    day_map = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6}
                                    day_offset = day_map.get(date_str.lower(), 0)
                                    week_start = current_dt - timedelta(days=current_dt.weekday())
                                    calendar_date = week_start + timedelta(days=day_offset)
                                else:
                                    # Assume it's a day of month
                                    day = int(date_str)
                                    calendar_date = datetime(week_year, week_month, day)
                            except:
                                # Fallback to current date if parsing fails
                                calendar_date = current_dt
                        else:
                            calendar_date = current_dt
                            
                        calendar_data.append({
                            'date': calendar_date.strftime('%Y-%m-%d'),
                            'time': time_str,
                            'currency': currency,
                            'country': self.countries.get(currency, currency),
                            'event': event_text,
                            'importance': importance,
                            'previous': previous,
                            'forecast': forecast,
                            'actual': actual
                        })
                    except Exception as inner_e:
                        self.logger.error(f"Error parsing calendar row: {str(inner_e)}")
                        continue
                
                # Move to next week
                current_dt += timedelta(days=7)
            
            # Convert to DataFrame
            if calendar_data:
                df = pd.DataFrame(calendar_data)
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                df = df.sort_values('datetime')
                return df
            else:
                self.logger.warning("No calendar data found, returning empty DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to fetch Forex Factory calendar: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_investing_calendar(
        self, 
        start_date: str, 
        end_date: str, 
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch economic calendar data from Investing.com."""
        self.log_action("fetch_data", f"Fetching calendar from Investing.com: {start_date} to {end_date}")
        
        try:
            # Get the session
            session = self._data_providers['investing_com']['session']
            
            # Investing.com's calendar filter requires specific parameters
            # The site uses POST requests to load calendar data
            url = f"{self._data_providers['investing_com']['base_url']}/economic-calendar/Service/getCalendarFilteredData"
            
            # Convert country codes to Investing.com's country IDs
            # This is a simplified mapping - actual implementation would need a complete mapping
            country_map = {
                'USD': '5', 'EUR': '72', 'GBP': '4', 'JPY': '35', 'AUD': '25',
                'CAD': '6', 'CHF': '12', 'NZD': '43', 'CNY': '37'
            }
            
            country_ids = []
            if countries:
                for country in countries:
                    if country in country_map:
                        country_ids.append(country_map[country])
            
            # Format dates for request
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Investing.com uses Unix timestamps
            start_timestamp = int(start_dt.timestamp())
            end_timestamp = int(end_dt.timestamp())
            
            # Build request payload
            payload = {
                'dateFrom': start_dt.strftime('%Y-%m-%d'),
                'dateTo': end_dt.strftime('%Y-%m-%d'),
                'timeZone': 0,  # UTC
                'timeFilter': 'timeRemain',
                'currentTab': 'custom',
                'limit_from': 0
            }
            
            if country_ids:
                payload['countries'] = ','.join(country_ids)
            
            # Add additional headers for the request
            headers = {
                'Referer': 'https://www.investing.com/economic-calendar/',
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-Requested-With': 'XMLHttpRequest'
            }
            
            # Make the request
            response = session.post(url, data=payload, headers=headers)
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch Investing.com calendar: {response.status_code}")
                return pd.DataFrame()
            
            # Parse the response - Investing.com returns HTML
            from bs4 import BeautifulSoup
            
            try:
                data = response.json()
                html = data.get('data', '')
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract calendar events
                event_rows = soup.select('tr.js-event-item')
                
                calendar_data = []
                for row in event_rows:
                    try:
                        # Get event data
                        event_id = row.get('id', '').replace('eventRowId_', '')
                        
                        # Time and date
                        date_str = row.select_one('td.first').text.strip() if row.select_one('td.first') else ""
                        time_str = row.select_one('td.time').text.strip() if row.select_one('td.time') else ""
                        
                        # Country and event name
                        country_elem = row.select_one('td.flagCur')
                        country_img = country_elem.select_one('span.ceFlags') if country_elem else None
                        country_code = country_img.get('title', '') if country_img else ""
                        
                        # Map Investing.com country names to currency codes
                        # This is a simplified mapping - actual implementation would need complete mapping
                        country_to_currency = {
                            'United States': 'USD', 'Eurozone': 'EUR', 'United Kingdom': 'GBP',
                            'Japan': 'JPY', 'Australia': 'AUD', 'Canada': 'CAD',
                            'Switzerland': 'CHF', 'New Zealand': 'NZD', 'China': 'CNY'
                        }
                        
                        currency = country_to_currency.get(country_code, country_code[:3])
                        
                        # Event name
                        event_elem = row.select_one('td.event')
                        event_text = event_elem.text.strip() if event_elem else ""
                        
                        # Importance (indicated by bull icons)
                        importance_elem = row.select_one('td.sentiment')
                        bull_icons = importance_elem.select('i.grayFullBullishIcon') if importance_elem else []
                        importance = "low"
                        if len(bull_icons) == 3:
                            importance = "high"
                        elif len(bull_icons) == 2:
                            importance = "medium"
                        
                        # Previous, forecast, actual values
                        previous_elem = row.select_one('td.prev')
                        previous = previous_elem.text.strip() if previous_elem else ""
                        
                        forecast_elem = row.select_one('td.forecast')
                        forecast = forecast_elem.text.strip() if forecast_elem else ""
                        
                        actual_elem = row.select_one('td.act')
                        actual = actual_elem.text.strip() if actual_elem else None
                        
                        # Format date and time
                        try:
                            datetime_str = f"{date_str} {time_str}"
                            event_dt = datetime.strptime(datetime_str, '%b %d, %Y %H:%M')
                            date_formatted = event_dt.strftime('%Y-%m-%d')
                            time_formatted = event_dt.strftime('%H:%M')
                        except:
                            # Fallback to current date if parsing fails
                            date_formatted = datetime.now().strftime('%Y-%m-%d')
                            time_formatted = time_str
                        
                        calendar_data.append({
                            'date': date_formatted,
                            'time': time_formatted,
                            'currency': currency,
                            'country': country_code,
                            'event': event_text,
                            'importance': importance,
                            'previous': previous,
                            'forecast': forecast,
                            'actual': actual
                        })
                    except Exception as inner_e:
                        self.logger.error(f"Error parsing Investing.com calendar row: {str(inner_e)}")
                        continue
                
                # Convert to DataFrame
                if calendar_data:
                    df = pd.DataFrame(calendar_data)
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                    df = df.sort_values('datetime')
                    return df
                else:
                    self.logger.warning("No Investing.com calendar data found, returning empty DataFrame")
                    return pd.DataFrame()
                    
            except Exception as parse_e:
                self.logger.error(f"Failed to parse Investing.com calendar data: {str(parse_e)}")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to fetch Investing.com calendar: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_trading_economics_calendar(
        self, 
        start_date: str, 
        end_date: str, 
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch economic calendar data from Trading Economics."""
        self.log_action("fetch_data", f"Fetching calendar from Trading Economics: {start_date} to {end_date}")
        
        try:
            # Get the session and headers
            base_url = self._data_providers['trading_economics']['base_url']
            headers = self._data_providers['trading_economics']['headers']
            
            # Build the URL with appropriate parameters
            url = f"{base_url}/calendar/country/all"
            
            # Trading Economics uses country names rather than currency codes for filtering
            country_params = []
            if countries:
                for currency in countries:
                    country_name = self.countries.get(currency)
                    if country_name:
                        # Format country name for the API (lowercase, spaces to hyphens)
                        formatted_name = country_name.lower().replace(' ', '-')
                        country_params.append(formatted_name)
            
            if country_params:
                url = f"{base_url}/calendar/country/{','.join(country_params)}"
            
            # Add date parameters
            url += f"/from/{start_date}/to/{end_date}"
            
            # Make the request
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch Trading Economics calendar: {response.status_code}")
                return pd.DataFrame()
            
            # Parse the JSON response
            data = response.json()
            
            calendar_data = []
            for event in data:
                try:
                    # Extract the data from the TE API response
                    date_str = event.get('Date', '')
                    country = event.get('Country', '')
                    event_name = event.get('Event', '')
                    importance = event.get('Importance', 1)
                    # Map importance level (1-3) to categories
                    importance_map = {1: 'low', 2: 'medium', 3: 'high'}
                    importance_category = importance_map.get(importance, 'low')
                    
                    # Previous, forecast, actual values
                    previous = event.get('Previous', '')
                    forecast = event.get('Forecast', '')
                    actual = event.get('Actual', None)
                    
                    # Map country to currency code
                    currency = None
                    for curr, country_name in self.countries.items():
                        if country_name.lower() == country.lower():
                            currency = curr
                            break
                    
                    if not currency:
                        currency = country[:3]  # Fallback to first 3 chars
                    
                    # Parse the date and time
                    try:
                        event_dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                        date_formatted = event_dt.strftime('%Y-%m-%d')
                        time_formatted = event_dt.strftime('%H:%M')
                    except:
                        # Fallback if parsing fails
                        date_formatted = start_date
                        time_formatted = '00:00'
                    
                    calendar_data.append({
                        'date': date_formatted,
                        'time': time_formatted,
                        'currency': currency,
                        'country': country,
                        'event': event_name,
                        'importance': importance_category,
                        'previous': previous,
                        'forecast': forecast,
                        'actual': actual
                    })
                except Exception as inner_e:
                    self.logger.error(f"Error parsing Trading Economics calendar event: {str(inner_e)}")
                    continue
            
            # Convert to DataFrame
            if calendar_data:
                df = pd.DataFrame(calendar_data)
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                df = df.sort_values('datetime')
                return df
            else:
                self.logger.warning("No Trading Economics calendar data found, returning empty DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to fetch Trading Economics calendar: {str(e)}")
            return pd.DataFrame()
    
    def get_economic_indicators(
        self, 
        country: str, 
        indicator: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get specific economic indicators for a country.
        
        Args:
            country: Country code or name to fetch indicators for
            indicator: Specific indicator to fetch (None for all)
            
        Returns:
            pd.DataFrame: Economic indicator data
        """
        self.log_action("get_economic_indicators", f"Fetching economic indicators for {country}")
        
        # Normalize country code
        country_code = self._normalize_country_code(country)
        
        # Check cache
        cache_key = f"economic_indicators_{country_code}"
        if indicator:
            cache_key += f"_{indicator}"
            
        cached_data = self._get_from_cache(cache_key, 'indicators')
        if cached_data is not None:
            self.logger.info(f"Using cached economic indicator data")
            return cached_data
        
        # In a real implementation, fetch from an actual API
        # For now, generate mock data
        
        # Generate mock indicator data
        indicator_data = self._generate_mock_economic_indicators(country_code, indicator)
        
        # Cache the data
        self._save_to_cache(indicator_data, cache_key, 'indicators')
        
        return indicator_data
    
    def get_central_bank_rates(self) -> pd.DataFrame:
        """
        Get current central bank interest rates for major currencies.
        
        Returns:
            pd.DataFrame: Central bank interest rates
        """
        self.log_action("get_central_bank_rates", "Fetching central bank interest rates")
        
        # Check cache
        cache_key = "central_bank_rates"
        cached_data = self._get_from_cache(cache_key, 'central_banks')
        if cached_data is not None:
            self.logger.info(f"Using cached central bank rates")
            return cached_data
        
        # Use the default economic data provider or fall back to trading_economics
        provider = self.default_providers.get('economic_data', 'trading_economics')
        
        rate_data = None
        
        # Ensure we have a connection to the data provider
        if not self._ensure_provider_connection(provider):
            self.logger.warning(f"Could not connect to {provider}, falling back to mock data")
            rate_data = self._generate_mock_central_bank_rates()
        else:
            try:
                # Fetch from the appropriate data provider
                if provider == "trading_economics":
                    rate_data = self._fetch_trading_economics_rates()
                else:
                    self.logger.warning(f"Unknown provider {provider} for central bank rates, falling back to mock data")
                    rate_data = self._generate_mock_central_bank_rates()
            except Exception as e:
                self.handle_error(e)
                self.logger.warning("Error fetching central bank rates, falling back to mock data")
                rate_data = self._generate_mock_central_bank_rates()
        
        # Cache the data
        if rate_data is not None and not rate_data.empty:
            self._save_to_cache(rate_data, cache_key, 'central_banks')
        
        return rate_data
    
    def _fetch_trading_economics_rates(self) -> pd.DataFrame:
        """Fetch central bank interest rates from Trading Economics."""
        self.log_action("fetch_data", "Fetching central bank rates from Trading Economics")
        
        try:
            # Get the session and headers
            base_url = self._data_providers['trading_economics']['base_url']
            headers = self._data_providers['trading_economics']['headers']
            
            # Trading Economics API endpoint for interest rates
            url = f"{base_url}/indicators/interest-rate"
            
            # Make the request
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch Trading Economics rates: {response.status_code}")
                return pd.DataFrame()
            
            # Parse the JSON response
            data = response.json()
            
            rates_data = []
            for rate_info in data:
                try:
                    country = rate_info.get('Country', '')
                    value = rate_info.get('Value', 0.0)
                    last_update = rate_info.get('Date', '')
                    
                    # Convert country to currency code
                    currency = None
                    for curr, country_name in self.countries.items():
                        if country_name.lower() == country.lower():
                            currency = curr
                            break
                    
                    if not currency:
                        # Skip countries not in our list
                        continue
                    
                    # Get central bank info
                    central_bank = self.central_banks.get(currency, {})
                    bank_name = central_bank.get('name', f"{country} Central Bank")
                    bank_short = central_bank.get('short', bank_name[:3])
                    
                    # Parse the date
                    try:
                        update_dt = datetime.strptime(last_update, '%Y-%m-%dT%H:%M:%S')
                        update_date = update_dt.strftime('%Y-%m-%d')
                    except:
                        # Fallback
                        update_date = datetime.now().strftime('%Y-%m-%d')
                    
                    rates_data.append({
                        'currency': currency,
                        'country': country,
                        'central_bank': bank_name,
                        'central_bank_short': bank_short,
                        'interest_rate': float(value),
                        'last_change': update_date,
                        'reference': 'Trading Economics'
                    })
                except Exception as inner_e:
                    self.logger.error(f"Error parsing rate data for {rate_info.get('Country', 'unknown')}: {str(inner_e)}")
                    continue
            
            # Convert to DataFrame
            if rates_data:
                df = pd.DataFrame(rates_data)
                return df
            else:
                self.logger.warning("No central bank rate data found, returning empty DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to fetch central bank rates: {str(e)}")
            return pd.DataFrame()
    
    def get_gdp_data(self, countries: List[str]) -> pd.DataFrame:
        """
        Get GDP data for specified countries.
        
        Args:
            countries: List of country codes to get GDP data for
            
        Returns:
            pd.DataFrame: GDP data for specified countries
        """
        self.log_action("get_gdp_data", f"Fetching GDP data for {countries}")
        
        # Normalize country codes
        country_codes = [self._normalize_country_code(c) for c in countries]
        
        # Check cache
        countries_key = "_".join(sorted(country_codes))
        cache_key = f"gdp_data_{countries_key}"
            
        cached_data = self._get_from_cache(cache_key, 'indicators')
        if cached_data is not None:
            self.logger.info(f"Using cached GDP data")
            return cached_data
        
        # Use the default economic data provider or fall back to trading_economics
        provider = self.default_providers.get('economic_data', 'trading_economics')
        
        gdp_data = None
        
        # Ensure we have a connection to the data provider
        if not self._ensure_provider_connection(provider):
            self.logger.warning(f"Could not connect to {provider}, falling back to mock data")
            gdp_data = self._generate_mock_gdp_data(country_codes)
        else:
            try:
                # Fetch from the appropriate data provider
                if provider == "trading_economics":
                    gdp_data = self._fetch_trading_economics_gdp(country_codes)
                else:
                    self.logger.warning(f"Unknown provider {provider} for GDP data, falling back to mock data")
                    gdp_data = self._generate_mock_gdp_data(country_codes)
            except Exception as e:
                self.handle_error(e)
                self.logger.warning("Error fetching GDP data, falling back to mock data")
                gdp_data = self._generate_mock_gdp_data(country_codes)
        
        # Cache the data
        if gdp_data is not None and not gdp_data.empty:
            self._save_to_cache(gdp_data, cache_key, 'indicators')
        
        return gdp_data
    
    def _fetch_trading_economics_gdp(self, countries: List[str]) -> pd.DataFrame:
        """Fetch GDP data from Trading Economics."""
        self.log_action("fetch_data", f"Fetching GDP data from Trading Economics for {countries}")
        
        try:
            # Get the session and headers
            base_url = self._data_providers['trading_economics']['base_url']
            headers = self._data_providers['trading_economics']['headers']
            
            gdp_data = []
            
            # For each country, fetch GDP data
            for currency in countries:
                country_name = self.countries.get(currency)
                if not country_name:
                    continue
                
                # Format country name for the API (lowercase, spaces to hyphens)
                formatted_name = country_name.lower().replace(' ', '-')
                
                # Trading Economics API endpoint for GDP
                url = f"{base_url}/country/{formatted_name}/gdp-growth"
                
                # Make the request
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    self.logger.warning(f"Failed to fetch GDP data for {currency}: {response.status_code}")
                    continue
                
                # Parse the JSON response
                data = response.json()
                
                for item in data:
                    try:
                        # Extract data from the response
                        value = item.get('Value', 0.0)
                        date_str = item.get('Date', '')
                        
                        # Parse the date
                        try:
                            data_dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                            data_date = data_dt.strftime('%Y-%m-%d')
                            quarter = f"Q{(data_dt.month-1)//3 + 1} {data_dt.year}"
                        except:
                            # Fallback
                            data_date = datetime.now().strftime('%Y-%m-%d')
                            quarter = f"Q{(datetime.now().month-1)//3 + 1} {datetime.now().year}"
                        
                        gdp_data.append({
                            'country': currency,
                            'indicator': 'GDP Growth Rate',
                            'date': data_date,
                            'value': float(value),
                            'unit': '%',
                            'frequency': 'Quarterly',
                            'period': quarter
                        })
                    except Exception as inner_e:
                        self.logger.error(f"Error parsing GDP data for {currency}: {str(inner_e)}")
                        continue
            
            # Convert to DataFrame
            if gdp_data:
                df = pd.DataFrame(gdp_data)
                
                # Add datetime column for sorting
                df['datetime'] = pd.to_datetime(df['date'])
                
                # Sort by country and date
                df = df.sort_values(['country', 'datetime'], ascending=[True, False])
                
                # Remove datetime column (used only for sorting)
                df = df.drop(columns=['datetime'])
                
                return df
            else:
                self.logger.warning("No GDP data found, returning empty DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to fetch GDP data: {str(e)}")
            return pd.DataFrame()
    
    def get_inflation_data(self, countries: List[str]) -> pd.DataFrame:
        """
        Get inflation data for specified countries.
        
        Args:
            countries: List of country codes to get inflation data for
            
        Returns:
            pd.DataFrame: Inflation data for specified countries
        """
        self.log_action("get_inflation_data", f"Fetching inflation data for {countries}")
        
        # Normalize country codes
        country_codes = [self._normalize_country_code(c) for c in countries]
        
        # Check cache
        countries_key = "_".join(sorted(country_codes))
        cache_key = f"inflation_data_{countries_key}"
            
        cached_data = self._get_from_cache(cache_key, 'indicators')
        if cached_data is not None:
            self.logger.info(f"Using cached inflation data")
            return cached_data
        
        # Use the default economic data provider or fall back to trading_economics
        provider = self.default_providers.get('economic_data', 'trading_economics')
        
        inflation_data = None
        
        # Ensure we have a connection to the data provider
        if not self._ensure_provider_connection(provider):
            self.logger.warning(f"Could not connect to {provider}, falling back to mock data")
            inflation_data = self._generate_mock_inflation_data(country_codes)
        else:
            try:
                # Fetch from the appropriate data provider
                if provider == "trading_economics":
                    inflation_data = self._fetch_trading_economics_inflation(country_codes)
                else:
                    self.logger.warning(f"Unknown provider {provider} for inflation data, falling back to mock data")
                    inflation_data = self._generate_mock_inflation_data(country_codes)
            except Exception as e:
                self.handle_error(e)
                self.logger.warning("Error fetching inflation data, falling back to mock data")
                inflation_data = self._generate_mock_inflation_data(country_codes)
        
        # Cache the data
        if inflation_data is not None and not inflation_data.empty:
            self._save_to_cache(inflation_data, cache_key, 'indicators')
        
        return inflation_data
    
    def _fetch_trading_economics_inflation(self, countries: List[str]) -> pd.DataFrame:
        """Fetch inflation data from Trading Economics."""
        self.log_action("fetch_data", f"Fetching inflation data from Trading Economics for {countries}")
        
        try:
            # Get the session and headers
            base_url = self._data_providers['trading_economics']['base_url']
            headers = self._data_providers['trading_economics']['headers']
            
            inflation_data = []
            
            # For each country, fetch inflation data
            for currency in countries:
                country_name = self.countries.get(currency)
                if not country_name:
                    continue
                
                # Format country name for the API (lowercase, spaces to hyphens)
                formatted_name = country_name.lower().replace(' ', '-')
                
                # Trading Economics API endpoint for inflation
                url = f"{base_url}/country/{formatted_name}/inflation-cpi"
                
                # Make the request
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    self.logger.warning(f"Failed to fetch inflation data for {currency}: {response.status_code}")
                    continue
                
                # Parse the JSON response
                data = response.json()
                
                for item in data:
                    try:
                        # Extract data from the response
                        value = item.get('Value', 0.0)
                        date_str = item.get('Date', '')
                        
                        # Parse the date
                        try:
                            data_dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                            data_date = data_dt.strftime('%Y-%m-%d')
                        except:
                            # Fallback
                            data_date = datetime.now().strftime('%Y-%m-%d')
                        
                        inflation_data.append({
                            'country': currency,
                            'indicator': 'Inflation Rate',
                            'date': data_date,
                            'value': float(value),
                            'unit': '%',
                            'frequency': 'Monthly'
                        })
                    except Exception as inner_e:
                        self.logger.error(f"Error parsing inflation data for {currency}: {str(inner_e)}")
                        continue
            
            # Convert to DataFrame
            if inflation_data:
                df = pd.DataFrame(inflation_data)
                
                # Add datetime column for sorting
                df['datetime'] = pd.to_datetime(df['date'])
                
                # Sort by country and date
                df = df.sort_values(['country', 'datetime'], ascending=[True, False])
                
                # Remove datetime column (used only for sorting)
                df = df.drop(columns=['datetime'])
                
                return df
            else:
                self.logger.warning("No inflation data found, returning empty DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to fetch inflation data: {str(e)}")
            return pd.DataFrame()
    
    # === News Analysis Methods ===
    
    def fetch_forex_news(
        self, 
        currencies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get recent news related to specific currencies.
        
        Args:
            currencies: List of currency codes to filter news for
            
        Returns:
            pd.DataFrame: Forex related news
        """
        self.log_action("fetch_forex_news", f"Fetching forex news for {currencies}")
        
        # Default to major currencies if none specified
        if not currencies:
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
            
        # Create cache key
        currencies_key = "_".join(sorted(currencies))
        cache_key = f"forex_news_{currencies_key}"
            
        # Check cache
        cached_data = self._get_from_cache(cache_key, 'news')
        if cached_data is not None:
            self.logger.info(f"Using cached forex news")
            return cached_data
        
        # Use the default news provider or fall back to news_api
        provider = self.default_providers.get('news', 'news_api')
        
        news_data = None
        
        # Ensure we have a connection to the data provider
        if not self._ensure_provider_connection(provider):
            self.logger.warning(f"Could not connect to {provider}, falling back to mock data")
            news_data = self._generate_mock_forex_news(currencies)
        else:
            try:
                # Fetch from the appropriate data provider
                if provider == "news_api":
                    news_data = self._fetch_news_api(currencies)
                elif provider == "investing_com":
                    news_data = self._fetch_investing_news(currencies)
                else:
                    self.logger.warning(f"Unknown provider {provider} for forex news, falling back to mock data")
                    news_data = self._generate_mock_forex_news(currencies)
            except Exception as e:
                self.handle_error(e)
                self.logger.warning("Error fetching forex news, falling back to mock data")
                news_data = self._generate_mock_forex_news(currencies)
        
        # Cache the data
        if news_data is not None and not news_data.empty:
            self._save_to_cache(news_data, cache_key, 'news')
        
        return news_data
    
    def _fetch_news_api(self, currencies: List[str]) -> pd.DataFrame:
        """Fetch forex news from News API."""
        self.log_action("fetch_data", f"Fetching forex news from News API for {currencies}")
        
        try:
            # Get API key
            api_key = self._data_providers['news_api']['api_key']
            base_url = self._data_providers['news_api']['base_url']
            
            # Build search queries based on currencies
            news_items = []
            
            # Common forex terms to improve search relevance
            forex_terms = ["forex", "currency", "exchange rate", "central bank"]
            
            # For each currency, fetch related news
            for currency in currencies:
                country = self.countries.get(currency, "")
                central_bank = self.central_banks.get(currency, {}).get('short', "")
                
                # Create search queries
                query_terms = [
                    f"({currency})",
                    f"({country} AND currency)",
                ]
                
                if central_bank:
                    query_terms.append(f"({central_bank})")
                
                # Add forex terms for better relevance
                query_string = " OR ".join(query_terms) + " AND (" + " OR ".join(forex_terms) + ")"
                
                # Parameters for the API request
                params = {
                    'apiKey': api_key,
                    'q': query_string,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 20  # Limit results per currency
                }
                
                # Make the request
                url = f"{base_url}/everything"
                response = requests.get(url, params=params)
                
                if response.status_code != 200:
                    self.logger.warning(f"Failed to fetch news for {currency}: {response.status_code}")
                    continue
                
                # Parse the JSON response
                data = response.json()
                articles = data.get('articles', [])
                
                for article in articles:
                    try:
                        # Extract article data
                        title = article.get('title', '')
                        description = article.get('description', '')
                        content = article.get('content', '')
                        url = article.get('url', '')
                        published_at = article.get('publishedAt', '')
                        source = article.get('source', {}).get('name', '')
                        
                        # Parse the date
                        try:
                            pub_dt = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
                            pub_date = pub_dt.strftime('%Y-%m-%d')
                            pub_time = pub_dt.strftime('%H:%M:%S')
                        except:
                            # Fallback
                            pub_date = datetime.now().strftime('%Y-%m-%d')
                            pub_time = datetime.now().strftime('%H:%M:%S')
                        
                        # Skip if the article doesn't actually mention the currency or country
                        text = (title + " " + description + " " + content).lower()
                        if currency.lower() not in text and country.lower() not in text:
                            continue
                        
                        # Add to news items
                        news_items.append({
                            'date': pub_date,
                            'time': pub_time,
                            'title': title,
                            'summary': description,
                            'content': content,
                            'url': url,
                            'source': source,
                            'related_currencies': currency,
                            'sentiment': None  # Will be calculated later
                        })
                    except Exception as inner_e:
                        self.logger.error(f"Error parsing news article: {str(inner_e)}")
                        continue
            
            # Convert to DataFrame
            if news_items:
                df = pd.DataFrame(news_items)
                
                # Remove duplicates based on URL
                df = df.drop_duplicates(subset=['url'])
                
                # Add datetime column
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                
                # Sort by datetime
                df = df.sort_values('datetime', ascending=False)
                
                return df
            else:
                self.logger.warning("No forex news found, returning empty DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to fetch forex news: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_investing_news(self, currencies: List[str]) -> pd.DataFrame:
        """Fetch forex news from Investing.com."""
        self.log_action("fetch_data", f"Fetching forex news from Investing.com for {currencies}")
        
        try:
            # Get session
            session = self._data_providers['investing_com']['session']
            base_url = self._data_providers['investing_com']['base_url']
            
            # Investing.com uses different URLs for different currency news
            news_items = []
            
            # Map of currencies to Investing.com URL paths
            currency_paths = {
                'EUR': 'eur-usd',
                'USD': 'us-dollar-index',
                'GBP': 'gbp-usd',
                'JPY': 'usd-jpy',
                'AUD': 'aud-usd',
                'CAD': 'usd-cad',
                'CHF': 'usd-chf',
                'NZD': 'nzd-usd'
            }
            
            # For each currency, fetch related news
            for currency in currencies:
                path = currency_paths.get(currency)
                if not path:
                    continue
                
                # URL for the currency news
                url = f"{base_url}/currencies/{path}-news"
                
                # Make the request
                response = session.get(url)
                
                if response.status_code != 200:
                    self.logger.warning(f"Failed to fetch Investing.com news for {currency}: {response.status_code}")
                    continue
                
                # Parse the HTML
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find news articles
                news_articles = soup.select('div.largeTitle article.js-article-item')
                
                for article in news_articles:
                    try:
                        # Extract article data
                        title_elem = article.select_one('a.title')
                        title = title_elem.text.strip() if title_elem else ""
                        url_path = title_elem['href'] if title_elem and 'href' in title_elem.attrs else ""
                        full_url = f"{base_url}{url_path}" if url_path.startswith('/') else url_path
                        
                        # Description/summary
                        desc_elem = article.select_one('p.description')
                        description = desc_elem.text.strip() if desc_elem else ""
                        
                        # Publication time
                        time_elem = article.select_one('span.date')
                        time_str = time_elem.text.strip() if time_elem else ""
                        
                        # Source (Investing.com or other)
                        source_elem = article.select_one('span.sourceName')
                        source = source_elem.text.strip() if source_elem else "Investing.com"
                        
                        # Parse the date
                        pub_date = datetime.now().strftime('%Y-%m-%d')
                        pub_time = "00:00:00"
                        
                        if time_str:
                            try:
                                # Investing.com uses relative times like "5 hours ago"
                                if "min ago" in time_str:
                                    mins = int(time_str.split()[0])
                                    pub_dt = datetime.now() - timedelta(minutes=mins)
                                elif "hour" in time_str:
                                    hours = int(time_str.split()[0])
                                    pub_dt = datetime.now() - timedelta(hours=hours)
                                elif "day" in time_str:
                                    days = int(time_str.split()[0])
                                    pub_dt = datetime.now() - timedelta(days=days)
                                else:
                                    # Try to parse as absolute date
                                    pub_dt = datetime.strptime(time_str, '%b %d, %Y')
                                
                                pub_date = pub_dt.strftime('%Y-%m-%d')
                                pub_time = pub_dt.strftime('%H:%M:%S')
                            except:
                                # Keep default
                                pass
                        
                        # Add to news items
                        news_items.append({
                            'date': pub_date,
                            'time': pub_time,
                            'title': title,
                            'summary': description,
                            'content': "",  # Would need to fetch the article page
                            'url': full_url,
                            'source': source,
                            'related_currencies': currency,
                            'sentiment': None  # Will be calculated later
                        })
                    except Exception as inner_e:
                        self.logger.error(f"Error parsing Investing.com news article: {str(inner_e)}")
                        continue
            
            # Convert to DataFrame
            if news_items:
                df = pd.DataFrame(news_items)
                
                # Remove duplicates based on URL
                df = df.drop_duplicates(subset=['url'])
                
                # Add datetime column
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                
                # Sort by datetime
                df = df.sort_values('datetime', ascending=False)
                
                return df
            else:
                self.logger.warning("No Investing.com forex news found, returning empty DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to fetch Investing.com forex news: {str(e)}")
            return pd.DataFrame()
    
    def analyze_news_impact(
        self, 
        news: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        currency: str
    ) -> Dict[str, Any]:
        """
        Analyze potential impact of news on a currency.
        
        Args:
            news: News data to analyze
            currency: Currency code to analyze impact for
            
        Returns:
            Dict[str, Any]: Analysis of news impact
        """
        self.log_action("analyze_news_impact", f"Analyzing news impact for {currency}")
        
        # Convert input to DataFrame if not already
        if isinstance(news, dict):
            news_df = pd.DataFrame([news])
        elif isinstance(news, list):
            news_df = pd.DataFrame(news)
        elif isinstance(news, pd.DataFrame):
            news_df = news
        else:
            raise ValueError(f"Unsupported news data type: {type(news)}")
        
        # Filter news for the specified currency
        currency_news = news_df[news_df['currency'].str.contains(currency, case=False, na=False)]
        
        if len(currency_news) == 0:
            return {
                "currency": currency,
                "impact": "neutral",
                "sentiment": 0.0,
                "key_topics": [],
                "summary": f"No news items found for {currency}."
            }
        
        # Analyze sentiment and impact of news
        # In a real implementation, this would use NLP techniques
        # For now, use the preset importance/sentiment
        
        avg_sentiment = currency_news.get('sentiment', currency_news.get('impact', 0)).mean()
        
        # Determine overall impact
        if abs(avg_sentiment) < 0.2:
            impact = "neutral"
        elif avg_sentiment >= 0.2:
            impact = "positive"
        else:
            impact = "negative"
            
        # Extract key topics
        if 'topics' in currency_news.columns:
            all_topics = []
            for topics in currency_news['topics']:
                if isinstance(topics, list):
                    all_topics.extend(topics)
            key_topics = pd.Series(all_topics).value_counts().head(5).index.tolist()
        else:
            key_topics = []
            
        # Generate summary
        summary = f"Analysis found {len(currency_news)} news items relevant to {currency}. "
        summary += f"Overall sentiment is {impact} ({avg_sentiment:.2f})."
        
        return {
            "currency": currency,
            "impact": impact,
            "sentiment": avg_sentiment,
            "key_topics": key_topics,
            "news_count": len(currency_news),
            "summary": summary
        }
    
    def classify_news_importance(
        self, 
        news: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Classify news by importance for trading.
        
        Args:
            news: News data to classify
            
        Returns:
            pd.DataFrame: News with importance classification
        """
        self.log_action("classify_news_importance", "Classifying news importance")
        
        # Convert input to DataFrame if not already
        if isinstance(news, dict):
            news_df = pd.DataFrame([news])
        elif isinstance(news, list):
            news_df = pd.DataFrame(news)
        elif isinstance(news, pd.DataFrame):
            news_df = news.copy()
        else:
            raise ValueError(f"Unsupported news data type: {type(news)}")
        
        # In a real implementation, this would use a trained classifier
        # For now, implement some heuristic rules
        
        # Initialize importance column if not present
        if 'importance' not in news_df.columns:
            news_df['importance'] = 'medium'
            
        # Rule 1: News containing central bank keywords is high importance
        central_bank_keywords = ['central bank', 'fed', 'federal reserve', 'ecb', 'boe', 'boj', 'rba', 'rate decision']
        pattern = '|'.join(central_bank_keywords)
        mask = news_df['title'].str.contains(pattern, case=False, na=False)
        news_df.loc[mask, 'importance'] = 'high'
        
        # Rule 2: News containing key economic indicators is high importance
        economic_keywords = ['gdp', 'inflation', 'cpi', 'unemployment', 'nfp', 'payroll', 'pmi', 'interest rate']
        pattern = '|'.join(economic_keywords)
        mask = news_df['title'].str.contains(pattern, case=False, na=False)
        news_df.loc[mask, 'importance'] = 'high'
        
        # Rule 3: News containing geopolitical keywords is medium-high importance
        geopolitical_keywords = ['war', 'conflict', 'sanction', 'tariff', 'trade war', 'election', 'brexit']
        pattern = '|'.join(geopolitical_keywords)
        mask = news_df['title'].str.contains(pattern, case=False, na=False)
        news_df.loc[mask, 'importance'] = 'medium-high'
        
        # Rule 4: News mentioning specific trading impacts is higher importance
        trading_keywords = ['rally', 'crash', 'surge', 'plunge', 'soar', 'tumble', 'volatile']
        pattern = '|'.join(trading_keywords)
        mask = news_df['title'].str.contains(pattern, case=False, na=False)
        # Increase importance by one level
        news_df.loc[mask & (news_df['importance'] == 'low'), 'importance'] = 'medium'
        news_df.loc[mask & (news_df['importance'] == 'medium'), 'importance'] = 'medium-high'
        news_df.loc[mask & (news_df['importance'] == 'medium-high'), 'importance'] = 'high'
        
        return news_df
    
    def extract_key_figures(
        self, 
        news: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Extract key numerical data from news.
        
        Args:
            news: News data to extract figures from
            
        Returns:
            Dict[str, Any]: Extracted key figures
        """
        self.log_action("extract_key_figures", "Extracting key figures from news")
        
        # Convert input to DataFrame if not already
        if isinstance(news, dict):
            news_df = pd.DataFrame([news])
        elif isinstance(news, list):
            news_df = pd.DataFrame(news)
        elif isinstance(news, pd.DataFrame):
            news_df = news
        else:
            raise ValueError(f"Unsupported news data type: {type(news)}")
        
        # In a real implementation, this would use NLP techniques to extract figures
        # For now, return mock extracted data
        
        # Extract content from all articles
        if 'content' in news_df.columns:
            all_content = ' '.join(news_df['content'].fillna('').astype(str))
        elif 'text' in news_df.columns:
            all_content = ' '.join(news_df['text'].fillna('').astype(str))
        else:
            all_content = ' '.join(news_df['title'].fillna('').astype(str))
        
        # In a real implementation, we would extract numbers and their context
        # For now, return a mock result
        
        return {
            "extracted_figures": {
                "interest_rates": {
                    "USD": "0.25-0.50%",
                    "EUR": "0.00%",
                    "GBP": "0.75%"
                },
                "gdp_growth": {
                    "USA": "2.8%",
                    "Eurozone": "1.3%",
                    "UK": "1.8%"
                },
                "inflation": {
                    "USA": "7.9%",
                    "Eurozone": "5.8%",
                    "UK": "6.2%"
                }
            },
            "context": {
                "period": "Latest reported figures",
                "source": "News articles analysis",
                "reliability": "medium"
            }
        }
    
    # === Fundamental Analysis Methods ===
    
    def calculate_interest_rate_differential(
        self, 
        currency_pair: str
    ) -> Dict[str, Any]:
        """
        Calculate interest rate differential between two currencies.
        
        Args:
            currency_pair: Currency pair to analyze (e.g., 'EUR/USD')
            
        Returns:
            Dict[str, Any]: Interest rate differential analysis
        """
        self.log_action("calculate_interest_rate_differential", f"Calculating interest rate differential for {currency_pair}")
        
        # Parse currency pair
        if '/' in currency_pair:
            base_currency, quote_currency = currency_pair.split('/')
        else:
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[3:]
            
        # Get current central bank rates
        rates_df = self.get_central_bank_rates()
        
        # Extract rates for the two currencies
        try:
            base_rate = rates_df.loc[rates_df['currency'] == base_currency, 'rate'].iloc[0]
            quote_rate = rates_df.loc[rates_df['currency'] == quote_currency, 'rate'].iloc[0]
        except (IndexError, KeyError) as e:
            self.logger.error(f"Failed to find interest rates for {currency_pair}: {str(e)}")
            return {
                "currency_pair": currency_pair,
                "base_currency": base_currency,
                "quote_currency": quote_currency,
                "base_rate": None,
                "quote_rate": None,
                "differential": None,
                "status": "error",
                "error": f"Rates not found for one or both currencies"
            }
            
        # Calculate differential
        differential = base_rate - quote_rate
        
        # Determine the carry trade direction
        carry_direction = None
        if differential > 0:
            carry_direction = f"Long {base_currency}, Short {quote_currency}"
        elif differential < 0:
            carry_direction = f"Long {quote_currency}, Short {base_currency}"
        else:
            carry_direction = "No carry advantage"
            
        # Get central bank names
        base_bank = self.central_banks.get(base_currency, {}).get('name', f"{base_currency} Central Bank")
        quote_bank = self.central_banks.get(quote_currency, {}).get('name', f"{quote_currency} Central Bank")
        
        return {
            "currency_pair": currency_pair,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "base_rate": base_rate,
            "quote_rate": quote_rate,
            "differential": differential,
            "base_central_bank": base_bank,
            "quote_central_bank": quote_bank,
            "carry_trade_direction": carry_direction,
            "status": "success"
        }
    
    def analyze_economic_strength(
        self, 
        country: str
    ) -> Dict[str, Any]:
        """
        Analyze overall economic strength of a country.
        
        Args:
            country: Country code or name to analyze
            
        Returns:
            Dict[str, Any]: Economic strength analysis
        """
        self.log_action("analyze_economic_strength", f"Analyzing economic strength for {country}")
        
        # Normalize country code
        country_code = self._normalize_country_code(country)
        country_name = self.countries.get(country_code, country)
        
        # Get key economic indicators
        gdp_data = self.get_gdp_data([country_code])
        inflation_data = self.get_inflation_data([country_code])
        
        # Get central bank rates
        rates_df = self.get_central_bank_rates()
        
        # Extract indicators for scoring
        try:
            # GDP growth
            gdp_growth = gdp_data.loc[gdp_data['country'] == country_code, 'growth_rate'].iloc[0]
            
            # Inflation
            inflation_rate = inflation_data.loc[inflation_data['country'] == country_code, 'rate'].iloc[0]
            
            # Interest rate
            interest_rate = rates_df.loc[rates_df['currency'] == country_code, 'rate'].iloc[0]
            
        except (IndexError, KeyError) as e:
            self.logger.error(f"Failed to find economic data for {country}: {str(e)}")
            return {
                "country": country_name,
                "country_code": country_code,
                "status": "error",
                "error": f"Economic data not complete for {country}"
            }
        
        # Calculate real interest rate (nominal rate - inflation)
        real_interest_rate = interest_rate - inflation_rate
        
        # Score economic indicators (simplified)
        gdp_score = self._score_gdp_growth(gdp_growth)
        inflation_score = self._score_inflation(inflation_rate)
        interest_score = self._score_interest_rate(interest_rate, inflation_rate)
        
        # Calculate overall score (weighted average)
        overall_score = (
            0.4 * gdp_score +      # GDP growth (40% weight)
            0.4 * inflation_score + # Inflation (40% weight)
            0.2 * interest_score    # Interest rates (20% weight)
        )
        
        # Determine economic strength category
        strength_category = self._categorize_economic_strength(overall_score)
        
        # Generate narrative
        narrative = self._generate_economic_narrative(
            country_name, gdp_growth, inflation_rate, interest_rate, real_interest_rate, overall_score
        )
        
        return {
            "country": country_name,
            "country_code": country_code,
            "gdp_growth": gdp_growth,
            "inflation_rate": inflation_rate,
            "interest_rate": interest_rate,
            "real_interest_rate": real_interest_rate,
            "gdp_score": gdp_score,
            "inflation_score": inflation_score,
            "interest_score": interest_score,
            "overall_score": overall_score,
            "strength_category": strength_category,
            "narrative": narrative,
            "status": "success"
        }
    
    def predict_currency_movement(
        self, 
        currency: str, 
        timeframe: str = "short_term"
    ) -> Dict[str, Any]:
        """
        Predict potential currency movement based on fundamentals.
        
        Args:
            currency: Currency code to analyze
            timeframe: Time horizon for prediction ('short_term', 'medium_term', or 'long_term')
            
        Returns:
            Dict[str, Any]: Currency movement prediction
        """
        self.log_action("predict_currency_movement", f"Predicting {timeframe} movement for {currency}")
        
        # Normalize currency code
        currency_code = currency.upper()
        
        # Get country associated with currency
        country = self.countries.get(currency_code, f"{currency_code} Region")
        
        # Analyze economic strength
        economic_analysis = self.analyze_economic_strength(currency_code)
        
        # Fetch recent news for the currency
        news_df = self.fetch_forex_news([currency_code])
        news_impact = self.analyze_news_impact(news_df, currency_code)
        
        # Get upcoming economic events
        start_date = datetime.now().strftime('%Y-%m-%d')
        if timeframe == "short_term":
            end_date = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
        elif timeframe == "medium_term":
            end_date = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
        else:  # long_term
            end_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
            
        calendar_df = self.get_economic_calendar(start_date, end_date, [currency_code])
        
        # Convert timeframe to weighting factors for different signals
        weights = self._get_timeframe_weights(timeframe)
        
        # Calculate prediction score components
        economic_score = economic_analysis.get('overall_score', 50) 
        news_score = 50 + (news_impact.get('sentiment', 0) * 50)
        event_score = self._calculate_event_impact_score(calendar_df, currency_code)
        
        # Calculate weighted prediction score
        prediction_score = (
            weights['economic'] * economic_score +
            weights['news'] * news_score +
            weights['events'] * event_score
        )
        
        # Interpret the score
        if prediction_score > 60:
            direction = "bullish"
            strength = min(int((prediction_score - 60) * 2.5), 100)
        elif prediction_score < 40:
            direction = "bearish"
            strength = min(int((40 - prediction_score) * 2.5), 100)
        else:
            direction = "neutral"
            strength = 100 - abs(50 - prediction_score) * 5
            
        # Generate narrative
        narrative = self._generate_movement_narrative(
            currency_code, country, direction, strength, economic_analysis, news_impact, timeframe
        )
        
        return {
            "currency": currency_code,
            "country": country,
            "timeframe": timeframe,
            "direction": direction,
            "strength": strength,
            "prediction_score": prediction_score,
            "economic_strength": economic_analysis.get('strength_category'),
            "news_sentiment": news_impact.get('impact'),
            "key_upcoming_events": self._extract_key_events(calendar_df, 3),
            "narrative": narrative,
            "status": "success"
        }
    
    def identify_key_events(
        self, 
        currency: str,
        days_ahead: int = 30
    ) -> pd.DataFrame:
        """
        Identify important upcoming events for a currency.
        
        Args:
            currency: Currency code to find events for
            days_ahead: Number of days to look ahead
            
        Returns:
            pd.DataFrame: Key upcoming events
        """
        self.log_action("identify_key_events", f"Identifying key events for {currency} in next {days_ahead} days")
        
        # Normalize currency code
        currency_code = currency.upper()
        
        # Get economic calendar for the specified period
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        calendar_df = self.get_economic_calendar(start_date, end_date, [currency_code])
        
        # Filter for important events
        if 'importance' in calendar_df.columns:
            important_events = calendar_df[calendar_df['importance'].isin(['high', 'medium-high'])]
        else:
            # If no importance column, use impact
            important_events = calendar_df[calendar_df['impact'].isin(['high', 'medium'])]
            
        # Sort by date and importance
        if 'importance' in important_events.columns:
            importance_order = {'high': 0, 'medium-high': 1, 'medium': 2, 'low': 3}
            important_events['importance_order'] = important_events['importance'].map(importance_order)
            important_events = important_events.sort_values(['date', 'importance_order'])
            important_events = important_events.drop('importance_order', axis=1)
        else:
            important_events = important_events.sort_values(['date', 'impact'])
            
        return important_events 

    # === Signal Generation Methods ===
    
    def generate_fundamental_signals(
        self, 
        currency_pair: str
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on fundamentals.
        
        Args:
            currency_pair: Currency pair to generate signals for
            
        Returns:
            Dict[str, Any]: Trading signals based on fundamentals
        """
        self.log_action("generate_fundamental_signals", f"Generating fundamental signals for {currency_pair}")
        
        # Parse currency pair
        if '/' in currency_pair:
            base_currency, quote_currency = currency_pair.split('/')
        else:
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[3:]
            
        # Analyze each currency
        base_analysis = self.predict_currency_movement(base_currency)
        quote_analysis = self.predict_currency_movement(quote_currency)
        
        # Get interest rate differential
        rate_diff = self.calculate_interest_rate_differential(currency_pair)
        
        # Calculate relative strength
        base_strength = base_analysis.get('prediction_score', 50)
        quote_strength = quote_analysis.get('prediction_score', 50)
        relative_strength = base_strength - quote_strength
        
        # Determine primary signal
        if relative_strength > 10:
            signal = "buy"
            strength = min(int(relative_strength), 100)
        elif relative_strength < -10:
            signal = "sell"
            strength = min(int(-relative_strength), 100)
        else:
            signal = "neutral"
            strength = 100 - abs(relative_strength) * 10
            
        # Adjust for interest rate differential
        interest_differential = rate_diff.get('differential', 0)
        if abs(interest_differential) > 0.5:
            if interest_differential > 0 and signal != "sell":
                # Higher interest rate for base currency strengthens buy signal
                strength += 10
            elif interest_differential < 0 and signal != "buy":
                # Higher interest rate for quote currency strengthens sell signal
                strength += 10
                
        # Cap strength at 100
        strength = min(strength, 100)
        
        # Get upcoming key events
        base_events = self.identify_key_events(base_currency, 14)
        quote_events = self.identify_key_events(quote_currency, 14)
        
        # Check for major upcoming events that could impact the signal
        major_events_upcoming = len(base_events[base_events['importance'] == 'high']) > 0 or \
                               len(quote_events[quote_events['importance'] == 'high']) > 0
                               
        # Generate trading signal with timeframes
        timeframes = {}
        for tf in ["short_term", "medium_term", "long_term"]:
            base_tf = self.predict_currency_movement(base_currency, tf)
            quote_tf = self.predict_currency_movement(quote_currency, tf)
            tf_relative_strength = base_tf.get('prediction_score', 50) - quote_tf.get('prediction_score', 50)
            
            if tf_relative_strength > 10:
                tf_signal = "buy"
                tf_strength = min(int(tf_relative_strength), 100)
            elif tf_relative_strength < -10:
                tf_signal = "sell"
                tf_strength = min(int(-tf_relative_strength), 100)
            else:
                tf_signal = "neutral"
                tf_strength = 100 - abs(tf_relative_strength) * 10
                
            timeframes[tf] = {
                "signal": tf_signal,
                "strength": tf_strength,
                "confidence": self._calculate_confidence(tf_strength, major_events_upcoming, tf)
            }
        
        # Generate supporting analysis text
        analysis_text = self._generate_signal_analysis(
            currency_pair, base_currency, quote_currency, signal, strength,
            base_analysis, quote_analysis, rate_diff, major_events_upcoming
        )
        
        return {
            "currency_pair": currency_pair,
            "signal": signal,
            "strength": strength,
            "confidence": self._calculate_confidence(strength, major_events_upcoming, "short_term"),
            "timeframes": timeframes,
            "base_currency": {
                "code": base_currency,
                "outlook": base_analysis.get('direction'),
                "strength": base_analysis.get('strength')
            },
            "quote_currency": {
                "code": quote_currency,
                "outlook": quote_analysis.get('direction'),
                "strength": quote_analysis.get('strength')
            },
            "interest_rate_differential": interest_differential,
            "major_events_upcoming": major_events_upcoming,
            "key_events": self._extract_key_events(pd.concat([base_events, quote_events]), 5),
            "analysis": analysis_text,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    
    def rate_fundamental_strength(
        self, 
        currency: str
    ) -> Dict[str, Any]:
        """
        Rate the fundamental strength of a currency.
        
        Args:
            currency: Currency code to rate
            
        Returns:
            Dict[str, Any]: Fundamental strength rating
        """
        self.log_action("rate_fundamental_strength", f"Rating fundamental strength for {currency}")
        
        # Normalize currency code
        currency_code = currency.upper()
        
        # Get economic analysis
        economic = self.analyze_economic_strength(currency_code)
        
        # Get news sentiment
        news = self.fetch_forex_news([currency_code])
        news_impact = self.analyze_news_impact(news, currency_code)
        
        # Get key indicators
        gdp_data = self.get_gdp_data([currency_code])
        inflation_data = self.get_inflation_data([currency_code])
        rate_data = self.get_central_bank_rates()
        
        # Calculate ratings for each fundamental factor
        ratings = {}
        
        # Economic growth rating
        try:
            gdp_growth = gdp_data.loc[gdp_data['country'] == currency_code, 'growth_rate'].iloc[0]
            if gdp_growth > 3.0:
                ratings['economic_growth'] = {'score': 90, 'description': 'Excellent growth'}
            elif gdp_growth > 2.0:
                ratings['economic_growth'] = {'score': 75, 'description': 'Strong growth'}
            elif gdp_growth > 1.0:
                ratings['economic_growth'] = {'score': 60, 'description': 'Moderate growth'}
            elif gdp_growth > 0:
                ratings['economic_growth'] = {'score': 50, 'description': 'Slow growth'}
            elif gdp_growth > -1.0:
                ratings['economic_growth'] = {'score': 40, 'description': 'Stagnant'}
            else:
                ratings['economic_growth'] = {'score': 25, 'description': 'Contraction'}
        except (IndexError, KeyError):
            ratings['economic_growth'] = {'score': 50, 'description': 'Unknown'}
            
        # Inflation rating
        try:
            inflation_rate = inflation_data.loc[inflation_data['country'] == currency_code, 'rate'].iloc[0]
            if 1.5 <= inflation_rate <= 2.5:
                ratings['inflation'] = {'score': 85, 'description': 'Optimal inflation'}
            elif 0.5 <= inflation_rate < 1.5 or 2.5 < inflation_rate <= 3.5:
                ratings['inflation'] = {'score': 70, 'description': 'Acceptable inflation'}
            elif 0 <= inflation_rate < 0.5 or 3.5 < inflation_rate <= 5:
                ratings['inflation'] = {'score': 50, 'description': 'Concerning inflation'}
            elif inflation_rate < 0:
                ratings['inflation'] = {'score': 30, 'description': 'Deflation'}
            else:
                ratings['inflation'] = {'score': 20, 'description': 'High inflation'}
        except (IndexError, KeyError):
            ratings['inflation'] = {'score': 50, 'description': 'Unknown'}
            
        # Interest rate rating
        try:
            interest_rate = rate_data.loc[rate_data['currency'] == currency_code, 'rate'].iloc[0]
            inflation_rate = inflation_data.loc[inflation_data['country'] == currency_code, 'rate'].iloc[0]
            real_rate = interest_rate - inflation_rate
            
            if real_rate > 2:
                ratings['interest_rate'] = {'score': 80, 'description': 'High real yield'}
            elif real_rate > 0.5:
                ratings['interest_rate'] = {'score': 70, 'description': 'Positive real yield'}
            elif real_rate > -0.5:
                ratings['interest_rate'] = {'score': 50, 'description': 'Neutral real yield'}
            elif real_rate > -2:
                ratings['interest_rate'] = {'score': 40, 'description': 'Negative real yield'}
            else:
                ratings['interest_rate'] = {'score': 20, 'description': 'Deeply negative real yield'}
        except (IndexError, KeyError):
            ratings['interest_rate'] = {'score': 50, 'description': 'Unknown'}
            
        # News sentiment rating
        sentiment = news_impact.get('sentiment', 0)
        if sentiment > 0.5:
            ratings['news_sentiment'] = {'score': 85, 'description': 'Very positive'}
        elif sentiment > 0.2:
            ratings['news_sentiment'] = {'score': 70, 'description': 'Positive'}
        elif sentiment > -0.2:
            ratings['news_sentiment'] = {'score': 50, 'description': 'Neutral'}
        elif sentiment > -0.5:
            ratings['news_sentiment'] = {'score': 30, 'description': 'Negative'}
        else:
            ratings['news_sentiment'] = {'score': 15, 'description': 'Very negative'}
            
        # Calculate overall rating
        weights = {
            'economic_growth': 0.35,
            'inflation': 0.25,
            'interest_rate': 0.25,
            'news_sentiment': 0.15
        }
        
        overall_score = sum(
            ratings[factor]['score'] * weights[factor] 
            for factor in weights.keys() 
            if factor in ratings
        )
        
        # Normalize score to account for missing factors
        available_weight = sum(weights[factor] for factor in ratings.keys())
        if available_weight > 0:
            overall_score = overall_score / available_weight * 100
            
        # Determine strength category
        if overall_score >= 80:
            strength_category = "Very Strong"
        elif overall_score >= 65:
            strength_category = "Strong"
        elif overall_score >= 50:
            strength_category = "Moderate"
        elif overall_score >= 35:
            strength_category = "Weak"
        else:
            strength_category = "Very Weak"
            
        return {
            "currency": currency_code,
            "country": self.countries.get(currency_code, f"{currency_code} Region"),
            "overall_score": overall_score,
            "strength_category": strength_category,
            "factor_ratings": ratings,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    
    def identify_divergence(
        self, 
        currency: str
    ) -> Dict[str, Any]:
        """
        Identify divergence between price and fundamentals.
        
        Args:
            currency: Currency code to analyze
            
        Returns:
            Dict[str, Any]: Divergence analysis
        """
        self.log_action("identify_divergence", f"Identifying divergence for {currency}")
        
        # Normalize currency code
        currency_code = currency.upper()
        
        # Get USD pair for analysis (since USD is the reference currency)
        if currency_code == "USD":
            currency_pair = "EUR/USD"  # Use EUR as reference for USD
            is_inverse = True
        else:
            currency_pair = f"{currency_code}/USD"
            is_inverse = False
            
        # Get fundamental strength
        strength = self.rate_fundamental_strength(currency_code)
        
        # Get price data
        # This would typically come from the MarketDataAgent, but we'll mock it
        # For a real implementation, you would call the MarketDataAgent instead
        price_data = self._get_mock_price_data(currency_pair)
        
        # Calculate recent price change
        if len(price_data) >= 2:
            current_price = price_data['close'].iloc[-1]
            start_price = price_data['close'].iloc[0]
            price_change_pct = ((current_price - start_price) / start_price) * 100
            
            if is_inverse:
                price_change_pct = -price_change_pct  # Invert for USD
                
            # Get 1-month and 3-month changes
            if len(price_data) >= 30:
                month_price = price_data['close'].iloc[-30]
                month_change_pct = ((current_price - month_price) / month_price) * 100
                if is_inverse:
                    month_change_pct = -month_change_pct
            else:
                month_change_pct = price_change_pct
                
            if len(price_data) >= 90:
                three_month_price = price_data['close'].iloc[-90]
                three_month_change_pct = ((current_price - three_month_price) / three_month_price) * 100
                if is_inverse:
                    three_month_change_pct = -three_month_change_pct
            else:
                three_month_change_pct = price_change_pct
        else:
            price_change_pct = 0
            month_change_pct = 0
            three_month_change_pct = 0
            
        # Calculate expected price movement based on fundamentals
        fundamental_score = strength.get('overall_score', 50)
        expected_bias = fundamental_score - 50  # Positive value suggests currency should strengthen
        
        # Check for divergence between price movement and fundamentals
        # Positive divergence means price is lower than fundamentals suggest it should be
        divergence = None
        magnitude = 0
        
        if expected_bias > 10 and price_change_pct < 0:
            # Strong fundamentals but price is falling
            divergence = "positive"
            magnitude = expected_bias - price_change_pct
        elif expected_bias < -10 and price_change_pct > 0:
            # Weak fundamentals but price is rising
            divergence = "negative"
            magnitude = abs(expected_bias) + price_change_pct
        elif expected_bias > 5 and price_change_pct < expected_bias - 5:
            # Price not rising as much as fundamentals suggest
            divergence = "positive"
            magnitude = expected_bias - price_change_pct
        elif expected_bias < -5 and price_change_pct > expected_bias + 5:
            # Price not falling as much as fundamentals suggest
            divergence = "negative"
            magnitude = abs(expected_bias) + price_change_pct
            
        # Normalize magnitude
        if divergence:
            magnitude = min(magnitude / 2, 100)
        
        # Generate trading opportunity signal
        if divergence == "positive" and magnitude > 30:
            opportunity = "strong_buy"
            explanation = f"Strong positive divergence suggests {currency_code} is undervalued."
        elif divergence == "positive" and magnitude > 15:
            opportunity = "buy"
            explanation = f"Positive divergence suggests {currency_code} may be undervalued."
        elif divergence == "negative" and magnitude > 30:
            opportunity = "strong_sell"
            explanation = f"Strong negative divergence suggests {currency_code} is overvalued."
        elif divergence == "negative" and magnitude > 15:
            opportunity = "sell"
            explanation = f"Negative divergence suggests {currency_code} may be overvalued."
        else:
            opportunity = "neutral"
            explanation = f"No significant divergence between {currency_code} price and fundamentals."
            
        return {
            "currency": currency_code,
            "currency_pair": currency_pair,
            "fundamental_score": fundamental_score,
            "fundamental_bias": "bullish" if expected_bias > 0 else "bearish" if expected_bias < 0 else "neutral",
            "price_change": {
                "recent": price_change_pct,
                "1_month": month_change_pct,
                "3_month": three_month_change_pct
            },
            "divergence_type": divergence,
            "divergence_magnitude": magnitude,
            "trading_opportunity": opportunity,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    
    # === Data Sources Integration Methods ===
    
    def connect_to_data_provider(
        self, 
        provider: str
    ) -> bool:
        """
        Connect to an external data provider.
        
        Args:
            provider: Name of the data provider to connect to
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        self.log_action("connect_to_data_provider", f"Connecting to data provider: {provider}")
        
        # Check if we have API credentials for this provider
        provider_key = f"{provider}_api_key"
        if provider_key not in self.api_keys:
            self.logger.warning(f"No API key found for provider {provider}")
            return False
            
        api_key = self.api_keys.get(provider_key)
        if not api_key:
            self.logger.warning(f"Empty API key for provider {provider}")
            return False
            
        # Different connection logic for each provider
        # In a real implementation, this would establish actual connections
        
        if provider == "trading_economics":
            return self._connect_to_trading_economics(api_key)
        elif provider == "forex_factory":
            return self._connect_to_forex_factory(api_key)
        elif provider == "investing_com":
            return self._connect_to_investing_com(api_key)
        elif provider == "news_api":
            return self._connect_to_news_api(api_key)
        else:
            self.logger.warning(f"Unknown provider: {provider}")
            return False
    
    def _connect_to_trading_economics(self, api_key: str) -> bool:
        """Connect to Trading Economics API."""
        try:
            # Actual Trading Economics API connection
            headers = {
                'Authorization': f'Client {api_key}'
            }
            # Test connection with a simple API call
            response = requests.get('https://api.tradingeconomics.com/markets/symbol/USD', headers=headers)
            
            if response.status_code == 200:
                self.logger.info("Connected to Trading Economics API successfully")
                self._data_providers = self._data_providers or {}
                self._data_providers['trading_economics'] = {
                    'session': requests.Session(),
                    'headers': headers,
                    'base_url': 'https://api.tradingeconomics.com',
                    'connected': True
                }
                return True
            else:
                self.logger.error(f"Failed to connect to Trading Economics API: {response.status_code}")
                return False
        except Exception as e:
            self.handle_error(e)
            return False
    
    def _connect_to_forex_factory(self, api_key: str) -> bool:
        """Connect to Forex Factory API."""
        try:
            # Forex Factory doesn't have a direct API, but we can use web scraping
            # For now, we'll set up a session with headers to mimic a browser
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            })
            
            # Test connection by accessing the calendar page
            response = session.get('https://www.forexfactory.com/calendar')
            
            if response.status_code == 200:
                self.logger.info("Connected to Forex Factory successfully")
                self._data_providers = self._data_providers or {}
                self._data_providers['forex_factory'] = {
                    'session': session,
                    'base_url': 'https://www.forexfactory.com',
                    'connected': True
                }
                return True
            else:
                self.logger.error(f"Failed to connect to Forex Factory: {response.status_code}")
                return False
        except Exception as e:
            self.handle_error(e)
            return False
    
    def _connect_to_investing_com(self, api_key: str) -> bool:
        """Connect to Investing.com API."""
        try:
            # Investing.com needs specific headers for its API/web access
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Accept-Language': 'en-US,en;q=0.5',
                'X-Requested-With': 'XMLHttpRequest',
                'Connection': 'keep-alive'
            })
            
            # Test connection with a basic page
            response = session.get('https://www.investing.com/economic-calendar/')
            
            if response.status_code == 200:
                self.logger.info("Connected to Investing.com successfully")
                self._data_providers = self._data_providers or {}
                self._data_providers['investing_com'] = {
                    'session': session,
                    'base_url': 'https://www.investing.com',
                    'connected': True
                }
                return True
            else:
                self.logger.error(f"Failed to connect to Investing.com: {response.status_code}")
                return False
        except Exception as e:
            self.handle_error(e)
            return False
    
    def _connect_to_news_api(self, api_key: str) -> bool:
        """Connect to News API."""
        try:
            # News API connection test
            params = {
                'apiKey': api_key,
                'q': 'forex',
                'pageSize': 1
            }
            response = requests.get('https://newsapi.org/v2/everything', params=params)
            
            if response.status_code == 200:
                self.logger.info("Connected to News API successfully")
                self._data_providers = self._data_providers or {}
                self._data_providers['news_api'] = {
                    'api_key': api_key,
                    'base_url': 'https://newsapi.org/v2',
                    'connected': True
                }
                return True
            else:
                self.logger.error(f"Failed to connect to News API: {response.status_code}, {response.json().get('message', '')}")
                return False
        except Exception as e:
            self.handle_error(e)
            return False
    
    def parse_economic_data(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]], str],
        data_type: str = "json"
    ) -> pd.DataFrame:
        """
        Parse and normalize economic data from different formats.
        
        Args:
            data: Raw data to parse
            data_type: Format of the data ('json', 'csv', 'xml')
            
        Returns:
            pd.DataFrame: Parsed and normalized data
        """
        self.log_action("parse_economic_data", f"Parsing {data_type} economic data")
        
        try:
            if data_type == "json":
                if isinstance(data, str):
                    # Parse JSON string
                    parsed_data = json.loads(data)
                else:
                    # Already a Python object
                    parsed_data = data
                    
                if isinstance(parsed_data, list):
                    # Convert list of dicts to DataFrame
                    df = pd.DataFrame(parsed_data)
                elif isinstance(parsed_data, dict):
                    if "data" in parsed_data and isinstance(parsed_data["data"], list):
                        # Common API response format
                        df = pd.DataFrame(parsed_data["data"])
                    else:
                        # Single record as dict
                        df = pd.DataFrame([parsed_data])
                else:
                    raise ValueError(f"Unsupported JSON structure: {type(parsed_data)}")
                    
            elif data_type == "csv":
                # Parse CSV data
                if isinstance(data, str):
                    # CSV string
                    df = pd.read_csv(pd.StringIO(data))
                else:
                    raise ValueError("CSV data must be a string")
                    
            elif data_type == "xml":
                # XML parsing would go here
                # This is a placeholder - real implementation would use a proper XML parser
                raise NotImplementedError("XML parsing not implemented")
                
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
            # Normalize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Handle date columns
            date_columns = [col for col in df.columns if 'date' in col or 'time' in col]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    # If conversion fails, leave as is
                    pass
                    
            return df
            
        except Exception as e:
            self.handle_error(e)
            self.logger.error(f"Failed to parse economic data: {str(e)}")
            # Return empty DataFrame on error
            return pd.DataFrame()
    
    def handle_data_provider_errors(
        self, 
        provider: str = None, 
        error: Exception = None
    ) -> Dict[str, Any]:
        """
        Handle errors from data providers.
        
        Args:
            provider: Name of the data provider (None for general error handling)
            error: The exception that occurred (None for status check)
            
        Returns:
            Dict[str, Any]: Error handling results and provider status
        """
        if error:
            self.log_action("handle_data_provider_errors", f"Handling error from {provider}: {str(error)}")
        else:
            self.log_action("handle_data_provider_errors", f"Checking provider status for {provider}")
        
        # If no provider specified, check all providers
        if not provider:
            results = {}
            for provider_name in self.default_providers.values():
                results[provider_name] = self.handle_data_provider_errors(provider_name)
            return results
            
        # If no error specified, just check provider status
        if not error:
            # Check provider connection
            connection_ok = self.connect_to_data_provider(provider)
            if connection_ok:
                return {
                    "provider": provider,
                    "status": "connected",
                    "error": None,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "provider": provider,
                    "status": "disconnected",
                    "error": "Could not connect to provider",
                    "timestamp": datetime.now().isoformat()
                }
                
        # Handle specific error types
        if isinstance(error, requests.exceptions.Timeout):
            self.logger.warning(f"Timeout connecting to {provider}, will retry")
            return {
                "provider": provider,
                "status": "timeout",
                "error": str(error),
                "action": "retry",
                "timestamp": datetime.now().isoformat()
            }
            
        elif isinstance(error, requests.exceptions.ConnectionError):
            self.logger.error(f"Connection error with {provider}")
            return {
                "provider": provider,
                "status": "connection_error",
                "error": str(error),
                "action": "pause_and_retry",
                "timestamp": datetime.now().isoformat()
            }
            
        elif isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response.status_code == 429:
                # Rate limit exceeded
                self.logger.warning(f"Rate limit exceeded for {provider}")
                return {
                    "provider": provider,
                    "status": "rate_limited",
                    "error": str(error),
                    "action": "wait_and_retry",
                    "timestamp": datetime.now().isoformat()
                }
            elif hasattr(error, 'response') and error.response.status_code == 401:
                # Authentication error
                self.logger.error(f"Authentication error with {provider}")
                return {
                    "provider": provider,
                    "status": "auth_error",
                    "error": str(error),
                    "action": "check_credentials",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Other HTTP error
                self.logger.error(f"HTTP error with {provider}: {str(error)}")
                return {
                    "provider": provider,
                    "status": "http_error",
                    "error": str(error),
                    "action": "evaluate_and_retry",
                    "timestamp": datetime.now().isoformat()
                }
                
        else:
            # Generic error
            self.logger.error(f"Error with {provider}: {str(error)}")
            return {
                "provider": provider,
                "status": "error",
                "error": str(error),
                "action": "log_and_continue",
                "timestamp": datetime.now().isoformat()
            }
    
    # === Helper Methods - Part 1 ===
    
    def _get_from_cache(
        self, 
        key: str, 
        data_type: str
    ) -> Optional[pd.DataFrame]:
        """
        Get data from cache if available and not expired.
        
        Args:
            key: Cache key
            data_type: Type of data (determines subfolder)
            
        Returns:
            Optional[pd.DataFrame]: Cached data or None if not available
        """
        if not self.cache_enabled:
            return None
            
        # Check memory cache first
        if key in self._data_cache:
            cache_entry = self._data_cache[key]
            age = datetime.now() - cache_entry['timestamp']
            if age.total_seconds() < self.cache_expiry_hours * 3600:
                return cache_entry['data']
                
        # Check disk cache
        cache_file = self.fundamentals_data_dir / data_type / f"{key}.csv"
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.total_seconds() < self.cache_expiry_hours * 3600:
                try:
                    return pd.read_csv(cache_file)
                except Exception as e:
                    self.logger.warning(f"Failed to read cache file {cache_file}: {e}")
                    
        return None
    
    def _save_to_cache(
        self, 
        data: pd.DataFrame, 
        key: str, 
        data_type: str
    ) -> bool:
        """
        Save data to cache.
        
        Args:
            data: Data to cache
            key: Cache key
            data_type: Type of data (determines subfolder)
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.cache_enabled:
            return False
            
        try:
            # Save to memory cache
            self._data_cache[key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            # Save to disk cache
            cache_dir = self.fundamentals_data_dir / data_type
            cache_dir.mkdir(exist_ok=True, parents=True)
            
            cache_file = cache_dir / f"{key}.csv"
            data.to_csv(cache_file, index=False)
            
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")
            return False
    
    def _normalize_country_code(self, country: str) -> str:
        """
        Normalize country to currency code.
        
        Args:
            country: Country name or code
            
        Returns:
            str: Normalized currency code
        """
        country = country.upper()
        
        # If it's already a currency code
        if country in self.countries:
            return country
            
        # Try to find by country name
        for code, name in self.countries.items():
            if country in name.upper():
                return code
                
        # Return original if no match found
        return country
    
    def _generate_mock_economic_calendar(
        self, 
        start_date: str, 
        end_date: str, 
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate mock economic calendar data for testing."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end - start).days + 1
        
        # Filter countries if specified
        if countries:
            country_list = countries
        else:
            country_list = list(self.countries.keys())
            
        # Create events list
        events = []
        current_date = start
        
        # Common economic events
        event_types = [
            ("Interest Rate Decision", "high"),
            ("GDP", "high"),
            ("CPI", "high"),
            ("Unemployment Rate", "high"),
            ("Retail Sales", "medium"),
            ("PMI", "medium"),
            ("Trade Balance", "medium"),
            ("Industrial Production", "medium"),
            ("Consumer Confidence", "medium"),
            ("Building Permits", "low"),
            ("Current Account", "low"),
            ("Housing Starts", "low")
        ]
        
        # Generate events
        for _ in range(min(days * 5, 100)):  # Max 5 events per day, 100 total
            currency = np.random.choice(country_list)
            country_name = self.countries.get(currency, currency)
            
            event_type, importance = event_types[np.random.randint(0, len(event_types))]
            
            # Randomize date within range
            days_offset = np.random.randint(0, days)
            event_date = (start + timedelta(days=days_offset)).strftime('%Y-%m-%d')
            
            # Randomize time
            hour = np.random.randint(7, 18)
            minute = np.random.choice([0, 15, 30, 45])
            event_time = f"{hour:02d}:{minute:02d}"
            
            # Add to events list
            events.append({
                'date': event_date,
                'time': event_time,
                'currency': currency,
                'country': country_name,
                'event': f"{country_name} {event_type}",
                'importance': importance,
                'previous': f"{np.random.uniform(-1, 5):.1f}%",
                'forecast': f"{np.random.uniform(-1, 5):.1f}%",
                'actual': None  # Will be filled after the event
            })
            
        # Sort by date and time
        events_df = pd.DataFrame(events)
        events_df['datetime'] = pd.to_datetime(events_df['date'] + ' ' + events_df['time'])
        events_df = events_df.sort_values('datetime')
        
        return events_df
    
    def _generate_mock_economic_indicators(
        self, 
        country_code: str, 
        indicator: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate mock economic indicator data for testing."""
        # Base indicators for all countries
        indicators = [
            "GDP Growth Rate",
            "Inflation Rate",
            "Unemployment Rate",
            "Interest Rate",
            "Government Debt to GDP",
            "Balance of Trade",
            "Current Account to GDP",
            "Industrial Production",
            "Retail Sales YoY",
            "Consumer Confidence"
        ]
        
        # Filter by indicator if specified
        if indicator:
            indicators = [ind for ind in indicators if indicator.lower() in ind.lower()]
            
        # Generate quarterly data for past 3 years
        dates = []
        current_date = datetime.now().replace(day=1)
        for i in range(12):  # 12 quarters = 3 years
            dates.append(current_date - timedelta(days=i*90))
            
        dates.reverse()  # Oldest first
        
        # Create random data for each indicator
        data = []
        
        for indicator_name in indicators:
            # Set base value and volatility based on indicator type
            if "GDP" in indicator_name:
                base = 2.0
                volatility = 0.8
            elif "Inflation" in indicator_name:
                base = 2.5
                volatility = 0.5
            elif "Unemployment" in indicator_name:
                base = 5.0
                volatility = 0.3
            elif "Interest Rate" in indicator_name:
                base = 1.5
                volatility = 0.25
            elif "Debt" in indicator_name:
                base = 60.0
                volatility = 3.0
            elif "Balance" in indicator_name or "Account" in indicator_name:
                base = 0.0
                volatility = 2.0
            else:
                base = 1.0
                volatility = 1.0
                
            # Generate time series with some autocorrelation
            values = [base]
            for i in range(len(dates) - 1):
                # New value with autocorrelation to previous value
                new_value = values[-1] * 0.8 + base * 0.2 + np.random.normal(0, volatility)
                values.append(new_value)
                
            # Add rows to data
            for i, date in enumerate(dates):
                data.append({
                    'country': country_code,
                    'indicator': indicator_name,
                    'date': date.strftime('%Y-%m-%d'),
                    'value': values[i],
                    'unit': self._get_indicator_unit(indicator_name)
                })
                
        return pd.DataFrame(data)
    
    def _get_indicator_unit(self, indicator_name: str) -> str:
        """Get appropriate unit for an economic indicator."""
        if "GDP" in indicator_name or "Growth" in indicator_name:
            return "%"
        elif "Inflation" in indicator_name:
            return "%"
        elif "Unemployment" in indicator_name:
            return "%"
        elif "Interest Rate" in indicator_name:
            return "%"
        elif "Debt" in indicator_name:
            return "% of GDP"
        elif "Balance" in indicator_name:
            return "Million USD"
        elif "Account" in indicator_name:
            return "% of GDP"
        elif "Production" in indicator_name:
            return "% YoY"
        elif "Sales" in indicator_name:
            return "% YoY"
        elif "Confidence" in indicator_name:
            return "Index Points"
        else:
            return "Value"
    
    # === Helper Methods - Part 2 ===
    
    def _generate_mock_central_bank_rates(self) -> pd.DataFrame:
        """Generate mock central bank interest rate data for testing."""
        # Create rates data for major currencies
        rates_data = []
        
        for currency, country in self.countries.items():
            # Set base rate depending on currency
            if currency == "USD":
                base_rate = 5.5
            elif currency == "EUR":
                base_rate = 4.0
            elif currency == "GBP":
                base_rate = 5.25
            elif currency == "JPY":
                base_rate = 0.1
            elif currency == "CHF":
                base_rate = 1.5
            elif currency == "AUD":
                base_rate = 4.35
            elif currency == "CAD":
                base_rate = 5.0
            elif currency == "NZD":
                base_rate = 5.5
            else:
                base_rate = 3.0 + np.random.uniform(-1, 2)
                
            # Get central bank info
            central_bank = self.central_banks.get(currency, {}).get('name', f"{country} Central Bank")
            central_bank_short = self.central_banks.get(currency, {}).get('short', central_bank)
            
            # Add small random variation
            actual_rate = round(base_rate + np.random.uniform(-0.1, 0.1), 2)
            
            # Add to data
            rates_data.append({
                'currency': currency,
                'country': country,
                'central_bank': central_bank,
                'central_bank_short': central_bank_short,
                'rate': actual_rate,
                'previous_rate': round(actual_rate + np.random.uniform(-0.25, 0.25), 2),
                'last_change_date': (datetime.now() - timedelta(days=np.random.randint(30, 180))).strftime('%Y-%m-%d'),
                'next_meeting_date': (datetime.now() + timedelta(days=np.random.randint(7, 60))).strftime('%Y-%m-%d')
            })
            
        return pd.DataFrame(rates_data)
    
    def _generate_mock_gdp_data(self, countries: List[str]) -> pd.DataFrame:
        """Generate mock GDP data for testing."""
        gdp_data = []
        
        for country in countries:
            # Set base GDP growth depending on country
            if country == "USD":
                base_growth = 2.5
            elif country == "EUR":
                base_growth = 1.0
            elif country == "GBP":
                base_growth = 1.5
            elif country == "JPY":
                base_growth = 1.0
            elif country == "CHF":
                base_growth = 1.8
            elif country == "AUD":
                base_growth = 2.0
            elif country == "CAD":
                base_growth = 1.8
            elif country == "NZD":
                base_growth = 2.2
            else:
                base_growth = 2.0
                
            # Add variation
            actual_growth = round(base_growth + np.random.uniform(-0.7, 0.7), 1)
            
            # Add to data
            gdp_data.append({
                'country': country,
                'growth_rate': actual_growth,
                'previous_growth_rate': round(actual_growth + np.random.uniform(-0.5, 0.5), 1),
                'annual_gdp': round(np.random.uniform(500, 25000) * (1 if country != "USD" else 1000)),
                'period': f"Q{np.random.randint(1, 5)} {datetime.now().year}",
                'updated_date': (datetime.now() - timedelta(days=np.random.randint(10, 90))).strftime('%Y-%m-%d')
            })
            
        return pd.DataFrame(gdp_data)
    
    def _generate_mock_inflation_data(self, countries: List[str]) -> pd.DataFrame:
        """Generate mock inflation data for testing."""
        inflation_data = []
        
        for country in countries:
            # Set base inflation depending on country
            if country == "USD":
                base_inflation = 3.7
            elif country == "EUR":
                base_inflation = 2.6
            elif country == "GBP":
                base_inflation = 4.0
            elif country == "JPY":
                base_inflation = 2.8
            elif country == "CHF":
                base_inflation = 1.6
            elif country == "AUD":
                base_inflation = 3.4
            elif country == "CAD":
                base_inflation = 3.8
            elif country == "NZD":
                base_inflation = 4.0
            else:
                base_inflation = 3.0
                
            # Add variation
            actual_inflation = round(base_inflation + np.random.uniform(-0.3, 0.3), 1)
            
            # Add to data
            inflation_data.append({
                'country': country,
                'rate': actual_inflation,
                'previous_rate': round(actual_inflation + np.random.uniform(-0.4, 0.4), 1),
                'core_rate': round(base_inflation - 0.5 + np.random.uniform(-0.2, 0.2), 1),
                'period': f"{datetime.now().strftime('%b %Y')}",
                'updated_date': (datetime.now() - timedelta(days=np.random.randint(5, 30))).strftime('%Y-%m-%d')
            })
            
        return pd.DataFrame(inflation_data)
    
    def _generate_mock_forex_news(self, currencies: List[str]) -> pd.DataFrame:
        """Generate mock forex news data for testing."""
        news_data = []
        
        # News topics for each currency
        topics = {
            "USD": ["Fed", "Interest Rates", "US Economy", "Inflation", "Employment"],
            "EUR": ["ECB", "EU Economy", "Eurozone", "Inflation", "German Economy"],
            "GBP": ["BoE", "UK Economy", "Brexit", "Inflation", "Housing Market"],
            "JPY": ["BoJ", "Japanese Economy", "Yen", "Inflation", "Stimulus"],
            "CHF": ["SNB", "Swiss Economy", "Safe Haven", "Banking", "Exports"],
            "AUD": ["RBA", "Australian Economy", "Commodities", "China Trade", "Housing"],
            "CAD": ["BoC", "Canadian Economy", "Oil Prices", "Housing", "Exports"],
            "NZD": ["RBNZ", "New Zealand Economy", "Dairy", "Trade", "Housing"]
        }
        
        # Generic topics for any currency
        generic_topics = [
            "Forex Market", "Currency Strength", "Central Banks", 
            "Economic Outlook", "Interest Rates", "Inflation", "Trade"
        ]
        
        # News sentiment ranges for each currency (this would reflect current market conditions)
        sentiment_ranges = {
            "USD": (-0.2, 0.5),  # Slightly positive bias
            "EUR": (-0.4, 0.3),  # Slightly negative bias
            "GBP": (-0.3, 0.3),  # Neutral
            "JPY": (-0.1, 0.4),  # Slightly positive bias
            "CHF": (0.0, 0.5),   # Positive bias
            "AUD": (-0.3, 0.4),  # Neutral
            "CAD": (-0.2, 0.5),  # Slightly positive bias
            "NZD": (-0.4, 0.3)   # Slightly negative bias
        }
        
        # News headlines to combine with topics
        headline_templates = [
            "{currency} {direction} as {topic} {impact}",
            "{topic} data {impact} {currency} outlook",
            "{entity} decision {impact} {currency} exchange rates",
            "Analysts: {currency} to {direction} on {topic} {trend}",
            "{topic}: What it means for {currency} traders",
            "{entity} {action} amid {topic} concerns",
            "{currency} {direction} to {timeframe} {level}",
            "Market Alert: {topic} drives {currency} volatility"
        ]
        
        # Possible values for template variables
        directions = ["rises", "falls", "surges", "drops", "rallies", "weakens", "strengthens", "stabilizes"]
        impacts = ["improves", "weighs on", "boosts", "dampens", "supports", "undermines"]
        trends = ["growth", "slowdown", "recovery", "contraction", "expansion", "trends"]
        entities = ["Central Bank", "Government", "Policymakers", "Regulators", "Treasury", "Finance Ministry"]
        actions = ["acts", "intervenes", "announces policy", "signals shift", "maintains stance", "raises concerns"]
        timeframes = ["1-year", "6-month", "multi-year", "session", "weekly", "monthly"]
        levels = ["high", "low", "resistance", "support", "average", "target"]
        
        # Generate news items
        for _ in range(min(len(currencies) * 10, 100)):  # Generate up to 10 news items per currency
            # Pick a random currency from the list
            currency = np.random.choice(currencies)
            
            # Get possible topics for this currency
            currency_topics = topics.get(currency, generic_topics)
            all_topics = currency_topics + generic_topics
            
            # Pick 1-3 topics for this news item
            num_topics = np.random.randint(1, 4)
            selected_topics = list(np.random.choice(all_topics, size=min(num_topics, len(all_topics)), replace=False))
            
            # Get sentiment range for this currency
            sentiment_min, sentiment_max = sentiment_ranges.get(currency, (-0.3, 0.3))
            sentiment = np.random.uniform(sentiment_min, sentiment_max)
            
            # Generate headline
            topic = np.random.choice(selected_topics)
            template = np.random.choice(headline_templates)
            headline = template.format(
                currency=currency,
                topic=topic,
                direction=np.random.choice(directions),
                impact=np.random.choice(impacts),
                trend=np.random.choice(trends),
                entity=np.random.choice(entities),
                action=np.random.choice(actions),
                timeframe=np.random.choice(timeframes),
                level=np.random.choice(levels)
            )
            
            # Generate publication date (within past 7 days)
            days_ago = np.random.randint(0, 7)
            hours_ago = np.random.randint(0, 24)
            pub_date = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            # Add to news data
            news_data.append({
                'title': headline,
                'currency': currency,
                'topics': selected_topics,
                'sentiment': sentiment,
                'importance': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2]),
                'source': np.random.choice(['Reuters', 'Bloomberg', 'Financial Times', 'Wall Street Journal', 'CNBC']),
                'url': f"https://example.com/news/{currency.lower()}/{pub_date.strftime('%Y%m%d')}/{np.random.randint(1000, 9999)}",
                'published_date': pub_date.strftime('%Y-%m-%d %H:%M:%S')
            })
            
        # Convert to DataFrame and sort by published date (most recent first)
        news_df = pd.DataFrame(news_data)
        news_df['published_date'] = pd.to_datetime(news_df['published_date'])
        news_df = news_df.sort_values('published_date', ascending=False)
        
        return news_df
    
    def _get_mock_price_data(self, currency_pair: str) -> pd.DataFrame:
        """Generate mock price data for testing."""
        # Generate 180 days of daily data
        dates = []
        current_date = datetime.now()
        for i in range(180):
            dates.append(current_date - timedelta(days=i))
            
        dates.reverse()  # Oldest first
        
        # Set base price depending on currency pair
        if currency_pair == "EUR/USD":
            base_price = 1.1000
            volatility = 0.0030
        elif currency_pair == "USD/JPY":
            base_price = 150.00
            volatility = 0.5000
        elif currency_pair == "GBP/USD":
            base_price = 1.2700
            volatility = 0.0040
        elif currency_pair == "AUD/USD":
            base_price = 0.6600
            volatility = 0.0025
        elif currency_pair == "USD/CAD":
            base_price = 1.3500
            volatility = 0.0030
        elif currency_pair == "NZD/USD":
            base_price = 0.6100
            volatility = 0.0025
        elif currency_pair == "USD/CHF":
            base_price = 0.8700
            volatility = 0.0030
        else:
            base_price = 1.0000
            volatility = 0.0030
            
        # Generate price series with trends and some random walks
        prices = [base_price]
        
        # Create a few trend periods
        trend_periods = np.random.randint(3, 7)  # 3-6 trend periods
        period_length = len(dates) // trend_periods
        trends = []
        
        for i in range(trend_periods):
            if i == 0:
                # First trend is random
                trend = np.random.choice([-1, 1]) * volatility / 3
            else:
                # Subsequent trends tend to reverse previous trend
                prev_trend = trends[-1]
                if np.random.random() < 0.7:  # 70% chance of reversal
                    trend = -np.sign(prev_trend) * volatility / 3 * np.random.uniform(0.5, 1.5)
                else:
                    trend = np.sign(prev_trend) * volatility / 3 * np.random.uniform(0.5, 1.5)
                    
            trends.append(trend)
            
        # Apply trends to price series
        for i in range(1, len(dates)):
            period_idx = min(i // period_length, trend_periods - 1)
            trend = trends[period_idx]
            
            # Previous price plus trend component plus random walk
            new_price = prices[-1] + trend + np.random.normal(0, volatility / 2)
            prices.append(new_price)
            
        # Create OHLC data
        data = []
        for i, date in enumerate(dates):
            price = prices[i]
            data.append({
                'datetime': date,
                'open': price * (1 + np.random.uniform(-0.0005, 0.0005)),
                'high': price * (1 + np.random.uniform(0.0005, 0.0020)),
                'low': price * (1 - np.random.uniform(0.0005, 0.0020)),
                'close': price,
                'volume': np.random.randint(1000, 10000)
            })
            
        return pd.DataFrame(data)
    
    def _score_gdp_growth(self, growth_rate: float) -> float:
        """Score GDP growth on a 0-100 scale."""
        if growth_rate >= 4.0:
            return 95.0  # Excellent growth
        elif growth_rate >= 3.0:
            return 85.0  # Very strong growth
        elif growth_rate >= 2.0:
            return 75.0  # Strong growth
        elif growth_rate >= 1.0:
            return 60.0  # Moderate growth
        elif growth_rate >= 0.0:
            return 50.0  # Weak growth
        elif growth_rate >= -1.0:
            return 35.0  # Contraction
        elif growth_rate >= -2.0:
            return 25.0  # Recession
        else:
            return 10.0  # Severe recession
    
    def _score_inflation(self, inflation_rate: float) -> float:
        """Score inflation on a 0-100 scale."""
        # Optimal inflation is around 2%
        if 1.5 <= inflation_rate <= 2.5:
            return 90.0  # Optimal inflation
        elif 1.0 <= inflation_rate < 1.5 or 2.5 < inflation_rate <= 3.0:
            return 80.0  # Good inflation
        elif 0.5 <= inflation_rate < 1.0 or 3.0 < inflation_rate <= 4.0:
            return 65.0  # Acceptable inflation
        elif 0.0 <= inflation_rate < 0.5 or 4.0 < inflation_rate <= 5.0:
            return 50.0  # Concerning inflation
        elif inflation_rate < 0.0:
            return 30.0  # Deflation
        else:
            return max(10.0, 100.0 - (inflation_rate - 5.0) * 10.0)  # High inflation
    
    def _score_interest_rate(self, interest_rate: float, inflation_rate: float) -> float:
        """Score interest rate on a 0-100 scale."""
        # Real interest rate (nominal - inflation)
        real_rate = interest_rate - inflation_rate
        
        if real_rate >= 2.0:
            return 85.0  # Very high real yield
        elif real_rate >= 1.0:
            return 75.0  # High real yield
        elif real_rate >= 0.0:
            return 65.0  # Positive real yield
        elif real_rate >= -1.0:
            return 50.0  # Slightly negative real yield
        elif real_rate >= -2.0:
            return 35.0  # Negative real yield
        else:
            return 20.0  # Very negative real yield
    
    def _categorize_economic_strength(self, score: float) -> str:
        """Categorize economic strength based on score."""
        if score >= 80:
            return "Very Strong"
        elif score >= 65:
            return "Strong"
        elif score >= 50:
            return "Moderate"
        elif score >= 35:
            return "Weak"
        else:
            return "Very Weak"
    
    def _generate_economic_narrative(
        self, 
        country: str, 
        gdp_growth: float, 
        inflation: float, 
        interest_rate: float,
        real_interest_rate: float,
        overall_score: float
    ) -> str:
        """Generate narrative about economic conditions."""
        strength = self._categorize_economic_strength(overall_score)
        
        narrative = f"The {country} economy is showing {strength.lower()} fundamentals. "
        
        # GDP component
        if gdp_growth >= 3.0:
            narrative += f"GDP growth is excellent at {gdp_growth:.1f}%, indicating robust economic expansion. "
        elif gdp_growth >= 2.0:
            narrative += f"GDP growth is strong at {gdp_growth:.1f}%, showing healthy economic activity. "
        elif gdp_growth >= 1.0:
            narrative += f"GDP growth is moderate at {gdp_growth:.1f}%, indicating steady but not robust growth. "
        elif gdp_growth >= 0.0:
            narrative += f"GDP growth is weak at {gdp_growth:.1f}%, showing minimal economic expansion. "
        else:
            narrative += f"GDP is contracting at {gdp_growth:.1f}%, indicating economic challenges. "
            
        # Inflation component
        if 1.5 <= inflation <= 2.5:
            narrative += f"Inflation at {inflation:.1f}% is at an optimal level. "
        elif inflation < 1.0:
            narrative += f"Inflation at {inflation:.1f}% is below target, indicating potential deflationary pressures. "
        elif inflation > 4.0:
            narrative += f"Inflation is high at {inflation:.1f}%, creating challenges for the economy. "
        else:
            narrative += f"Inflation stands at {inflation:.1f}%. "
            
        # Interest rate component
        narrative += f"The nominal interest rate is {interest_rate:.2f}%, "
        if real_interest_rate > 1.0:
            narrative += f"resulting in a positive real yield of {real_interest_rate:.2f}%. "
        elif real_interest_rate >= 0:
            narrative += f"providing a slightly positive real yield of {real_interest_rate:.2f}%. "
        else:
            narrative += f"giving a negative real yield of {real_interest_rate:.2f}%. "
            
        return narrative
            
    def _get_timeframe_weights(self, timeframe: str) -> Dict[str, float]:
        """Get weighting factors for different signals based on timeframe."""
        if timeframe == "short_term":
            return {
                'economic': 0.2,  # 20% weight on economic data
                'news': 0.5,      # 50% weight on news
                'events': 0.3      # 30% weight on upcoming events
            }
        elif timeframe == "medium_term":
            return {
                'economic': 0.5,  # 50% weight on economic data
                'news': 0.3,      # 30% weight on news
                'events': 0.2      # 20% weight on upcoming events
            }
        else:  # long_term
            return {
                'economic': 0.7,  # 70% weight on economic data
                'news': 0.2,      # 20% weight on news
                'events': 0.1      # 10% weight on upcoming events
            }
            
    def _calculate_event_impact_score(self, events_df: pd.DataFrame, currency: str) -> float:
        """Calculate the potential impact score of upcoming events."""
        if len(events_df) == 0:
            return 50.0  # Neutral impact if no events
            
        # Filter for the currency and next 14 days
        if 'currency' in events_df.columns:
            currency_events = events_df[events_df['currency'] == currency]
        else:
            currency_events = events_df
            
        if len(currency_events) == 0:
            return 50.0
            
        # Map importance to numeric values
        importance_map = {
            'high': 5.0,
            'medium-high': 3.0,
            'medium': 2.0,
            'low': 0.5
        }
        
        # Calculate impact score
        total_impact = 0
        count = 0
        
        for _, event in currency_events.iterrows():
            impact = importance_map.get(event.get('importance', 'medium'), 2.0)
            total_impact += impact
            count += 1
            
        # Normalize to 0-100 scale
        if count > 0:
            # More high-impact events = higher score
            base_score = 50.0
            impact_score = min(100, base_score + (total_impact / count) * 5)
            return impact_score
        else:
            return 50.0
            
    def _extract_key_events(self, events_df: pd.DataFrame, count: int = 3) -> List[Dict[str, Any]]:
        """Extract key upcoming events from events DataFrame."""
        if len(events_df) == 0:
            return []
            
        # Sort by importance and date
        if 'importance' in events_df.columns:
            importance_order = {'high': 0, 'medium-high': 1, 'medium': 2, 'low': 3}
            events_df['importance_order'] = events_df['importance'].map(importance_order)
            sorted_events = events_df.sort_values(['importance_order', 'date'])
            sorted_events = sorted_events.drop('importance_order', axis=1)
        else:
            sorted_events = events_df.sort_values('date')
            
        # Extract top events
        key_events = []
        for _, event in sorted_events.head(count).iterrows():
            key_events.append({
                'date': event.get('date', 'Unknown'),
                'time': event.get('time', 'Unknown'),
                'currency': event.get('currency', 'Unknown'),
                'event': event.get('event', 'Unknown'),
                'importance': event.get('importance', 'medium')
            })
            
        return key_events
            
    def _generate_movement_narrative(
        self, 
        currency: str, 
        country: str, 
        direction: str, 
        strength: float,
        economic: Dict[str, Any],
        news: Dict[str, Any],
        timeframe: str
    ) -> str:
        """Generate narrative about currency movement prediction."""
        if direction == "bullish":
            direction_text = "rise"
            direction_adj = "bullish"
        elif direction == "bearish":
            direction_text = "fall"
            direction_adj = "bearish"
        else:
            direction_text = "remain stable"
            direction_adj = "neutral"
            
        strength_text = "strongly " if strength > 70 else ""
        
        timeframe_text = {
            "short_term": "short term (1-2 weeks)",
            "medium_term": "medium term (1-3 months)",
            "long_term": "long term (3-12 months)"
        }.get(timeframe, timeframe)
        
        narrative = f"The {currency} is forecast to {strength_text}{direction_text} in the {timeframe_text}, "
        narrative += f"with a {direction_adj} outlook "
        
        if strength > 70:
            narrative += f"and strong conviction. "
        elif strength > 50:
            narrative += f"and moderate conviction. "
        else:
            narrative += f"but low conviction. "
            
        # Add economic analysis
        econ_strength = economic.get('strength_category', 'Moderate')
        narrative += f"The {country} economy is showing {econ_strength.lower()} fundamentals. "
        
        # Add news sentiment if available
        if news:
            sentiment = news.get('impact', 'neutral')
            news_count = news.get('news_count', 0)
            if news_count > 0:
                narrative += f"Recent news sentiment is {sentiment} based on {news_count} news items. "
                
        # Add key factors based on timeframe
        if timeframe == "short_term":
            narrative += "Short-term movement is likely to be influenced primarily by news flow and upcoming economic releases. "
        elif timeframe == "medium_term":
            narrative += "Medium-term outlook is balanced between economic fundamentals and evolving market sentiment. "
        else:
            narrative += "Long-term direction will be predominantly determined by structural economic factors and monetary policy. "
            
        return narrative
    
    def _generate_signal_analysis(
        self,
        currency_pair: str,
        base_currency: str,
        quote_currency: str,
        signal: str,
        strength: float,
        base_analysis: Dict[str, Any],
        quote_analysis: Dict[str, Any],
        rate_diff: Dict[str, Any],
        major_events: bool
    ) -> str:
        """Generate in-depth analysis text for trading signal."""
        # Start with the main signal
        if signal == "buy":
            direction = "buying"
            direction_text = "strengthen against"
        elif signal == "sell":
            direction = "selling"
            direction_text = "weaken against"
        else:
            direction = "neutral on"
            direction_text = "trade sideways against"
            
        analysis = f"Technical analysis suggests {direction} {currency_pair} "
        
        if strength > 80:
            analysis += "with strong conviction. "
        elif strength > 60:
            analysis += "with moderate conviction. "
        else:
            analysis += "with low conviction. "
            
        # Add relative fundamental analysis
        base_strength = base_analysis.get('strength', 50)
        quote_strength = quote_analysis.get('strength', 50)
        base_direction = base_analysis.get('direction', 'neutral')
        quote_direction = quote_analysis.get('direction', 'neutral')
        
        analysis += f"Fundamentally, {base_currency} is expected to {direction_text} {quote_currency} as "
        
        # Explain why
        if base_direction == "bullish" and quote_direction == "bearish":
            analysis += f"the {base_currency} outlook is bullish while {quote_currency} outlook is bearish. "
        elif base_direction == "bullish" and quote_direction != "bullish":
            analysis += f"the {base_currency} outlook is bullish. "
        elif base_direction != "bearish" and quote_direction == "bearish":
            analysis += f"the {quote_currency} outlook is bearish. "
        elif base_direction == "bearish" and quote_direction == "bullish":
            analysis += f"the {base_currency} outlook is bearish while {quote_currency} outlook is bullish. "
        else:
            analysis += f"relative economic strength favors the {base_currency if base_strength > quote_strength else quote_currency}. "
            
        # Add interest rate differential info
        if 'differential' in rate_diff:
            diff = rate_diff['differential']
            if abs(diff) >= 0.5:
                if diff > 0:
                    analysis += f"The interest rate differential of {diff:.2f}% favors the {base_currency}. "
                else:
                    analysis += f"The interest rate differential of {abs(diff):.2f}% favors the {quote_currency}. "
                    
        # Add warning about upcoming events if applicable
        if major_events:
            analysis += "Caution is advised due to major economic events on the horizon that could impact price action. "
            
        return analysis
    
    def _calculate_confidence(
        self, 
        strength: float,
        major_events: bool,
        timeframe: str
    ) -> str:
        """Calculate confidence level for a prediction or signal."""
        # Base confidence on strength
        if strength > 80:
            base_confidence = "high"
        elif strength > 60:
            base_confidence = "medium-high"
        elif strength > 40:
            base_confidence = "medium"
        elif strength > 20:
            base_confidence = "medium-low"
        else:
            base_confidence = "low"
            
        # Adjust for major events
        if major_events and timeframe == "short_term":
            # Downgrade confidence one level for short-term if major events pending
            if base_confidence == "high":
                return "medium-high"
            elif base_confidence == "medium-high":
                return "medium"
            elif base_confidence == "medium":
                return "medium-low"
            else:
                return "low"
                
        return base_confidence
    
    # === LangGraph Integration ===
    
    def setup_node(self) -> Callable:
        """
        Set up this agent as a node in a LangGraph workflow.
        
        Returns:
            Callable: Function that can be used as a node in a LangGraph
        """
        def fundamentals_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """LangGraph node function for the FundamentalsAgent."""
            # Extract task from state
            task = state.get('task', {})
            task_type = task.get('type', '')
            
            if not task_type:
                return {
                    **state,
                    'error': 'No task type specified for FundamentalsAgent',
                    'status': 'error'
                }
                
            # Execute the task
            try:
                result = self.run_task(task)
                
                # Update state with the result
                return {
                    **state,
                    'result': result,
                    'status': result.get('status', 'success')
                }
                
            except Exception as e:
                self.handle_error(e)
                return {
                    **state,
                    'error': str(e),
                    'status': 'error'
                }
                
        return fundamentals_node 