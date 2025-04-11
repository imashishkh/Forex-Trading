#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentiment Agent implementation for the Forex Trading Platform

This module provides a comprehensive implementation of a sentiment analyst agent
that retrieves and analyzes sentiment data from news, social media, and market positioning
affecting forex markets.
"""

import os
import json
import logging
import requests
import traceback
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
from datetime import datetime, timedelta
from pathlib import Path
import abc
import re

# Data analysis imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# NLP and API imports
import openai
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Base Agent class
from utils.base_agent import BaseAgent, AgentState, AgentMessage

# LangGraph imports
import langgraph.graph as lg
from langgraph.checkpoint.memory import MemorySaver


class SentimentAgent(BaseAgent):
    """
    Sentiment Agent for analyzing market sentiment for forex trading.
    
    This agent is responsible for retrieving and analyzing sentiment data from various sources,
    including news articles, social media, and market positioning data. It provides insights
    on market sentiment, potential currency movements, and trading signals based on sentiment analysis.
    """
    
    def __init__(
        self,
        agent_name: str = "sentiment_agent",
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Sentiment Agent.
        
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
        self.sentiment_data_dir = self.data_dir / 'sentiment'
        
        # Ensure data directories exist
        self.sentiment_data_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.api_keys = self.config.get('api_credentials', {})
        
        # Cache settings
        self.cache_enabled = self.config.get('system', {}).get('cache_enabled', True)
        self.cache_expiry_hours = self.config.get('system', {}).get('cache_expiry_hours', 24)
        
        # Internal cache for frequently accessed data
        self._data_cache = {}
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            'strongly_bullish': 0.75,
            'bullish': 0.25,
            'neutral_upper': 0.25,
            'neutral_lower': -0.25,
            'bearish': -0.25,
            'strongly_bearish': -0.75
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
        
        # Default data providers
        self.default_providers = {
            'news': 'news_api',
            'social_media': 'twitter_api',
            'market_positioning': 'cot_reports',
            'analyst_ratings': 'investing_com'
        }
        
        # Override provider settings if provided
        if config and 'data_providers' in config:
            self.default_providers.update(config['data_providers'])
            
        # Initialize NLTK sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        self.log_action("init", f"Sentiment Agent initialized")
    
    def initialize(self) -> bool:
        """
        Set up the Sentiment Agent and its resources.
        
        This method handles resource allocation, connection to external
        services, and any other initialization tasks.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.log_action("initialize", "Sentiment Agent initialization started")
        
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
            
            self.log_action("initialize", "Sentiment Agent initialized successfully")
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
        """Initialize the data cache for frequently accessed data."""
        self._data_cache = {
            'news_sentiment': {},
            'social_media_sentiment': {},
            'analyst_ratings': {},
            'market_positioning': {},
            'combined_sentiment': {}
        }
    
    def _get_from_cache(self, key: str, data_type: str) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if available and not expired.
        
        Args:
            key: Cache key for the data
            data_type: Type of data (used for organization)
            
        Returns:
            Optional[pd.DataFrame]: Cached data if available, None otherwise
        """
        if not self.cache_enabled:
            return None
            
        cache_key = f"{data_type}_{key}"
        
        if cache_key in self._data_cache:
            cached_item = self._data_cache[cache_key]
            timestamp = cached_item.get('timestamp')
            
            if timestamp:
                # Check if cache is expired
                now = datetime.now()
                diff = now - timestamp
                
                if diff.total_seconds() < self.cache_expiry_hours * 3600:
                    self.logger.debug(f"Using cached data for {cache_key}")
                    return cached_item.get('data')
                    
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, key: str, data_type: str) -> bool:
        """
        Save data to cache.
        
        Args:
            data: Data to cache
            key: Cache key for the data
            data_type: Type of data (used for organization)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.cache_enabled:
            return False
            
        try:
            cache_key = f"{data_type}_{key}"
            
            self._data_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
            return False
    
    def _normalize_currency_pair(self, currency_pair: str) -> str:
        """
        Normalize currency pair to standard format with underscore separator (e.g. EUR_USD).
        Handles various input formats: EUR/USD, EURUSD, etc.
        
        Args:
            currency_pair: Currency pair in any common format
            
        Returns:
            str: Normalized currency pair (BASE_QUOTE format with underscore)
        """
        # Remove any whitespace
        pair = currency_pair.strip().upper()
        
        # Handle slash format (EUR/USD)
        if '/' in pair:
            pair = pair.replace('/', '_')
        
        # Handle period format (EUR.USD)
        if '.' in pair:
            pair = pair.replace('.', '_')
        
        # Handle direct format without separator (EURUSD)
        if '_' not in pair and len(pair) == 6:
            pair = f"{pair[:3]}_{pair[3:]}"
        
        # For any other formats, do our best to standardize
        if '_' not in pair and len(pair) > 6:
            # Try to find common currency codes
            common_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
            
            for currency in common_currencies:
                if pair.startswith(currency):
                    remainder = pair[len(currency):]
                    if remainder in common_currencies:
                        pair = f"{currency}_{remainder}"
                        break
            
            # If we still don't have a normalized pair, handle special case of 3-letter vs 2-letter
            if '_' not in pair:
                if len(pair) == 5:  # Like USDJP
                    pair = f"{pair[:3]}_{pair[3:]}"
        
        # Log if we couldn't normalize properly
        if '_' not in pair:
            self.logger.warning(f"Could not normalize currency pair: {currency_pair}, using as-is")
        
        return pair
    
    def analyze(self, currency_pairs=None):
        """
        Analyze market sentiment for the specified currency pairs
        
        Args:
            currency_pairs (list, optional): List of currency pairs to analyze.
                If None, generates sentiment for all currency combinations.
        
        Returns:
            dict: Sentiment analysis results by currency pair
        """
        self.logger.info("Running sentiment analysis")
        
        # If no currency pairs specified, generate all combinations
        if currency_pairs is None:
            currencies = list(self.currency_keywords.keys())
            currency_pairs = []
            for i, base in enumerate(currencies):
                for quote in currencies[i+1:]:
                    currency_pairs.append(f"{base}/{quote}")
                    currency_pairs.append(f"{quote}/{base}")
        
        results = {}
        
        # Get sentiment data from each source
        for source in self.data_sources:
            if source == 'news':
                news_sentiment = self._analyze_news_sentiment()
                
                # Process for each currency pair
                for pair in currency_pairs:
                    base, quote = pair.split('/')
                    pair_sentiment = self._calculate_pair_sentiment(news_sentiment, base, quote)
                    
                    if pair not in results:
                        results[pair] = {}
                    
                    results[pair]['news'] = pair_sentiment
            
            elif source == 'twitter':
                twitter_sentiment = self._analyze_twitter_sentiment()
                
                # Process for each currency pair
                for pair in currency_pairs:
                    base, quote = pair.split('/')
                    pair_sentiment = self._calculate_pair_sentiment(twitter_sentiment, base, quote)
                    
                    if pair not in results:
                        results[pair] = {}
                    
                    results[pair]['twitter'] = pair_sentiment
            
            elif source == 'reddit':
                reddit_sentiment = self._analyze_reddit_sentiment()
                
                # Process for each currency pair
                for pair in currency_pairs:
                    base, quote = pair.split('/')
                    pair_sentiment = self._calculate_pair_sentiment(reddit_sentiment, base, quote)
                    
                    if pair not in results:
                        results[pair] = {}
                    
                    results[pair]['reddit'] = pair_sentiment
        
        # Calculate combined sentiment scores for each pair
        for pair in results:
            combined_score = 0
            source_count = 0
            
            for source, sentiment in results[pair].items():
                combined_score += sentiment['score']
                source_count += 1
            
            average_score = combined_score / source_count if source_count > 0 else 0
            
            # Generate trading signal based on sentiment score
            if average_score > 0.2:
                signal = 'buy'
                strength = 'high' if average_score > 0.5 else 'medium'
            elif average_score < -0.2:
                signal = 'sell'
                strength = 'high' if average_score < -0.5 else 'medium'
            else:
                signal = 'neutral'
                strength = 'low'
            
            results[pair]['combined'] = {
                'score': average_score,
                'signal': signal,
                'strength': strength,
                'sources': list(results[pair].keys())
            }
        
        return results
    
    def _analyze_news_sentiment(self):
        """
        Analyze sentiment from news sources
        
        Returns:
            dict: Sentiment scores by currency
        """
        self.logger.info("Analyzing news sentiment")
        
        cache_file = os.path.join(self.cache_dir, 'news_sentiment.json')
        
        # Check if we have cached data that's less than 12 hours old
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=12):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # In a real implementation, this would fetch news articles from APIs
        # For demonstration, we'll generate sample data
        
        # Generate sample news sentiment for each currency
        sentiment = {}
        
        # Seed with a timestamp to keep values consistent
        timestamp = int(datetime.now().timestamp())
        np.random.seed(timestamp)
        
        for currency in self.currency_keywords:
            # Generate a random sentiment score between -1.0 and 1.0
            # Use a normal distribution centered around 0 with std=0.3
            score = np.clip(np.random.normal(0, 0.3), -1.0, 1.0)
            
            # Generate some sample headlines
            headlines = self._generate_sample_headlines(currency, 5, sentiment_bias=score)
            
            sentiment[currency] = {
                'score': score,
                'headlines': headlines,
                'article_count': len(headlines),
                'timestamp': datetime.now().isoformat()
            }
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(sentiment, f)
        
        return sentiment
    
    def _analyze_twitter_sentiment(self):
        """
        Analyze sentiment from Twitter (X)
        
        Returns:
            dict: Sentiment scores by currency
        """
        self.logger.info("Analyzing Twitter sentiment")
        
        cache_file = os.path.join(self.cache_dir, 'twitter_sentiment.json')
        
        # Check if we have cached data that's less than 6 hours old
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=6):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # In a real implementation, this would fetch tweets from the Twitter API
        # For demonstration, we'll generate sample data
        
        # Generate sample Twitter sentiment for each currency
        sentiment = {}
        
        # Seed with a timestamp to keep values consistent but different from news
        timestamp = int(datetime.now().timestamp()) + 1000
        np.random.seed(timestamp)
        
        for currency in self.currency_keywords:
            # Generate a random sentiment score between -1.0 and 1.0
            # Twitter tends to be more volatile, so use a wider std=0.4
            score = np.clip(np.random.normal(0, 0.4), -1.0, 1.0)
            
            # Generate some sample tweets
            tweets = self._generate_sample_tweets(currency, 8, sentiment_bias=score)
            
            # Calculate tweet volume (made up metric for demonstration)
            volume = int(np.random.normal(1000, 300))
            
            sentiment[currency] = {
                'score': score,
                'tweets': tweets,
                'tweet_count': len(tweets),
                'volume': volume,
                'timestamp': datetime.now().isoformat()
            }
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(sentiment, f)
        
        return sentiment
    
    def _analyze_reddit_sentiment(self):
        """
        Analyze sentiment from Reddit
        
        Returns:
            dict: Sentiment scores by currency
        """
        self.logger.info("Analyzing Reddit sentiment")
        
        cache_file = os.path.join(self.cache_dir, 'reddit_sentiment.json')
        
        # Check if we have cached data that's less than 8 hours old
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < timedelta(hours=8):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # In a real implementation, this would fetch posts from the Reddit API
        # For demonstration, we'll generate sample data
        
        # Generate sample Reddit sentiment for each currency
        sentiment = {}
        
        # Seed with a timestamp to keep values consistent but different from others
        timestamp = int(datetime.now().timestamp()) + 2000
        np.random.seed(timestamp)
        
        for currency in self.currency_keywords:
            # Generate a random sentiment score between -1.0 and 1.0
            # Reddit tends to have more extreme views, so use std=0.5
            score = np.clip(np.random.normal(0, 0.5), -1.0, 1.0)
            
            # Generate some sample posts
            posts = self._generate_sample_reddit_posts(currency, 6, sentiment_bias=score)
            
            # Calculate engagement metrics (made up for demonstration)
            upvotes = int(np.random.normal(500, 200))
            comments = int(upvotes * np.random.uniform(0.2, 0.5))
            
            sentiment[currency] = {
                'score': score,
                'posts': posts,
                'post_count': len(posts),
                'upvotes': upvotes,
                'comments': comments,
                'timestamp': datetime.now().isoformat()
            }
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(sentiment, f)
        
        return sentiment
    
    def _calculate_pair_sentiment(self, sentiment_data, base_currency, quote_currency):
        """
        Calculate sentiment for a currency pair based on individual currency sentiment
        
        Args:
            sentiment_data (dict): Sentiment data by currency
            base_currency (str): Base currency code
            quote_currency (str): Quote currency code
        
        Returns:
            dict: Sentiment analysis for the currency pair
        """
        # Extract sentiment scores for base and quote currencies
        base_score = sentiment_data.get(base_currency, {}).get('score', 0)
        quote_score = sentiment_data.get(quote_currency, {}).get('score', 0)
        
        # Calculate the pair sentiment (positive means base is stronger than quote)
        pair_score = base_score - quote_score
        
        # Determine signal based on the score
        if pair_score > 0.2:
            signal = 'buy'
            strength = 'high' if pair_score > 0.5 else 'medium'
        elif pair_score < -0.2:
            signal = 'sell'
            strength = 'high' if pair_score < -0.5 else 'medium'
        else:
            signal = 'neutral'
            strength = 'low'
        
        # Create summary
        if 'headlines' in sentiment_data.get(base_currency, {}):
            # This is news data
            base_mentions = sentiment_data[base_currency]['article_count'] if base_currency in sentiment_data else 0
            quote_mentions = sentiment_data[quote_currency]['article_count'] if quote_currency in sentiment_data else 0
            data_type = 'news'
        elif 'tweets' in sentiment_data.get(base_currency, {}):
            # This is Twitter data
            base_mentions = sentiment_data[base_currency]['tweet_count'] if base_currency in sentiment_data else 0
            quote_mentions = sentiment_data[quote_currency]['tweet_count'] if quote_currency in sentiment_data else 0
            data_type = 'tweets'
        elif 'posts' in sentiment_data.get(base_currency, {}):
            # This is Reddit data
            base_mentions = sentiment_data[base_currency]['post_count'] if base_currency in sentiment_data else 0
            quote_mentions = sentiment_data[quote_currency]['post_count'] if quote_currency in sentiment_data else 0
            data_type = 'posts'
        else:
            base_mentions = 0
            quote_mentions = 0
            data_type = 'data'
        
        # Format the results
        return {
            'score': pair_score,
            'base_score': base_score,
            'quote_score': quote_score,
            'base_mentions': base_mentions,
            'quote_mentions': quote_mentions,
            'signal': signal,
            'strength': strength,
            'data_type': data_type
        }
    
    def _generate_sample_headlines(self, currency, count=5, sentiment_bias=0):
        """
        Generate sample news headlines for a currency
        
        Args:
            currency (str): Currency code
            count (int): Number of headlines to generate
            sentiment_bias (float): Sentiment bias (-1.0 to 1.0)
        
        Returns:
            list: List of sample headlines
        """
        # Define templates for positive, neutral, and negative headlines
        positive_templates = [
            "{currency} strengthens as economic data exceeds expectations",
            "{currency} rallies amid positive economic outlook",
            "Strong GDP figures boost {currency} against major peers",
            "{country} central bank comments drive {currency} higher",
            "Investors bullish on {currency} following {country} policy announcement",
            "{currency} gains as inflation concerns ease in {country}",
            "Trade surplus widens, {currency} rises against basket of currencies"
        ]
        
        neutral_templates = [
            "{currency} steady ahead of {country} economic data release",
            "{currency} trades in narrow range as markets await central bank decision",
            "{country} economic outlook remains stable, {currency} flat",
            "Traders cautious on {currency} ahead of key economic figures",
            "{currency} shows little movement as {country} reports mixed economic data",
            "Market participants neutral on {currency} direction"
        ]
        
        negative_templates = [
            "{currency} weakens after disappointing economic data",
            "Downbeat economic forecast weighs on {currency}",
            "{currency} slides as {country} inflation rises above expectations",
            "Political uncertainty in {country} pressures {currency} lower",
            "{country} central bank dovish tone sends {currency} tumbling",
            "Trade deficit concerns impact {currency} sentiment",
            "{currency} falls as risk aversion drives investors elsewhere"
        ]
        
        # Get country name
        country_mapping = {
            'USD': 'US',
            'EUR': 'Eurozone',
            'GBP': 'UK',
            'JPY': 'Japan',
            'AUD': 'Australia',
            'CAD': 'Canada',
            'CHF': 'Switzerland',
            'NZD': 'New Zealand'
        }
        country = country_mapping.get(currency, currency)
        
        # Select templates based on sentiment bias
        if sentiment_bias > 0.3:
            # More positive headlines
            pos_count = int(count * 0.7)
            neu_count = count - pos_count
            neg_count = 0
        elif sentiment_bias < -0.3:
            # More negative headlines
            neg_count = int(count * 0.7)
            neu_count = count - neg_count
            pos_count = 0
        else:
            # Balanced headlines with slight bias
            pos_count = int(count * (0.4 + sentiment_bias * 0.3))
            neg_count = int(count * (0.4 - sentiment_bias * 0.3))
            neu_count = count - pos_count - neg_count
        
        headlines = []
        
        # Generate positive headlines
        for i in range(pos_count):
            template = np.random.choice(positive_templates)
            headline = template.format(currency=currency, country=country)
            headlines.append(headline)
        
        # Generate neutral headlines
        for i in range(neu_count):
            template = np.random.choice(neutral_templates)
            headline = template.format(currency=currency, country=country)
            headlines.append(headline)
        
        # Generate negative headlines
        for i in range(neg_count):
            template = np.random.choice(negative_templates)
            headline = template.format(currency=currency, country=country)
            headlines.append(headline)
        
        # Shuffle the headlines
        np.random.shuffle(headlines)
        
        return headlines
    
    def _generate_sample_tweets(self, currency, count=8, sentiment_bias=0):
        """
        Generate sample tweets for a currency
        
        Args:
            currency (str): Currency code
            count (int): Number of tweets to generate
            sentiment_bias (float): Sentiment bias (-1.0 to 1.0)
        
        Returns:
            list: List of sample tweets
        """
        # Define templates for positive, neutral, and negative tweets
        positive_templates = [
            "Just going long on {currency}, the trend looks extremely bullish! #forex #trading",
            "{currency} looking strong after the recent economic data. Time to buy? 📈",
            "Bullish on {currency} for the next few weeks. Technical indicators all pointing up! 🚀",
            "{currency} is my top forex pick this week. Strong fundamentals and technicals align.",
            "Central bank policy is super bullish for {currency}. Loading up my position! 💰",
            "Finally {currency} breaking out of consolidation to the upside as expected! #forextrading"
        ]
        
        neutral_templates = [
            "{currency} in a tight range today. Waiting for a breakout either way before trading.",
            "Anyone have thoughts on {currency} direction? Charts look confusing to me right now. #forex",
            "Keeping my powder dry on {currency} trades until we get more clarity from the central bank.",
            "{currency} has been choppy lately, staying on the sidelines for now. #forextrading",
            "Neither bullish nor bearish on {currency} at these levels. Need more conviction."
        ]
        
        negative_templates = [
            "Shorting {currency} here seems like easy money. Clear downtrend forming! #forex",
            "{currency} looks super weak after that economic report. Bearish for now 📉",
            "Just closed my long {currency} position for a loss. This doesn't look good...",
            "The {currency} rally was clearly a bull trap. Now the real move down begins! 🐻",
            "Central bank policy is going to crush {currency}. Time to short! #forextrading",
            "Technical breakdown on {currency} charts. This could drop much further! 📉"
        ]
        
        # Select templates based on sentiment bias
        if sentiment_bias > 0.3:
            # More positive tweets
            pos_count = int(count * 0.7)
            neu_count = int(count * 0.2)
            neg_count = count - pos_count - neu_count
        elif sentiment_bias < -0.3:
            # More negative tweets
            neg_count = int(count * 0.7)
            neu_count = int(count * 0.2)
            pos_count = count - neg_count - neu_count
        else:
            # Balanced tweets with slight bias
            pos_count = int(count * (0.4 + sentiment_bias * 0.3))
            neg_count = int(count * (0.4 - sentiment_bias * 0.3))
            neu_count = count - pos_count - neg_count
        
        tweets = []
        
        # Generate positive tweets
        for i in range(pos_count):
            template = np.random.choice(positive_templates)
            tweet = template.format(currency=currency)
            tweets.append(tweet)
        
        # Generate neutral tweets
        for i in range(neu_count):
            template = np.random.choice(neutral_templates)
            tweet = template.format(currency=currency)
            tweets.append(tweet)
        
        # Generate negative tweets
        for i in range(neg_count):
            template = np.random.choice(negative_templates)
            tweet = template.format(currency=currency)
            tweets.append(tweet)
        
        # Shuffle the tweets
        np.random.shuffle(tweets)
        
        return tweets
    
    def _generate_sample_reddit_posts(self, currency, count=6, sentiment_bias=0):
        """
        Generate sample Reddit posts for a currency
        
        Args:
            currency (str): Currency code
            count (int): Number of posts to generate
            sentiment_bias (float): Sentiment bias (-1.0 to 1.0)
        
        Returns:
            list: List of sample Reddit posts
        """
        # Define templates for positive, neutral, and negative posts
        positive_templates = [
            "[ANALYSIS] Why {currency} is poised for a major rally in the coming weeks",
            "Technical Analysis: {currency} forming a perfect bullish pattern, here's my trade plan",
            "Fundamentals for {currency} are the strongest they've been in years - Discussion",
            "Loaded up on {currency} today, here's why I'm extremely bullish going forward",
            "{currency} bullish thesis - Multiple factors aligning for a strong move up"
        ]
        
        neutral_templates = [
            "What's everyone's take on {currency} right now? Can't decide if I should enter or wait",
            "{currency} Analysis Request - Conflicting signals on different timeframes",
            "Thoughts on {currency} after yesterday's price action? Seems range-bound",
            "Historical patterns of {currency} in this economic environment - Data Analysis",
            "Comparing {currency} movement to previous cycles - What can we learn?"
        ]
        
        negative_templates = [
            "[ANALYSIS] Why I'm shorting {currency} and expect a significant drop",
            "Technical breakdown: {currency} showing classic bear signals across multiple timeframes",
            "The fundamental case against {currency} - Economic headwinds ahead",
            "Closed all my {currency} longs and reversed to short - Here's my reasoning",
            "{currency} bearish outlook - Multiple red flags appearing in recent data"
        ]
        
        # Select templates based on sentiment bias
        if sentiment_bias > 0.3:
            # More positive posts
            pos_count = int(count * 0.7)
            neu_count = int(count * 0.2)
            neg_count = count - pos_count - neu_count
        elif sentiment_bias < -0.3:
            # More negative posts
            neg_count = int(count * 0.7)
            neu_count = int(count * 0.2)
            pos_count = count - neg_count - neu_count
        else:
            # Balanced posts with slight bias
            pos_count = int(count * (0.4 + sentiment_bias * 0.3))
            neg_count = int(count * (0.4 - sentiment_bias * 0.3))
            neu_count = count - pos_count - neg_count
        
        posts = []
        
        # Generate positive posts
        for i in range(pos_count):
            template = np.random.choice(positive_templates)
            post = {
                "title": template.format(currency=currency),
                "subreddit": "r/Forex",
                "upvotes": int(np.random.normal(50, 30)),
                "comments": int(np.random.normal(15, 10)),
                "sentiment": "positive"
            }
            posts.append(post)
        
        # Generate neutral posts
        for i in range(neu_count):
            template = np.random.choice(neutral_templates)
            post = {
                "title": template.format(currency=currency),
                "subreddit": "r/Forex",
                "upvotes": int(np.random.normal(20, 15)),
                "comments": int(np.random.normal(10, 8)),
                "sentiment": "neutral"
            }
            posts.append(post)
        
        # Generate negative posts
        for i in range(neg_count):
            template = np.random.choice(negative_templates)
            post = {
                "title": template.format(currency=currency),
                "subreddit": "r/Forex",
                "upvotes": int(np.random.normal(40, 25)),
                "comments": int(np.random.normal(12, 9)),
                "sentiment": "negative"
            }
            posts.append(post)
        
        # Shuffle the posts
        np.random.shuffle(posts)
        
        return posts

    # === Sentiment Data Fetching Methods ===
    
    def get_news_sentiment(self, currency_pair: str, days_back: int = 7) -> pd.DataFrame:
        """
        Analyze sentiment from news articles for a currency pair.
        
        Args:
            currency_pair: Currency pair to analyze
            days_back: Number of days to look back for news
            
        Returns:
            pd.DataFrame: News articles with sentiment analysis
        """
        self.log_action("get_news_sentiment", f"Fetching news sentiment for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        base_currency, quote_currency = pair.split('_')
        
        # Check cache
        cache_key = f"{pair}_{days_back}d"
        cached_data = self._get_from_cache(cache_key, 'news_sentiment')
        if cached_data is not None:
            self.logger.info(f"Using cached news sentiment data for {pair}")
            return cached_data
        
        # Use the default news provider
        provider = self.default_providers.get('news', 'news_api')
        
        # Fetch news data based on provider
        news_data = None
        if provider == 'news_api':
            news_data = self._fetch_news_api(pair, days_back)
        elif provider == 'investing_com':
            news_data = self._fetch_investing_com_news(pair, days_back)
        elif provider == 'mock':
            news_data = self._generate_mock_news(pair, days_back)
        else:
            self.logger.warning(f"Unknown news provider: {provider}, using mock data")
            news_data = self._generate_mock_news(pair, days_back)
        
        # Analyze sentiment for each news article
        if news_data is not None and not news_data.empty:
            # Add sentiment analysis
            news_data['sentiment'] = news_data['title'].apply(lambda x: self.analyze_text_sentiment(x))
            news_data['sentiment_score'] = news_data['sentiment'].apply(lambda x: x.get('score', 0))
            news_data['sentiment_label'] = news_data['sentiment'].apply(lambda x: x.get('label', 'neutral'))
            
            # Save to cache
            self._save_to_cache(news_data, cache_key, 'news_sentiment')
            
            return news_data
        else:
            self.logger.warning(f"No news data found for {pair}")
            return pd.DataFrame()
    
    def get_social_media_sentiment(self, currency_pair: str, days_back: int = 7) -> pd.DataFrame:
        """
        Analyze sentiment from social media for a currency pair.
        
        Args:
            currency_pair: Currency pair to analyze
            days_back: Number of days to look back for social media posts
            
        Returns:
            pd.DataFrame: Social media posts with sentiment analysis
        """
        self.log_action("get_social_media_sentiment", f"Fetching social media sentiment for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        base_currency, quote_currency = pair.split('_')
        
        # Check cache
        cache_key = f"{pair}_{days_back}d"
        cached_data = self._get_from_cache(cache_key, 'social_media_sentiment')
        if cached_data is not None:
            self.logger.info(f"Using cached social media sentiment data for {pair}")
            return cached_data
        
        # Use the default social media provider
        provider = self.default_providers.get('social_media', 'twitter_api')
        
        # Fetch social media data based on provider
        social_data = None
        if provider == 'twitter_api':
            social_data = self._fetch_twitter_data(pair, days_back)
        elif provider == 'reddit_api':
            social_data = self._fetch_reddit_data(pair, days_back)
        elif provider == 'stocktwits_api':
            social_data = self._fetch_stocktwits_data(pair, days_back)
        elif provider == 'mock':
            social_data = self._generate_mock_social_media(pair, days_back)
        else:
            self.logger.warning(f"Unknown social media provider: {provider}, using mock data")
            social_data = self._generate_mock_social_media(pair, days_back)
        
        # Analyze sentiment for each social media post
        if social_data is not None and not social_data.empty:
            # Add sentiment analysis
            social_data['sentiment'] = social_data['content'].apply(lambda x: self.analyze_text_sentiment(x))
            social_data['sentiment_score'] = social_data['sentiment'].apply(lambda x: x.get('score', 0))
            social_data['sentiment_label'] = social_data['sentiment'].apply(lambda x: x.get('label', 'neutral'))
            
            # Save to cache
            self._save_to_cache(social_data, cache_key, 'social_media_sentiment')
            
            return social_data
        else:
            self.logger.warning(f"No social media data found for {pair}")
            return pd.DataFrame()
    
    def get_analyst_ratings(self, currency_pair: str) -> pd.DataFrame:
        """
        Get professional analyst opinions for a currency pair.
        
        Args:
            currency_pair: Currency pair to analyze
            
        Returns:
            pd.DataFrame: Analyst ratings with sentiment metrics
        """
        self.log_action("get_analyst_ratings", f"Fetching analyst ratings for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        base_currency, quote_currency = pair.split('_')
        
        # Check cache
        cache_key = f"{pair}"
        cached_data = self._get_from_cache(cache_key, 'analyst_ratings')
        if cached_data is not None:
            self.logger.info(f"Using cached analyst ratings data for {pair}")
            return cached_data
        
        # Use the default analyst ratings provider
        provider = self.default_providers.get('analyst_ratings', 'investing_com')
        
        # Fetch analyst ratings based on provider
        ratings_data = None
        if provider == 'investing_com':
            ratings_data = self._fetch_investing_com_ratings(pair)
        elif provider == 'forexfactory_api':
            ratings_data = self._fetch_forexfactory_ratings(pair)
        elif provider == 'tradingview_api':
            ratings_data = self._fetch_tradingview_ratings(pair)
        elif provider == 'mock':
            ratings_data = self._generate_mock_analyst_ratings(pair)
        else:
            self.logger.warning(f"Unknown analyst ratings provider: {provider}, using mock data")
            ratings_data = self._generate_mock_analyst_ratings(pair)
        
        # Process and standardize the ratings data
        if ratings_data is not None and not ratings_data.empty:
            # Save to cache
            self._save_to_cache(ratings_data, cache_key, 'analyst_ratings')
            
            return ratings_data
        else:
            self.logger.warning(f"No analyst ratings found for {pair}")
            return pd.DataFrame()
    
    def get_market_positioning(self, currency_pair: str) -> pd.DataFrame:
        """
        Get data on market positioning for a currency pair.
        
        Args:
            currency_pair: Currency pair to analyze
            
        Returns:
            pd.DataFrame: Market positioning data with sentiment metrics
        """
        self.log_action("get_market_positioning", f"Fetching market positioning for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        base_currency, quote_currency = pair.split('_')
        
        # Check cache
        cache_key = f"{pair}"
        cached_data = self._get_from_cache(cache_key, 'market_positioning')
        if cached_data is not None:
            self.logger.info(f"Using cached market positioning data for {pair}")
            return cached_data
        
        # Use the default market positioning provider
        provider = self.default_providers.get('market_positioning', 'cot_reports')
        
        # Fetch market positioning based on provider
        positioning_data = None
        if provider == 'cot_reports':
            positioning_data = self._fetch_cot_reports(pair)
        elif provider == 'oanda_positioning':
            positioning_data = self._fetch_oanda_positioning(pair)
        elif provider == 'fxcm_ssi':
            positioning_data = self._fetch_fxcm_ssi(pair)
        elif provider == 'mock':
            positioning_data = self._generate_mock_market_positioning(pair)
        else:
            self.logger.warning(f"Unknown market positioning provider: {provider}, using mock data")
            positioning_data = self._generate_mock_market_positioning(pair)
        
        # Process and standardize the positioning data
        if positioning_data is not None and not positioning_data.empty:
            # Save to cache
            self._save_to_cache(positioning_data, cache_key, 'market_positioning')
            
            return positioning_data
        else:
            self.logger.warning(f"No market positioning data found for {pair}")
            return pd.DataFrame()
            
    # Fetch methods are placeholders for actual API implementations
    def _fetch_news_api(self, currency_pair: str, days_back: int = 7) -> list:
        """
        Fetch news about a currency pair from NewsAPI.org.
        
        Args:
            currency_pair: Currency pair in format like EUR_USD
            days_back: Number of days to look back for news
            
        Returns:
            list: List of news articles from NewsAPI
        """
        try:
            # Check if NewsAPI key is configured
            if not self.config.get('news_api_key'):
                self.logger.warning("NewsAPI key not configured, using mock data")
                return self._generate_mock_news(currency_pair, 'news_api', days_back)
                
            # Import NewsAPI library
            try:
                from newsapi import NewsApiClient
            except ImportError:
                self.logger.error("NewsAPI client not installed. Install with: pip install newsapi-python")
                return self._generate_mock_news(currency_pair, 'news_api', days_back)
                
            # Normalize the currency pair
            pair = self._normalize_currency_pair(currency_pair)
            base, quote = pair.split('_')
            
            # Create search queries
            queries = [
                f"{base}/{quote}",          # EUR/USD
                f"{base} {quote}",          # EUR USD
                f"{base}-{quote}",          # EUR-USD
                f"{base}{quote}",           # EURUSD
                f"{base} {quote} forex",    # EUR USD forex
                f"{base} {quote} currency", # EUR USD currency
            ]
            
            # Add currency names if common ones
            currency_names = {
                'USD': 'dollar',
                'EUR': 'euro',
                'GBP': 'pound sterling',
                'JPY': 'yen',
                'AUD': 'australian dollar',
                'CAD': 'canadian dollar',
                'CHF': 'swiss franc',
                'NZD': 'new zealand dollar'
            }
            
            if base in currency_names:
                queries.append(currency_names[base])
            if quote in currency_names:
                queries.append(currency_names[quote])
                
            self.logger.info(f"Querying NewsAPI for {pair} with {len(queries)} queries")
            
            # Initialize NewsAPI client
            newsapi = NewsApiClient(api_key=self.config['news_api_key'])
            
            # Calculate date range
            from_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            all_articles = []
            
            # Make queries to NewsAPI
            for query in queries:
                try:
                    self.logger.debug(f"Querying NewsAPI for: {query}")
                    
                    response = newsapi.get_everything(
                        q=query,
                        from_param=from_date,
                        to=to_date,
                        language='en',
                        sort_by='relevancy',
                        page_size=100
                    )
                    
                    if response and 'articles' in response:
                        # Add currency pair to each article for reference
                        for article in response['articles']:
                            article['currency_pair'] = pair
                            
                        all_articles.extend(response['articles'])
                        self.logger.debug(f"Found {len(response['articles'])} articles for query '{query}'")
                    
                except Exception as e:
                    self.logger.error(f"Error querying NewsAPI for '{query}': {str(e)}")
                
            # Remove duplicates based on URL
            unique_articles = []
            seen_urls = set()
            
            for article in all_articles:
                if article['url'] not in seen_urls:
                    seen_urls.add(article['url'])
                    unique_articles.append(article)
            
            self.logger.info(f"Found {len(unique_articles)} unique news articles for {pair} from NewsAPI")
            
            if not unique_articles:
                self.logger.warning(f"No news found for {pair} on NewsAPI, falling back to mock data")
                return self._generate_mock_news(currency_pair, 'news_api', days_back)
                
            return unique_articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            self.logger.info("Falling back to mock data")
            return self._generate_mock_news(currency_pair, 'news_api', days_back)
    
    def _fetch_investing_com_news(self, currency_pair: str, days_back: int = 7) -> list:
        """
        Fetch news about a currency pair from Investing.com via web scraping.
        
        Args:
            currency_pair: Currency pair in format like EUR_USD
            days_back: Number of days to look back for news
            
        Returns:
            list: List of news articles from Investing.com
        """
        try:
            # Check if web scraping dependencies are available
            try:
                import requests
                from bs4 import BeautifulSoup
                import datetime
                import time
                import random
            except ImportError as e:
                self.logger.error(f"Required libraries not installed: {str(e)}. Install with: pip install requests beautifulsoup4")
                return self._generate_mock_news(currency_pair, 'investing', days_back)
            
            # Normalize the currency pair
            pair = self._normalize_currency_pair(currency_pair)
            base, quote = pair.split('_')
            
            # Create search mappings - these are the URL slugs used by Investing.com
            pair_to_slug = {
                'EUR_USD': 'eur-usd',
                'GBP_USD': 'gbp-usd',
                'USD_JPY': 'usd-jpy',
                'AUD_USD': 'aud-usd',
                'USD_CAD': 'usd-cad',
                'USD_CHF': 'usd-chf',
                'NZD_USD': 'nzd-usd',
                'EUR_GBP': 'eur-gbp',
                'EUR_JPY': 'eur-jpy',
                'GBP_JPY': 'gbp-jpy',
            }
            
            # If we don't have a direct slug mapping, use forex news page
            slug = pair_to_slug.get(pair, 'forex')
            
            # Calculate date ranges
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days_back)
            
            # Define headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            }
            
            articles = []
            
            # URLs to fetch
            urls = []
            
            # Add currency-specific URL if we have a mapping
            if slug in pair_to_slug.values():
                urls.append(f"https://www.investing.com/currencies/{slug}-news")
            
            # Add general forex news URL
            urls.append("https://www.investing.com/news/forex-news")
            
            # If USD is involved, add economic news
            if 'USD' in pair:
                urls.append("https://www.investing.com/news/economic-indicators")
            
            self.logger.info(f"Fetching Investing.com news for {pair} from {len(urls)} URLs")
            
            for url in urls:
                try:
                    self.logger.info(f"Fetching news from: {url}")
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find news articles - structure depends on the page
                    article_containers = soup.select('div.largeTitle article') or soup.select('article.js-article-item')
                    
                    self.logger.info(f"Found {len(article_containers)} article containers on {url}")
                    
                    for article in article_containers:
                        try:
                            # Extract title
                            title_element = article.select_one('a.title') or article.select_one('a.articleTitle')
                            if not title_element:
                                continue
                                
                            title = title_element.text.strip()
                            link = title_element.get('href')
                            
                            # Make relative URLs absolute
                            if link and link.startswith('/'):
                                link = f"https://www.investing.com{link}"
                            
                            # Extract date
                            date_element = article.select_one('span.articleDetails span') or article.select_one('span.date')
                            if date_element:
                                date_text = date_element.text.strip()
                                # Parse various date formats like "2 hours ago", "Jul 21, 2023"
                                try:
                                    if 'ago' in date_text.lower():
                                        # Handle relative dates
                                        if 'hour' in date_text.lower():
                                            hours = int(date_text.split()[0])
                                            publish_date = (datetime.datetime.now() - datetime.timedelta(hours=hours)).isoformat()
                                        elif 'day' in date_text.lower():
                                            days = int(date_text.split()[0])
                                            publish_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
                                        elif 'min' in date_text.lower():
                                            minutes = int(date_text.split()[0])
                                            publish_date = (datetime.datetime.now() - datetime.timedelta(minutes=minutes)).isoformat()
                                        else:
                                            publish_date = datetime.datetime.now().isoformat()
                                    else:
                                        # Try to parse absolute date
                                        date_obj = datetime.datetime.strptime(date_text, "%b %d, %Y")
                                        publish_date = date_obj.isoformat()
                                except Exception as e:
                                    publish_date = datetime.datetime.now().isoformat()
                                    self.logger.debug(f"Could not parse date '{date_text}': {str(e)}")
                            else:
                                publish_date = datetime.datetime.now().isoformat()
                            
                            # Extract description if available
                            description_element = article.select_one('p.description')
                            description = description_element.text.strip() if description_element else ""
                            
                            # Create article object
                            article_obj = {
                                'title': title,
                                'description': description,
                                'url': link,
                                'publishedAt': publish_date,
                                'source': {
                                    'name': 'Investing.com',
                                    'url': 'https://www.investing.com'
                                },
                                'currency_pair': pair
                            }
                            
                            # Check if this is a duplicate
                            if not any(a.get('url') == article_obj['url'] for a in articles):
                                articles.append(article_obj)
                                
                        except Exception as e:
                            self.logger.debug(f"Error parsing article: {str(e)}")
                    
                    # Be nice to the website - add some delay between requests
                    time.sleep(random.uniform(1.0, 3.0))
                    
                except Exception as e:
                    self.logger.error(f"Error fetching news from {url}: {str(e)}")
            
            self.logger.info(f"Found {len(articles)} unique news articles from Investing.com for {pair}")
            
            if not articles:
                self.logger.warning(f"No news found on Investing.com for {pair}, falling back to mock data")
                return self._generate_mock_news(currency_pair, 'investing', days_back)
                
            return articles
            
        except Exception as e:
            self.logger.error(f"Error scraping Investing.com: {str(e)}")
            self.logger.info("Falling back to mock data")
            return self._generate_mock_news(currency_pair, 'investing', days_back)
    
    def _fetch_twitter_data(self, currency_pair: str, count: int = 100) -> list:
        """
        Fetch tweets about a currency pair from Twitter API.
        
        Args:
            currency_pair: Currency pair in format like EUR_USD
            count: Number of tweets to fetch
            
        Returns:
            list: List of tweets
        """
        try:
            # Check if Twitter API credentials are configured
            twitter_credentials = {
                'bearer_token': self.config.get('twitter_bearer_token'),
                'api_key': self.config.get('twitter_api_key'),
                'api_secret': self.config.get('twitter_api_secret'),
                'access_token': self.config.get('twitter_access_token'),
                'access_secret': self.config.get('twitter_access_secret')
            }
            
            # If no Twitter credentials are configured, fall back to mock data
            if not twitter_credentials['bearer_token'] and not twitter_credentials['api_key']:
                self.logger.warning("Twitter API credentials not configured, falling back to mock data")
                return self._generate_mock_tweets(currency_pair, count)
                
            # Normalize the currency pair
            pair = self._normalize_currency_pair(currency_pair)
            base, quote = pair.split('_')
            
            # Import Twitter API libraries - we use tweepy if available
            try:
                import tweepy
            except ImportError:
                self.logger.error("Tweepy library not installed. Install with: pip install tweepy")
                return self._generate_mock_tweets(currency_pair, count)
            
            # Set up Twitter client
            if twitter_credentials['bearer_token']:
                # Use v2 API with bearer token
                client = tweepy.Client(
                    bearer_token=twitter_credentials['bearer_token'],
                    consumer_key=twitter_credentials['api_key'],
                    consumer_secret=twitter_credentials['api_secret'],
                    access_token=twitter_credentials['access_token'],
                    access_token_secret=twitter_credentials['access_secret']
                )
                
                # Prepare search queries - we'll use multiple to increase coverage
                search_queries = [
                    f"#{base}{quote}",
                    f"#{base}/{quote}",
                    f"#{base}_{quote}",
                    f"forex {base} {quote}",
                    f"currency {base} {quote}",
                    f"trading {base} {quote}"
                ]
                
                all_tweets = []
                
                # Search for each query
                for query in search_queries:
                    self.logger.info(f"Searching Twitter for: {query}")
                    try:
                        # Twitter API v2 search
                        response = client.search_recent_tweets(
                            query=query,
                            max_results=min(100, count),  # API limits to 100 per request
                            tweet_fields=['created_at', 'text', 'author_id', 'public_metrics']
                        )
                        
                        if not response or not response.data:
                            continue
                            
                        for tweet in response.data:
                            tweet_data = {
                                'id': tweet.id,
                                'text': tweet.text,
                                'created_at': tweet.created_at.isoformat() if hasattr(tweet, 'created_at') else None,
                                'author_id': tweet.author_id,
                                'like_count': tweet.public_metrics.get('like_count', 0) if hasattr(tweet, 'public_metrics') else 0,
                                'retweet_count': tweet.public_metrics.get('retweet_count', 0) if hasattr(tweet, 'public_metrics') else 0,
                                'query': query,
                                'currency_pair': pair
                            }
                            
                            # Avoid duplicates
                            if not any(t.get('id') == tweet_data['id'] for t in all_tweets):
                                all_tweets.append(tweet_data)
                                
                        self.logger.info(f"Found {len(response.data)} tweets for {query}")
                        
                    except Exception as e:
                        self.logger.error(f"Error searching Twitter for {query}: {str(e)}")
                
                if not all_tweets:
                    self.logger.warning(f"No tweets found for {pair}, falling back to mock data")
                    return self._generate_mock_tweets(currency_pair, count)
                
                self.logger.info(f"Total unique tweets found: {len(all_tweets)}")
                return all_tweets[:count]  # Limit to requested count
                
            else:
                self.logger.warning("Twitter bearer token not provided, falling back to mock data")
                return self._generate_mock_tweets(currency_pair, count)
                
        except Exception as e:
            self.logger.error(f"Error fetching data from Twitter API: {str(e)}")
            self.logger.info("Falling back to mock data")
            return self._generate_mock_tweets(currency_pair, count)
    
    def _fetch_reddit_data(self, currency_pair: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch data from Reddit API."""
        # This would be replaced with actual API call implementation
        self.logger.info(f"Fetching mock social media data for {currency_pair}")
        return self._generate_mock_social_media(currency_pair, days_back)
        
    def _fetch_stocktwits_data(self, currency_pair: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch data from StockTwits API."""
        # This would be replaced with actual API call implementation
        self.logger.info(f"Fetching mock social media data for {currency_pair}")
        return self._generate_mock_social_media(currency_pair, days_back)
        
    def _fetch_investing_com_ratings(self, currency_pair: str) -> pd.DataFrame:
        """Fetch analyst ratings from Investing.com."""
        # This would be replaced with actual web scraping implementation
        self.logger.info(f"Fetching mock analyst ratings for {currency_pair}")
        return self._generate_mock_analyst_ratings(currency_pair)
        
    def _fetch_forexfactory_ratings(self, currency_pair: str) -> pd.DataFrame:
        """Fetch analyst ratings from ForexFactory."""
        # This would be replaced with actual web scraping implementation
        self.logger.info(f"Fetching mock analyst ratings for {currency_pair}")
        return self._generate_mock_analyst_ratings(currency_pair)
        
    def _fetch_tradingview_ratings(self, currency_pair: str) -> pd.DataFrame:
        """Fetch analyst ratings from TradingView."""
        # This would be replaced with actual web scraping implementation
        self.logger.info(f"Fetching mock analyst ratings for {currency_pair}")
        return self._generate_mock_analyst_ratings(currency_pair)
        
    def _fetch_cot_reports(self, currency_pair: str, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch Commitment of Traders (COT) reports from CFTC.
        
        Args:
            currency_pair: Currency pair to analyze
            days_back: Number of days to look back
            
        Returns:
            pd.DataFrame: COT data with positions and date
        """
        self.log_action("_fetch_cot_reports", f"Fetching COT reports for {currency_pair}")
        
        try:
            import requests
            from io import StringIO
            
            # Map currency pair to COT codes
            # These are the codes used in the CFTC reports for major currencies
            currency_code_mapping = {
                'EUR_USD': '099741',  # Euro
                'GBP_USD': '096742',  # British Pound
                'USD_JPY': '097741',  # Japanese Yen
                'AUD_USD': '232741',  # Australian Dollar
                'USD_CAD': '090741',  # Canadian Dollar
                'NZD_USD': '112741',  # New Zealand Dollar
                'USD_CHF': '092741',  # Swiss Franc
                'USD_MXN': '095741',  # Mexican Peso
            }
            
            normalized_pair = self._normalize_currency_pair(currency_pair)
            
            # Check if we have a code for this currency pair
            if normalized_pair not in currency_code_mapping:
                self.logger.warning(f"No COT code available for {currency_pair}, using mock data")
                return self._generate_mock_cot_data(currency_pair, days_back)
            
            # Get the currency code
            code = currency_code_mapping[normalized_pair]
            
            # Calculate the date range for the COT reports (they're published every Friday)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for the API
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            # CFTC legacy reports URL
            url = f"https://www.cftc.gov/dea/newcot/deacot.txt"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the fixed-width file format
            content = StringIO(response.text)
            
            # Read the fixed-width file data
            # Example column structure for legacy COT reports (simplified)
            # This will need to be adjusted based on the actual format
            data = []
            
            for line in content:
                # Check if line contains our currency code
                if code in line:
                    parts = line.split()
                    if len(parts) >= 10:  # Ensure we have enough parts
                        try:
                            report_date_str = parts[0]
                            
                            # Convert the date format from YYMMDD to YYYY-MM-DD
                            report_date = datetime.strptime(report_date_str, '%y%m%d')
                            
                            # Skip if outside our date range
                            if report_date < start_date or report_date > end_date:
                                continue
                                
                            # Extract the positioning data
                            # Note: Column positions may vary, this is an example
                            non_commercial_long = int(parts[2])
                            non_commercial_short = int(parts[3])
                            commercial_long = int(parts[5])
                            commercial_short = int(parts[6])
                            
                            # Calculate net positions
                            non_commercial_net = non_commercial_long - non_commercial_short
                            commercial_net = commercial_long - commercial_short
                            
                            data.append({
                                'date': report_date.strftime('%Y-%m-%d'),
                                'currency_pair': currency_pair,
                                'non_commercial_long': non_commercial_long,
                                'non_commercial_short': non_commercial_short,
                                'non_commercial_net': non_commercial_net,
                                'commercial_long': commercial_long,
                                'commercial_short': commercial_short,
                                'commercial_net': commercial_net,
                                'timestamp': report_date
                            })
                        except (ValueError, IndexError) as e:
                            self.logger.warning(f"Error parsing COT line: {e}")
                            continue
            
            if data:
                return pd.DataFrame(data)
            else:
                self.logger.warning(f"No COT data found for {currency_pair}, using mock data")
                return self._generate_mock_cot_data(currency_pair, days_back)
                
        except Exception as e:
            self.logger.error(f"Error fetching COT reports: {e}")
            # Fall back to mock data on error
            return self._generate_mock_cot_data(currency_pair, days_back)
    
    def _fetch_oanda_positioning(self, currency_pair: str) -> pd.DataFrame:
        """Fetch positioning data from OANDA."""
        # This would be replaced with actual API call implementation
        self.logger.info(f"Fetching mock market positioning for {currency_pair}")
        return self._generate_mock_market_positioning(currency_pair)
        
    def _fetch_fxcm_ssi(self, currency_pair: str) -> pd.DataFrame:
        """Fetch SSI data from FXCM."""
        # This would be replaced with actual API call implementation
        self.logger.info(f"Fetching mock market positioning for {currency_pair}")
        return self._generate_mock_market_positioning(currency_pair)

    # === Sentiment Analysis and NLP Methods ===
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text using OpenAI or NLTK.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results with score and label
        """
        if not text:
            return {'score': 0, 'label': 'neutral'}
        
        try:
            # First try to use OpenAI if available
            if self.llm and hasattr(self, 'api_keys') and 'openai' in self.api_keys:
                return self._analyze_sentiment_with_openai(text)
            
            # Fall back to NLTK SentimentIntensityAnalyzer
            return self._analyze_sentiment_with_nltk(text)
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            # Return neutral as fallback
            return {'score': 0, 'label': 'neutral'}
    
    def _analyze_sentiment_with_openai(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using OpenAI API.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results with score and label
        """
        try:
            # Use the LLM to analyze sentiment if available
            if self.llm:
                prompt = f"""
                Analyze the sentiment of the following text related to forex/currency trading.
                Provide a score from -1.0 (extremely bearish) to 1.0 (extremely bullish),
                where 0.0 is neutral.
                
                Text: "{text}"
                
                Return only a JSON object with two keys:
                - score: the sentiment score as a float
                - label: one of "bullish", "bearish", or "neutral"
                """
                
                # Get response from LLM
                response = self.llm.invoke(prompt)
                
                # Extract JSON from response
                import json
                import re
                
                # Extract JSON content using regex
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # Parse the JSON
                    try:
                        sentiment_data = json.loads(json_str)
                        return {
                            'score': float(sentiment_data.get('score', 0)),
                            'label': sentiment_data.get('label', 'neutral')
                        }
                    except json.JSONDecodeError:
                        pass
                        
                # If we couldn't extract JSON, use a simpler approach
                if "bullish" in response.lower():
                    score = 0.5
                    label = "bullish"
                elif "bearish" in response.lower():
                    score = -0.5
                    label = "bearish"
                else:
                    score = 0
                    label = "neutral"
                
                return {'score': score, 'label': label}
                
        except Exception as e:
            self.logger.error(f"Error with OpenAI sentiment analysis: {e}")
            
        # Fall back to NLTK if OpenAI fails
        return self._analyze_sentiment_with_nltk(text)
    
    def _analyze_sentiment_with_nltk(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using NLTK's SentimentIntensityAnalyzer.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results with score and label
        """
        sentiment = self.sia.polarity_scores(text)
        
        # Extract compound score and determine label
        score = sentiment['compound']
        
        if score >= self.sentiment_thresholds['bullish']:
            label = 'bullish'
        elif score <= self.sentiment_thresholds['bearish']:
            label = 'bearish'
        else:
            label = 'neutral'
            
        return {'score': score, 'label': label}
    
    def classify_sentiment(self, text: str) -> str:
        """
        Classify text as bullish, bearish, or neutral.
        
        Args:
            text: Text to classify
            
        Returns:
            str: Sentiment classification label
        """
        sentiment = self.analyze_text_sentiment(text)
        return sentiment.get('label', 'neutral')
    
    def extract_sentiment_factors(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract factors influencing sentiment from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List[Dict[str, Any]]: List of factors with their sentiment impact
        """
        factors = []
        
        try:
            # Use the LLM to extract factors if available
            if self.llm:
                prompt = f"""
                Analyze the following text related to forex/currency trading and identify factors
                that influence market sentiment. For each factor, provide:
                1. The factor name or description
                2. Its sentiment impact (positive, negative, or neutral)
                3. The strength of impact (low, medium, high)
                
                Text: "{text}"
                
                Return a JSON array of objects with these keys:
                - factor: description of the factor
                - impact: "positive", "negative", or "neutral"
                - strength: "low", "medium", or "high"
                """
                
                # Get response from LLM
                response = self.llm.invoke(prompt)
                
                # Extract JSON from response
                import json
                import re
                
                # Extract JSON content using regex
                json_match = re.search(r'(\[.*\])', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    # Parse the JSON
                    try:
                        factors = json.loads(json_str)
                        return factors
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            self.logger.error(f"Error extracting sentiment factors: {e}")
        
        # If LLM extraction fails or isn't available, use a simple keyword approach
        keywords = {
            'positive': ['increase', 'rise', 'gain', 'bull', 'uptrend', 'growth', 'strong'],
            'negative': ['decrease', 'fall', 'drop', 'bear', 'downtrend', 'recession', 'weak']
        }
        
        for impact, words in keywords.items():
            for word in words:
                if word in text.lower():
                    factors.append({
                        'factor': f"Contains keyword '{word}'",
                        'impact': 'positive' if impact == 'positive' else 'negative',
                        'strength': 'medium'
                    })
                    
        return factors
    
    def calculate_sentiment_score(self, texts: List[str], weights: Optional[List[float]] = None) -> float:
        """
        Calculate an overall sentiment score for a list of texts.
        
        Args:
            texts: List of texts to analyze
            weights: Optional list of weights for each text (must be same length as texts)
            
        Returns:
            float: Overall sentiment score from -1.0 to 1.0
        """
        if not texts:
            return 0.0
            
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0] * len(texts)
        elif len(weights) != len(texts):
            self.logger.warning("Weights list length doesn't match texts length, using equal weights")
            weights = [1.0] * len(texts)
            
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
            
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted sentiment scores
        sentiments = [self.analyze_text_sentiment(text) for text in texts]
        scores = [s.get('score', 0) for s in sentiments]
        
        # Compute weighted average
        weighted_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        
        return weighted_score
    
    def get_combined_sentiment(self, currency_pair: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get combined sentiment from all available sources.
        
        Args:
            currency_pair: Currency pair to analyze
            days_back: Number of days to look back
            
        Returns:
            Dict[str, Any]: Combined sentiment analysis
        """
        self.log_action("get_combined_sentiment", f"Calculating combined sentiment for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        
        # Check cache
        cache_key = f"{pair}_{days_back}d"
        cached_data = self._get_from_cache(cache_key, 'combined_sentiment')
        if cached_data is not None:
            self.logger.info(f"Using cached combined sentiment data for {pair}")
            return cached_data
        
        # Get data from all sources
        news_sentiment = self.get_news_sentiment(pair, days_back)
        social_sentiment = self.get_social_media_sentiment(pair, days_back)
        analyst_ratings = self.get_analyst_ratings(pair)
        market_positioning = self.get_market_positioning(pair)
        
        # Calculate overall sentiment score
        sentiment_scores = []
        sources = []
        
        # News sentiment (weighted 0.3)
        if not news_sentiment.empty:
            avg_news_score = news_sentiment['sentiment_score'].mean()
            sentiment_scores.append(avg_news_score)
            sources.append({
                'source': 'news',
                'score': avg_news_score,
                'weight': 0.3,
                'count': len(news_sentiment)
            })
        
        # Social media sentiment (weighted 0.2)
        if not social_sentiment.empty:
            avg_social_score = social_sentiment['sentiment_score'].mean()
            sentiment_scores.append(avg_social_score)
            sources.append({
                'source': 'social_media',
                'score': avg_social_score,
                'weight': 0.2,
                'count': len(social_sentiment)
            })
        
        # Analyst ratings (weighted 0.3)
        if not analyst_ratings.empty:
            avg_analyst_score = analyst_ratings['score'].mean() if 'score' in analyst_ratings.columns else 0
            sentiment_scores.append(avg_analyst_score)
            sources.append({
                'source': 'analyst_ratings',
                'score': avg_analyst_score,
                'weight': 0.3,
                'count': len(analyst_ratings)
            })
        
        # Market positioning (weighted 0.2)
        if not market_positioning.empty:
            avg_positioning_score = market_positioning['score'].mean() if 'score' in market_positioning.columns else 0
            sentiment_scores.append(avg_positioning_score)
            sources.append({
                'source': 'market_positioning',
                'score': avg_positioning_score,
                'weight': 0.2,
                'count': len(market_positioning)
            })
        
        # Calculate weighted score
        overall_score = 0
        total_weight = 0
        
        for source in sources:
            overall_score += source['score'] * source['weight']
            total_weight += source['weight']
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Determine sentiment label
        if overall_score >= self.sentiment_thresholds['strongly_bullish']:
            label = 'strongly_bullish'
        elif overall_score >= self.sentiment_thresholds['bullish']:
            label = 'bullish'
        elif overall_score <= self.sentiment_thresholds['strongly_bearish']:
            label = 'strongly_bearish'
        elif overall_score <= self.sentiment_thresholds['bearish']:
            label = 'bearish'
        else:
            label = 'neutral'
        
        # Create result
        result = {
            'currency_pair': pair,
            'overall_score': overall_score,
            'sentiment_label': label,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to cache
        self._save_to_cache(pd.DataFrame([result]), cache_key, 'combined_sentiment')
        
        return result

    # === Sentiment Tracking Methods ===
    
    def track_sentiment_changes(self, currency_pair: str, timeframe: str = 'daily') -> pd.DataFrame:
        """
        Track changes in sentiment over time.
        
        Args:
            currency_pair: Currency pair to analyze
            timeframe: Time resolution ('hourly', 'daily', 'weekly')
            
        Returns:
            pd.DataFrame: Time series of sentiment data
        """
        self.log_action("track_sentiment_changes", f"Tracking sentiment changes for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        
        # Determine number of days to look back based on timeframe
        if timeframe == 'hourly':
            days_back = 7
            resample_freq = 'H'
        elif timeframe == 'daily':
            days_back = 30
            resample_freq = 'D'
        elif timeframe == 'weekly':
            days_back = 90
            resample_freq = 'W'
        else:
            self.logger.warning(f"Unknown timeframe: {timeframe}, defaulting to daily")
            days_back = 30
            resample_freq = 'D'
        
        # Get sentiment data
        news_sentiment = self.get_news_sentiment(pair, days_back)
        social_sentiment = self.get_social_media_sentiment(pair, days_back)
        
        # Combine data sources
        sentiment_data = []
        
        if not news_sentiment.empty:
            news_sentiment['source'] = 'news'
            news_sentiment['weight'] = 0.6
            sentiment_data.append(news_sentiment)
            
        if not social_sentiment.empty:
            social_sentiment['source'] = 'social_media'
            social_sentiment['weight'] = 0.4
            sentiment_data.append(social_sentiment)
            
        if not sentiment_data:
            self.logger.warning(f"No sentiment data available for {pair}")
            return pd.DataFrame()
            
        # Combine all data
        combined_data = pd.concat(sentiment_data, ignore_index=True)
        
        # Add timestamp column if not present
        if 'timestamp' not in combined_data.columns:
            if 'date' in combined_data.columns:
                combined_data['timestamp'] = pd.to_datetime(combined_data['date'])
            else:
                # Use current time as fallback
                combined_data['timestamp'] = datetime.now()
        
        # Convert timestamp to datetime if needed
        combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
        
        # Set timestamp as index
        combined_data.set_index('timestamp', inplace=True)
        
        # Resample by timeframe and calculate weighted average sentiment
        resampled_data = (
            combined_data
            .groupby(pd.Grouper(freq=resample_freq))
            .apply(lambda x: np.average(x['sentiment_score'], weights=x['weight']) if len(x) > 0 else np.nan)
        ).reset_index()
        
        resampled_data.columns = ['timestamp', 'sentiment_score']
        
        # Fill any missing values with forward fill, then backward fill
        resampled_data['sentiment_score'] = resampled_data['sentiment_score'].fillna(method='ffill').fillna(method='bfill')
        
        # Add sentiment category based on score
        def categorize_sentiment(score):
            if score >= self.sentiment_thresholds['strongly_bullish']:
                return 'strongly_bullish'
            elif score >= self.sentiment_thresholds['bullish']:
                return 'bullish'
            elif score <= self.sentiment_thresholds['strongly_bearish']:
                return 'strongly_bearish'
            elif score <= self.sentiment_thresholds['bearish']:
                return 'bearish'
            else:
                return 'neutral'
                
        resampled_data['sentiment_category'] = resampled_data['sentiment_score'].apply(categorize_sentiment)
        
        # Add change from previous period
        resampled_data['change'] = resampled_data['sentiment_score'].diff()
        
        # Add period-over-period change (7-period for weekly momentum)
        resampled_data['momentum'] = resampled_data['sentiment_score'].diff(7)
        
        return resampled_data
    
    def detect_sentiment_extremes(self, currency_pair: str) -> Dict[str, Any]:
        """
        Detect extremely positive or negative sentiment.
        
        Args:
            currency_pair: Currency pair to analyze
            
        Returns:
            Dict[str, Any]: Information about extreme sentiment if found
        """
        self.log_action("detect_sentiment_extremes", f"Detecting sentiment extremes for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        
        # Get sentiment changes data
        sentiment_changes = self.track_sentiment_changes(pair, 'daily')
        
        if sentiment_changes.empty:
            return {'status': 'no_data', 'message': f"No sentiment data available for {pair}"}
        
        # Get the most recent sentiment score
        latest_score = sentiment_changes['sentiment_score'].iloc[-1]
        latest_category = sentiment_changes['sentiment_category'].iloc[-1]
        
        # Check if sentiment is at an extreme
        is_extreme = False
        extreme_type = None
        
        if latest_score >= self.sentiment_thresholds['strongly_bullish']:
            is_extreme = True
            extreme_type = 'bullish'
        elif latest_score <= self.sentiment_thresholds['strongly_bearish']:
            is_extreme = True
            extreme_type = 'bearish'
            
        # Calculate z-score to measure how extreme the sentiment is compared to recent history
        mean_score = sentiment_changes['sentiment_score'].mean()
        std_score = sentiment_changes['sentiment_score'].std()
        
        # Avoid division by zero
        if std_score > 0:
            z_score = (latest_score - mean_score) / std_score
        else:
            z_score = 0
            
        # Check if z-score indicates extreme (>2 standard deviations)
        if abs(z_score) >= 2:
            is_extreme = True
            extreme_type = 'bullish' if z_score > 0 else 'bearish'
            
        # Get the trend (average of the last 3 days' changes)
        recent_changes = sentiment_changes['change'].tail(3).mean()
        trend = 'increasing' if recent_changes > 0 else 'decreasing' if recent_changes < 0 else 'stable'
        
        return {
            'currency_pair': pair,
            'latest_score': latest_score,
            'sentiment_category': latest_category,
            'is_extreme': is_extreme,
            'extreme_type': extreme_type if is_extreme else None,
            'z_score': z_score,
            'trend': trend,
            'timestamp': datetime.now().isoformat()
        }
    
    def correlate_sentiment_with_price(self, sentiment_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlation between sentiment and price.
        
        Args:
            sentiment_data: DataFrame with sentiment time series
            price_data: DataFrame with price time series
            
        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        self.log_action("correlate_sentiment_with_price", "Analyzing sentiment-price correlation")
        
        if sentiment_data.empty or price_data.empty:
            return {'status': 'no_data', 'message': "Insufficient data for correlation analysis"}
            
        # Ensure both DataFrames have datetime index
        if 'timestamp' in sentiment_data.columns:
            sentiment_data.set_index('timestamp', inplace=True)
        
        if 'timestamp' in price_data.columns:
            price_data.set_index('timestamp', inplace=True)
        elif 'datetime' in price_data.columns:
            price_data.set_index('datetime', inplace=True)
            
        # Resample both to daily frequency for consistent comparison
        sentiment_daily = sentiment_data['sentiment_score'].resample('D').mean()
        
        # Assume price_data has 'close' column, adjust if needed
        if 'close' in price_data.columns:
            price_daily = price_data['close'].resample('D').last()
        else:
            # Try to find a suitable price column
            price_cols = [col for col in price_data.columns if col.lower() in ('close', 'price', 'adj_close', 'adjusted_close')]
            if price_cols:
                price_daily = price_data[price_cols[0]].resample('D').last()
            else:
                # Use the first numeric column as fallback
                numeric_cols = price_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_daily = price_data[numeric_cols[0]].resample('D').last()
                else:
                    return {'status': 'error', 'message': "No suitable price column found in price_data"}
                    
        # Calculate daily returns
        price_returns = price_daily.pct_change().dropna()
        
        # Align the series with an inner join
        joined = pd.concat([sentiment_daily, price_returns], axis=1, join='inner').dropna()
        
        if len(joined) < 5:  # Need at least a few data points for meaningful correlation
            return {'status': 'insufficient_data', 'message': "Not enough overlapping data points for correlation analysis"}
            
        # Calculate correlation
        correlation = joined.corr().iloc[0, 1]
        
        # Calculate lagged correlations to see if sentiment leads price
        lags = {}
        for lag in range(1, min(6, len(joined) // 2)):  # Up to 5 days, or half the data length
            lagged_sentiment = sentiment_daily.shift(lag)
            lagged_joined = pd.concat([lagged_sentiment, price_returns], axis=1, join='inner').dropna()
            if len(lagged_joined) >= 5:
                lags[lag] = lagged_joined.corr().iloc[0, 1]
                
        # Find the lag with the strongest correlation
        best_lag = max(lags.items(), key=lambda x: abs(x[1]), default=(0, 0))
        
        return {
            'current_correlation': correlation,
            'lagged_correlations': lags,
            'best_lag': best_lag[0],
            'best_lag_correlation': best_lag[1],
            'sentiment_leads_price': best_lag[0] > 0 and abs(best_lag[1]) > abs(correlation),
            'data_points': len(joined),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def identify_sentiment_divergence(self, sentiment_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify divergence between sentiment and price.
        
        Args:
            sentiment_data: DataFrame with sentiment time series
            price_data: DataFrame with price time series
            
        Returns:
            Dict[str, Any]: Divergence analysis results
        """
        self.log_action("identify_sentiment_divergence", "Analyzing sentiment-price divergence")
        
        if sentiment_data.empty or price_data.empty:
            return {'status': 'no_data', 'message': "Insufficient data for divergence analysis"}
            
        # Ensure both DataFrames have datetime index
        if 'timestamp' in sentiment_data.columns:
            sentiment_data.set_index('timestamp', inplace=True)
        
        if 'timestamp' in price_data.columns:
            price_data.set_index('timestamp', inplace=True)
        elif 'datetime' in price_data.columns:
            price_data.set_index('datetime', inplace=True)
            
        # Resample both to daily frequency for consistent comparison
        sentiment_daily = sentiment_data['sentiment_score'].resample('D').mean()
        
        # Assume price_data has 'close' column, adjust if needed
        if 'close' in price_data.columns:
            price_daily = price_data['close'].resample('D').last()
        else:
            # Try to find a suitable price column
            price_cols = [col for col in price_data.columns if col.lower() in ('close', 'price', 'adj_close', 'adjusted_close')]
            if price_cols:
                price_daily = price_data[price_cols[0]].resample('D').last()
            else:
                # Use the first numeric column as fallback
                numeric_cols = price_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_daily = price_data[numeric_cols[0]].resample('D').last()
                else:
                    return {'status': 'error', 'message': "No suitable price column found in price_data"}
        
        # Calculate the 10-day change for both price and sentiment
        sentiment_change = sentiment_daily.pct_change(10).dropna()
        price_change = price_daily.pct_change(10).dropna()
        
        # Align the series with an inner join
        joined = pd.concat([sentiment_change, price_change], axis=1, join='inner').dropna()
        joined.columns = ['sentiment_change', 'price_change']
        
        if len(joined) < 5:  # Need at least a few data points for meaningful analysis
            return {'status': 'insufficient_data', 'message': "Not enough overlapping data points for divergence analysis"}
        
        # Check for divergence (sentiment and price moving in opposite directions)
        joined['divergence'] = joined['sentiment_change'] * joined['price_change'] < 0
        
        # Calculate 30-day trend for sentiment and price
        sentiment_trend = sentiment_daily.rolling(30).mean().diff().dropna()
        price_trend = price_daily.rolling(30).mean().diff().dropna()
        
        # Align the trend series with an inner join
        trend_joined = pd.concat([sentiment_trend, price_trend], axis=1, join='inner').dropna()
        trend_joined.columns = ['sentiment_trend', 'price_trend']
        
        if len(trend_joined) < 5:
            return {'status': 'insufficient_data', 'message': "Not enough overlapping data points for trend divergence analysis"}
            
        # Check for trend divergence (sentiment and price trending in opposite directions)
        trend_joined['trend_divergence'] = trend_joined['sentiment_trend'] * trend_joined['price_trend'] < 0
        
        # Get the most recent data
        latest_divergence = joined['divergence'].iloc[-1]
        latest_trend_divergence = trend_joined['trend_divergence'].iloc[-1] if not trend_joined.empty else False
        
        # Calculate the proportion of days with divergence
        divergence_percent = joined['divergence'].mean() * 100
        trend_divergence_percent = trend_joined['trend_divergence'].mean() * 100 if not trend_joined.empty else 0
        
        return {
            'current_divergence': bool(latest_divergence),
            'current_trend_divergence': bool(latest_trend_divergence),
            'divergence_percent': divergence_percent,
            'trend_divergence_percent': trend_divergence_percent,
            'data_points': len(joined),
            'analysis_timestamp': datetime.now().isoformat()
        }

    # === Signal Generation Methods ===
    
    def generate_sentiment_signals(self, currency_pair: str) -> Dict[str, Any]:
        """
        Generate trading signals based on sentiment.
        
        Args:
            currency_pair: Currency pair to analyze
            
        Returns:
            Dict[str, Any]: Trading signal information
        """
        self.log_action("generate_sentiment_signals", f"Generating sentiment signals for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        
        # Get combined sentiment
        sentiment = self.get_combined_sentiment(pair)
        
        # Check for extreme sentiment (potential contrarian signals)
        extremes = self.detect_sentiment_extremes(pair)
        
        # Generate signals based on sentiment and extremes
        signal = None
        signal_strength = 0.0
        is_contrarian = False
        confidence = 'low'
        
        # Regular sentiment-based signal
        if sentiment['sentiment_label'] in ['strongly_bullish', 'bullish']:
            signal = 'buy'
            signal_strength = abs(sentiment['overall_score'])
        elif sentiment['sentiment_label'] in ['strongly_bearish', 'bearish']:
            signal = 'sell'
            signal_strength = abs(sentiment['overall_score'])
        
        # Adjust for extreme sentiment (contrarian signal)
        if extremes.get('is_extreme', False):
            # If sentiment is extremely positive/negative, consider a contrarian signal
            z_score = extremes.get('z_score', 0)
            if abs(z_score) > 3:  # Very extreme sentiment
                is_contrarian = True
                if signal == 'buy':
                    signal = 'sell'
                elif signal == 'sell':
                    signal = 'buy'
                signal_strength = min(0.7, signal_strength)  # Reduce strength for contrarian signals
        
        # Determine confidence level based on signal strength and data quality
        source_count = sum(source.get('count', 0) for source in sentiment.get('sources', []))
        
        if source_count >= 50 and signal_strength >= 0.6:
            confidence = 'high'
        elif source_count >= 20 and signal_strength >= 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Adjust confidence for contrarian signals
        if is_contrarian:
            if confidence == 'high':
                confidence = 'medium'
            elif confidence == 'medium':
                confidence = 'low'
        
        # Generate timeframes for the signal
        timeframes = {
            'short_term': {
                'direction': signal,
                'strength': signal_strength,
                'confidence': confidence,
                'horizon': '1-3 days'
            },
            'medium_term': {
                'direction': signal,
                'strength': max(0.0, signal_strength - 0.1),  # Slightly lower confidence for medium term
                'confidence': self._adjust_confidence_down(confidence),
                'horizon': '1-2 weeks'
            },
            'long_term': None  # Sentiment is not as reliable for long-term signals
        }
        
        return {
            'currency_pair': pair,
            'signal': signal,
            'strength': signal_strength,
            'confidence': confidence,
            'is_contrarian': is_contrarian,
            'sentiment_score': sentiment['overall_score'],
            'sentiment_label': sentiment['sentiment_label'],
            'timeframes': timeframes,
            'timestamp': datetime.now().isoformat()
        }
    
    def _adjust_confidence_down(self, confidence: str) -> str:
        """Adjust confidence level down one step."""
        if confidence == 'high':
            return 'medium'
        elif confidence == 'medium':
            return 'low'
        return 'low'
    
    def combine_sentiment_sources(self, sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine sentiment from multiple sources into a single DataFrame.
        
        Args:
            sources: Dictionary of source name to DataFrame
            
        Returns:
            pd.DataFrame: Combined sentiment data
        """
        combined_data = []
        
        for source_name, data in sources.items():
            if data is None or data.empty:
                continue
                
            # Add source column if not present
            if 'source' not in data.columns:
                data['source'] = source_name
                
            combined_data.append(data)
            
        if not combined_data:
            return pd.DataFrame()
            
        return pd.concat(combined_data, ignore_index=True)
    
    def weight_sentiment_factors(self, factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply weights to different sentiment factors.
        
        Args:
            factors: List of sentiment factors with impact and strength
            
        Returns:
            List[Dict[str, Any]]: Factors with weights added
        """
        if not factors:
            return []
            
        # Weight mapping for impact
        impact_weights = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }
        
        # Weight mapping for strength
        strength_weights = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }
        
        # Apply weights
        for factor in factors:
            impact = factor.get('impact', 'neutral')
            strength = factor.get('strength', 'medium')
            
            impact_weight = impact_weights.get(impact, 0.0)
            strength_weight = strength_weights.get(strength, 0.5)
            
            factor['weight'] = impact_weight * strength_weight
            
        return factors
    
    # === Data Visualization Methods ===
    
    def plot_sentiment_trends(self, currency_pair: str, timeframe: str = 'daily') -> Figure:
        """
        Plot sentiment trends over time.
        
        Args:
            currency_pair: Currency pair to analyze
            timeframe: Time resolution ('hourly', 'daily', 'weekly')
            
        Returns:
            Figure: Matplotlib figure with sentiment trends
        """
        self.log_action("plot_sentiment_trends", f"Plotting sentiment trends for {currency_pair}")
        
        # Get sentiment changes data
        sentiment_changes = self.track_sentiment_changes(currency_pair, timeframe)
        
        if sentiment_changes.empty:
            # Create empty figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No sentiment data available for {currency_pair}", 
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(f"Sentiment Trends for {currency_pair}")
            return fig
            
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot sentiment score
        ax.plot(sentiment_changes['timestamp'], sentiment_changes['sentiment_score'], 
                marker='o', linestyle='-', color='blue', label='Sentiment Score')
        
        # Add horizontal lines for thresholds
        ax.axhline(y=self.sentiment_thresholds['strongly_bullish'], color='green', linestyle='--', alpha=0.7, 
                  label='Strongly Bullish Threshold')
        ax.axhline(y=self.sentiment_thresholds['bullish'], color='lightgreen', linestyle='--', alpha=0.7, 
                  label='Bullish Threshold')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=self.sentiment_thresholds['bearish'], color='pink', linestyle='--', alpha=0.7, 
                  label='Bearish Threshold')
        ax.axhline(y=self.sentiment_thresholds['strongly_bearish'], color='red', linestyle='--', alpha=0.7, 
                  label='Strongly Bearish Threshold')
        
        # Color the background based on sentiment zones
        ax.axhspan(self.sentiment_thresholds['strongly_bullish'], 1.0, color='green', alpha=0.1)
        ax.axhspan(self.sentiment_thresholds['bullish'], self.sentiment_thresholds['strongly_bullish'], color='lightgreen', alpha=0.1)
        ax.axhspan(self.sentiment_thresholds['bearish'], self.sentiment_thresholds['bullish'], color='gray', alpha=0.1)
        ax.axhspan(self.sentiment_thresholds['strongly_bearish'], self.sentiment_thresholds['bearish'], color='pink', alpha=0.1)
        ax.axhspan(-1.0, self.sentiment_thresholds['strongly_bearish'], color='red', alpha=0.1)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.set_title(f'Sentiment Trends for {currency_pair} ({timeframe} resolution)')
        
        # Set y-axis limits
        ax.set_ylim(-1.1, 1.1)
        
        # Add legend
        ax.legend(loc='best')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        return fig
    
    def create_sentiment_dashboard(self, currency_pairs: List[str]) -> Dict[str, Any]:
        """
        Create a dashboard of sentiment data for multiple currency pairs.
        
        Args:
            currency_pairs: List of currency pairs to analyze
            
        Returns:
            Dict[str, Any]: Dashboard data
        """
        self.log_action("create_sentiment_dashboard", f"Creating sentiment dashboard for {len(currency_pairs)} pairs")
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'currency_pairs': {},
            'summary': {}
        }
        
        # Analyze each currency pair
        for pair in currency_pairs:
            normalized_pair = self._normalize_currency_pair(pair)
            
            # Get combined sentiment
            sentiment = self.get_combined_sentiment(normalized_pair)
            
            # Generate signal
            signal = self.generate_sentiment_signals(normalized_pair)
            
            # Store in dashboard
            dashboard['currency_pairs'][normalized_pair] = {
                'sentiment_score': sentiment.get('overall_score', 0),
                'sentiment_label': sentiment.get('sentiment_label', 'neutral'),
                'signal': signal.get('signal', None),
                'signal_strength': signal.get('strength', 0),
                'confidence': signal.get('confidence', 'low'),
                'is_contrarian': signal.get('is_contrarian', False)
            }
        
        # Generate summary stats
        bullish_count = sum(1 for pair in dashboard['currency_pairs'].values() 
                           if pair['sentiment_label'] in ['bullish', 'strongly_bullish'])
        bearish_count = sum(1 for pair in dashboard['currency_pairs'].values() 
                           if pair['sentiment_label'] in ['bearish', 'strongly_bearish'])
        neutral_count = len(currency_pairs) - bullish_count - bearish_count
        
        dashboard['summary'] = {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'market_bias': 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'neutral',
            'most_bullish': max(dashboard['currency_pairs'].items(), key=lambda x: x[1]['sentiment_score'])[0] if dashboard['currency_pairs'] else None,
            'most_bearish': min(dashboard['currency_pairs'].items(), key=lambda x: x[1]['sentiment_score'])[0] if dashboard['currency_pairs'] else None
        }
        
        return dashboard
    
    def generate_sentiment_report(self, currency_pair: str) -> Dict[str, Any]:
        """
        Generate a comprehensive sentiment report for a currency pair.
        
        Args:
            currency_pair: Currency pair to analyze
            
        Returns:
            Dict[str, Any]: Comprehensive sentiment report
        """
        self.log_action("generate_sentiment_report", f"Generating sentiment report for {currency_pair}")
        
        # Normalize currency pair
        pair = self._normalize_currency_pair(currency_pair)
        base_currency, quote_currency = pair.split('_')
        
        # Get data from all sources
        news_sentiment = self.get_news_sentiment(pair)
        social_sentiment = self.get_social_media_sentiment(pair)
        analyst_ratings = self.get_analyst_ratings(pair)
        market_positioning = self.get_market_positioning(pair)
        
        # Get combined sentiment
        combined_sentiment = self.get_combined_sentiment(pair)
        
        # Generate signal
        signal = self.generate_sentiment_signals(pair)
        
        # Check for extreme sentiment
        extremes = self.detect_sentiment_extremes(pair)
        
        # Track sentiment changes
        sentiment_changes = self.track_sentiment_changes(pair, 'daily')
        
        # Create the report
        report = {
            'currency_pair': pair,
            'base_currency': base_currency,
            'quote_currency': quote_currency,
            'timestamp': datetime.now().isoformat(),
            'combined_sentiment': combined_sentiment,
            'signal': signal,
            'extreme_sentiment': extremes,
            'source_summary': {
                'news': {
                    'count': len(news_sentiment) if not news_sentiment.empty else 0,
                    'avg_score': news_sentiment['sentiment_score'].mean() if not news_sentiment.empty else None,
                    'latest': news_sentiment.iloc[-1].to_dict() if not news_sentiment.empty else None
                },
                'social_media': {
                    'count': len(social_sentiment) if not social_sentiment.empty else 0,
                    'avg_score': social_sentiment['sentiment_score'].mean() if not social_sentiment.empty else None,
                    'latest': social_sentiment.iloc[-1].to_dict() if not social_sentiment.empty else None
                },
                'analyst_ratings': {
                    'count': len(analyst_ratings) if not analyst_ratings.empty else 0,
                    'latest': analyst_ratings.iloc[-1].to_dict() if not analyst_ratings.empty else None
                },
                'market_positioning': {
                    'count': len(market_positioning) if not market_positioning.empty else 0,
                    'latest': market_positioning.iloc[-1].to_dict() if not market_positioning.empty else None
                }
            },
            'sentiment_trends': {
                'weekly_change': sentiment_changes['change'].tail(7).sum() if not sentiment_changes.empty else None,
                'momentum': sentiment_changes['momentum'].iloc[-1] if not sentiment_changes.empty else None,
                'direction': 'increasing' if sentiment_changes['change'].tail(3).mean() > 0 else 'decreasing' if sentiment_changes['change'].tail(3).mean() < 0 else 'stable' if not sentiment_changes.empty else None
            },
            'key_factors': self.extract_sentiment_factors(
                " ".join([
                    news_sentiment.iloc[-1]['title'] if not news_sentiment.empty else "",
                    social_sentiment.iloc[-1]['content'] if not social_sentiment.empty else ""
                ])
            )
        }
        
        # Generate narrative
        report['narrative'] = self._generate_sentiment_narrative(report)
        
        return report
    
    def _generate_sentiment_narrative(self, report: Dict[str, Any]) -> str:
        """
        Generate a narrative description of the sentiment analysis.
        
        Args:
            report: Sentiment report data
            
        Returns:
            str: Narrative description
        """
        pair = report.get('currency_pair', '')
        base = report.get('base_currency', '')
        quote = report.get('quote_currency', '')
        
        sentiment = report.get('combined_sentiment', {})
        sentiment_label = sentiment.get('sentiment_label', 'neutral')
        sentiment_score = sentiment.get('overall_score', 0)
        
        signal = report.get('signal', {})
        signal_direction = signal.get('signal', None)
        signal_strength = signal.get('strength', 0)
        signal_confidence = signal.get('confidence', 'low')
        is_contrarian = signal.get('is_contrarian', False)
        
        trends = report.get('sentiment_trends', {})
        trend_direction = trends.get('direction', 'stable')
        
        # Start building the narrative
        narrative = f"Market sentiment for {pair} is currently {sentiment_label}"
        
        if sentiment_label in ['strongly_bullish', 'bullish']:
            narrative += f", with a positive sentiment score of {sentiment_score:.2f}."
        elif sentiment_label in ['strongly_bearish', 'bearish']:
            narrative += f", with a negative sentiment score of {sentiment_score:.2f}."
        else:
            narrative += f", with a neutral sentiment score of {sentiment_score:.2f}."
            
        # Add trend information
        if trend_direction == 'increasing':
            narrative += f" The sentiment has been improving over the recent period."
        elif trend_direction == 'decreasing':
            narrative += f" The sentiment has been deteriorating over the recent period."
        else:
            narrative += f" The sentiment has been relatively stable recently."
            
        # Add signal information
        if signal_direction:
            narrative += f" Based on the current sentiment analysis, a {signal_confidence} confidence {signal_direction.upper()} signal is generated"
            
            if is_contrarian:
                narrative += " (contrarian signal based on extreme sentiment)."
            else:
                narrative += "."
                
        # Add source information
        news_count = report.get('source_summary', {}).get('news', {}).get('count', 0)
        social_count = report.get('source_summary', {}).get('social_media', {}).get('count', 0)
        
        if news_count > 0 or social_count > 0:
            narrative += f" This analysis is based on {news_count} news articles and {social_count} social media posts."
            
        # Add key factors if available
        key_factors = report.get('key_factors', [])
        if key_factors:
            narrative += " Key factors influencing sentiment include: "
            factor_texts = []
            
            for factor in key_factors[:3]:  # Include top 3 factors
                impact = factor.get('impact', 'neutral')
                if impact == 'positive':
                    factor_texts.append(f"{factor.get('factor', 'unknown factor')} (positive)")
                elif impact == 'negative':
                    factor_texts.append(f"{factor.get('factor', 'unknown factor')} (negative)")
                    
            narrative += ", ".join(factor_texts) + "."
            
        return narrative
    
    # === Mock Data Generation Methods ===
    
    def _generate_mock_news(self, currency_pair: str, days_back: int = 7) -> pd.DataFrame:
        """Generate mock news data for testing."""
        base_currency, quote_currency = self._normalize_currency_pair(currency_pair).split('_')
        
        # Get country names
        base_country = self.countries.get(base_currency, base_currency)
        quote_country = self.countries.get(quote_currency, quote_currency)
        
        # News titles and sentiments
        news_templates = [
            (f"{base_country} economy shows strong growth", 0.7),
            (f"{base_country} inflation rises more than expected", -0.4),
            (f"{base_country} central bank hints at rate hike", 0.6),
            (f"{base_country} unemployment falls to record low", 0.8),
            (f"{base_country} trade deficit widens", -0.3),
            (f"{quote_country} economy slows down", 0.5),  # Good for base currency
            (f"{quote_country} inflation slows", -0.2),
            (f"{quote_country} central bank keeps rates unchanged", 0.1),
            (f"Analysts bullish on {base_currency}", 0.7),
            (f"Traders cautious about {quote_currency} outlook", 0.4),
            (f"{base_currency}/{quote_currency} likely to rise, says expert", 0.6),
            (f"{base_currency}/{quote_currency} faces resistance at key levels", -0.2),
            (f"Economic data supports stronger {base_currency}", 0.5),
            (f"Political uncertainty weighs on {base_currency}", -0.5),
            (f"Market sentiment improves for {base_currency}", 0.4)
        ]
        
        # Generate random news
        news = []
        end_date = datetime.now()
        
        for i in range(50):  # Generate 50 news items
            days_ago = np.random.randint(0, days_back)
            hours_ago = np.random.randint(0, 24)
            minutes_ago = np.random.randint(0, 60)
            
            news_date = end_date - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            # Select random news template
            title, base_sentiment = news_templates[np.random.randint(0, len(news_templates))]
            
            # Add some random variation to sentiment
            sentiment_variation = np.random.uniform(-0.2, 0.2)
            sentiment = base_sentiment + sentiment_variation
            sentiment = max(-1.0, min(1.0, sentiment))  # Clip to [-1.0, 1.0]
            
            news.append({
                'title': title,
                'date': news_date.strftime('%Y-%m-%d'),
                'time': news_date.strftime('%H:%M'),
                'timestamp': news_date,
                'source': np.random.choice(['Financial Times', 'Bloomberg', 'Reuters', 'CNBC', 'Wall Street Journal']),
                'url': f"https://example.com/news/{i}",
                'currency_pair': currency_pair
            })
            
        return pd.DataFrame(news)
    
    def _generate_mock_social_media(self, currency_pair: str, days_back: int = 7) -> pd.DataFrame:
        """Generate mock social media data for testing."""
        base_currency, quote_currency = self._normalize_currency_pair(currency_pair).split('_')
        
        # Social media content templates
        templates = [
            (f"Feeling bullish on {base_currency}/{quote_currency} today! #forex #trading", 0.7),
            (f"Just went long on {base_currency}/{quote_currency}, technical setup looks great", 0.6),
            (f"Not sure about {base_currency}/{quote_currency}, seems overvalued", -0.3),
            (f"Bearish on {base_currency} after disappointing economic data #trading", -0.7),
            (f"Anyone else watching {base_currency}/{quote_currency}? Looks like it's about to break out", 0.4),
            (f"Taking profits on my {base_currency}/{quote_currency} position today", 0.1),
            (f"Shorting {base_currency}/{quote_currency}, expecting pullback", -0.6),
            (f"{base_currency} looking strong against {quote_currency} #forex #trading", 0.5),
            (f"Technical indicators suggest {base_currency}/{quote_currency} will drop soon", -0.5),
            (f"Just closed my {base_currency}/{quote_currency} position at a loss. Market is unpredictable", -0.4)
        ]
        
        # Generate random posts
        posts = []
        end_date = datetime.now()
        
        for i in range(100):  # Generate 100 social media posts
            days_ago = np.random.randint(0, days_back)
            hours_ago = np.random.randint(0, 24)
            minutes_ago = np.random.randint(0, 60)
            
            post_date = end_date - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            # Select random template
            content, base_sentiment = templates[np.random.randint(0, len(templates))]
            
            # Add some random variation to sentiment
            sentiment_variation = np.random.uniform(-0.3, 0.3)
            sentiment = base_sentiment + sentiment_variation
            sentiment = max(-1.0, min(1.0, sentiment))  # Clip to [-1.0, 1.0]
            
            posts.append({
                'content': content,
                'date': post_date.strftime('%Y-%m-%d'),
                'time': post_date.strftime('%H:%M'),
                'timestamp': post_date,
                'platform': np.random.choice(['Twitter', 'Reddit', 'StockTwits', 'Trading Forum']),
                'likes': np.random.randint(0, 100),
                'currency_pair': currency_pair
            })
            
        return pd.DataFrame(posts)
    
    def _generate_mock_analyst_ratings(self, currency_pair: str) -> pd.DataFrame:
        """Generate mock analyst ratings for testing."""
        ratings = []
        current_date = datetime.now()
        
        # Generate 10 analyst ratings
        for i in range(10):
            days_ago = np.random.randint(0, 30)  # Ratings from the last 30 days
            rating_date = current_date - timedelta(days=days_ago)
            
            # Generate random rating
            direction = np.random.choice(['buy', 'sell', 'hold'], p=[0.5, 0.3, 0.2])
            
            # Map direction to score
            if direction == 'buy':
                score = np.random.uniform(0.3, 0.8)
            elif direction == 'sell':
                score = np.random.uniform(-0.8, -0.3)
            else:  # hold
                score = np.random.uniform(-0.2, 0.2)
                
            ratings.append({
                'analyst': f"Analyst {i+1}",
                'firm': np.random.choice(['Goldman Sachs', 'JP Morgan', 'Morgan Stanley', 'Citi', 'BofA', 'Barclays']),
                'rating': direction,
                'score': score,
                'target': np.random.uniform(0.9, 1.1),  # Random price target (normalized)
                'date': rating_date.strftime('%Y-%m-%d'),
                'currency_pair': currency_pair
            })
            
        return pd.DataFrame(ratings)
    
    def _generate_mock_market_positioning(self, currency_pair: str) -> pd.DataFrame:
        """Generate mock market positioning data for testing."""
        positions = []
        current_date = datetime.now()
        
        # Generate positioning data for the last 12 weeks
        for i in range(12):
            days_ago = i * 7  # Weekly data
            position_date = current_date - timedelta(days=days_ago)
            
            # Generate random positioning data
            long_positions = np.random.randint(10000, 50000)
            short_positions = np.random.randint(10000, 50000)
            
            # Calculate net positioning and sentiment score
            net_positioning = long_positions - short_positions
            total_positions = long_positions + short_positions
            
            # Calculate percent long/short
            percent_long = long_positions / total_positions
            percent_short = short_positions / total_positions
            
            # Calculate sentiment score (-1 to 1)
            score = (percent_long - percent_short) * 2  # Scale to [-1, 1]
            
            positions.append({
                'date': position_date.strftime('%Y-%m-%d'),
                'long_positions': long_positions,
                'short_positions': short_positions,
                'net_positioning': net_positioning,
                'percent_long': percent_long,
                'percent_short': percent_short,
                'score': score,
                'currency_pair': currency_pair
            })
            
        return pd.DataFrame(positions)
        
    # === LangGraph Node Implementation ===
    
    def setup_node(self) -> Callable:
        """
        Set up the LangGraph node for this agent.
        
        Returns:
            Callable: Sentiment node function
        """
        def sentiment_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """LangGraph node implementation for the sentiment agent."""
            # Extract the task from the state
            task = state.get('task', {})
            
            # Extract parameters
            action = task.get('action', '')
            params = task.get('parameters', {})
            
            result = None
            
            try:
                # Route to the appropriate method based on the action
                if action == 'get_news_sentiment':
                    currency_pair = params.get('currency_pair', '')
                    days_back = params.get('days_back', 7)
                    result = self.get_news_sentiment(currency_pair, days_back)
                    
                elif action == 'get_social_media_sentiment':
                    currency_pair = params.get('currency_pair', '')
                    days_back = params.get('days_back', 7)
                    result = self.get_social_media_sentiment(currency_pair, days_back)
                    
                elif action == 'get_analyst_ratings':
                    currency_pair = params.get('currency_pair', '')
                    result = self.get_analyst_ratings(currency_pair)
                    
                elif action == 'get_market_positioning':
                    currency_pair = params.get('currency_pair', '')
                    result = self.get_market_positioning(currency_pair)
                    
                elif action == 'get_combined_sentiment':
                    currency_pair = params.get('currency_pair', '')
                    days_back = params.get('days_back', 7)
                    result = self.get_combined_sentiment(currency_pair, days_back)
                    
                elif action == 'generate_sentiment_signals':
                    currency_pair = params.get('currency_pair', '')
                    result = self.generate_sentiment_signals(currency_pair)
                    
                elif action == 'track_sentiment_changes':
                    currency_pair = params.get('currency_pair', '')
                    timeframe = params.get('timeframe', 'daily')
                    result = self.track_sentiment_changes(currency_pair, timeframe)
                    
                elif action == 'generate_sentiment_report':
                    currency_pair = params.get('currency_pair', '')
                    result = self.generate_sentiment_report(currency_pair)
                    
                else:
                    result = {
                        'status': 'error',
                        'message': f"Unknown action: {action}"
                    }
                    
            except Exception as e:
                result = {
                    'status': 'error',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
                
            # Update the state with the result
            state['result'] = result
            
            return state
            
        return sentiment_node 

    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sentiment analysis tasks on forex market data.
        
        This method implements the abstract method from BaseAgent and serves
        as the primary entry point for sentiment analysis operations.
        
        Args:
            task: Task description and parameters including:
                - type: Type of analysis to perform (e.g., "news", "social", "combined")
                - currency: The currency or currency pair to analyze
                - sources: Optional specific sources to analyze
                - timeframe: Optional timeframe for analysis

        Returns:
            Dict[str, Any]: Task execution results including:
                - status: "success" or "error"
                - sentiment: Sentiment analysis results
                - confidence: Confidence scores
                - sources: Source breakdown of sentiment
        """
        self.log_action("run_task", f"Running task: {task.get('type', 'Unknown')}")
        
        # Update state
        self.state["tasks"].append(task)
        self.last_active = datetime.now()
        self.state["last_active"] = self.last_active
        
        try:
            task_type = task.get("type", "combined")
            currency = task.get("currency", "EUR/USD")
            
            # Normalize currency format if needed
            currency = currency.replace("_", "/") if "_" in currency else currency
            
            # Set up timeframe
            timeframe = task.get("timeframe", "24h")
            
            if task_type == "news":
                # Analyze news sentiment
                sources = task.get("sources", self.news_sources)
                news_sentiment = self.analyze_news_sentiment(currency, sources, timeframe)
                
                return {
                    "status": "success",
                    "sentiment": news_sentiment.get("sentiment", 0),
                    "confidence": news_sentiment.get("confidence", 0),
                    "sources": news_sentiment.get("sources", {}),
                    "articles": news_sentiment.get("articles", []),
                    "timestamp": datetime.now().isoformat()
                }
                
            elif task_type == "social":
                # Analyze social media sentiment
                platforms = task.get("platforms", self.social_platforms)
                social_sentiment = self.analyze_social_sentiment(currency, platforms, timeframe)
                
                return {
                    "status": "success",
                    "sentiment": social_sentiment.get("sentiment", 0),
                    "confidence": social_sentiment.get("confidence", 0),
                    "platforms": social_sentiment.get("platforms", {}),
                    "posts": social_sentiment.get("sample_posts", []),
                    "timestamp": datetime.now().isoformat()
                }
                
            elif task_type == "combined" or task_type == "all":
                # Combine news and social sentiment analysis
                combined_sentiment = self.get_combined_sentiment(currency, timeframe)
                
                return {
                    "status": "success",
                    "sentiment": combined_sentiment.get("sentiment", 0),
                    "confidence": combined_sentiment.get("confidence", 0),
                    "breakdown": {
                        "news": combined_sentiment.get("news_sentiment", 0),
                        "social": combined_sentiment.get("social_sentiment", 0)
                    },
                    "impact": combined_sentiment.get("market_impact", "neutral"),
                    "timestamp": datetime.now().isoformat()
                }
                
            elif task_type == "historical":
                # Get historical sentiment trends
                days = int(task.get("days", 7))
                historical_data = self.get_historical_sentiment(currency, days)
                
                return {
                    "status": "success",
                    "historical_data": historical_data,
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