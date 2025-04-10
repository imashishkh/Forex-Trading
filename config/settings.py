#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration settings for the Forex Trading Platform
"""

import os
import yaml
import json
from pathlib import Path

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

def load_config(config_path=None):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str, optional): Path to the configuration file. 
                                     If None, uses the default path.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the configuration
    with open(config_path, 'r') as file:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(file)
        elif config_path.endswith('.json'):
            config = json.load(file)
        else:
            raise ValueError("Config file must be YAML or JSON format")
    
    return config

def save_config(config, config_path=None):
    """
    Save configuration to file
    
    Args:
        config (dict): Configuration dictionary
        config_path (str, optional): Path to save the configuration to.
                                    If None, uses the default path.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(config_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the configuration
    with open(config_path, 'w') as file:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, file, default_flow_style=False)
        elif config_path.endswith('.json'):
            json.dump(config, file, indent=4)
        else:
            raise ValueError("Config file must be YAML or JSON format")

def get_default_config():
    """
    Generate default configuration
    
    Returns:
        dict: Default configuration dictionary
    """
    return {
        'platform': {
            'name': 'Forex Trading AI Platform',
            'version': '0.1.0',
            'cycle_interval': 60,  # seconds
            'log_level': 'INFO',
        },
        'market_data': {
            'api_provider': 'example_provider',
            'api_key': 'your_api_key_here',
            'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'data_dir': str(Path('../data/market_data').resolve()),
        },
        'technical_analysis': {
            'indicators': {
                'moving_averages': ['SMA', 'EMA', 'WMA'],
                'oscillators': ['RSI', 'MACD', 'Stochastic'],
                'volatility': ['Bollinger Bands', 'ATR'],
                'trend': ['ADX', 'Ichimoku']
            },
            'lookback_periods': {
                'short_term': 14,
                'medium_term': 50,
                'long_term': 200
            }
        },
        'fundamentals': {
            'data_sources': ['economic_calendar', 'central_bank_announcements', 'economic_indicators'],
            'api_keys': {
                'news_api': 'your_news_api_key_here',
                'economic_data': 'your_economic_data_api_key_here'
            }
        },
        'sentiment': {
            'data_sources': ['twitter', 'news', 'reddit'],
            'api_keys': {
                'twitter': 'your_twitter_api_key_here',
                'news': 'your_news_api_key_here'
            },
            'models': {
                'nlp_model': 'distilbert-base-uncased',
                'sentiment_threshold': 0.5
            }
        },
        'risk_management': {
            'max_risk_per_trade': 0.02,  # 2% of account
            'max_open_positions': 5,
            'max_risk_per_currency': 0.05,  # 5% of account
            'max_daily_drawdown': 0.05,  # 5% of account
            'stop_loss_atr_multiplier': 2.0
        },
        'portfolio_management': {
            'account_size': 10000,
            'leverage': 30,
            'position_sizing_method': 'risk_based',  # Alternatives: 'fixed_lot', 'percent_based'
            'broker': {
                'name': 'example_broker',
                'api_key': 'your_broker_api_key_here',
                'api_secret': 'your_broker_api_secret_here'
            }
        }
    } 