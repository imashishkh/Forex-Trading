#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for the Portfolio Manager Agent
"""

DEFAULT_CONFIG = {
    # Account Settings
    'account_size': 10000,         # Initial account size in USD
    'leverage': 30,                # Maximum leverage allowed
    'max_open_positions': 5,       # Maximum number of open positions
    'position_sizing_method': 'risk_based',  # 'fixed', 'risk_based', 'position_based'
    
    # Risk Management
    'risk': {
        'max_drawdown_pct': 20,     # Maximum drawdown percentage allowed
        'max_risk_per_trade_pct': 2, # Maximum risk per trade (percentage of account)
        'min_risk_reward_ratio': 1.5, # Minimum risk-reward ratio for new trades
        'min_margin_level': 200,     # Minimum margin level for new trades
        'stop_loss_method': 'atr',   # Method for setting stop loss: 'fixed', 'atr', 'support_resistance'
        'take_profit_method': 'risk_reward', # Method for setting take profit: 'fixed', 'risk_reward', 'fibonacci'
        'trailing_stop_activation': 1.0, # Activation level for trailing stop (% of price)
        'trailing_stop_distance': 0.5   # Distance for trailing stop (% of price)
    },
    
    # Trading Sessions
    'trading_sessions': {
        'enabled': True,              # Whether to consider trading sessions
        'preferred_sessions': ['London-New York'],  # Preferred trading sessions
        'session_volume_threshold': 0.7  # Minimum session volume threshold
    },
    
    # Signal Alignment
    'signal_alignment': {
        'min_aligned_signals': 2,    # Minimum number of aligned signals required
        'technical_weight': 0.4,     # Weight of technical analysis in decision
        'fundamental_weight': 0.3,   # Weight of fundamental analysis in decision
        'sentiment_weight': 0.3      # Weight of sentiment analysis in decision
    },
    
    # Position Management
    'position_management': {
        'partial_close': {
            'enabled': True,          # Whether to enable partial closing of positions
            'thresholds': [           # Thresholds for partial closing
                {'profit_pct': 1.0, 'close_pct': 0.25},
                {'profit_pct': 2.0, 'close_pct': 0.25},
                {'profit_pct': 3.0, 'close_pct': 0.25}
            ]
        },
        'pyramiding': {
            'enabled': False,         # Whether to enable pyramiding (adding to winning positions)
            'max_additions': 2,       # Maximum number of additions to a position
            'min_profit_pct': 0.5     # Minimum profit percentage to allow pyramiding
        },
        'timeout': {
            'enabled': True,          # Whether to enable position timeout
            'days': 30,               # Maximum days to hold a position
            'only_losing': True       # Only apply timeout to losing positions
        }
    },
    
    # Broker Settings
    'broker': {
        'name': 'default',            # Broker name
        'api_key': '',                # API key
        'api_secret': '',             # API secret
        'account_id': '',             # Account ID
        'base_url': '',               # Base URL for API
        'paper_trading': False         # Whether to use paper trading
    },
    
    # Backtesting Settings
    'backtesting': {
        'enabled': False,             # Whether this is a backtesting configuration
        'start_date': '2023-01-01',   # Start date for backtesting
        'end_date': '2023-12-31',     # End date for backtesting
        'commission_pct': 0.1,        # Commission percentage
        'slippage_pips': 1,           # Slippage in pips
        'include_spread': True        # Whether to include spread in backtesting
    },
    
    # Cache Settings
    'cache': {
        'portfolio_file': 'portfolio.json',  # Filename for portfolio state
        'history_days': 90            # Number of days to keep history
    },
    
    # Logging
    'logging': {
        'level': 'INFO',              # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        'to_file': True,              # Whether to log to file
        'log_trades': True,           # Whether to log individual trades
        'log_signals': False          # Whether to log all signals
    }
}

def get_config(config_file=None):
    """
    Get configuration for the Portfolio Manager Agent
    
    Args:
        config_file (str, optional): Path to configuration file. If None, use default configuration.
    
    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_file:
        import json
        import os
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Update configuration with user settings
                _update_nested_dict(config, user_config)
            except Exception as e:
                print(f"Error loading configuration file: {e}")
    
    return config

def _update_nested_dict(d, u):
    """
    Update nested dictionary recursively
    
    Args:
        d (dict): Dictionary to update
        u (dict): Dictionary with updated values
    """
    import collections
    
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def save_config(config, config_file):
    """
    Save configuration to file
    
    Args:
        config (dict): Configuration dictionary
        config_file (str): Path to configuration file
    """
    import json
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4) 