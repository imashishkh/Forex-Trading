#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Manager for Forex Trading Platform

This module provides a centralized configuration management system
that loads configuration from multiple sources, validates it,
and provides a consistent interface for accessing configuration values.
"""

import os
import json
import argparse
import logging
from typing import Any, Dict, List, Optional, Union, Set, TypeVar, cast
from pathlib import Path
import copy
import re
from datetime import datetime

# For schema validation
try:
    import jsonschema
    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Type definitions
T = TypeVar('T')
ConfigDict = Dict[str, Any]

class ConfigManager:
    """
    Singleton configuration manager for the forex trading platform.
    
    This class handles loading configuration from multiple sources,
    validating it, and providing a consistent interface for accessing
    configuration values throughout the application.
    """
    
    # Singleton instance
    _instance = None
    
    # Configuration schema definition
    # This defines the structure and validation rules for the configuration
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "api_credentials": {
                "type": "object",
                "properties": {
                    "openai": {
                        "type": "object",
                        "properties": {
                            "api_key": {"type": "string"},
                            "model": {"type": "string", "default": "gpt-3.5-turbo"}
                        },
                        "required": ["api_key"]
                    },
                    "oanda": {
                        "type": "object",
                        "properties": {
                            "api_key": {"type": "string"},
                            "account_id": {"type": "string"},
                            "api_url": {"type": "string"}
                        },
                        "required": ["api_key", "account_id", "api_url"]
                    }
                }
            },
            "trading": {
                "type": "object",
                "properties": {
                    "risk_per_trade": {"type": "number", "minimum": 0.01, "maximum": 10.0},
                    "max_open_trades": {"type": "integer", "minimum": 1},
                    "default_stop_loss": {"type": "number", "minimum": 0.1},
                    "default_take_profit": {"type": "number", "minimum": 0.1},
                    "allowed_instruments": {"type": "array", "items": {"type": "string"}},
                    "trading_sessions": {"type": "array", "items": {"type": "string", "enum": ["asian", "london", "new_york", "all"]}},
                    "paper_trading_mode": {"type": "boolean"}
                },
                "required": ["risk_per_trade", "max_open_trades", "default_stop_loss", "default_take_profit"]
            },
            "risk_management": {
                "type": "object",
                "properties": {
                    "max_daily_loss": {"type": "number", "minimum": 0.1},
                    "max_drawdown": {"type": "number", "minimum": 0.1},
                    "risk_to_reward_minimum": {"type": "number", "minimum": 0.5},
                    "correlation_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "max_risk_per_currency": {"type": "number", "minimum": 0.1},
                    "position_sizing_model": {"type": "string", "enum": ["fixed", "fixed_dollar", "volatility", "kelly"]}
                },
                "required": ["max_daily_loss", "max_drawdown"]
            },
            "system": {
                "type": "object",
                "properties": {
                    "data_storage_path": {"type": "string"},
                    "cache_enabled": {"type": "boolean"},
                    "cache_expiry_days": {"type": "integer", "minimum": 1},
                    "timezone": {"type": "string"},
                    "debug_mode": {"type": "boolean"},
                    "lang_graph_enabled": {"type": "boolean"},
                    "langsmith_enabled": {"type": "boolean"},
                    "langsmith_project": {"type": "string"}
                },
                "required": ["data_storage_path"]
            },
            "logging": {
                "type": "object",
                "properties": {
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "log_file_path": {"type": "string"},
                    "log_rotation": {"type": "boolean"},
                    "log_retention_days": {"type": "integer", "minimum": 1},
                    "trade_logging": {"type": "boolean"},
                    "performance_logging": {"type": "boolean"},
                    "console_log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]}
                },
                "required": ["log_level"]
            },
            "agents": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "model_name": {"type": "string"},
                        "temperature": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "specific_settings": {"type": "object"}
                    },
                    "required": ["enabled"]
                }
            }
        },
        "required": ["trading", "system", "logging"]
    }
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "api_credentials": {
            "openai": {
                "model": "gpt-3.5-turbo"
            },
            "oanda": {
                "api_url": "https://api-fxpractice.oanda.com"  # Default to practice
            }
        },
        "trading": {
            "risk_per_trade": 1.0,  # 1% of account size per trade
            "max_open_trades": 5,
            "default_stop_loss": 1.5,  # 1.5% stop loss
            "default_take_profit": 3.0,  # 3% take profit (2:1 reward-to-risk)
            "allowed_instruments": [
                "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", 
                "USD_CAD", "EUR_JPY", "GBP_JPY"
            ],
            "trading_sessions": ["all"],
            "paper_trading_mode": True
        },
        "risk_management": {
            "max_daily_loss": 3.0,  # 3% max daily loss
            "max_drawdown": 10.0,  # 10% max drawdown
            "risk_to_reward_minimum": 1.5,
            "correlation_threshold": 0.7,
            "max_risk_per_currency": 5.0,  # 5% max risk per currency
            "position_sizing_model": "fixed"
        },
        "system": {
            "data_storage_path": "data",
            "cache_enabled": True,
            "cache_expiry_days": 7,
            "timezone": "UTC",
            "debug_mode": False,
            "lang_graph_enabled": True,
            "langsmith_enabled": False
        },
        "logging": {
            "log_level": "INFO",
            "log_file_path": "logs/forex_trading.log",
            "log_rotation": True,
            "log_retention_days": 30,
            "trade_logging": True,
            "performance_logging": True,
            "console_log_level": "INFO"
        },
        "agents": {
            "portfolio_manager": {
                "enabled": True,
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.1,
                "specific_settings": {
                    "portfolio_rebalance_frequency": "daily",
                    "risk_adjustment_enabled": True
                }
            },
            "market_analyzer": {
                "enabled": True,
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.2,
                "specific_settings": {
                    "indicators": ["RSI", "MACD", "Moving_Average"],
                    "timeframes": ["1h", "4h", "1d"]
                }
            },
            "news_analyzer": {
                "enabled": True,
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.3,
                "specific_settings": {
                    "news_sources": ["forexfactory", "investing.com", "bloomberg"],
                    "sentiment_analysis": True
                }
            }
        }
    }
    
    def __new__(cls, *args, **kwargs):
        """
        Implement singleton pattern.
        
        Returns:
            ConfigManager: Singleton instance of ConfigManager
        """
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_file: Optional[str] = None, cmd_args: Optional[List[str]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to a JSON configuration file
            cmd_args: Command-line arguments to parse
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        self._logger = logging.getLogger("ConfigManager")
        
        # Initialize configuration with defaults
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Track modified configuration keys
        self._modified_keys: Set[str] = set()
        
        # Load configuration from sources in order of increasing precedence
        self._load_from_env_vars()
        
        if config_file:
            self.load_config_file(config_file)
            
        if cmd_args:
            self._load_from_cmd_args(cmd_args)
        
        # Validate the configuration
        self.validate_config()
        
        # Set up config modification timestamp tracking
        self._last_modified = datetime.now()
        
        self._initialized = True
        
    def _load_from_env_vars(self) -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with FOREX_
        and use double underscore to indicate nesting, e.g.,
        FOREX_TRADING__RISK_PER_TRADE=1.5 for trading.risk_per_trade
        """
        prefix = "FOREX_"
        pattern = re.compile(f"^{prefix}(.+)$")
        
        for key, value in os.environ.items():
            match = pattern.match(key)
            if not match:
                continue
                
            config_key = match.group(1)
            
            # Convert double underscore to nested keys
            parts = config_key.split("__")
            
            # Convert string value to appropriate type
            typed_value = self._convert_string_value(value)
            
            # Set in configuration
            self._set_nested_value(self._config, parts, typed_value)
            self._modified_keys.add(".".join(parts))
            
    def _convert_string_value(self, value: str) -> Any:
        """
        Convert string values to appropriate types.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value with appropriate type
        """
        # Try to convert to boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
            
        # Try to convert to number
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
            
        # Try to convert to list (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]
            
        # Return as string
        return value
        
    def _load_from_cmd_args(self, args: List[str]) -> None:
        """
        Load configuration from command-line arguments.
        
        Args:
            args: Command-line arguments to parse
        """
        parser = argparse.ArgumentParser(description="Forex Trading Platform")
        
        # Add common arguments
        parser.add_argument("--config", help="Path to configuration file")
        parser.add_argument("--risk-per-trade", type=float, help="Risk percentage per trade")
        parser.add_argument("--max-open-trades", type=int, help="Maximum number of open trades")
        parser.add_argument("--paper-trading", action="store_true", help="Enable paper trading mode")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
        
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        # Update configuration
        if parsed_args.risk_per_trade is not None:
            self.set_config("trading.risk_per_trade", parsed_args.risk_per_trade)
            
        if parsed_args.max_open_trades is not None:
            self.set_config("trading.max_open_trades", parsed_args.max_open_trades)
            
        if parsed_args.paper_trading:
            self.set_config("trading.paper_trading_mode", True)
            
        if parsed_args.debug:
            self.set_config("system.debug_mode", True)
            self.set_config("logging.log_level", "DEBUG")
            self.set_config("logging.console_log_level", "DEBUG")
            
        if parsed_args.log_level:
            self.set_config("logging.log_level", parsed_args.log_level)
            
    def _get_nested_value(self, config: ConfigDict, keys: List[str], default: Any = None) -> Any:
        """
        Get a nested value from the configuration dictionary.
        
        Args:
            config: Configuration dictionary
            keys: List of keys to navigate the nested structure
            default: Default value to return if key not found
            
        Returns:
            Value from the configuration or default
        """
        current = config
        
        for key in keys:
            if not isinstance(current, dict):
                return default
                
            if key not in current:
                return default
                
            current = current[key]
            
        return current
        
    def _set_nested_value(self, config: ConfigDict, keys: List[str], value: Any) -> None:
        """
        Set a nested value in the configuration dictionary.
        
        Args:
            config: Configuration dictionary
            keys: List of keys to navigate the nested structure
            value: Value to set
        """
        current = config
        
        # Navigate to the parent of the final key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
                
            current = current[key]
            
        # Set the value
        current[keys[-1]] = value
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot notation for nested keys)
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        # Split key into parts
        parts = key.split(".")
        
        # Get nested value
        return self._get_nested_value(self._config, parts, default)
        
    def set_config(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot notation for nested keys)
            value: Value to set
        """
        # Split key into parts
        parts = key.split(".")
        
        # Set nested value
        self._set_nested_value(self._config, parts, value)
        
        # Track modification
        self._modified_keys.add(key)
        self._last_modified = datetime.now()
        
    def load_config_file(self, filepath: str) -> bool:
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to the configuration file (JSON)
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        filepath = os.path.expanduser(filepath)
        
        try:
            with open(filepath, 'r') as f:
                file_config = json.load(f)
                
            # Update configuration
            self._update_config_recursive(self._config, file_config)
            
            self._logger.info(f"Loaded configuration from {filepath}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to load configuration from {filepath}: {e}")
            return False
            
    def _update_config_recursive(self, target: ConfigDict, source: ConfigDict) -> None:
        """
        Update configuration recursively.
        
        Args:
            target: Target configuration dictionary
            source: Source configuration dictionary
        """
        for key, value in source.items():
            # If both values are dictionaries, update recursively
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config_recursive(target[key], value)
            else:
                # Otherwise, just update the value
                target[key] = copy.deepcopy(value)
                
                # Track the dotted key path for this change
                self._modified_keys.add(self._get_dotted_key_path(target, key))
                
    def _get_dotted_key_path(self, target: ConfigDict, key: str) -> str:
        """
        Get the full dotted key path for a modified key.
        This is a best-effort method and may not always work correctly.
        
        Args:
            target: Target dictionary containing the key
            key: The key that was modified
            
        Returns:
            str: Dotted key path
        """
        # This is a simplistic implementation that doesn't handle all cases
        # A more robust solution would track the path during recursion
        for full_key in self._modified_keys:
            if full_key.endswith(f".{key}") or full_key == key:
                return full_key
                
        return key
                
    def save_config_file(self, filepath: str) -> bool:
        """
        Save current configuration to a file.
        
        Args:
            filepath: Path to the configuration file (JSON)
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        filepath = os.path.expanduser(filepath)
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=2)
                
            self._logger.info(f"Saved configuration to {filepath}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save configuration to {filepath}: {e}")
            return False
            
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration specific to an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dict[str, Any]: Agent-specific configuration
        """
        # Get agent-specific configuration with fallback to empty dict
        agent_config = self.get_config(f"agents.{agent_name}", {})
        
        # Create a new dictionary with agent configuration and some global settings
        config = {
            "agent_name": agent_name,
            "trading": self.get_config("trading", {}),
            "risk_management": self.get_config("risk_management", {}),
            "system": self.get_config("system", {}),
            "logging": self.get_config("logging", {})
        }
        
        # Add API credentials if needed
        if agent_name == "portfolio_manager":
            config["api_credentials"] = {
                "oanda": self.get_config("api_credentials.oanda", {})
            }
            
        if "llm" in agent_config.get("specific_settings", {}):
            config["api_credentials"] = {
                "openai": self.get_config("api_credentials.openai", {})
            }
            
        # Add agent-specific configuration
        config.update(agent_config)
        
        return config
        
    def validate_config(self) -> bool:
        """
        Ensure all required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if not SCHEMA_VALIDATION_AVAILABLE:
            self._logger.warning("jsonschema package not available, skipping configuration validation")
            return True
            
        try:
            jsonschema.validate(instance=self._config, schema=self.CONFIG_SCHEMA)
            self._logger.debug("Configuration validated successfully")
            return True
        except jsonschema.exceptions.ValidationError as e:
            self._logger.error(f"Configuration validation failed: {e}")
            
            # Log the specific error path
            path = ".".join(str(p) for p in e.path)
            self._logger.error(f"Validation failed at path: {path}")
            self._logger.error(f"Schema: {e.schema}")
            
            return False
            
    def get_modified_keys(self) -> Set[str]:
        """
        Get the set of modified configuration keys.
        
        Returns:
            Set[str]: Set of modified keys
        """
        return set(self._modified_keys)
        
    def get_last_modified_time(self) -> datetime:
        """
        Get the timestamp of the last configuration modification.
        
        Returns:
            datetime: Last modification timestamp
        """
        return self._last_modified
        
    def reset_to_defaults(self) -> None:
        """
        Reset the configuration to default values.
        """
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._modified_keys = set()
        self._last_modified = datetime.now()
        
    def as_dict(self) -> Dict[str, Any]:
        """
        Get a copy of the entire configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Copy of the configuration
        """
        return copy.deepcopy(self._config)
        
    def update_from_dict(self, config: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.
        
        Args:
            config: Dictionary with configuration values
        """
        self._update_config_recursive(self._config, config)
        self._last_modified = datetime.now()
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section: Section name (top-level key)
            
        Returns:
            Dict[str, Any]: Configuration section
        """
        return copy.deepcopy(self.get_config(section, {}))
        
    @property
    def api_credentials(self) -> Dict[str, Any]:
        """Get API credentials configuration section."""
        return self.get_section("api_credentials")
        
    @property
    def trading(self) -> Dict[str, Any]:
        """Get trading configuration section."""
        return self.get_section("trading")
        
    @property
    def risk_management(self) -> Dict[str, Any]:
        """Get risk management configuration section."""
        return self.get_section("risk_management")
        
    @property
    def system(self) -> Dict[str, Any]:
        """Get system configuration section."""
        return self.get_section("system")
        
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.get_section("logging")
        
    @property
    def agents(self) -> Dict[str, Any]:
        """Get agents configuration section."""
        return self.get_section("agents")
        
    def __str__(self) -> str:
        """
        String representation of the configuration.
        
        Returns:
            str: String representation
        """
        return f"ConfigManager(keys={len(self._config)}, modified={len(self._modified_keys)})"
        
    def __repr__(self) -> str:
        """
        Detailed string representation.
        
        Returns:
            str: Detailed string representation
        """
        return f"ConfigManager(modified={sorted(self._modified_keys)}, last_modified={self._last_modified})" 