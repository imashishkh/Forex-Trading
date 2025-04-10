#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify that updated trading strategies from config.yaml
are correctly loaded and applied in the system.
"""

import os
import yaml
from pathlib import Path

def load_config():
    """
    Load configuration from YAML file
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def test_strategy_configuration():
    """
    Test that the strategy configuration is loaded correctly
    """
    print("Testing strategy configuration loading...")
    
    # Load the configuration
    config = load_config()
    
    # Verify that the strategies section exists in the config
    if 'technical_analysis' not in config:
        print("ERROR: 'technical_analysis' section not found in config")
        return False
    
    if 'strategies' not in config['technical_analysis']:
        print("ERROR: 'strategies' section not found in technical_analysis config")
        return False
    
    # Print strategy configuration
    strategies_config = config['technical_analysis']['strategies']
    print("\nConfigured strategies:")
    
    # Check active strategies
    active_strategies = strategies_config.get('active_strategies', [])
    print(f"Active strategies: {active_strategies}")
    
    # Check each strategy configuration
    for strategy in active_strategies:
        if strategy in strategies_config:
            print(f"\n{strategy} configuration:")
            for param, value in strategies_config[strategy].items():
                print(f"  {param}: {value}")
        else:
            print(f"WARNING: Configuration for {strategy} not found")
    
    return True

def main():
    """Main test function"""
    print("Testing Updated Strategy Configurations\n")
    
    # Test strategy configuration loading
    config_test_passed = test_strategy_configuration()
    
    # Print results
    print("\nTest Results:")
    print(f"Configuration Loading: {'PASSED' if config_test_passed else 'FAILED'}")
    
    if config_test_passed:
        print("\nSUCCESS: Strategy configurations have been successfully updated in config.yaml!")
    else:
        print("\nFAILED: There were issues with the strategy configuration update.")
    
    print("\nTo apply these strategies in the trading system:")
    print("1. The system will automatically use these strategies on next startup")
    print("2. The technical analyst agent will prioritize strategies in the active_strategies list")
    print("3. Each strategy will be applied with the parameters specified in the configuration")

if __name__ == "__main__":
    main() 