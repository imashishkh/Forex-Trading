#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line interface for the Portfolio Manager Agent
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

from portfolio_manager_agent.agent import PortfolioManagerAgent
from portfolio_manager_agent.config import get_config, save_config
from utils.logger import setup_trading_logger, log_portfolio_summary

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Portfolio Manager Agent CLI')
    
    # Subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the Portfolio Manager Agent')
    run_parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    run_parser.add_argument('--market-data', '-m', type=str, help='Path to market data file or directory')
    run_parser.add_argument('--technical', '-t', type=str, help='Path to technical analysis signals file')
    run_parser.add_argument('--fundamental', '-f', type=str, help='Path to fundamental analysis insights file')
    run_parser.add_argument('--sentiment', '-s', type=str, help='Path to sentiment analysis insights file')
    run_parser.add_argument('--risk', '-r', type=str, help='Path to risk assessment file')
    run_parser.add_argument('--output', '-o', type=str, help='Path to output directory')
    run_parser.add_argument('--backtest', '-b', action='store_true', help='Run in backtest mode')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize configuration file')
    init_parser.add_argument('--output', '-o', type=str, required=True, help='Path to output configuration file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check portfolio status')
    status_parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='View or modify configuration')
    config_parser.add_argument('--view', '-v', action='store_true', help='View current configuration')
    config_parser.add_argument('--set', '-s', nargs=2, action='append', metavar=('KEY', 'VALUE'), 
                              help='Set configuration value (e.g., --set risk.max_risk_per_trade_pct 1.5)')
    config_parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    
    return parser.parse_args()

def load_data(file_path, data_type):
    """
    Load data from file
    
    Args:
        file_path (str): Path to data file
        data_type (str): Type of data ('market', 'technical', 'fundamental', 'sentiment', 'risk')
    
    Returns:
        dict: Loaded data
    """
    if not file_path or not os.path.exists(file_path):
        print(f"Error: {data_type} data file '{file_path}' not found.")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process market data
        if data_type == 'market' and isinstance(data, dict):
            for symbol, symbol_data in data.items():
                if isinstance(symbol_data, list):
                    # Convert list of dict/list to DataFrame
                    data[symbol] = pd.DataFrame(symbol_data)
        
        return data
    except Exception as e:
        print(f"Error loading {data_type} data: {e}")
        return {}

def save_results(results, output_dir):
    """
    Save results to output directory
    
    Args:
        results (dict): Results from portfolio manager
        output_dir (str): Output directory
    """
    if not output_dir:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save portfolio summary
    summary_file = os.path.join(output_dir, f'portfolio_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(results['portfolio_summary'], f, indent=2)
    
    # Save performance metrics
    metrics_file = os.path.join(output_dir, f'performance_metrics_{timestamp}.json')
    with open(metrics_file, 'w') as f:
        json.dump(results['performance_metrics'], f, indent=2)
    
    # Save trades
    trades_file = os.path.join(output_dir, f'trades_{timestamp}.json')
    with open(trades_file, 'w') as f:
        json.dump({
            'position_updates': results['position_updates'],
            'new_trades': results['new_trades']
        }, f, indent=2)
    
    print(f"Results saved to {output_dir}")

def display_status(portfolio_manager):
    """
    Display portfolio status
    
    Args:
        portfolio_manager (PortfolioManagerAgent): Portfolio manager instance
    """
    portfolio = portfolio_manager.portfolio
    
    # Display portfolio summary
    print("\n=== Portfolio Summary ===")
    print(f"Account Balance: ${portfolio['account_balance']:.2f}")
    print(f"Equity: ${portfolio['equity']:.2f}")
    print(f"Used Margin: ${portfolio['used_margin']:.2f}")
    print(f"Free Margin: ${portfolio['free_margin']:.2f}")
    print(f"Margin Level: {portfolio['margin_level']:.2f}%")
    
    # Display open positions
    print("\n=== Open Positions ===")
    if not portfolio['open_positions']:
        print("No open positions")
    else:
        for position in portfolio['open_positions']:
            print(f"{position['symbol']} ({position['direction'].upper()}):")
            print(f"  Entry: {position['entry_price']:.5f} | Current: {position['current_price']:.5f}")
            print(f"  P&L: ${position['current_pnl']:.2f} ({position['current_pnl_percentage']:.2f}%)")
            print(f"  Size: {position['size']:.2f} | Value: ${position['value']:.2f}")
            print(f"  SL: {position['stop_loss']:.5f} | TP: {position['take_profit']:.5f}")
            print()
    
    # Display performance metrics
    if portfolio['closed_positions']:
        win_count = sum(1 for p in portfolio['closed_positions'] if p['pnl'] > 0)
        loss_count = len(portfolio['closed_positions']) - win_count
        win_rate = (win_count / len(portfolio['closed_positions'])) * 100
        
        print("=== Performance ===")
        print(f"Total Trades: {len(portfolio['closed_positions'])}")
        print(f"Win Rate: {win_rate:.2f}% ({win_count} wins, {loss_count} losses)")
        
        if portfolio['closed_positions']:
            total_profit = sum(p['pnl'] for p in portfolio['closed_positions'] if p['pnl'] > 0)
            total_loss = sum(p['pnl'] for p in portfolio['closed_positions'] if p['pnl'] < 0)
            total_pnl = total_profit + total_loss
            
            print(f"Total P&L: ${total_pnl:.2f}")
            print(f"Total Profit: ${total_profit:.2f}")
            print(f"Total Loss: ${total_loss:.2f}")
            
            if total_loss != 0:
                profit_factor = abs(total_profit / total_loss)
                print(f"Profit Factor: {profit_factor:.2f}")

def get_nested_dict_value(d, key_path):
    """
    Get value from nested dictionary using dot notation
    
    Args:
        d (dict): Dictionary
        key_path (str): Key path in dot notation (e.g., 'risk.max_risk_per_trade_pct')
    
    Returns:
        Any: Value at key path or None if not found
    """
    keys = key_path.split('.')
    current = d
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    
    return current

def set_nested_dict_value(d, key_path, value):
    """
    Set value in nested dictionary using dot notation
    
    Args:
        d (dict): Dictionary
        key_path (str): Key path in dot notation (e.g., 'risk.max_risk_per_trade_pct')
        value (Any): Value to set
    
    Returns:
        bool: True if successful, False otherwise
    """
    keys = key_path.split('.')
    current = d
    
    # Navigate to the parent dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            return False
        current = current[key]
    
    # Set the value
    try:
        # Try to convert value to appropriate type
        last_key = keys[-1]
        if last_key in current and isinstance(current[last_key], (int, float)):
            # Convert to number
            try:
                if isinstance(current[last_key], int):
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                pass
        elif value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        
        current[last_key] = value
        return True
    except Exception:
        return False

def main():
    """
    Main function
    """
    args = parse_args()
    
    if args.command == 'init':
        # Initialize configuration file
        config = get_config()
        save_config(config, args.output)
        print(f"Initialized configuration file at {args.output}")
        return
    
    # Get configuration
    config_file = getattr(args, 'config', None)
    config = get_config(config_file)
    
    if args.command == 'config':
        # View or modify configuration
        if args.view:
            print(json.dumps(config, indent=2))
        
        if args.set:
            for key, value in args.set:
                if set_nested_dict_value(config, key, value):
                    print(f"Set {key} = {value}")
                else:
                    print(f"Error: Could not set {key}")
            
            if config_file:
                save_config(config, config_file)
                print(f"Updated configuration saved to {config_file}")
            else:
                print("Warning: Configuration file not specified. Changes not saved.")
        
        return
    
    # Create portfolio manager
    logger = setup_trading_logger('portfolio_manager')
    portfolio_manager = PortfolioManagerAgent(config)
    
    if args.command == 'status':
        # Display portfolio status
        display_status(portfolio_manager)
        return
    
    if args.command == 'run':
        # Load data
        market_data = load_data(args.market_data, 'market')
        technical_signals = load_data(args.technical, 'technical')
        fundamental_insights = load_data(args.fundamental, 'fundamental')
        sentiment_insights = load_data(args.sentiment, 'sentiment')
        risk_assessment = load_data(args.risk, 'risk')
        
        if not market_data:
            print("Error: Market data is required")
            return
        
        # Run portfolio manager
        results = portfolio_manager.manage_portfolio(
            market_data,
            technical_signals,
            fundamental_insights,
            sentiment_insights,
            risk_assessment
        )
        
        # Display results
        print("\n=== Portfolio Management Results ===")
        print(f"Portfolio Summary: {json.dumps(results['portfolio_summary'], indent=2)}")
        
        if results['position_updates']:
            print(f"\nPosition Updates: {len(results['position_updates'])}")
            for update in results['position_updates'][:5]:  # Show first 5
                print(f"- {update}")
            if len(results['position_updates']) > 5:
                print(f"...and {len(results['position_updates']) - 5} more")
        
        if results['new_trades']:
            print(f"\nNew Trades: {len(results['new_trades'])}")
            for trade in results['new_trades']:
                print(f"- {trade['direction'].upper()} {trade['symbol']} @ {trade['price']:.5f}")
        
        # Save results
        if args.output:
            save_results(results, args.output)

if __name__ == '__main__':
    main() 