#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WalletManager Demo

This script demonstrates the functionality of the WalletManager class by connecting to
a real OANDA account and performing various account management operations.
"""

import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from tabulate import tabulate
import pandas as pd

# Import the WalletManager class
from wallet_manager import WalletManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WalletManagerDemo")

def print_section(title):
    """Print a section title with a horizontal line."""
    line = "-" * 60
    print(f"\n{line}\n{title}\n{line}")
    
def display_dict_as_table(data, headers=None):
    """Display a dictionary as a formatted table."""
    if headers is None:
        headers = ["Property", "Value"]
    
    table_data = []
    for key, value in data.items():
        if isinstance(value, dict):
            value = str(value)
        table_data.append([key, value])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    """Main function to run the WalletManager demo."""
    # Load environment variables
    load_dotenv()
    
    print_section("WALLET MANAGER DEMO")
    print("Initializing WalletManager and connecting to OANDA account...")
    
    # Initialize the WalletManager
    wallet_manager = WalletManager()
    
    # Connect to the OANDA account
    connected = wallet_manager.connect_to_oanda_account()
    
    if not connected:
        print("Failed to connect to OANDA account. Please check your credentials.")
        return
    
    print("✅ Successfully connected to OANDA account")
    
    # Display account details
    print_section("ACCOUNT DETAILS")
    account_details = wallet_manager.get_account_details()
    display_dict_as_table({
        "Account ID": wallet_manager.account_id,
        "Account Name": account_details.get("alias", "Unknown"),
        "Currency": account_details.get("currency", "Unknown"),
        "Balance": account_details.get("balance", "Unknown"),
        "Created At": account_details.get("createdTime", "Unknown"),
    })
    
    # Display account summary
    print_section("ACCOUNT SUMMARY")
    account_summary = wallet_manager.get_account_summary()
    display_dict_as_table({
        "Balance": account_summary.get("balance", "Unknown"),
        "NAV": account_summary.get("NAV", "Unknown"),
        "Margin Used": account_summary.get("marginUsed", "Unknown"),
        "Margin Available": account_summary.get("marginAvailable", "Unknown"),
        "Open Trade Count": account_summary.get("openTradeCount", "Unknown"),
        "Open Position Count": account_summary.get("openPositionCount", "Unknown"),
        "Unrealized PL": account_summary.get("unrealizedPL", "Unknown"),
    })
    
    # Get open positions
    print_section("OPEN POSITIONS")
    # Use the export_account_data method to get open positions
    export_data = wallet_manager.export_account_data("./")
    open_positions = export_data.get("open_positions", [])
    
    if not open_positions:
        print("No open positions found.")
    else:
        positions_data = []
        for position in open_positions:
            instrument = position.get("instrument", "Unknown")
            long_units = float(position.get("long", {}).get("units", 0))
            short_units = float(position.get("short", {}).get("units", 0))
            units = long_units + short_units  # Will be negative for short positions
            direction = "LONG" if units > 0 else "SHORT"
            unrealized_pl = float(position.get("unrealizedPL", 0))
            
            positions_data.append([
                instrument,
                direction,
                abs(units),
                unrealized_pl
            ])
        
        print(tabulate(positions_data, 
                      headers=["Instrument", "Direction", "Units", "Unrealized PL"],
                      tablefmt="grid"))
    
    # Get open orders
    print_section("OPEN ORDERS")
    # Use the export_account_data method to get open orders
    open_orders = export_data.get("open_orders", [])
    
    if not open_orders:
        print("No open orders found.")
    else:
        orders_data = []
        for order in open_orders:
            order_id = order.get("id", "Unknown")
            instrument = order.get("instrument", "Unknown")
            order_type = order.get("type", "Unknown")
            units = order.get("units", "0")
            price = order.get("price", "Market")
            
            orders_data.append([
                order_id,
                instrument,
                order_type,
                units,
                price
            ])
        
        print(tabulate(orders_data, 
                      headers=["Order ID", "Instrument", "Type", "Units", "Price"],
                      tablefmt="grid"))
    
    # Get recent transaction history
    print_section("RECENT TRANSACTIONS")
    from_time = datetime.utcnow() - timedelta(days=7)
    transactions = wallet_manager.get_transaction_history(
        from_time=from_time,
        count=10
    )
    
    if not transactions:
        print("No recent transactions found.")
    else:
        tx_data = []
        for tx in transactions:
            tx_id = tx.get("id", "Unknown")
            tx_type = tx.get("type", "Unknown")
            tx_time = tx.get("time", "Unknown")
            instrument = tx.get("instrument", "N/A")
            units = tx.get("units", "N/A")
            pl = tx.get("pl", "N/A")
            
            tx_data.append([
                tx_id,
                tx_type,
                tx_time,
                instrument,
                units,
                pl
            ])
        
        print(tabulate(tx_data, 
                      headers=["ID", "Type", "Time", "Instrument", "Units", "P/L"],
                      tablefmt="grid"))
    
    # Check margin health
    print_section("MARGIN HEALTH")
    margin_health = wallet_manager.check_margin_health()
    display_dict_as_table(margin_health)
    
    # Calculate realized PnL
    print_section("PROFIT AND LOSS")
    realized_pnl_7d = wallet_manager.calculate_realized_pnl(timeframe="7d")
    realized_pnl_30d = wallet_manager.calculate_realized_pnl(timeframe="30d")
    
    display_dict_as_table({
        "Realized P/L (7 days)": realized_pnl_7d,
        "Realized P/L (30 days)": realized_pnl_30d,
    })
    
    # Check drawdown protection
    print_section("RISK METRICS")
    drawdown = wallet_manager.check_drawdown_protection()
    
    # Set max drawdown percentage if not already set
    if not hasattr(wallet_manager, 'max_drawdown_percentage') or wallet_manager.max_drawdown_percentage is None:
        wallet_manager.set_max_drawdown_protection(0.10)  # 10% default
    
    display_dict_as_table({
        "Current Drawdown": f"{drawdown.get('current_drawdown', 0) * 100:.2f}%",
        "Max Allowed Drawdown": f"{wallet_manager.max_drawdown_percentage * 100:.2f}%",
        "Action Needed": drawdown.get('action_needed', False),
    })
    
    print_section("DEMO COMPLETED")
    print("WalletManager functionality demonstrated successfully.")

if __name__ == "__main__":
    main() 