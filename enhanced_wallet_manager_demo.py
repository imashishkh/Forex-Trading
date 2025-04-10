#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced WalletManager Demo

This script provides a real-time interactive demonstration of the WalletManager
class, showing account status, positions, orders, and risk metrics for an OANDA
trading account.
"""

import os
import sys
import time
import threading
import curses
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Any, Optional
from tabulate import tabulate
from dotenv import load_dotenv
import logging
import pandas as pd

# Import the WalletManager class
from wallet_manager import WalletManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_wallet_manager_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedWalletManagerDemo")

class EnhancedWalletManagerDemo:
    """
    Real-time interactive demo for the WalletManager class.
    Provides a comprehensive view of an OANDA trading account.
    """
    
    def __init__(self):
        """Initialize the demo with a WalletManager instance."""
        load_dotenv()
        self.wallet_manager = WalletManager()
        self.refresh_interval = 5  # seconds
        self.should_exit = False
        self.data = {
            "account_details": {},
            "account_summary": {},
            "open_positions": [],
            "open_orders": [],
            "recent_transactions": [],
            "margin_health": {},
            "risk_metrics": {},
            "instruments": []
        }
        self.connected = False
        self.update_thread = None
        self.last_updated = None
    
    def connect(self) -> bool:
        """Connect to the OANDA account."""
        self.connected = self.wallet_manager.connect_to_oanda_account()
        return self.connected
    
    def update_data(self) -> None:
        """Update all data from the OANDA account."""
        try:
            # Get account details
            self.data["account_details"] = self.wallet_manager.get_account_details()
            
            # Get account summary
            self.data["account_summary"] = self.wallet_manager.get_account_summary()
            
            # Get open positions and orders using export_account_data
            export_data = self.wallet_manager.export_account_data("./tmp_export")
            self.data["open_positions"] = export_data.get("open_positions", [])
            self.data["open_orders"] = export_data.get("open_orders", [])
            
            # Delete temporary export file
            if os.path.exists("./tmp_export"):
                os.remove("./tmp_export")
            
            # Get recent transactions (last day)
            from_time = datetime.now(UTC) - timedelta(days=1)
            self.data["recent_transactions"] = self.wallet_manager.get_transaction_history(
                from_time=from_time,
                count=10
            )
            
            # Get margin health
            self.data["margin_health"] = self.wallet_manager.check_margin_health()
            
            # Get risk metrics (drawdown)
            self.data["risk_metrics"] = self.wallet_manager.check_drawdown_protection()
            
            # Get instruments (top 5 most traded)
            all_instruments = self.wallet_manager.get_account_instruments()
            # Sort by pip location as a proxy for popularity (lower is more popular)
            all_instruments.sort(key=lambda x: x.get('pipLocation', 0))
            self.data["instruments"] = all_instruments[:5]
            
            # Update last updated timestamp
            self.last_updated = datetime.now(UTC)
        except Exception as e:
            logger.error(f"Error updating data: {str(e)}")
    
    def start_update_thread(self) -> None:
        """Start a background thread to update data periodically."""
        if self.update_thread is not None and self.update_thread.is_alive():
            return
        
        def update_job():
            while not self.should_exit:
                self.update_data()
                time.sleep(self.refresh_interval)
        
        self.update_thread = threading.Thread(target=update_job)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_update_thread(self) -> None:
        """Stop the background update thread."""
        self.should_exit = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
    
    def format_account_details(self) -> str:
        """Format account details for display."""
        details = self.data["account_details"]
        summary = self.data["account_summary"]
        
        if not details or not summary:
            return "No account details available."
        
        data = {
            "Account ID": self.wallet_manager.account_id,
            "Account Name": details.get("alias", "Unknown"),
            "Currency": details.get("currency", "Unknown"),
            "Balance": f"{float(summary.get('balance', 0)):.2f}",
            "NAV": f"{float(summary.get('NAV', 0)):.2f}",
            "Margin Used": f"{float(summary.get('marginUsed', 0)):.2f}",
            "Margin Available": f"{float(summary.get('marginAvailable', 0)):.2f}",
            "Unrealized P/L": f"{float(summary.get('unrealizedPL', 0)):.2f}",
            "Open Positions": summary.get("openPositionCount", 0),
            "Open Orders": summary.get("openTradeCount", 0)
        }
        
        table_data = [[k, v] for k, v in data.items()]
        return tabulate(table_data, headers=["Property", "Value"], tablefmt="grid")
    
    def format_positions(self) -> str:
        """Format open positions for display."""
        positions = self.data["open_positions"]
        
        if not positions:
            return "No open positions."
        
        positions_data = []
        for position in positions:
            instrument = position.get("instrument", "Unknown")
            long_units = float(position.get("long", {}).get("units", 0))
            short_units = float(position.get("short", {}).get("units", 0))
            units = long_units + short_units  # Will be negative for short positions
            direction = "LONG" if units > 0 else "SHORT"
            avg_price = float(position.get("long" if units > 0 else "short", {}).get("averagePrice", 0))
            unrealized_pl = float(position.get("unrealizedPL", 0))
            
            positions_data.append([
                instrument,
                direction,
                abs(units),
                f"{avg_price:.5f}",
                f"{unrealized_pl:.2f}"
            ])
        
        return tabulate(positions_data, 
                        headers=["Instrument", "Direction", "Units", "Avg Price", "Unrealized P/L"],
                        tablefmt="grid")
    
    def format_orders(self) -> str:
        """Format open orders for display."""
        orders = self.data["open_orders"]
        
        if not orders:
            return "No open orders."
        
        orders_data = []
        for order in orders:
            order_id = order.get("id", "Unknown")
            instrument = order.get("instrument", "Unknown")
            order_type = order.get("type", "Unknown")
            units = order.get("units", "0")
            price = order.get("price", "Market")
            time_in_force = order.get("timeInForce", "Unknown")
            
            orders_data.append([
                order_id[:8] + "...",  # Truncate ID for display
                instrument,
                order_type,
                units,
                price,
                time_in_force
            ])
        
        return tabulate(orders_data, 
                       headers=["Order ID", "Instrument", "Type", "Units", "Price", "Time in Force"],
                       tablefmt="grid")
    
    def format_transactions(self) -> str:
        """Format recent transactions for display."""
        transactions = self.data["recent_transactions"]
        
        if not transactions:
            return "No recent transactions."
        
        tx_data = []
        for tx in transactions:
            tx_id = tx.get("id", "Unknown")
            tx_type = tx.get("type", "Unknown")
            tx_time = tx.get("time", "Unknown")
            instrument = tx.get("instrument", "N/A")
            units = tx.get("units", "N/A")
            pl = tx.get("pl", "N/A")
            
            tx_data.append([
                tx_id[:8] + "...",  # Truncate ID for display
                tx_type,
                tx_time.split('T')[0] + " " + tx_time.split('T')[1][:8],  # Format datetime
                instrument,
                units,
                pl
            ])
        
        return tabulate(tx_data, 
                       headers=["ID", "Type", "Time", "Instrument", "Units", "P/L"],
                       tablefmt="grid")
    
    def format_risk_metrics(self) -> str:
        """Format risk metrics for display."""
        margin_health = self.data["margin_health"]
        risk_metrics = self.data["risk_metrics"]
        
        if not margin_health or not risk_metrics:
            return "No risk metrics available."
        
        data = {
            "Margin Status": margin_health.get("status", "Unknown"),
            "Risk Level": margin_health.get("risk_level", "Unknown"),
            "Margin Usage": f"{margin_health.get('margin_usage_ratio', 0) * 100:.2f}%",
            "Margin Used": f"{float(margin_health.get('margin_used', 0)):.2f}",
            "Current Drawdown": f"{risk_metrics.get('current_drawdown', 0) * 100:.2f}%",
            "Max Allowed Drawdown": f"{self.wallet_manager.max_drawdown_percentage * 100:.2f}%",
            "Action Needed": "Yes" if risk_metrics.get('action_needed', False) else "No"
        }
        
        table_data = [[k, v] for k, v in data.items()]
        return tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid")
    
    def format_instruments(self) -> str:
        """Format instrument information for display."""
        instruments = self.data["instruments"]
        
        if not instruments:
            return "No instruments available."
        
        instrument_data = []
        for instrument in instruments:
            name = instrument.get("name", "Unknown")
            display_name = instrument.get("displayName", "Unknown")
            pip_location = instrument.get("pipLocation", 0)
            margin_rate = float(instrument.get("marginRate", 0)) * 100
            
            instrument_data.append([
                name,
                display_name,
                10 ** pip_location,
                f"{margin_rate:.2f}%"
            ])
        
        return tabulate(instrument_data, 
                       headers=["Name", "Display Name", "Pip Value", "Margin Rate"],
                       tablefmt="grid")
    
    def display_console(self) -> None:
        """Display the demo in console mode (non-curses)."""
        try:
            # Make initial connection
            if not self.connected:
                print("Connecting to OANDA account...")
                if not self.connect():
                    print("Failed to connect to OANDA account. Please check your credentials.")
                    return
                print("✅ Successfully connected to OANDA account")
            
            # Start the update thread
            self.update_data()  # Initial data fetch
            self.start_update_thread()
            
            # Main display loop
            try:
                while not self.should_exit:
                    # Clear screen (cross-platform)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # Print header
                    print("\n" + "=" * 80)
                    print("ENHANCED WALLET MANAGER DEMO - REAL-TIME OANDA ACCOUNT MONITORING")
                    print("=" * 80)
                    
                    if self.last_updated:
                        print(f"Last Updated: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        print(f"Auto-refresh every {self.refresh_interval} seconds. Press Ctrl+C to exit.\n")
                    
                    # Account details and summary
                    print("\n" + "=" * 80)
                    print("ACCOUNT OVERVIEW")
                    print("=" * 80)
                    print(self.format_account_details())
                    
                    # Open positions
                    print("\n" + "=" * 80)
                    print("OPEN POSITIONS")
                    print("=" * 80)
                    print(self.format_positions())
                    
                    # Open orders
                    print("\n" + "=" * 80)
                    print("OPEN ORDERS")
                    print("=" * 80)
                    print(self.format_orders())
                    
                    # Risk metrics
                    print("\n" + "=" * 80)
                    print("RISK METRICS")
                    print("=" * 80)
                    print(self.format_risk_metrics())
                    
                    # Recent transactions
                    print("\n" + "=" * 80)
                    print("RECENT TRANSACTIONS")
                    print("=" * 80)
                    print(self.format_transactions())
                    
                    # Available instruments
                    print("\n" + "=" * 80)
                    print("TOP INSTRUMENTS")
                    print("=" * 80)
                    print(self.format_instruments())
                    
                    # Wait for next refresh
                    time.sleep(self.refresh_interval)
                    
            except KeyboardInterrupt:
                print("\nExiting demo...")
        finally:
            self.stop_update_thread()
    
    def run(self) -> None:
        """Run the demo."""
        self.display_console()

def main():
    """Main function to run the Enhanced WalletManager Demo."""
    demo = EnhancedWalletManagerDemo()
    demo.run()

if __name__ == "__main__":
    main() 