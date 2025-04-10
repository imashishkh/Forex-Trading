#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the PaperTradingSystem.

This script simulates a few trades on EUR/USD and prints the results,
including account balance changes.
"""

import logging
import time
from datetime import datetime

from paper_trading import PaperTradingSystem, OrderSide, OrderType, OrderStatus

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("paper_trading_test")

def print_balance_change(desc, old_balance, new_balance):
    """Print a formatted balance change message."""
    change = new_balance - old_balance
    percentage = (change / old_balance) * 100 if old_balance else 0
    print(f"{desc}: {old_balance:.2f} -> {new_balance:.2f} (Change: {change:.2f}, {percentage:.2f}%)")

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)

def main():
    print_section_header("PAPER TRADING SYSTEM TEST")
    
    # Create paper trading system
    print("Creating paper trading system...")
    paper_trading = PaperTradingSystem(logger=logger)
    
    # Initialize account with $10,000
    print_section_header("ACCOUNT INITIALIZATION")
    initial_balance = 10000.0
    paper_trading.initialize_account(initial_balance, "USD")
    
    current_balance = initial_balance
    
    # Initial price for EUR/USD
    print_section_header("INITIAL MARKET PRICES")
    eur_usd_bid = 1.0800
    eur_usd_ask = 1.0802
    print(f"EUR/USD: Bid: {eur_usd_bid}, Ask: {eur_usd_ask}, Spread: {(eur_usd_ask - eur_usd_bid)*10000:.1f} pips")
    
    # Update price
    paper_trading.process_price_update("EUR_USD", eur_usd_bid, eur_usd_ask)
    
    # TRADE 1: Market buy order
    print_section_header("TRADE 1: MARKET BUY ORDER")
    print(f"Executing market buy order for 10,000 units of EUR/USD at ask price {eur_usd_ask}...")
    
    order_id = paper_trading.execute_trade("EUR_USD", 10000, OrderSide.BUY, OrderType.MARKET)
    
    # Get account summary after trade
    account_summary = paper_trading.get_account_summary()
    print_balance_change("Account balance after market buy", current_balance, account_summary["balance"])
    current_balance = account_summary["balance"]
    
    # Get positions
    positions = paper_trading.get_open_positions()
    print("\nOpen positions:")
    for pos_id, pos in positions.items():
        print(f"- ID: {pos_id[:8]}..., Instrument: {pos.instrument}, Side: {pos.side.value}, Units: {pos.units}, Open Price: {pos.open_price:.4f}")
    
    # Price moves up (good for our buy position)
    print_section_header("PRICE MOVEMENT: UP")
    eur_usd_bid = 1.0820
    eur_usd_ask = 1.0822
    print(f"EUR/USD price update: Bid: {eur_usd_bid}, Ask: {eur_usd_ask}")
    
    paper_trading.process_price_update("EUR_USD", eur_usd_bid, eur_usd_ask)
    
    # Check floating P&L
    floating_pnl = paper_trading.calculate_floating_pnl()
    print(f"Floating P&L: {floating_pnl:.2f}")
    
    # Get account summary
    account_summary = paper_trading.get_account_summary()
    print(f"Equity: {account_summary['equity']:.2f}")
    
    # TRADE 2: Place a limit sell order above current price
    print_section_header("TRADE 2: LIMIT SELL ORDER")
    limit_price = 1.0830
    print(f"Placing limit sell order for 5,000 units of EUR/USD at {limit_price}...")
    
    order_id = paper_trading.place_limit_order("EUR_USD", 5000, OrderSide.SELL, limit_price)
    
    # Check pending orders
    pending_orders = [order for order_id, order in paper_trading.orders.items() 
                     if order.status == OrderStatus.PENDING]
    print(f"\nPending orders: {len(pending_orders)}")
    for order in pending_orders:
        print(f"- ID: {order.id[:8]}..., Type: {order.type.value}, Side: {order.side.value}, Price: {order.price:.4f}")
    
    # Price moves up more, triggering the limit sell
    print_section_header("PRICE MOVEMENT: TRIGGER LIMIT")
    eur_usd_bid = 1.0832  # This should trigger our limit sell
    eur_usd_ask = 1.0834
    print(f"EUR/USD price update: Bid: {eur_usd_bid}, Ask: {eur_usd_ask}")
    
    paper_trading.process_price_update("EUR_USD", eur_usd_bid, eur_usd_ask)
    
    # Check if the order was triggered
    pending_orders = [order for order_id, order in paper_trading.orders.items() 
                     if order.status == OrderStatus.PENDING]
    print(f"\nPending orders: {len(pending_orders)}")
    
    # Get positions
    positions = paper_trading.get_open_positions()
    print("\nOpen positions:")
    for pos_id, pos in positions.items():
        print(f"- ID: {pos_id[:8]}..., Instrument: {pos.instrument}, Side: {pos.side.value}, Units: {pos.units}, Open Price: {pos.open_price:.4f}")
    
    # Get account summary after trade
    account_summary = paper_trading.get_account_summary()
    print_balance_change("Account balance after limit sell", current_balance, account_summary["balance"])
    current_balance = account_summary["balance"]
    
    # TRADE 3: Place a stop buy order
    print_section_header("TRADE 3: STOP BUY ORDER")
    stop_price = 1.0850
    print(f"Placing stop buy order for 7,500 units of EUR/USD at {stop_price}...")
    
    order_id = paper_trading.place_stop_order("EUR_USD", 7500, OrderSide.BUY, stop_price)
    
    # Check pending orders
    pending_orders = [order for order_id, order in paper_trading.orders.items() 
                     if order.status == OrderStatus.PENDING]
    print(f"\nPending orders: {len(pending_orders)}")
    for order in pending_orders:
        print(f"- ID: {order.id[:8]}..., Type: {order.type.value}, Side: {order.side.value}, Price: {order.price:.4f}")
    
    # Price moves up more, triggering the stop buy
    print_section_header("PRICE MOVEMENT: TRIGGER STOP")
    eur_usd_bid = 1.0850  # This should trigger our stop buy
    eur_usd_ask = 1.0852
    print(f"EUR/USD price update: Bid: {eur_usd_bid}, Ask: {eur_usd_ask}")
    
    paper_trading.process_price_update("EUR_USD", eur_usd_bid, eur_usd_ask)
    
    # Check if the order was triggered
    pending_orders = [order for order_id, order in paper_trading.orders.items() 
                     if order.status == OrderStatus.PENDING]
    print(f"\nPending orders: {len(pending_orders)}")
    
    # Get positions
    positions = paper_trading.get_open_positions()
    print("\nOpen positions:")
    for pos_id, pos in positions.items():
        print(f"- ID: {pos_id[:8]}..., Instrument: {pos.instrument}, Side: {pos.side.value}, Units: {pos.units}, Open Price: {pos.open_price:.4f}")
    
    # Get account summary after trade
    account_summary = paper_trading.get_account_summary()
    print_balance_change("Account balance after stop buy", current_balance, account_summary["balance"])
    current_balance = account_summary["balance"]
    
    # Price moves down (bad for our position)
    print_section_header("PRICE MOVEMENT: DOWN")
    eur_usd_bid = 1.0820
    eur_usd_ask = 1.0822
    print(f"EUR/USD price update: Bid: {eur_usd_bid}, Ask: {eur_usd_ask}")
    
    paper_trading.process_price_update("EUR_USD", eur_usd_bid, eur_usd_ask)
    
    # Check floating P&L
    floating_pnl = paper_trading.calculate_floating_pnl()
    print(f"Floating P&L: {floating_pnl:.2f}")
    
    # Close all positions
    print_section_header("CLOSING ALL POSITIONS")
    for position_id in list(paper_trading.positions.keys()):
        print(f"Closing position {position_id[:8]}...")
        paper_trading.close_position(position_id)
    
    # Get account summary after closing all positions
    account_summary = paper_trading.get_account_summary()
    print_balance_change("Final account balance", initial_balance, account_summary["balance"])
    
    # Print performance metrics
    print_section_header("PERFORMANCE METRICS")
    metrics = paper_trading.calculate_performance_metrics()
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Profitable trades: {metrics['profitable_trades']}")
    print(f"Losing trades: {metrics['losing_trades']}")
    print(f"Win rate: {metrics['win_rate']*100:.2f}%")
    print(f"Total profit: {metrics['total_profit']:.2f}")
    print(f"Total loss: {metrics['total_loss']:.2f}")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 