#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper Trading System for Forex Trading Platform

This module provides a comprehensive PaperTradingSystem class that simulates
trading without real money. It includes account simulation, trade execution,
position tracking, market simulation, and performance tracking.
"""

import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class OrderType(Enum):
    """Order types supported by the paper trading system."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(Enum):
    """Order sides (buy/sell) supported by the paper trading system."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Possible statuses for orders in the paper trading system."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    TRIGGERED = "TRIGGERED"


@dataclass
class Order:
    """Represents an order in the paper trading system."""
    id: str
    instrument: str
    units: float
    side: OrderSide
    type: OrderType
    status: OrderStatus
    price: Optional[float] = None
    executed_price: Optional[float] = None
    created_at: datetime = datetime.now()
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None


@dataclass
class Position:
    """Represents a position in the paper trading system."""
    id: str
    instrument: str
    units: float
    side: OrderSide
    open_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = datetime.now()
    closed_at: Optional[datetime] = None
    closed_price: Optional[float] = None
    pnl: Optional[float] = None


class MarketCondition(Enum):
    """Different market conditions that can be simulated."""
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    HIGH_LIQUIDITY = "HIGH_LIQUIDITY"


class PaperTradingSystem:
    """
    A comprehensive paper trading system for simulating forex trading without real money.
    
    This class provides functionality for:
    1. Account simulation
    2. Trade simulation
    3. Position tracking
    4. Market simulation
    5. Performance tracking
    
    It's designed to provide a realistic trading experience for testing strategies
    and learning without financial risk.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the paper trading system.
        
        Args:
            logger: Logger instance for the paper trading system
        """
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger("paper_trading")
        else:
            self.logger = logger
            
        # Account data
        self.account_currency = None
        self.initial_balance = 0.0
        self.balance = 0.0
        self.margin_rate = 0.02  # Default 2% margin requirement
        
        # Orders and positions
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.closed_positions: Dict[str, Position] = {}
        
        # Market data
        self.current_prices: Dict[str, Dict[str, float]] = {}
        self.market_condition = MarketCondition.NORMAL
        self.default_spread_pips = 2.0
        self.spread_multipliers = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 2.0,
            MarketCondition.TRENDING_UP: 1.2,
            MarketCondition.TRENDING_DOWN: 1.2,
            MarketCondition.LOW_LIQUIDITY: 2.5,
            MarketCondition.HIGH_LIQUIDITY: 0.8
        }
        
        # Performance tracking
        self.trade_history = pd.DataFrame(columns=[
            'id', 'instrument', 'units', 'side', 'open_price', 'close_price',
            'opened_at', 'closed_at', 'pnl', 'pnl_percentage'
        ])
        self.balance_history = pd.DataFrame(columns=['timestamp', 'balance'])
        self.equity_history = pd.DataFrame(columns=['timestamp', 'equity'])
        
        # System settings
        self.auto_update_positions = True
        
        self.logger.info("Paper Trading System initialized")

    # Account simulation methods will go here
    
    def initialize_account(self, balance: float, currency: str, margin_rate: float = 0.02) -> None:
        """
        Initialize a simulated trading account.
        
        Args:
            balance: Initial account balance
            currency: Base currency for the account
            margin_rate: Required margin rate as a decimal (default: 0.02 or 2%)
        """
        self.initial_balance = balance
        self.balance = balance
        self.account_currency = currency
        self.margin_rate = margin_rate
        
        # Initialize history with starting balance
        timestamp = datetime.now()
        self.balance_history = pd.DataFrame({
            'timestamp': [timestamp],
            'balance': [balance]
        })
        self.equity_history = pd.DataFrame({
            'timestamp': [timestamp],
            'equity': [balance]
        })
        
        self.logger.info(f"Account initialized with {balance} {currency}")

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get the current account summary.
        
        Returns:
            Dict[str, Any]: Summary of account including balance, equity, margin, etc.
        """
        # Calculate current equity (balance + unrealized PnL)
        floating_pnl = self.calculate_floating_pnl()
        equity = self.balance + floating_pnl
        
        # Calculate margin values
        margin_used = self.calculate_margin_used()
        margin_available = self.calculate_margin_available()
        
        # Open positions count
        open_positions_count = len(self.positions)
        
        # Pending orders count
        pending_orders_count = sum(
            1 for order in self.orders.values() 
            if order.status == OrderStatus.PENDING
        )
        
        return {
            "account_id": id(self),  # Use object id as account id
            "currency": self.account_currency,
            "balance": self.balance,
            "equity": equity,
            "margin_used": margin_used,
            "margin_available": margin_available,
            "margin_level": (equity / margin_used * 100) if margin_used > 0 else None,
            "floating_pnl": floating_pnl,
            "open_positions_count": open_positions_count,
            "pending_orders_count": pending_orders_count,
            "margin_rate": self.margin_rate,
            "updated_at": datetime.now()
        }

    def update_account_balance(self, amount: float, reason: str = "") -> float:
        """
        Update the account balance.
        
        Args:
            amount: Amount to add (positive) or subtract (negative) from balance
            reason: Reason for the balance update (for logging)
            
        Returns:
            float: New account balance
        """
        previous_balance = self.balance
        self.balance += amount
        
        # Record in balance history
        self.balance_history = pd.concat([
            self.balance_history,
            pd.DataFrame({
                'timestamp': [datetime.now()],
                'balance': [self.balance]
            })
        ], ignore_index=True)
        
        # Update equity history as well
        floating_pnl = self.calculate_floating_pnl()
        equity = self.balance + floating_pnl
        self.equity_history = pd.concat([
            self.equity_history,
            pd.DataFrame({
                'timestamp': [datetime.now()],
                'equity': [equity]
            })
        ], ignore_index=True)
        
        if reason:
            self.logger.info(f"Balance updated from {previous_balance} to {self.balance}: {reason}")
        else:
            self.logger.info(f"Balance updated from {previous_balance} to {self.balance}")
            
        return self.balance

    def calculate_margin_used(self) -> float:
        """
        Calculate margin currently in use by open positions.
        
        Returns:
            float: Amount of margin being used
        """
        margin_used = 0.0
        
        for position in self.positions.values():
            # Calculate position value in account currency
            if position.instrument in self.current_prices:
                current_price = self.current_prices[position.instrument]["mid"]
                position_value = abs(position.units) * current_price
                
                # Apply margin rate to calculate margin requirement
                position_margin = position_value * self.margin_rate
                margin_used += position_margin
            
        return margin_used

    def calculate_margin_available(self) -> float:
        """
        Calculate available margin for new positions.
        
        Returns:
            float: Amount of margin available
        """
        # Calculate current equity (balance + unrealized PnL)
        floating_pnl = self.calculate_floating_pnl()
        equity = self.balance + floating_pnl
        
        # Calculate used margin
        margin_used = self.calculate_margin_used()
        
        # Available margin is equity minus used margin
        margin_available = equity - margin_used
        
        return max(0.0, margin_available)
    
    def check_margin_call(self) -> bool:
        """
        Check if account is in a margin call situation.
        
        Returns:
            bool: True if margin call triggered, False otherwise
        """
        # Calculate current equity and margin level
        floating_pnl = self.calculate_floating_pnl()
        equity = self.balance + floating_pnl
        margin_used = self.calculate_margin_used()
        
        if margin_used > 0:
            margin_level = (equity / margin_used) * 100
            
            # Typical margin call level is 100% (meaning equity equals margin used)
            if margin_level < 100:
                self.logger.warning(f"Margin call triggered! Margin level: {margin_level:.2f}%")
                return True
                
        return False

    # Trade simulation methods will go here
    
    def execute_trade(self, instrument: str, units: float, side: Union[OrderSide, str], 
                     order_type: Union[OrderType, str], price: Optional[float] = None) -> Optional[str]:
        """
        Simulate trade execution with various order types.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            units: Number of units to trade
            side: Buy or sell order
            order_type: Type of order (MARKET, LIMIT, STOP)
            price: Price for limit and stop orders (not needed for market orders)
            
        Returns:
            Optional[str]: Order ID if successful, None if failed
        """
        # Convert string enums to proper Enum values if needed
        if isinstance(side, str):
            side = OrderSide(side)
        if isinstance(order_type, str):
            order_type = OrderType(order_type)
            
        # Validate inputs
        if units <= 0:
            self.logger.error(f"Cannot execute trade: units must be positive")
            return None
            
        if not instrument in self.current_prices:
            self.logger.error(f"Cannot execute trade: no price data for {instrument}")
            return None
            
        # Handle market orders immediately
        if order_type == OrderType.MARKET:
            return self.place_market_order(instrument, units, side)
            
        # Handle limit and stop orders (require price)
        if price is None:
            self.logger.error(f"Cannot place {order_type.value} order: price not specified")
            return None
            
        if order_type == OrderType.LIMIT:
            return self.place_limit_order(instrument, units, side, price)
            
        if order_type == OrderType.STOP:
            return self.place_stop_order(instrument, units, side, price)
            
        return None

    def place_market_order(self, instrument: str, units: float, side: OrderSide) -> Optional[str]:
        """
        Simulate placing a market order.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            units: Number of units to trade
            side: Buy or sell order
            
        Returns:
            Optional[str]: Order ID if successful, None if failed
        """
        # Check if we have current price for the instrument
        if instrument not in self.current_prices:
            self.logger.error(f"Cannot place market order: no price data for {instrument}")
            return None
            
        # Get bid/ask price based on side
        if side == OrderSide.BUY:
            price = self.current_prices[instrument]["ask"]
        else:  # SELL
            price = self.current_prices[instrument]["bid"]
            
        # Apply slippage
        executed_price = self.simulate_slippage(price, instrument, side)
        
        # Create the order
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            instrument=instrument,
            units=units,
            side=side,
            type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            price=price,
            executed_price=executed_price,
            created_at=datetime.now(),
            filled_at=datetime.now()
        )
        
        # Store the order
        self.orders[order_id] = order
        
        # Open the position immediately since it's a market order
        position_id = self.open_position(instrument, units, side, executed_price)
        
        if position_id:
            self.logger.info(
                f"Market order executed: {side.value} {units} {instrument} @ {executed_price}"
            )
            return order_id
        else:
            # Revert the order status if position couldn't be opened
            order.status = OrderStatus.REJECTED
            self.logger.error(f"Market order rejected: couldn't open position")
            return None

    def place_limit_order(self, instrument: str, units: float, side: OrderSide, price: float) -> str:
        """
        Simulate placing a limit order.
        
        A limit order is an order to buy at or below a specified price, or to sell at or above
        a specified price.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            units: Number of units to trade
            side: Buy or sell order
            price: Limit price for the order
            
        Returns:
            str: Order ID
        """
        # For limit orders, check if price is valid
        if side == OrderSide.BUY and price >= self.current_prices[instrument]["ask"]:
            # Buy limit must be below current ask price
            self.logger.warning(
                f"Buy limit price {price} is above current ask {self.current_prices[instrument]['ask']}. "
                f"Order will execute immediately."
            )
            return self.place_market_order(instrument, units, side)
            
        if side == OrderSide.SELL and price <= self.current_prices[instrument]["bid"]:
            # Sell limit must be above current bid price
            self.logger.warning(
                f"Sell limit price {price} is below current bid {self.current_prices[instrument]['bid']}. "
                f"Order will execute immediately."
            )
            return self.place_market_order(instrument, units, side)
            
        # Create the pending order
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            instrument=instrument,
            units=units,
            side=side,
            type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
            price=price,
            created_at=datetime.now()
        )
        
        # Store the order
        self.orders[order_id] = order
        
        self.logger.info(
            f"Limit order placed: {side.value} {units} {instrument} @ {price}"
        )
        
        return order_id

    def place_stop_order(self, instrument: str, units: float, side: OrderSide, price: float) -> str:
        """
        Simulate placing a stop order.
        
        A stop order is an order to buy at or above a specified price, or to sell at or below
        a specified price.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            units: Number of units to trade
            side: Buy or sell order
            price: Stop price for the order
            
        Returns:
            str: Order ID
        """
        # For stop orders, check if price is valid
        if side == OrderSide.BUY and price <= self.current_prices[instrument]["ask"]:
            # Buy stop must be above current ask price
            self.logger.warning(
                f"Buy stop price {price} is below current ask {self.current_prices[instrument]['ask']}. "
                f"Order will execute immediately."
            )
            return self.place_market_order(instrument, units, side)
            
        if side == OrderSide.SELL and price >= self.current_prices[instrument]["bid"]:
            # Sell stop must be below current bid price
            self.logger.warning(
                f"Sell stop price {price} is above current bid {self.current_prices[instrument]['bid']}. "
                f"Order will execute immediately."
            )
            return self.place_market_order(instrument, units, side)
            
        # Create the pending order
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            instrument=instrument,
            units=units,
            side=side,
            type=OrderType.STOP,
            status=OrderStatus.PENDING,
            price=price,
            created_at=datetime.now()
        )
        
        # Store the order
        self.orders[order_id] = order
        
        self.logger.info(
            f"Stop order placed: {side.value} {units} {instrument} @ {price}"
        )
        
        return order_id

    def modify_order(self, order_id: str, new_parameters: Dict[str, Any]) -> bool:
        """
        Simulate modifying an existing order.
        
        Args:
            order_id: ID of the order to modify
            new_parameters: Dict of parameters to update (price, units, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if order_id not in self.orders:
            self.logger.error(f"Cannot modify order: order {order_id} not found")
            return False
            
        order = self.orders[order_id]
        
        # Can only modify pending orders
        if order.status != OrderStatus.PENDING:
            self.logger.error(f"Cannot modify order: order {order_id} is {order.status.value}")
            return False
            
        # Update allowed parameters
        modified = False
        
        if "price" in new_parameters:
            order.price = new_parameters["price"]
            modified = True
            
        if "units" in new_parameters:
            if new_parameters["units"] <= 0:
                self.logger.error("Cannot modify order: units must be positive")
                return False
            order.units = new_parameters["units"]
            modified = True
            
        if modified:
            self.logger.info(f"Order {order_id} modified")
            return True
        else:
            self.logger.warning(f"No parameters were modified for order {order_id}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        Simulate canceling an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if successful, False otherwise
        """
        if order_id not in self.orders:
            self.logger.error(f"Cannot cancel order: order {order_id} not found")
            return False
            
        order = self.orders[order_id]
        
        # Can only cancel pending orders
        if order.status != OrderStatus.PENDING:
            self.logger.error(f"Cannot cancel order: order {order_id} is {order.status.value}")
            return False
            
        # Update order status
        order.status = OrderStatus.CANCELED
        order.canceled_at = datetime.now()
        
        self.logger.info(f"Order {order_id} canceled")
        return True

    # Position tracking methods will go here
    
    def open_position(self, instrument: str, units: float, side: OrderSide, price: float,
                      stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Optional[str]:
        """
        Open a simulated position.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            units: Number of units for the position
            side: Buy or sell position
            price: Opening price for the position
            stop_loss: Optional stop loss price level
            take_profit: Optional take profit price level
            
        Returns:
            Optional[str]: Position ID if successful, None if failed
        """
        # Check if we have enough margin available
        position_value = units * price
        margin_required = position_value * self.margin_rate
        
        if margin_required > self.calculate_margin_available():
            self.logger.error(
                f"Cannot open position: insufficient margin. " 
                f"Required: {margin_required}, Available: {self.calculate_margin_available()}"
            )
            return None
            
        # Check if position already exists for this instrument
        # If so, we'll merge with it or reduce it rather than creating a new one
        existing_position = None
        for pos in self.positions.values():
            if pos.instrument == instrument:
                existing_position = pos
                break
                
        if existing_position:
            # Position exists - handle accordingly
            if (existing_position.side == side):
                # Same direction - increase position size
                existing_position.units += units
                
                # Recalculate average open price
                total_value = (existing_position.units - units) * existing_position.open_price + units * price
                existing_position.open_price = total_value / existing_position.units
                
                self.logger.info(
                    f"Position {existing_position.id} increased: {side.value} {units} more {instrument} @ {price}"
                )
                return existing_position.id
                
            else:
                # Opposite direction - reduce or close position
                if units < existing_position.units:
                    # Reduce position
                    pnl = self.calculate_position_pnl(existing_position.id, price)
                    existing_position.units -= units
                    
                    # Update account balance with realized PnL
                    self.update_account_balance(pnl, f"Partial position close: {instrument}")
                    
                    self.logger.info(
                        f"Position {existing_position.id} decreased: {side.value} {units} {instrument} @ {price}"
                    )
                    return existing_position.id
                    
                elif units == existing_position.units:
                    # Close position fully
                    return self.close_position(existing_position.id, price)
                    
                else:
                    # Close position and open new one in opposite direction
                    self.close_position(existing_position.id, price)
                    remaining_units = units - existing_position.units
                    return self.open_position(instrument, remaining_units, side, price, stop_loss, take_profit)
        
        # Create new position
        position_id = str(uuid.uuid4())
        position = Position(
            id=position_id,
            instrument=instrument,
            units=units,
            side=side,
            open_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=datetime.now()
        )
        
        # Store the position
        self.positions[position_id] = position
        
        self.logger.info(
            f"Position opened: {side.value} {units} {instrument} @ {price}"
        )
        
        return position_id

    def close_position(self, position_id: str, price: Optional[float] = None) -> bool:
        """
        Close a simulated position.
        
        Args:
            position_id: ID of the position to close
            price: Optional closing price (if None, will use current market price)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if position_id not in self.positions:
            self.logger.error(f"Cannot close position: position {position_id} not found")
            return False
            
        position = self.positions[position_id]
        
        # Get closing price
        if price is None:
            if position.instrument not in self.current_prices:
                self.logger.error(f"Cannot close position: no price data for {position.instrument}")
                return False
                
            if position.side == OrderSide.BUY:
                price = self.current_prices[position.instrument]["bid"]  # Sell at bid
            else:
                price = self.current_prices[position.instrument]["ask"]  # Buy at ask
        
        # Calculate PnL
        pnl = self.calculate_position_pnl(position_id, price)
        
        # Update position
        position.closed_price = price
        position.closed_at = datetime.now()
        position.pnl = pnl
        
        # Move to closed positions
        self.closed_positions[position_id] = position
        del self.positions[position_id]
        
        # Update account balance
        self.update_account_balance(pnl, f"Position close: {position.instrument}")
        
        # Add to trade history
        pnl_percentage = (pnl / (position.units * position.open_price)) * 100
        self.trade_history = pd.concat([
            self.trade_history,
            pd.DataFrame({
                'id': [position_id],
                'instrument': [position.instrument],
                'units': [position.units],
                'side': [position.side.value],
                'open_price': [position.open_price],
                'close_price': [price],
                'opened_at': [position.opened_at],
                'closed_at': [position.closed_at],
                'pnl': [pnl],
                'pnl_percentage': [pnl_percentage]
            })
        ], ignore_index=True)
        
        self.logger.info(
            f"Position closed: {position.side.value} {position.units} {position.instrument} "
            f"@ {price}, PnL: {pnl:.2f}"
        )
        
        return True

    def modify_position(self, position_id: str, stop_loss: Optional[float] = None, 
                        take_profit: Optional[float] = None) -> bool:
        """
        Modify a simulated position.
        
        Args:
            position_id: ID of the position to modify
            stop_loss: New stop loss level (None to leave unchanged)
            take_profit: New take profit level (None to leave unchanged)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if position_id not in self.positions:
            self.logger.error(f"Cannot modify position: position {position_id} not found")
            return False
            
        position = self.positions[position_id]
        modified = False
        
        # Update stop loss if provided
        if stop_loss is not None:
            position.stop_loss = stop_loss
            modified = True
            
        # Update take profit if provided
        if take_profit is not None:
            position.take_profit = take_profit
            modified = True
            
        if modified:
            self.logger.info(f"Position {position_id} modified")
            return True
        else:
            self.logger.warning(f"No parameters were modified for position {position_id}")
            return False

    def get_open_positions(self) -> Dict[str, Position]:
        """
        Get all open positions.
        
        Returns:
            Dict[str, Position]: Dictionary of open positions
        """
        return self.positions

    def calculate_position_value(self, position_id: str) -> float:
        """
        Calculate the current value of a position.
        
        Args:
            position_id: ID of the position
            
        Returns:
            float: Current value of the position
        """
        if position_id not in self.positions:
            self.logger.error(f"Cannot calculate position value: position {position_id} not found")
            return 0.0
            
        position = self.positions[position_id]
        
        if position.instrument not in self.current_prices:
            self.logger.error(f"Cannot calculate position value: no price data for {position.instrument}")
            return 0.0
            
        # Use mid price for value calculation
        current_price = self.current_prices[position.instrument]["mid"]
        position_value = position.units * current_price
        
        return position_value

    def calculate_position_pnl(self, position_id: str, current_price: Optional[float] = None) -> float:
        """
        Calculate profit/loss for a specific position.
        
        Args:
            position_id: ID of the position
            current_price: Optional current price (if None, will use market price)
            
        Returns:
            float: Position profit/loss
        """
        if position_id not in self.positions and position_id not in self.closed_positions:
            self.logger.error(f"Cannot calculate PnL: position {position_id} not found")
            return 0.0
            
        if position_id in self.positions:
            position = self.positions[position_id]
            
            # For open positions, use provided price or get current market price
            if current_price is None:
                if position.instrument not in self.current_prices:
                    self.logger.error(f"Cannot calculate PnL: no price data for {position.instrument}")
                    return 0.0
                    
                if position.side == OrderSide.BUY:
                    current_price = self.current_prices[position.instrument]["bid"]  # Sell at bid
                else:
                    current_price = self.current_prices[position.instrument]["ask"]  # Buy at ask
        else:
            # For closed positions, use the recorded closing price
            position = self.closed_positions[position_id]
            if current_price is None:
                current_price = position.closed_price
                
        # Calculate PnL based on position side
        if position.side == OrderSide.BUY:
            # For buy positions, profit when closing price > opening price
            pnl = position.units * (current_price - position.open_price)
        else:
            # For sell positions, profit when closing price < opening price
            pnl = position.units * (position.open_price - current_price)
            
        return pnl

    def calculate_floating_pnl(self) -> float:
        """
        Calculate floating (unrealized) profit/loss for all open positions.
        
        Returns:
            float: Total floating profit/loss
        """
        total_pnl = 0.0
        
        for position_id in self.positions:
            pnl = self.calculate_position_pnl(position_id)
            total_pnl += pnl
            
        return total_pnl

    def check_position_triggers(self) -> None:
        """
        Check if any positions need to be closed due to stop loss or take profit triggers.
        """
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position.instrument not in self.current_prices:
                continue
                
            current_bid = self.current_prices[position.instrument]["bid"]
            current_ask = self.current_prices[position.instrument]["ask"]
            
            # Check stop loss
            if position.stop_loss is not None:
                if (position.side == OrderSide.BUY and current_bid <= position.stop_loss) or \
                   (position.side == OrderSide.SELL and current_ask >= position.stop_loss):
                    self.logger.info(f"Stop loss triggered for position {position_id}")
                    positions_to_close.append((position_id, position.stop_loss))
                    continue
                    
            # Check take profit
            if position.take_profit is not None:
                if (position.side == OrderSide.BUY and current_bid >= position.take_profit) or \
                   (position.side == OrderSide.SELL and current_ask <= position.take_profit):
                    self.logger.info(f"Take profit triggered for position {position_id}")
                    positions_to_close.append((position_id, position.take_profit))
                    continue
                    
        # Close triggered positions
        for position_id, price in positions_to_close:
            self.close_position(position_id, price)

    # Market simulation methods will go here
    
    def process_price_update(self, instrument: str, bid: float, ask: float) -> None:
        """
        Process a price update for an instrument.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
            bid: Bid price
            ask: Ask price
        """
        # Calculate mid price
        mid = (bid + ask) / 2
        
        # Update current prices
        self.current_prices[instrument] = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "timestamp": datetime.now()
        }
        
        # If auto-update is enabled, check for order triggers and position triggers
        if self.auto_update_positions:
            self.check_order_triggers()
            self.check_position_triggers()
            
            # Update equity history
            floating_pnl = self.calculate_floating_pnl()
            equity = self.balance + floating_pnl
            
            self.equity_history = pd.concat([
                self.equity_history,
                pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'equity': [equity]
                })
            ], ignore_index=True)
            
            # Check for margin call
            self.check_margin_call()

    def check_order_triggers(self) -> None:
        """
        Check if any pending orders should be triggered based on current prices.
        """
        orders_to_process = []
        
        for order_id, order in self.orders.items():
            # Only check pending orders
            if order.status != OrderStatus.PENDING:
                continue
                
            # Skip if no price data for the instrument
            if order.instrument not in self.current_prices:
                continue
                
            current_bid = self.current_prices[order.instrument]["bid"]
            current_ask = self.current_prices[order.instrument]["ask"]
            
            # Check if order should be triggered
            if order.type == OrderType.LIMIT:
                # Limit buy: trigger when ask price <= limit price
                if order.side == OrderSide.BUY and current_ask <= order.price:
                    orders_to_process.append((order_id, "LIMIT_BUY", current_ask))
                    
                # Limit sell: trigger when bid price >= limit price
                elif order.side == OrderSide.SELL and current_bid >= order.price:
                    orders_to_process.append((order_id, "LIMIT_SELL", current_bid))
                    
            elif order.type == OrderType.STOP:
                # Stop buy: trigger when ask price >= stop price
                if order.side == OrderSide.BUY and current_ask >= order.price:
                    orders_to_process.append((order_id, "STOP_BUY", current_ask))
                    
                # Stop sell: trigger when bid price <= stop price
                elif order.side == OrderSide.SELL and current_bid <= order.price:
                    orders_to_process.append((order_id, "STOP_SELL", current_bid))
        
        # Process triggered orders
        for order_id, trigger_type, price in orders_to_process:
            self.execute_triggered_order(order_id, price)

    def execute_triggered_order(self, order_id: str, executed_price: float) -> None:
        """
        Execute a triggered pending order.
        
        Args:
            order_id: ID of the order to execute
            executed_price: Price at which the order is executed
        """
        if order_id not in self.orders:
            return
            
        order = self.orders[order_id]
        
        # Apply slippage
        final_price = self.simulate_slippage(executed_price, order.instrument, order.side)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.executed_price = final_price
        order.filled_at = datetime.now()
        
        # Open position for the executed order
        position_id = self.open_position(
            order.instrument, 
            order.units, 
            order.side, 
            final_price
        )
        
        if position_id:
            self.logger.info(
                f"{order.type.value} order {order_id} executed: {order.side.value} "
                f"{order.units} {order.instrument} @ {final_price}"
            )
        else:
            order.status = OrderStatus.REJECTED
            self.logger.error(f"Order {order_id} rejected: couldn't open position")

    def apply_spread(self, mid_price: float, instrument: str) -> Tuple[float, float]:
        """
        Apply realistic spread to a mid price.
        
        Args:
            mid_price: Mid price to apply spread to
            instrument: Trading instrument (for instrument-specific spreads)
            
        Returns:
            Tuple[float, float]: (bid, ask) prices
        """
        # Base spread in pips (default is 2.0)
        base_spread_pips = self.default_spread_pips
        
        # Apply market condition multiplier
        spread_multiplier = self.spread_multipliers.get(self.market_condition, 1.0)
        
        # Calculate actual spread in price units
        # For forex, we need to convert pips to price movement
        # For most forex pairs, 1 pip = 0.0001, except JPY pairs where 1 pip = 0.01
        is_jpy_pair = "_JPY" in instrument or instrument.startswith("JPY_")
        pip_value = 0.01 if is_jpy_pair else 0.0001
        
        spread_value = base_spread_pips * pip_value * spread_multiplier
        
        # Calculate bid and ask
        half_spread = spread_value / 2
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        
        return bid, ask

    def simulate_slippage(self, price: float, instrument: str, side: OrderSide) -> float:
        """
        Simulate realistic slippage for an order.
        
        Args:
            price: Original execution price
            instrument: Trading instrument
            side: Order side (buy/sell)
            
        Returns:
            float: Price after slippage
        """
        # Base slippage factor (0.0001 = 1 pip for most pairs)
        is_jpy_pair = "_JPY" in instrument or instrument.startswith("JPY_")
        base_slippage = 0.01 if is_jpy_pair else 0.0001
        
        # Adjust slippage based on market condition
        if self.market_condition == MarketCondition.VOLATILE:
            slippage_factor = 2.0
        elif self.market_condition == MarketCondition.LOW_LIQUIDITY:
            slippage_factor = 3.0
        else:
            slippage_factor = 1.0
            
        # Random slippage between 0 and max_slippage
        max_slippage = base_slippage * slippage_factor
        slippage = np.random.uniform(0, max_slippage)
        
        # Apply slippage (negative for buy, positive for sell)
        if side == OrderSide.BUY:
            # Buy orders get executed at a higher price (worse for buyer)
            executed_price = price + slippage
        else:
            # Sell orders get executed at a lower price (worse for seller)
            executed_price = price - slippage
            
        return executed_price

    def simulate_market_conditions(self, condition: Union[MarketCondition, str]) -> None:
        """
        Simulate specific market conditions.
        
        Args:
            condition: Market condition to simulate
        """
        # Convert string to enum if needed
        if isinstance(condition, str):
            condition = MarketCondition(condition)
            
        self.market_condition = condition
        self.logger.info(f"Market condition set to {condition.value}")
        
        # Update current prices with new spreads
        for instrument, prices in self.current_prices.items():
            mid_price = prices["mid"]
            bid, ask = self.apply_spread(mid_price, instrument)
            
            self.current_prices[instrument]["bid"] = bid
            self.current_prices[instrument]["ask"] = ask

    # Performance tracking methods will go here

    def track_trade_history(self) -> pd.DataFrame:
        """
        Track history of all trades.
        
        Returns:
            pd.DataFrame: DataFrame with all closed trades
        """
        return self.trade_history

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics based on trade history.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if len(self.trade_history) == 0:
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_percentage": 0.0,
                "sharpe_ratio": 0.0
            }
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        profitable_trades = len(self.trade_history[self.trade_history['pnl'] > 0])
        losing_trades = len(self.trade_history[self.trade_history['pnl'] < 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit and loss metrics
        total_profit = self.trade_history[self.trade_history['pnl'] > 0]['pnl'].sum()
        total_loss = abs(self.trade_history[self.trade_history['pnl'] < 0]['pnl'].sum())
        
        average_profit = total_profit / profitable_trades if profitable_trades > 0 else 0.0
        average_loss = total_loss / losing_trades if losing_trades > 0 else 0.0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate expectancy: (Win Rate × Average Win) − (Loss Rate × Average Loss)
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0.0
        expectancy = (win_rate * average_profit) - (loss_rate * average_loss)
        
        # Calculate drawdown
        equity_curve = self.equity_history.copy()
        if len(equity_curve) > 1:
            # Calculate running maximum
            equity_curve['equity_peak'] = equity_curve['equity'].cummax()
            # Calculate drawdown in currency units
            equity_curve['drawdown'] = equity_curve['equity_peak'] - equity_curve['equity']
            # Calculate drawdown as percentage
            equity_curve['drawdown_pct'] = equity_curve['drawdown'] / equity_curve['equity_peak']
            
            max_drawdown = equity_curve['drawdown'].max()
            max_drawdown_percentage = equity_curve['drawdown_pct'].max() * 100.0
        else:
            max_drawdown = 0.0
            max_drawdown_percentage = 0.0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(self.balance_history) > 1:
            # Calculate daily returns
            balance_history_daily = self.balance_history.copy()
            balance_history_daily.set_index('timestamp', inplace=True)
            # Resample to daily frequency
            daily_balance = balance_history_daily.resample('D').last()
            daily_balance.fillna(method='ffill', inplace=True)
            
            if len(daily_balance) > 1:
                daily_returns = daily_balance['balance'].pct_change().dropna()
                sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
            "max_drawdown_percentage": max_drawdown_percentage,
            "sharpe_ratio": sharpe_ratio
        }

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dict[str, Any]: Performance report with metrics and summary
        """
        # Get account summary
        account_summary = self.get_account_summary()
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        # Get current market conditions
        market_conditions = {
            "condition": self.market_condition.value,
            "instruments": {
                instr: {
                    "bid": data["bid"],
                    "ask": data["ask"],
                    "spread": data["ask"] - data["bid"],
                    "timestamp": data["timestamp"].isoformat()
                }
                for instr, data in self.current_prices.items()
            }
        }
        
        # Get position information
        positions = {
            pos_id: {
                "instrument": pos.instrument,
                "units": pos.units,
                "side": pos.side.value,
                "open_price": pos.open_price,
                "current_pnl": self.calculate_position_pnl(pos_id),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "opened_at": pos.opened_at.isoformat()
            }
            for pos_id, pos in self.positions.items()
        }
        
        # Get pending orders
        pending_orders = {
            order_id: {
                "instrument": order.instrument,
                "units": order.units,
                "side": order.side.value,
                "type": order.type.value,
                "price": order.price,
                "created_at": order.created_at.isoformat()
            }
            for order_id, order in self.orders.items()
            if order.status == OrderStatus.PENDING
        }
        
        # Compile report
        report = {
            "timestamp": datetime.now().isoformat(),
            "account": account_summary,
            "performance_metrics": metrics,
            "market_conditions": market_conditions,
            "positions": positions,
            "pending_orders": pending_orders,
            "trade_count": len(self.trade_history),
            "last_trades": self.trade_history.tail(10).to_dict('records') if len(self.trade_history) > 0 else []
        }
        
        return report

    def export_trade_data(self, filepath: str) -> bool:
        """
        Export trade data to a CSV file.
        
        Args:
            filepath: Path to export the data to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Export trade history
            self.trade_history.to_csv(filepath, index=False)
            
            self.logger.info(f"Trade data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting trade data: {str(e)}")
            return False
            
    def export_performance_report(self, filepath: str) -> bool:
        """
        Export performance report to a JSON file.
        
        Args:
            filepath: Path to export the report to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Generate report
            report = self.generate_performance_report()
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Performance report exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting performance report: {str(e)}")
            return False

    def visualize_performance(self) -> Figure:
        """
        Create visualizations of performance.
        
        Returns:
            Figure: Matplotlib figure with performance visualizations
        """
        if len(self.equity_history) < 2:
            # Not enough data for visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Not enough data for visualization", 
                    horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 1. Equity curve
        equity_df = self.equity_history.copy()
        equity_df.set_index('timestamp', inplace=True)
        equity_df['balance'] = self.balance_history.set_index('timestamp')['balance']
        
        axes[0].plot(equity_df.index, equity_df['equity'], label='Equity', color='blue')
        axes[0].plot(equity_df.index, equity_df['balance'], label='Balance', color='green')
        axes[0].set_title('Account Equity and Balance Over Time')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # Calculate and plot drawdown
        if len(equity_df) > 1:
            equity_df['equity_peak'] = equity_df['equity'].cummax()
            equity_df['drawdown_pct'] = (equity_df['equity_peak'] - equity_df['equity']) / equity_df['equity_peak'] * 100
            
            axes[1].fill_between(equity_df.index, 0, equity_df['drawdown_pct'], color='red', alpha=0.3)
            axes[1].set_title('Drawdown Percentage')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Drawdown %')
            axes[1].grid(True)
        
        # 3. Trade results by instrument
        if len(self.trade_history) > 0:
            # Group by instrument
            by_instrument = self.trade_history.groupby('instrument')['pnl'].sum()
            colors = ['green' if x > 0 else 'red' for x in by_instrument]
            
            by_instrument.plot(kind='bar', ax=axes[2], color=colors)
            axes[2].set_title('Profit/Loss by Instrument')
            axes[2].set_xlabel('Instrument')
            axes[2].set_ylabel('Profit/Loss')
            axes[2].grid(True)
        
        plt.tight_layout()
        
        return fig

    def reset(self) -> None:
        """
        Reset the paper trading system.
        """
        # Keep the logger but reset everything else
        logger = self.logger
        
        # Re-initialize
        self.__init__(logger=logger)
        
        self.logger.info("Paper Trading System reset")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("paper_trading_example")
    
    # Create paper trading system
    paper_trading = PaperTradingSystem(logger=logger)
    
    # Initialize account
    paper_trading.initialize_account(10000.0, "USD")
    
    # Update some prices
    paper_trading.process_price_update("EUR_USD", 1.1000, 1.1002)
    paper_trading.process_price_update("GBP_USD", 1.2500, 1.2503)
    
    # Execute some trades
    paper_trading.execute_trade("EUR_USD", 10000, OrderSide.BUY, OrderType.MARKET)
    paper_trading.execute_trade("GBP_USD", 5000, OrderSide.SELL, OrderType.MARKET)
    
    # Modify a position
    positions = paper_trading.get_open_positions()
    if positions:
        first_position_id = next(iter(positions))
        paper_trading.modify_position(first_position_id, stop_loss=1.0900, take_profit=1.1100)
    
    # Update prices to simulate market movement
    paper_trading.process_price_update("EUR_USD", 1.1050, 1.1052)
    paper_trading.process_price_update("GBP_USD", 1.2520, 1.2523)
    
    # Generate report
    report = paper_trading.generate_performance_report()
    print(json.dumps(report, indent=2))
    
    # Close all positions
    for position_id in list(paper_trading.positions.keys()):
        paper_trading.close_position(position_id)
    
    # Export data
    paper_trading.export_trade_data("trade_data.csv")
    paper_trading.export_performance_report("performance_report.json")
    
    # Create visualization
    fig = paper_trading.visualize_performance()
    plt.savefig("performance_visualization.png")
    plt.close(fig) 