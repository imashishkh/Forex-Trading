#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Portfolio Manager Agent implementation for the Forex Trading Platform

This module provides a comprehensive implementation of a portfolio manager agent
that handles trade execution, position management, portfolio tracking, and reporting
for the forex trading platform.
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path

# OANDA API
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    TakeProfitDetails,
    StopLossDetails
)

# Base Agent class
from utils.base_agent import BaseAgent, AgentState, AgentMessage

# Config Manager
from utils.config_manager import ConfigManager

# Metrics
from utils.metrics import calculate_drawdown, calculate_sharpe_ratio

# LangGraph imports
import langgraph.graph as lg
from langgraph.checkpoint import MemorySaver


class PortfolioManagerAgent(BaseAgent):
    """
    Portfolio Manager Agent for executing and managing trades in a forex trading portfolio.
    
    This agent is responsible for:
    1. Trade execution via the OANDA API
    2. Position management
    3. Portfolio tracking and performance metrics
    4. Trade management strategies like trailing stops
    5. Reporting and logging of trading activity
    """
    
    def __init__(
        self,
        agent_name: str = "portfolio_manager_agent",
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Portfolio Manager Agent.
        
        Args:
            agent_name: Name identifier for the agent
            llm: Language model to use (defaults to OpenAI if None)
            config: Configuration parameters for the agent
            logger: Logger instance for the agent
        """
        # Initialize BaseAgent
        super().__init__(agent_name, llm, config, logger)
        
        # If config not provided, use ConfigManager
        if config is None:
            config_manager = ConfigManager()
            self.config = config_manager.as_dict()
        
        # OANDA API configuration
        self.oanda_config = self.config.get('api_credentials', {}).get('oanda', {})
        self.api_key = self.oanda_config.get('api_key')
        self.account_id = self.oanda_config.get('account_id')
        self.api_url = self.oanda_config.get('api_url')
        
        # Determine environment (practice or live)
        self.is_practice = 'practice' in self.api_url.lower() if self.api_url else True
        
        # OANDA API client
        self.api_client = None
        
        # Data storage paths
        self.data_dir = Path(self.config.get('system', {}).get('data_storage_path', 'data'))
        self.portfolio_data_dir = self.data_dir / 'portfolio_data'
        
        # Ensure data directories exist
        self.portfolio_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Trade history file
        self.trade_history_file = self.portfolio_data_dir / 'trade_history.csv'
        self.performance_file = self.portfolio_data_dir / 'performance.csv'
        
        # Initialize internal tracking
        self._init_trade_tracking()
        
        self.log_action("init", f"Initialized Portfolio Manager Agent")
    
    def _init_trade_tracking(self) -> None:
        """Initialize trade tracking data structures"""
        # Load trade history if exists, create new one if not
        if self.trade_history_file.exists():
            try:
                self.trade_history = pd.read_csv(self.trade_history_file)
            except Exception as e:
                self.log_action("init", f"Error loading trade history: {str(e)}")
                self._create_new_trade_history()
        else:
            self._create_new_trade_history()
            
        # Load performance data if exists
        if self.performance_file.exists():
            try:
                self.performance_data = pd.read_csv(self.performance_file)
            except Exception as e:
                self.log_action("init", f"Error loading performance data: {str(e)}")
                self._create_new_performance_data()
        else:
            self._create_new_performance_data()
    
    def _create_new_trade_history(self) -> None:
        """Create a new trade history DataFrame"""
        self.trade_history = pd.DataFrame(columns=[
            'trade_id', 'instrument', 'direction', 'units', 'entry_price', 
            'exit_price', 'entry_time', 'exit_time', 'profit_loss',
            'profit_loss_pips', 'stop_loss', 'take_profit', 'status'
        ])
        self.trade_history.to_csv(self.trade_history_file, index=False)
    
    def _create_new_performance_data(self) -> None:
        """Create a new performance tracking DataFrame"""
        self.performance_data = pd.DataFrame(columns=[
            'timestamp', 'balance', 'equity', 'open_positions', 'floating_pl',
            'daily_pl', 'drawdown_percentage', 'drawdown_amount'
        ])
        self.performance_data.to_csv(self.performance_file, index=False)
    
    def initialize(self) -> bool:
        """
        Initialize the Portfolio Manager Agent and connect to the OANDA API.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.log_action("initialize", "Initializing Portfolio Manager Agent")
        
        try:
            # Validate required configuration
            if not all([self.api_key, self.account_id, self.api_url]):
                self.log_action("initialize", "Missing OANDA API credentials")
                self.handle_error(ValueError("Missing required OANDA API credentials"))
                return False
            
            # Initialize OANDA API client
            self.api_client = API(access_token=self.api_key, environment=self.api_url)
            
            # Test connection by fetching account details
            self._test_api_connection()
            
            # Update status
            self.status = "ready"
            self.state["status"] = "ready"
            
            self.log_action("initialize", "Portfolio Manager Agent initialized successfully")
            return True
            
        except Exception as e:
            self.handle_error(e)
            self.status = "error"
            self.state["status"] = "error"
            return False
    
    def _test_api_connection(self) -> bool:
        """
        Test connection to the OANDA API.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Request account details to verify connection
            r = accounts.AccountSummary(accountID=self.account_id)
            self.api_client.request(r)
            
            return True
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to connect to OANDA API: {str(e)}")

    def manage_portfolio(self, market_data, technical_signals, fundamental_insights, sentiment_insights, risk_assessment):
        """
        Manage the trading portfolio based on analysis from other agents
        
        Args:
            market_data (dict): Market data by symbol
            technical_signals (dict): Technical analysis signals
            fundamental_insights (dict): Fundamental analysis insights
            sentiment_insights (dict): Sentiment analysis insights
            risk_assessment (dict): Risk assessment from Risk Manager
        
        Returns:
            dict: Portfolio management results
        """
        self.logger.info("Managing portfolio")
        
        # Update portfolio with latest market data
        self._update_portfolio_values(market_data)
        
        # Check and update existing positions
        position_updates = self._manage_existing_positions(market_data, technical_signals, risk_assessment)
        
        # Identify new trading opportunities
        new_trades = self._identify_trading_opportunities(
            market_data, technical_signals, fundamental_insights, sentiment_insights, risk_assessment
        )
        
        # Execute trades
        executed_trades = self._execute_trades(new_trades, market_data)
        
        # Generate portfolio performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Save updated portfolio state
        self._save_portfolio()
        
        # Return management results
        results = {
            'portfolio_summary': self._get_portfolio_summary(),
            'position_updates': position_updates,
            'new_trades': executed_trades,
            'performance_metrics': performance_metrics
        }
        
        return results
    
    def _update_portfolio_values(self, market_data):
        """
        Update portfolio values based on current market data
        
        Args:
            market_data (dict): Current market data
        """
        total_equity = self.portfolio['account_balance']
        total_used_margin = 0
        
        # Update open positions with current market values
        for position in self.portfolio['open_positions']:
            symbol = position['symbol']
            
            # Get current price data for the symbol
            symbol_data = market_data.get(symbol, None)
            if symbol_data is None:
                continue
            
            # Get current price
            if isinstance(symbol_data, dict) and 'close' in symbol_data:
                current_price = symbol_data['close'].iloc[-1]
            elif isinstance(symbol_data, pd.DataFrame) and 'close' in symbol_data.columns:
                current_price = symbol_data['close'].iloc[-1]
            else:
                continue
            
            # Update position market value
            position_size = position['size']
            entry_price = position['entry_price']
            
            if position['direction'] == 'buy':
                pnl = (current_price - entry_price) * position_size
            else:  # sell
                pnl = (entry_price - current_price) * position_size
            
            position['current_price'] = current_price
            position['current_pnl'] = pnl
            position['current_pnl_percentage'] = (pnl / position['value']) * 100
            
            # Update equity with unrealized P&L
            total_equity += pnl
            
            # Calculate used margin
            position_margin = position['value'] / self.leverage
            total_used_margin += position_margin
            position['margin'] = position_margin
        
        # Update portfolio values
        self.portfolio['equity'] = total_equity
        self.portfolio['used_margin'] = total_used_margin
        self.portfolio['free_margin'] = total_equity - total_used_margin
        
        if total_used_margin > 0:
            self.portfolio['margin_level'] = (total_equity / total_used_margin) * 100
        else:
            self.portfolio['margin_level'] = 100
        
        # Add to account history
        self.portfolio['account_history'].append({
            'timestamp': datetime.now().isoformat(),
            'balance': self.portfolio['account_balance'],
            'equity': total_equity,
            'used_margin': total_used_margin,
            'free_margin': self.portfolio['free_margin']
        })
    
    def _manage_existing_positions(self, market_data, technical_signals, risk_assessment):
        """
        Check and update existing positions
        
        Args:
            market_data (dict): Current market data
            technical_signals (dict): Technical analysis signals
            risk_assessment (dict): Risk assessment from Risk Manager
        
        Returns:
            list: Position updates
        """
        position_updates = []
        positions_to_close = []
        
        for i, position in enumerate(self.portfolio['open_positions']):
            symbol = position['symbol']
            
            # Get current price data for the symbol
            symbol_data = market_data.get(symbol, None)
            if symbol_data is None:
                continue
            
            # Get current price
            if isinstance(symbol_data, dict) and 'close' in symbol_data:
                current_price = symbol_data['close'].iloc[-1]
            elif isinstance(symbol_data, pd.DataFrame) and 'close' in symbol_data.columns:
                current_price = symbol_data['close'].iloc[-1]
            else:
                continue
            
            # Check if stop loss has been hit
            stop_loss = position['stop_loss']
            if position['direction'] == 'buy' and current_price <= stop_loss:
                # Close long position due to stop loss
                position['exit_reason'] = 'stop_loss'
                positions_to_close.append(i)
                position_updates.append({
                    'position_id': position['id'],
                    'action': 'close',
                    'reason': 'stop_loss',
                    'price': current_price
                })
            elif position['direction'] == 'sell' and current_price >= stop_loss:
                # Close short position due to stop loss
                position['exit_reason'] = 'stop_loss'
                positions_to_close.append(i)
                position_updates.append({
                    'position_id': position['id'],
                    'action': 'close',
                    'reason': 'stop_loss',
                    'price': current_price
                })
            
            # Check if take profit has been hit
            take_profit = position['take_profit']
            if position['direction'] == 'buy' and current_price >= take_profit:
                # Close long position due to take profit
                position['exit_reason'] = 'take_profit'
                positions_to_close.append(i)
                position_updates.append({
                    'position_id': position['id'],
                    'action': 'close',
                    'reason': 'take_profit',
                    'price': current_price
                })
            elif position['direction'] == 'sell' and current_price <= take_profit:
                # Close short position due to take profit
                position['exit_reason'] = 'take_profit'
                positions_to_close.append(i)
                position_updates.append({
                    'position_id': position['id'],
                    'action': 'close',
                    'reason': 'take_profit',
                    'price': current_price
                })
            
            # Check if trailing stop has been activated and hit
            if 'trailing_stop' in position and position['trailing_stop']['active']:
                trailing_stop = position['trailing_stop']['level']
                if position['direction'] == 'buy' and current_price <= trailing_stop:
                    # Close long position due to trailing stop
                    position['exit_reason'] = 'trailing_stop'
                    positions_to_close.append(i)
                    position_updates.append({
                        'position_id': position['id'],
                        'action': 'close',
                        'reason': 'trailing_stop',
                        'price': current_price
                    })
                elif position['direction'] == 'sell' and current_price >= trailing_stop:
                    # Close short position due to trailing stop
                    position['exit_reason'] = 'trailing_stop'
                    positions_to_close.append(i)
                    position_updates.append({
                        'position_id': position['id'],
                        'action': 'close',
                        'reason': 'trailing_stop',
                        'price': current_price
                    })
            
            # Update trailing stop if needed
            if 'trailing_stop' in position and not position['trailing_stop']['active']:
                # Check if price has moved enough to activate trailing stop
                activation_level = position['trailing_stop']['activation_level']
                if position['direction'] == 'buy' and current_price >= activation_level:
                    # Activate trailing stop for long position
                    distance = position['trailing_stop']['distance']
                    position['trailing_stop']['active'] = True
                    position['trailing_stop']['level'] = current_price - distance
                    position_updates.append({
                        'position_id': position['id'],
                        'action': 'update_trailing_stop',
                        'active': True,
                        'level': position['trailing_stop']['level']
                    })
                elif position['direction'] == 'sell' and current_price <= activation_level:
                    # Activate trailing stop for short position
                    distance = position['trailing_stop']['distance']
                    position['trailing_stop']['active'] = True
                    position['trailing_stop']['level'] = current_price + distance
                    position_updates.append({
                        'position_id': position['id'],
                        'action': 'update_trailing_stop',
                        'active': True,
                        'level': position['trailing_stop']['level']
                    })
            elif 'trailing_stop' in position and position['trailing_stop']['active']:
                # Update trailing stop level if price moved favorably
                distance = position['trailing_stop']['distance']
                current_level = position['trailing_stop']['level']
                
                if position['direction'] == 'buy':
                    new_level = current_price - distance
                    if new_level > current_level:
                        position['trailing_stop']['level'] = new_level
                        position_updates.append({
                            'position_id': position['id'],
                            'action': 'update_trailing_stop',
                            'active': True,
                            'level': new_level
                        })
                else:  # sell
                    new_level = current_price + distance
                    if new_level < current_level:
                        position['trailing_stop']['level'] = new_level
                        position_updates.append({
                            'position_id': position['id'],
                            'action': 'update_trailing_stop',
                            'active': True,
                            'level': new_level
                        })
            
            # Check for signal reversal or timeout
            position_age = datetime.now() - datetime.fromisoformat(position['timestamp'])
            
            # Get current signals for this symbol
            symbol_signals = technical_signals.get(symbol, {}).get('signals', {})
            current_signal = symbol_signals.get('overall', 'NEUTRAL')
            
            # Check for signal reversal
            if (position['direction'] == 'buy' and current_signal == 'SELL' or 
                position['direction'] == 'sell' and current_signal == 'BUY'):
                # Close position due to signal reversal
                if position['current_pnl'] > 0:  # Only close if profitable
                    position['exit_reason'] = 'signal_reversal'
                    positions_to_close.append(i)
                    position_updates.append({
                        'position_id': position['id'],
                        'action': 'close',
                        'reason': 'signal_reversal',
                        'price': current_price
                    })
            
            # Check for position timeout (close after 30 days if not profitable)
            if position_age.days >= 30 and position['current_pnl'] <= 0:
                position['exit_reason'] = 'timeout'
                positions_to_close.append(i)
                position_updates.append({
                    'position_id': position['id'],
                    'action': 'close',
                    'reason': 'timeout',
                    'price': current_price
                })
        
        # Close positions (in reverse order to avoid index issues)
        for i in sorted(positions_to_close, reverse=True):
            position = self.portfolio['open_positions'][i]
            self._close_position(position, position['current_price'], position['exit_reason'])
            self.portfolio['open_positions'].pop(i)
        
        return position_updates
    
    def _identify_trading_opportunities(self, market_data, technical_signals, fundamental_insights, sentiment_insights, risk_assessment):
        """
        Identify new trading opportunities
        
        Args:
            market_data (dict): Current market data
            technical_signals (dict): Technical analysis signals
            fundamental_insights (dict): Fundamental analysis insights
            sentiment_insights (dict): Sentiment analysis insights
            risk_assessment (dict): Risk assessment from Risk Manager
        
        Returns:
            list: Potential trades
        """
        potential_trades = []
        
        # Check if we have exceeded maximum open positions
        max_positions = self.config.get('max_open_positions', 5)
        if len(self.portfolio['open_positions']) >= max_positions:
            self.logger.info(f"Maximum open positions ({max_positions}) reached. No new trades.")
            return potential_trades
        
        # Check margin level
        margin_level = self.portfolio['margin_level']
        min_margin_level = 200  # Minimum margin level to open new positions
        if margin_level < min_margin_level:
            self.logger.warning(f"Margin level too low ({margin_level}%). No new trades.")
            return potential_trades
        
        # Process each symbol
        for symbol in market_data:
            # Skip if we already have an open position for this symbol
            if any(p['symbol'] == symbol for p in self.portfolio['open_positions']):
                continue
            
            # Get symbol data
            symbol_data = market_data[symbol]
            if not isinstance(symbol_data, pd.DataFrame) or 'close' not in symbol_data.columns:
                continue
            
            # Get signals for this symbol
            tech_signal = technical_signals.get(symbol, {}).get('signals', {}).get('overall', 'NEUTRAL')
            fund_signal = fundamental_insights.get(symbol, {}).get('recommendation', {}).get('direction', 'neutral')
            sent_signal = sentiment_insights.get(symbol, {}).get('combined', {}).get('signal', 'neutral')
            
            # Get risk assessment
            symbol_risk = risk_assessment.get(symbol, {})
            
            # Check if signals align
            signals = [
                tech_signal.lower() if tech_signal.lower() in ['buy', 'sell'] else 'neutral',
                fund_signal,
                sent_signal
            ]
            
            buy_signals = signals.count('buy')
            sell_signals = signals.count('sell')
            
            # Minimum required aligned signals
            min_aligned_signals = 2
            
            # Check if we have enough aligned signals
            if buy_signals >= min_aligned_signals:
                # Check trading session
                if self._is_good_trading_session(symbol):
                    # Create potential buy trade
                    trade = self._create_potential_trade(symbol, 'buy', symbol_data, symbol_risk)
                    potential_trades.append(trade)
            elif sell_signals >= min_aligned_signals:
                # Check trading session
                if self._is_good_trading_session(symbol):
                    # Create potential sell trade
                    trade = self._create_potential_trade(symbol, 'sell', symbol_data, symbol_risk)
                    potential_trades.append(trade)
        
        # Sort trades by potential (highest first)
        potential_trades.sort(key=lambda x: x['potential'], reverse=True)
        
        # Limit to maximum new trades
        max_new_trades = max_positions - len(self.portfolio['open_positions'])
        return potential_trades[:max_new_trades]
    
    def _is_good_trading_session(self, symbol):
        """
        Check if current time is in a good trading session for the symbol
        
        Args:
            symbol (str): Symbol to check
        
        Returns:
            bool: True if it's a good trading session, False otherwise
        """
        # Get active forex sessions
        active_sessions = get_forex_sessions()
        
        # If no active sessions (weekend), don't trade
        if not active_sessions:
            return False
        
        # Get currency pair components
        if '/' in symbol:
            base, quote = symbol.split('/')
        else:
            # For symbols that don't use the standard format, assume it's always tradable
            return True
        
        # Check if the currencies are in major trading sessions
        major_currencies = {
            'USD': 'New York',
            'EUR': 'London',
            'GBP': 'London',
            'JPY': 'Tokyo',
            'AUD': 'Sydney',
            'NZD': 'Sydney',
            'CAD': 'New York',
            'CHF': 'London'
        }
        
        base_session = major_currencies.get(base)
        quote_session = major_currencies.get(quote)
        
        # If either currency's major session is active, it's a good time to trade
        if base_session in active_sessions or quote_session in active_sessions:
            return True
        
        # If both London and New York sessions are active, it's good for any major pair
        if 'London' in active_sessions and 'New York' in active_sessions:
            return True
        
        # Otherwise, not an optimal trading session
        return False
    
    def _create_potential_trade(self, symbol, direction, price_data, risk_assessment):
        """
        Create a potential trade
        
        Args:
            symbol (str): Symbol to trade
            direction (str): Trade direction ('buy' or 'sell')
            price_data (pd.DataFrame): Price data for the symbol
            risk_assessment (dict): Risk assessment for the symbol
        
        Returns:
            dict: Potential trade
        """
        # Get current price
        current_price = price_data['close'].iloc[-1]
        
        # Get position sizing from risk assessment
        position_sizing = risk_assessment.get('position_sizing', {})
        position_size = position_sizing.get('position_size', 0)
        position_value = position_sizing.get('position_value', 0)
        
        # Get risk management levels
        risk_management = risk_assessment.get('risk_management', {})
        stop_loss = risk_management.get('stop_loss', 0)
        take_profit = risk_management.get('take_profit', 0)
        
        # Get trailing stop parameters
        trailing_stop = risk_management.get('trailing_stop', {})
        trailing_activation = trailing_stop.get('activation', 0)
        trailing_distance = trailing_stop.get('distance', 0)
        
        # Calculate potential reward and risk
        if direction == 'buy':
            potential_reward = take_profit - current_price
            potential_risk = current_price - stop_loss
            trailing_activation_level = current_price + trailing_activation
        else:  # sell
            potential_reward = current_price - take_profit
            potential_risk = stop_loss - current_price
            trailing_activation_level = current_price - trailing_activation
        
        # Calculate risk-reward ratio
        risk_reward_ratio = potential_reward / potential_risk if potential_risk > 0 else 0
        
        # Calculate trade potential score (higher is better)
        potential = risk_reward_ratio
        
        # Adjust for market conditions
        market_conditions = risk_assessment.get('market_conditions', {})
        if market_conditions.get('trend', 'sideways') == 'uptrend' and direction == 'buy':
            potential *= 1.2
        elif market_conditions.get('trend', 'sideways') == 'downtrend' and direction == 'sell':
            potential *= 1.2
        
        # Create potential trade
        trade = {
            'symbol': symbol,
            'direction': direction,
            'price': current_price,
            'size': position_size,
            'value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'potential': potential,
            'trailing_stop': {
                'active': False,
                'activation_level': trailing_activation_level,
                'distance': trailing_distance
            }
        }
        
        return trade
    
    def _execute_trades(self, potential_trades, market_data):
        """
        Execute the potential trades
        
        Args:
            potential_trades (list): List of potential trades
            market_data (dict): Current market data
        
        Returns:
            list: Executed trades
        """
        executed_trades = []
        
        for trade in potential_trades:
            # Validate that we have enough margin
            required_margin = trade['value'] / self.leverage
            
            if required_margin > self.portfolio['free_margin']:
                self.logger.warning(f"Not enough free margin to execute trade for {trade['symbol']}")
                continue
            
            # Create position
            position = {
                'id': str(uuid.uuid4()),
                'symbol': trade['symbol'],
                'direction': trade['direction'],
                'entry_price': trade['price'],
                'size': trade['size'],
                'value': trade['value'],
                'stop_loss': trade['stop_loss'],
                'take_profit': trade['take_profit'],
                'trailing_stop': trade['trailing_stop'],
                'timestamp': datetime.now().isoformat(),
                'current_price': trade['price'],
                'current_pnl': 0,
                'current_pnl_percentage': 0,
                'margin': required_margin
            }
            
            # Add position to portfolio
            self.portfolio['open_positions'].append(position)
            
            # Add to executed trades
            executed_trades.append({
                'position_id': position['id'],
                'symbol': position['symbol'],
                'direction': position['direction'],
                'price': position['entry_price'],
                'size': position['size'],
                'value': position['value'],
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit']
            })
            
            self.logger.info(f"Executed {position['direction']} trade for {position['symbol']} at {position['entry_price']}")
        
        return executed_trades
    
    def _close_position(self, position, exit_price, reason):
        """
        Close a position and update portfolio
        
        Args:
            position (dict): Position to close
            exit_price (float): Exit price
            reason (str): Reason for closing
        """
        # Calculate profit/loss
        entry_price = position['entry_price']
        position_size = position['size']
        
        if position['direction'] == 'buy':
            pnl = (exit_price - entry_price) * position_size
        else:  # sell
            pnl = (entry_price - exit_price) * position_size
        
        pnl_percentage = (pnl / position['value']) * 100
        
        # Update position with exit details
        closed_position = position.copy()
        closed_position['exit_price'] = exit_price
        closed_position['exit_timestamp'] = datetime.now().isoformat()
        closed_position['pnl'] = pnl
        closed_position['pnl_percentage'] = pnl_percentage
        closed_position['exit_reason'] = reason
        
        # Add to closed positions
        self.portfolio['closed_positions'].append(closed_position)
        
        # Update account balance
        self.portfolio['account_balance'] += pnl
        
        self.logger.info(f"Closed {position['direction']} position for {position['symbol']} at {exit_price}. " +
                        f"P&L: {pnl:.2f} ({pnl_percentage:.2f}%). Reason: {reason}")
    
    def _calculate_performance_metrics(self):
        """
        Calculate portfolio performance metrics
        
        Returns:
            dict: Performance metrics
        """
        # Get closed positions
        closed_positions = self.portfolio['closed_positions']
        
        if not closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_profit': 0,
                'average_loss': 0,
                'largest_profit': 0,
                'largest_loss': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Calculate trade metrics
        total_trades = len(closed_positions)
        profitable_trades = sum(1 for p in closed_positions if p['pnl'] > 0)
        losing_trades = total_trades - profitable_trades
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        gross_profit = sum(p['pnl'] for p in closed_positions if p['pnl'] > 0)
        gross_loss = sum(p['pnl'] for p in closed_positions if p['pnl'] < 0)
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        average_profit = gross_profit / profitable_trades if profitable_trades > 0 else 0
        average_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        
        largest_profit = max([p['pnl'] for p in closed_positions]) if closed_positions else 0
        largest_loss = min([p['pnl'] for p in closed_positions]) if closed_positions else 0
        
        # Calculate equity curve
        account_history = self.portfolio['account_history']
        equity_curve = pd.Series([a['equity'] for a in account_history])
        
        # Calculate Sharpe ratio
        equity_returns = equity_curve.pct_change().dropna()
        sharpe_ratio = calculate_sharpe_ratio(equity_returns) if not equity_returns.empty else 0
        
        # Calculate max drawdown
        drawdown_result = calculate_drawdown(equity_curve)
        max_drawdown = drawdown_result['max_drawdown']
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_profit': average_profit,
            'average_loss': average_loss,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _get_portfolio_summary(self):
        """
        Get a summary of the current portfolio
        
        Returns:
            dict: Portfolio summary
        """
        return {
            'account_balance': self.portfolio['account_balance'],
            'equity': self.portfolio['equity'],
            'used_margin': self.portfolio['used_margin'],
            'free_margin': self.portfolio['free_margin'],
            'margin_level': self.portfolio['margin_level'],
            'open_positions_count': len(self.portfolio['open_positions']),
            'open_positions_value': sum(p['value'] for p in self.portfolio['open_positions']),
            'unrealized_pnl': sum(p['current_pnl'] for p in self.portfolio['open_positions']),
            'realized_pnl': sum(p['pnl'] for p in self.portfolio['closed_positions'])
        }
    
    def _save_portfolio(self):
        """
        Save the portfolio state to file
        """
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)

    # === Trade Execution Methods ===
    
    def execute_trade(
        self, 
        instrument: str, 
        units: float, 
        side: str, 
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a trade via the OANDA API.
        
        Args:
            instrument: Currency pair to trade (e.g., 'EUR_USD')
            units: Number of units to trade. Positive for buy, negative for sell.
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', or 'stop'
            price: Price for limit or stop orders
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            
        Returns:
            Dict containing order details and status
            
        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If there's an issue with the API connection
        """
        self.log_action("execute_trade", f"Executing {order_type} {side} order for {instrument}: {units} units")
        
        # Normalize instrument format (replace / with _)
        instrument = instrument.replace('/', '_')
        
        # Validate side
        side = side.lower()
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
        
        # Convert units based on side
        if side == 'sell':
            units = -abs(units)
        else:
            units = abs(units)
        
        # Execute order based on type
        try:
            if order_type.lower() == 'market':
                return self.place_market_order(instrument, units, stop_loss, take_profit)
            elif order_type.lower() == 'limit':
                if price is None:
                    raise ValueError("Price must be specified for limit orders")
                return self.place_limit_order(instrument, units, price, stop_loss, take_profit)
            elif order_type.lower() == 'stop':
                if price is None:
                    raise ValueError("Price must be specified for stop orders")
                return self.place_stop_order(instrument, units, price, stop_loss, take_profit)
            else:
                raise ValueError(f"Invalid order type: {order_type}. Must be 'market', 'limit', or 'stop'")
        except V20Error as e:
            self.handle_error(e)
            raise ConnectionError(f"Failed to execute trade: {str(e)}")
    
    def place_market_order(
        self, 
        instrument: str, 
        units: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a market order to execute immediately at current market price.
        
        Args:
            instrument: Currency pair to trade (e.g., 'EUR_USD')
            units: Number of units to trade. Positive for buy, negative for sell.
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            
        Returns:
            Dict containing order details and status
        """
        self.log_action("place_market_order", f"Placing market order for {instrument}: {units} units")
        
        # Create order request data
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK"  # Fill Or Kill - the order must be filled immediately or canceled
            }
        }
        
        # Add stop loss if specified
        if stop_loss is not None:
            order_data["order"]["stopLossOnFill"] = {"price": str(stop_loss)}
            
        # Add take profit if specified
        if take_profit is not None:
            order_data["order"]["takeProfitOnFill"] = {"price": str(take_profit)}
        
        # Create and send the order request
        order_request = orders.OrderCreate(accountID=self.account_id, data=order_data)
        
        try:
            response = self.api_client.request(order_request)
            
            # Log the trade
            trade_details = {
                'instrument': instrument,
                'direction': 'buy' if units > 0 else 'sell',
                'units': abs(units),
                'order_type': 'market',
                'order_id': response.get('orderCreateTransaction', {}).get('id'),
                'time': datetime.now().isoformat(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'executed'
            }
            self.log_trade(trade_details)
            
            return {
                'success': True,
                'order_id': response.get('orderCreateTransaction', {}).get('id'),
                'trade_id': response.get('orderFillTransaction', {}).get('tradeOpened', {}).get('tradeID'),
                'details': response
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def place_limit_order(
        self, 
        instrument: str, 
        units: float, 
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a limit order that will execute when price reaches the specified level.
        
        Args:
            instrument: Currency pair to trade (e.g., 'EUR_USD')
            units: Number of units to trade. Positive for buy, negative for sell.
            price: Price level at which the limit order should execute
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            
        Returns:
            Dict containing order details and status
        """
        self.log_action("place_limit_order", f"Placing limit order for {instrument}: {units} units at {price}")
        
        # Create order request data
        order_data = {
            "order": {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(units),
                "price": str(price),
                "timeInForce": "GTC"  # Good Till Cancelled
            }
        }
        
        # Add stop loss if specified
        if stop_loss is not None:
            order_data["order"]["stopLossOnFill"] = {"price": str(stop_loss)}
            
        # Add take profit if specified
        if take_profit is not None:
            order_data["order"]["takeProfitOnFill"] = {"price": str(take_profit)}
        
        # Create and send the order request
        order_request = orders.OrderCreate(accountID=self.account_id, data=order_data)
        
        try:
            response = self.api_client.request(order_request)
            
            # Log the trade
            trade_details = {
                'instrument': instrument,
                'direction': 'buy' if units > 0 else 'sell',
                'units': abs(units),
                'order_type': 'limit',
                'price': price,
                'order_id': response.get('orderCreateTransaction', {}).get('id'),
                'time': datetime.now().isoformat(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'pending'
            }
            self.log_trade(trade_details)
            
            return {
                'success': True,
                'order_id': response.get('orderCreateTransaction', {}).get('id'),
                'details': response
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def place_stop_order(
        self, 
        instrument: str, 
        units: float, 
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a stop order that will execute when price reaches the specified level.
        
        Args:
            instrument: Currency pair to trade (e.g., 'EUR_USD')
            units: Number of units to trade. Positive for buy, negative for sell.
            price: Price level at which the stop order should execute
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            
        Returns:
            Dict containing order details and status
        """
        self.log_action("place_stop_order", f"Placing stop order for {instrument}: {units} units at {price}")
        
        # Create order request data
        order_data = {
            "order": {
                "type": "STOP",
                "instrument": instrument,
                "units": str(units),
                "price": str(price),
                "timeInForce": "GTC"  # Good Till Cancelled
            }
        }
        
        # Add stop loss if specified
        if stop_loss is not None:
            order_data["order"]["stopLossOnFill"] = {"price": str(stop_loss)}
            
        # Add take profit if specified
        if take_profit is not None:
            order_data["order"]["takeProfitOnFill"] = {"price": str(take_profit)}
        
        # Create and send the order request
        order_request = orders.OrderCreate(accountID=self.account_id, data=order_data)
        
        try:
            response = self.api_client.request(order_request)
            
            # Log the trade
            trade_details = {
                'instrument': instrument,
                'direction': 'buy' if units > 0 else 'sell',
                'units': abs(units),
                'order_type': 'stop',
                'price': price,
                'order_id': response.get('orderCreateTransaction', {}).get('id'),
                'time': datetime.now().isoformat(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'pending'
            }
            self.log_trade(trade_details)
            
            return {
                'success': True,
                'order_id': response.get('orderCreateTransaction', {}).get('id'),
                'details': response
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def modify_order(self, order_id: str, new_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            new_parameters: Dict of parameters to update (e.g., 'price', 'units')
            
        Returns:
            Dict containing updated order details and status
        """
        self.log_action("modify_order", f"Modifying order {order_id}")
        
        # Create order data for modification
        order_data = {}
        
        # Add parameters to modify
        for key, value in new_parameters.items():
            if key in ['price', 'units', 'timeInForce']:
                order_data[key] = str(value)
        
        # Create and send the order modify request
        order_request = orders.OrderReplace(accountID=self.account_id, orderID=order_id, data=order_data)
        
        try:
            response = self.api_client.request(order_request)
            
            return {
                'success': True,
                'order_id': response.get('orderCreateTransaction', {}).get('id'),
                'details': response
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dict containing cancellation details and status
        """
        self.log_action("cancel_order", f"Cancelling order {order_id}")
        
        # Create and send the order cancel request
        order_request = orders.OrderCancel(accountID=self.account_id, orderID=order_id)
        
        try:
            response = self.api_client.request(order_request)
            
            return {
                'success': True,
                'order_id': order_id,
                'details': response
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    # === Position Management Methods ===
    
    def get_open_positions(self) -> Dict[str, Any]:
        """
        Get all currently open positions in the account.
        
        Returns:
            Dict containing open positions and summary data
        """
        self.log_action("get_open_positions", "Fetching open positions")
        
        # Create and send the positions request
        positions_request = positions.OpenPositions(accountID=self.account_id)
        
        try:
            response = self.api_client.request(positions_request)
            positions_data = response.get('positions', [])
            
            # Process and normalize position data
            processed_positions = []
            for position in positions_data:
                # Extract details
                instrument = position.get('instrument', '')
                
                # Get long and short position data
                long_units = float(position.get('long', {}).get('units', 0))
                short_units = float(position.get('short', {}).get('units', 0))
                
                # Calculate total position
                units = long_units + short_units  # Short will be negative
                
                # Determine net direction
                direction = 'long' if units > 0 else 'short'
                
                # Combine data
                position_info = {
                    'instrument': instrument,
                    'units': abs(units),
                    'direction': direction,
                    'long_units': long_units,
                    'short_units': short_units,
                    'unrealized_pl': float(position.get('unrealizedPL', 0)),
                    'pl': float(position.get('pl', 0)),
                    'position_id': position.get('id', ''),
                    'margin_used': float(position.get('marginUsed', 0)),
                    'details': position
                }
                
                processed_positions.append(position_info)
            
            # Calculate summary data
            total_positions = len(processed_positions)
            total_unrealized_pl = sum(p['unrealized_pl'] for p in processed_positions)
            total_margin_used = sum(p['margin_used'] for p in processed_positions)
            
            return {
                'success': True,
                'positions': processed_positions,
                'count': total_positions,
                'total_unrealized_pl': total_unrealized_pl,
                'total_margin_used': total_margin_used
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details,
                'positions': []
            }
    
    def close_position(self, position_id: str) -> Dict[str, Any]:
        """
        Close a specific position.
        
        Args:
            position_id: ID of the position to close
            
        Returns:
            Dict containing closing details and status
        """
        self.log_action("close_position", f"Closing position {position_id}")
        
        # Get all positions to find the correct one
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions to close',
                'details': positions_data
            }
        
        # Find the position by ID
        position_to_close = None
        for position in positions_data.get('positions', []):
            if position.get('position_id') == position_id:
                position_to_close = position
                break
        
        if not position_to_close:
            return {
                'success': False,
                'error': f'Position with ID {position_id} not found',
                'details': None
            }
        
        # Create position close request
        instrument = position_to_close.get('instrument')
        
        # Create the position close request
        close_data = {"longUnits": "ALL"}
        if position_to_close.get('direction') == 'short':
            close_data = {"shortUnits": "ALL"}
        
        position_request = positions.PositionClose(
            accountID=self.account_id,
            instrument=instrument,
            data=close_data
        )
        
        try:
            response = self.api_client.request(position_request)
            
            # Update trade log with close details
            self._update_trade_log_on_close(position_to_close, response)
            
            return {
                'success': True,
                'position_id': position_id,
                'instrument': instrument,
                'details': response
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open positions.
        
        Returns:
            Dict containing details of closed positions and status
        """
        self.log_action("close_all_positions", "Closing all open positions")
        
        # Get all positions
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions to close',
                'details': positions_data
            }
        
        positions_list = positions_data.get('positions', [])
        if not positions_list:
            return {
                'success': True,
                'message': 'No open positions to close',
                'closed': []
            }
        
        # Close each position
        closed_positions = []
        failed_positions = []
        
        for position in positions_list:
            position_id = position.get('position_id')
            result = self.close_position(position_id)
            
            if result.get('success', False):
                closed_positions.append({
                    'position_id': position_id,
                    'instrument': position.get('instrument'),
                    'details': result
                })
            else:
                failed_positions.append({
                    'position_id': position_id,
                    'instrument': position.get('instrument'),
                    'error': result.get('error')
                })
        
        return {
            'success': len(failed_positions) == 0,
            'closed_count': len(closed_positions),
            'failed_count': len(failed_positions),
            'closed_positions': closed_positions,
            'failed_positions': failed_positions
        }
    
    def modify_position_stops(
        self, 
        position_id: str, 
        stop_loss: Optional[float] = None, 
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modify stop loss and take profit levels for a position.
        
        Args:
            position_id: ID of the position to modify
            stop_loss: New stop loss price level
            take_profit: New take profit price level
            
        Returns:
            Dict containing modification details and status
        """
        self.log_action("modify_position_stops", f"Modifying stops for position {position_id}")
        
        # Get all positions to find the trades associated with the position
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions to modify',
                'details': positions_data
            }
        
        # Find the position by ID
        position_to_modify = None
        for position in positions_data.get('positions', []):
            if position.get('position_id') == position_id:
                position_to_modify = position
                break
        
        if not position_to_modify:
            return {
                'success': False,
                'error': f'Position with ID {position_id} not found',
                'details': None
            }
        
        # Get the trades in this position
        trades_list = position_to_modify.get('details', {}).get('trades', [])
        if not trades_list:
            return {
                'success': False,
                'error': 'No trades found for this position',
                'details': None
            }
        
        # Modify each trade in the position
        modified_trades = []
        failed_trades = []
        
        for trade in trades_list:
            trade_id = trade.get('id')
            
            # Prepare modification data
            trade_data = {}
            
            if stop_loss is not None:
                trade_data['stopLoss'] = {"price": str(stop_loss)}
                
            if take_profit is not None:
                trade_data['takeProfit'] = {"price": str(take_profit)}
                
            # Submit modification if we have changes
            if trade_data:
                trade_request = trades.TradeCRCDO(
                    accountID=self.account_id,
                    tradeID=trade_id,
                    data=trade_data
                )
                
                try:
                    response = self.api_client.request(trade_request)
                    modified_trades.append({
                        'trade_id': trade_id,
                        'details': response
                    })
                except V20Error as e:
                    self.handle_error(e)
                    failed_trades.append({
                        'trade_id': trade_id,
                        'error': str(e)
                    })
        
        return {
            'success': len(failed_trades) == 0,
            'position_id': position_id,
            'modified_trades_count': len(modified_trades),
            'failed_trades_count': len(failed_trades),
            'modified_trades': modified_trades,
            'failed_trades': failed_trades
        }
    
    def partial_close(self, position_id: str, units: float) -> Dict[str, Any]:
        """
        Partially close a position by reducing its size.
        
        Args:
            position_id: ID of the position to partially close
            units: Number of units to close
            
        Returns:
            Dict containing closing details and status
        """
        self.log_action("partial_close", f"Partially closing position {position_id}, {units} units")
        
        # Get all positions to find the correct one
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions to close',
                'details': positions_data
            }
        
        # Find the position by ID
        position_to_close = None
        for position in positions_data.get('positions', []):
            if position.get('position_id') == position_id:
                position_to_close = position
                break
        
        if not position_to_close:
            return {
                'success': False,
                'error': f'Position with ID {position_id} not found',
                'details': None
            }
        
        # Create position close request
        instrument = position_to_close.get('instrument')
        direction = position_to_close.get('direction')
        
        # Ensure units don't exceed position size
        max_units = position_to_close.get('units', 0)
        if units > max_units:
            units = max_units
        
        # Create the position close request
        close_data = {"longUnits": str(units)}
        if direction == 'short':
            close_data = {"shortUnits": str(units)}
        
        position_request = positions.PositionClose(
            accountID=self.account_id,
            instrument=instrument,
            data=close_data
        )
        
        try:
            response = self.api_client.request(position_request)
            
            # Update trade log with partial close details
            self._update_trade_log_on_partial_close(position_to_close, units, response)
            
            return {
                'success': True,
                'position_id': position_id,
                'instrument': instrument,
                'units_closed': units,
                'details': response
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def _update_trade_log_on_close(self, position: Dict[str, Any], close_response: Dict[str, Any]) -> None:
        """Update trade history when a position is closed"""
        try:
            # Extract details from position and close response
            instrument = position.get('instrument')
            direction = position.get('direction')
            units = position.get('units')
            profit_loss = float(close_response.get('orderFillTransaction', {}).get('pl', 0))
            
            # Find trade in history and update it
            for i, trade in enumerate(self.trade_history.to_dict('records')):
                if trade.get('trade_id') == position.get('position_id'):
                    # Update the trade record
                    self.trade_history.at[i, 'exit_price'] = close_response.get('orderFillTransaction', {}).get('price')
                    self.trade_history.at[i, 'exit_time'] = datetime.now().isoformat()
                    self.trade_history.at[i, 'profit_loss'] = profit_loss
                    self.trade_history.at[i, 'status'] = 'closed'
                    
                    # Save updated trade history
                    self.trade_history.to_csv(self.trade_history_file, index=False)
                    break
        except Exception as e:
            self.log_action("update_trade_log", f"Error updating trade log: {str(e)}")
    
    def _update_trade_log_on_partial_close(
        self, 
        position: Dict[str, Any], 
        units_closed: float, 
        close_response: Dict[str, Any]
    ) -> None:
        """Update trade history when a position is partially closed"""
        try:
            # Extract details from position and close response
            instrument = position.get('instrument')
            direction = position.get('direction')
            units = position.get('units')
            profit_loss = float(close_response.get('orderFillTransaction', {}).get('pl', 0))
            
            # Calculate remaining units
            remaining_units = units - units_closed
            
            # Find trade in history and update it
            for i, trade in enumerate(self.trade_history.to_dict('records')):
                if trade.get('trade_id') == position.get('position_id'):
                    # If all units closed, mark as closed
                    if remaining_units <= 0:
                        self.trade_history.at[i, 'exit_price'] = close_response.get('orderFillTransaction', {}).get('price')
                        self.trade_history.at[i, 'exit_time'] = datetime.now().isoformat()
                        self.trade_history.at[i, 'profit_loss'] = profit_loss
                        self.trade_history.at[i, 'status'] = 'closed'
                    else:
                        # Update with partial close details
                        self.trade_history.at[i, 'units'] = remaining_units
                        partial_profit = self.trade_history.at[i, 'profit_loss'] or 0
                        self.trade_history.at[i, 'profit_loss'] = partial_profit + profit_loss
                        self.trade_history.at[i, 'status'] = 'partial_close'
                    
                    # Save updated trade history
                    self.trade_history.to_csv(self.trade_history_file, index=False)
                    break
        except Exception as e:
            self.log_action("update_trade_log", f"Error updating trade log: {str(e)}")
    
    # === Portfolio Tracking Methods ===
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account balance, margin, and other details.
        
        Returns:
            Dict containing account summary data
        """
        self.log_action("get_account_summary", "Fetching account summary")
        
        # Create and send the account summary request
        account_request = accounts.AccountSummary(accountID=self.account_id)
        
        try:
            response = self.api_client.request(account_request)
            account = response.get('account', {})
            
            # Extract key account metrics
            summary = {
                'balance': float(account.get('balance', 0)),
                'unrealized_pl': float(account.get('unrealizedPL', 0)),
                'realized_pl': float(account.get('realizedPL', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'margin_used': float(account.get('marginUsed', 0)),
                'margin_rate': float(account.get('marginRate', 0)),
                'open_trade_count': int(account.get('openTradeCount', 0)),
                'pending_order_count': int(account.get('pendingOrderCount', 0)),
                'currency': account.get('currency', 'USD'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate additional metrics
            summary['equity'] = summary['balance'] + summary['unrealized_pl']
            summary['margin_used_percent'] = (summary['margin_used'] / summary['equity']) * 100 if summary['equity'] > 0 else 0
            
            # Update performance tracking
            self._update_performance_tracking(summary)
            
            return {
                'success': True,
                'summary': summary,
                'details': account
            }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculate key portfolio metrics including exposure, risk, and performance.
        
        Returns:
            Dict containing portfolio metrics
        """
        self.log_action("calculate_portfolio_metrics", "Calculating portfolio metrics")
        
        # Get account summary
        account_data = self.get_account_summary()
        if not account_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get account data',
                'details': account_data
            }
        
        # Get open positions
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions data',
                'details': positions_data
            }
        
        summary = account_data.get('summary', {})
        positions = positions_data.get('positions', [])
        
        # Calculate basic metrics
        equity = summary.get('equity', 0)
        balance = summary.get('balance', 0)
        unrealized_pl = summary.get('unrealized_pl', 0)
        
        # Calculate exposure metrics
        total_long_exposure = sum(p['units'] for p in positions if p['direction'] == 'long')
        total_short_exposure = sum(p['units'] for p in positions if p['direction'] == 'short')
        total_exposure = total_long_exposure + total_short_exposure
        net_exposure = total_long_exposure - total_short_exposure
        
        # Calculate exposure ratios
        exposure_ratio = (total_exposure / equity) if equity > 0 else 0
        net_exposure_ratio = (net_exposure / equity) if equity > 0 else 0
        
        # Get historical performance metrics
        drawdown = self.calculate_drawdown()
        sharpe = self.calculate_sharpe_ratio()
        win_rate = self.calculate_win_rate()
        
        # Build metrics dictionary
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'balance': balance,
            'equity': equity,
            'unrealized_pl': unrealized_pl,
            'realized_pl': summary.get('realized_pl', 0),
            'margin_used': summary.get('margin_used', 0),
            'margin_available': summary.get('margin_available', 0),
            
            # Exposure metrics
            'total_exposure': total_exposure,
            'long_exposure': total_long_exposure,
            'short_exposure': total_short_exposure,
            'net_exposure': net_exposure,
            'exposure_ratio': exposure_ratio,
            'net_exposure_ratio': net_exposure_ratio,
            
            # Risk metrics
            'margin_used_percent': summary.get('margin_used_percent', 0),
            'open_positions_count': len(positions),
            'current_drawdown': drawdown.get('current_drawdown_percentage', 0),
            'max_drawdown': drawdown.get('max_drawdown_percentage', 0),
            
            # Performance metrics
            'sharpe_ratio': sharpe.get('sharpe_ratio', 0),
            'win_rate': win_rate.get('win_rate', 0),
            'win_loss_ratio': win_rate.get('win_loss_ratio', 0),
            'average_win': win_rate.get('average_win', 0),
            'average_loss': win_rate.get('average_loss', 0),
            'expectancy': win_rate.get('expectancy', 0),
            
            # Currency exposure
            'currency_exposure': self._calculate_currency_exposure(positions)
        }
        
        return {
            'success': True,
            'metrics': metrics
        }
    
    def track_performance(self, timeframe: str = 'daily') -> Dict[str, Any]:
        """
        Track performance over a specified timeframe.
        
        Args:
            timeframe: Time period for tracking ('daily', 'weekly', 'monthly')
            
        Returns:
            Dict containing performance metrics over the specified timeframe
        """
        self.log_action("track_performance", f"Tracking performance over {timeframe} timeframe")
        
        # Load performance data
        if not hasattr(self, 'performance_data') or self.performance_data.empty:
            if self.performance_file.exists():
                try:
                    self.performance_data = pd.read_csv(self.performance_file)
                except Exception as e:
                    self.log_action("track_performance", f"Error loading performance data: {str(e)}")
                    self._create_new_performance_data()
            else:
                self._create_new_performance_data()
        
        # Convert timestamp to datetime for filtering
        if 'timestamp' in self.performance_data.columns:
            self.performance_data['timestamp'] = pd.to_datetime(self.performance_data['timestamp'])
        
        # Determine start date based on timeframe
        today = datetime.now()
        if timeframe == 'daily':
            start_date = today - timedelta(days=1)
        elif timeframe == 'weekly':
            start_date = today - timedelta(weeks=1)
        elif timeframe == 'monthly':
            start_date = today - timedelta(days=30)
        elif timeframe == 'yearly':
            start_date = today - timedelta(days=365)
        else:  # All time
            start_date = datetime.min
        
        # Filter performance data by timeframe
        period_data = self.performance_data[self.performance_data['timestamp'] >= start_date]
        
        if period_data.empty:
            return {
                'success': True,
                'message': f'No performance data available for {timeframe} timeframe',
                'data': {}
            }
        
        # Calculate key metrics for the period
        start_balance = period_data['balance'].iloc[0]
        end_balance = period_data['balance'].iloc[-1]
        
        # Calculate profit/loss
        absolute_profit = end_balance - start_balance
        percentage_profit = (absolute_profit / start_balance) * 100 if start_balance > 0 else 0
        
        # Calculate maximum drawdown in period
        max_drawdown = period_data['drawdown_percentage'].max() if 'drawdown_percentage' in period_data.columns else 0
        
        # Calculate other metrics if we have enough data points
        metrics = {
            'timeframe': timeframe,
            'start_date': start_date.isoformat(),
            'end_date': today.isoformat(),
            'start_balance': start_balance,
            'end_balance': end_balance,
            'absolute_profit': absolute_profit,
            'percentage_profit': percentage_profit,
            'max_drawdown': max_drawdown
        }
        
        # Add daily/weekly return data if we have enough points
        if len(period_data) >= 2:
            # Calculate returns
            period_data['return'] = period_data['balance'].pct_change()
            
            # Calculate volatility
            volatility = period_data['return'].std() * (252 ** 0.5)  # Annualized
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            avg_return = period_data['return'].mean()
            sharpe = (avg_return * 252) / volatility if volatility > 0 else 0
            
            metrics.update({
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'average_daily_return': avg_return
            })
        
        return {
            'success': True,
            'metrics': metrics,
            'data_points': len(period_data)
        }
    
    def calculate_drawdown(self) -> Dict[str, Any]:
        """
        Calculate current and maximum drawdown.
        
        Returns:
            Dict containing drawdown metrics
        """
        self.log_action("calculate_drawdown", "Calculating drawdown")
        
        # Load performance data
        if not hasattr(self, 'performance_data') or self.performance_data.empty:
            if self.performance_file.exists():
                try:
                    self.performance_data = pd.read_csv(self.performance_file)
                except Exception as e:
                    self.log_action("calculate_drawdown", f"Error loading performance data: {str(e)}")
                    self._create_new_performance_data()
            else:
                self._create_new_performance_data()
        
        # Get current account summary
        account_data = self.get_account_summary()
        if not account_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get account data',
                'details': account_data
            }
        
        current_equity = account_data.get('summary', {}).get('equity', 0)
        
        # Calculate drawdown using the imported function
        drawdown_results = calculate_drawdown(self.performance_data, current_equity)
        
        return drawdown_results
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Calculate Sharpe ratio for the portfolio.
        
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
            
        Returns:
            Dict containing Sharpe ratio and related metrics
        """
        self.log_action("calculate_sharpe_ratio", "Calculating Sharpe ratio")
        
        # Load performance data
        if not hasattr(self, 'performance_data') or self.performance_data.empty:
            if self.performance_file.exists():
                try:
                    self.performance_data = pd.read_csv(self.performance_file)
                except Exception as e:
                    self.log_action("calculate_sharpe_ratio", f"Error loading performance data: {str(e)}")
                    self._create_new_performance_data()
            else:
                self._create_new_performance_data()
        
        # Calculate using the imported function
        sharpe_results = calculate_sharpe_ratio(self.performance_data, risk_free_rate)
        
        return sharpe_results
    
    def calculate_win_rate(self) -> Dict[str, Any]:
        """
        Calculate win rate for closed trades.
        
        Returns:
            Dict containing win rate and related metrics
        """
        self.log_action("calculate_win_rate", "Calculating win rate")
        
        # Load trade history
        if not hasattr(self, 'trade_history') or self.trade_history.empty:
            if self.trade_history_file.exists():
                try:
                    self.trade_history = pd.read_csv(self.trade_history_file)
                except Exception as e:
                    self.log_action("calculate_win_rate", f"Error loading trade history: {str(e)}")
                    self._create_new_trade_history()
            else:
                self._create_new_trade_history()
        
        # Filter for closed trades
        closed_trades = self.trade_history[self.trade_history['status'] == 'closed']
        
        if closed_trades.empty:
            return {
                'win_rate': 0,
                'win_count': 0,
                'loss_count': 0,
                'total_trades': 0,
                'win_loss_ratio': 0,
                'average_win': 0,
                'average_loss': 0,
                'expectancy': 0,
                'profit_factor': 0
            }
        
        # Count wins and losses
        profitable_trades = closed_trades[closed_trades['profit_loss'] > 0]
        losing_trades = closed_trades[closed_trades['profit_loss'] <= 0]
        
        win_count = len(profitable_trades)
        loss_count = len(losing_trades)
        total_trades = len(closed_trades)
        
        # Calculate win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate win/loss ratio
        win_loss_ratio = win_count / loss_count if loss_count > 0 else float('inf')
        
        # Calculate average win and loss
        average_win = profitable_trades['profit_loss'].mean() if win_count > 0 else 0
        average_loss = abs(losing_trades['profit_loss'].mean()) if loss_count > 0 else 0
        
        # Calculate expectancy: (Win Rate × Average Win) - (Loss Rate × Average Loss)
        loss_rate = 1 - win_rate
        expectancy = (win_rate * average_win) - (loss_rate * average_loss)
        
        # Calculate profit factor: Total Profit / Total Loss
        total_profit = profitable_trades['profit_loss'].sum()
        total_loss = abs(losing_trades['profit_loss'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'win_rate': win_rate,
            'win_count': win_count,
            'loss_count': loss_count,
            'total_trades': total_trades,
            'win_loss_ratio': win_loss_ratio,
            'average_win': average_win,
            'average_loss': average_loss,
            'expectancy': expectancy,
            'profit_factor': profit_factor
        }
    
    def _update_performance_tracking(self, account_summary: Dict[str, Any]) -> None:
        """Update performance tracking with latest account data"""
        try:
            # Create new performance record
            new_record = {
                'timestamp': datetime.now(),
                'balance': account_summary.get('balance', 0),
                'equity': account_summary.get('equity', 0),
                'open_positions': account_summary.get('open_trade_count', 0),
                'floating_pl': account_summary.get('unrealized_pl', 0)
            }
            
            # Calculate daily P/L
            if len(self.performance_data) > 0:
                last_balance = self.performance_data['balance'].iloc[-1]
                new_record['daily_pl'] = new_record['balance'] - last_balance
            else:
                new_record['daily_pl'] = 0
            
            # Calculate drawdown
            max_equity = self.performance_data['equity'].max() if len(self.performance_data) > 0 else new_record['equity']
            if new_record['equity'] > max_equity:
                max_equity = new_record['equity']
            
            drawdown_amount = max_equity - new_record['equity']
            drawdown_percentage = (drawdown_amount / max_equity) * 100 if max_equity > 0 else 0
            
            new_record['drawdown_amount'] = drawdown_amount
            new_record['drawdown_percentage'] = drawdown_percentage
            
            # Append to performance data
            self.performance_data = pd.concat([self.performance_data, pd.DataFrame([new_record])], ignore_index=True)
            
            # Save to file
            self.performance_data.to_csv(self.performance_file, index=False)
            
        except Exception as e:
            self.log_action("update_performance", f"Error updating performance tracking: {str(e)}")
    
    def _calculate_currency_exposure(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate exposure by currency"""
        currencies = {}
        
        for position in positions:
            instrument = position.get('instrument', '')
            units = position.get('units', 0)
            direction = position.get('direction', 'long')
            
            # Skip if no instrument or units
            if not instrument or units == 0:
                continue
            
            # Split into base and quote currencies
            if '_' in instrument:
                base, quote = instrument.split('_')
            elif '/' in instrument:
                base, quote = instrument.split('/')
            else:
                # Can't parse, skip
                continue
            
            # Add to base currency (positive for long, negative for short)
            exposure = units if direction == 'long' else -units
            currencies[base] = currencies.get(base, 0) + exposure
            
            # Add to quote currency (negative for long, positive for short)
            reverse_exposure = -exposure  # Opposite exposure for quote currency
            currencies[quote] = currencies.get(quote, 0) + reverse_exposure
        
        return currencies
    
    # === Trade Management Methods ===
    
    def trail_stop_loss(self, position_id: str, distance: float) -> Dict[str, Any]:
        """
        Implement trailing stop loss for a position.
        
        Args:
            position_id: ID of the position to trail
            distance: Distance in pips for the trailing stop
            
        Returns:
            Dict containing trailing stop details and status
        """
        self.log_action("trail_stop_loss", f"Setting trailing stop for position {position_id}, distance: {distance}")
        
        # Get position details
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions data',
                'details': positions_data
            }
        
        # Find the position
        position = None
        for pos in positions_data.get('positions', []):
            if pos.get('position_id') == position_id:
                position = pos
                break
                
        if not position:
            return {
                'success': False,
                'error': f'Position {position_id} not found',
                'details': None
            }
        
        # Get current price for the instrument
        instrument = position.get('instrument')
        
        # Create pricing request
        pricing_request = pricing.PricingInfo(
            accountID=self.account_id,
            params={"instruments": instrument}
        )
        
        try:
            response = self.api_client.request(pricing_request)
            prices = response.get('prices', [])
            
            if not prices:
                return {
                    'success': False,
                    'error': f'No price data available for {instrument}',
                    'details': None
                }
            
            # Get bid/ask prices
            price_data = prices[0]
            bid = float(price_data.get('bids', [{}])[0].get('price', 0))
            ask = float(price_data.get('asks', [{}])[0].get('price', 0))
            
            # Calculate stop loss price based on direction
            direction = position.get('direction', 'long')
            current_price = ask if direction == 'long' else bid
            
            # Calculate stop price
            if direction == 'long':
                stop_price = current_price - distance
            else:
                stop_price = current_price + distance
            
            # Round to appropriate number of decimal places
            # This is a simplification - in real implementation, get exact precision from instrument details
            decimal_places = 5 if 'JPY' not in instrument else 3
            stop_price = round(stop_price, decimal_places)
            
            # Apply the new stop loss
            result = self.modify_position_stops(position_id, stop_loss=stop_price)
            
            if result.get('success', False):
                return {
                    'success': True,
                    'position_id': position_id,
                    'instrument': instrument,
                    'direction': direction,
                    'current_price': current_price,
                    'stop_price': stop_price,
                    'distance': distance,
                    'details': result
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to modify stop loss',
                    'details': result
                }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def break_even_stop(self, position_id: str, min_profit: float) -> Dict[str, Any]:
        """
        Move stop loss to break even after minimum profit is reached.
        
        Args:
            position_id: ID of the position to modify
            min_profit: Minimum profit in pips before moving to break even
            
        Returns:
            Dict containing break even stop details and status
        """
        self.log_action("break_even_stop", f"Setting break even stop for position {position_id}, min profit: {min_profit}")
        
        # Get position details
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions data',
                'details': positions_data
            }
        
        # Find the position
        position = None
        for pos in positions_data.get('positions', []):
            if pos.get('position_id') == position_id:
                position = pos
                break
                
        if not position:
            return {
                'success': False,
                'error': f'Position {position_id} not found',
                'details': None
            }
        
        # Get current price for the instrument
        instrument = position.get('instrument')
        
        # Create pricing request
        pricing_request = pricing.PricingInfo(
            accountID=self.account_id,
            params={"instruments": instrument}
        )
        
        try:
            # Get price data
            response = self.api_client.request(pricing_request)
            prices = response.get('prices', [])
            
            if not prices:
                return {
                    'success': False,
                    'error': f'No price data available for {instrument}',
                    'details': None
                }
            
            # Get bid/ask prices
            price_data = prices[0]
            bid = float(price_data.get('bids', [{}])[0].get('price', 0))
            ask = float(price_data.get('asks', [{}])[0].get('price', 0))
            
            # Get position information
            direction = position.get('direction', 'long')
            trades = position.get('details', {}).get('trades', [])
            
            if not trades:
                return {
                    'success': False,
                    'error': 'No trades found for this position',
                    'details': None
                }
            
            # Get entry price from first trade in position
            trade = trades[0]
            entry_price = float(trade.get('price', 0))
            
            if entry_price == 0:
                return {
                    'success': False,
                    'error': 'Could not determine entry price',
                    'details': None
                }
            
            # Calculate current profit in pips
            current_price = bid if direction == 'long' else ask
            pip_multiplier = 100 if 'JPY' in instrument else 10000
            
            if direction == 'long':
                profit_pips = (current_price - entry_price) * pip_multiplier
            else:
                profit_pips = (entry_price - current_price) * pip_multiplier
            
            # Check if profit exceeds minimum required
            if profit_pips < min_profit:
                return {
                    'success': True,
                    'action': 'none',
                    'message': f'Current profit {profit_pips:.1f} pips is below minimum {min_profit} pips',
                    'position_id': position_id,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'profit_pips': profit_pips
                }
            
            # Set stop loss to entry price (break even)
            result = self.modify_position_stops(position_id, stop_loss=entry_price)
            
            if result.get('success', False):
                return {
                    'success': True,
                    'action': 'moved_to_break_even',
                    'position_id': position_id,
                    'instrument': instrument,
                    'direction': direction,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'profit_pips': profit_pips,
                    'details': result
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to modify stop loss',
                    'details': result
                }
            
        except V20Error as e:
            self.handle_error(e)
            return {
                'success': False,
                'error': str(e),
                'details': e.details
            }
    
    def scale_in(self, position_id: str, additional_units: float) -> Dict[str, Any]:
        """
        Scale into an existing position by adding more units.
        
        Args:
            position_id: ID of the position to scale into
            additional_units: Number of additional units to add
            
        Returns:
            Dict containing scale-in details and status
        """
        self.log_action("scale_in", f"Scaling into position {position_id}, additional units: {additional_units}")
        
        # Get position details
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions data',
                'details': positions_data
            }
        
        # Find the position
        position = None
        for pos in positions_data.get('positions', []):
            if pos.get('position_id') == position_id:
                position = pos
                break
                
        if not position:
            return {
                'success': False,
                'error': f'Position {position_id} not found',
                'details': None
            }
        
        # Get position details
        instrument = position.get('instrument')
        direction = position.get('direction')
        
        # Determine units based on direction
        units = additional_units
        if direction == 'short':
            units = -additional_units
        
        # Place a market order to add to the position
        result = self.place_market_order(instrument, units)
        
        if result.get('success', False):
            return {
                'success': True,
                'position_id': position_id,
                'instrument': instrument,
                'direction': direction,
                'additional_units': additional_units,
                'details': result
            }
        else:
            return {
                'success': False,
                'error': 'Failed to add units to position',
                'details': result
            }
    
    def scale_out(self, position_id: str, units_to_close: float) -> Dict[str, Any]:
        """
        Scale out of an existing position by reducing units.
        
        Args:
            position_id: ID of the position to scale out of
            units_to_close: Number of units to close
            
        Returns:
            Dict containing scale-out details and status
        """
        self.log_action("scale_out", f"Scaling out of position {position_id}, units to close: {units_to_close}")
        
        # Use the partial close method to implement scaling out
        return self.partial_close(position_id, units_to_close)
    
    def implement_martingale(self, position_id: str, factor: float = 2.0) -> Dict[str, Any]:
        """
        Implement martingale strategy for losing positions by doubling position size.
        
        Args:
            position_id: ID of the position to apply strategy to
            factor: Multiplication factor for increasing position size (default: 2.0)
            
        Returns:
            Dict containing martingale implementation details and status
        """
        self.log_action("implement_martingale", f"Applying martingale strategy to position {position_id}, factor: {factor}")
        
        # Get position details
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions data',
                'details': positions_data
            }
        
        # Find the position
        position = None
        for pos in positions_data.get('positions', []):
            if pos.get('position_id') == position_id:
                position = pos
                break
                
        if not position:
            return {
                'success': False,
                'error': f'Position {position_id} not found',
                'details': None
            }
        
        # Get position details
        instrument = position.get('instrument')
        direction = position.get('direction')
        current_units = position.get('units')
        unrealized_pl = position.get('unrealized_pl')
        
        # Check if position is losing money
        if unrealized_pl >= 0:
            return {
                'success': True,
                'action': 'none',
                'message': 'Position is not losing money, martingale not applied',
                'position_id': position_id,
                'unrealized_pl': unrealized_pl
            }
        
        # Calculate new units using martingale factor
        additional_units = current_units * factor
        
        # Close current position and open a new larger one
        close_result = self.close_position(position_id)
        
        if not close_result.get('success', False):
            return {
                'success': False,
                'error': 'Failed to close existing position',
                'details': close_result
            }
        
        # Determine units based on direction for new position
        units = additional_units
        if direction == 'short':
            units = -additional_units
        
        # Open new larger position
        new_position = self.place_market_order(instrument, units)
        
        if new_position.get('success', False):
            return {
                'success': True,
                'action': 'martingale_applied',
                'position_id': position_id,
                'old_units': current_units,
                'new_units': additional_units,
                'factor': factor,
                'instrument': instrument,
                'direction': direction,
                'new_position': new_position
            }
        else:
            return {
                'success': False,
                'error': 'Failed to open new larger position',
                'details': new_position
            }
    
    def implement_anti_martingale(self, position_id: str, factor: float = 2.0) -> Dict[str, Any]:
        """
        Implement anti-martingale strategy for winning positions by increasing position size.
        
        Args:
            position_id: ID of the position to apply strategy to
            factor: Multiplication factor for increasing position size (default: 2.0)
            
        Returns:
            Dict containing anti-martingale implementation details and status
        """
        self.log_action("implement_anti_martingale", f"Applying anti-martingale strategy to position {position_id}, factor: {factor}")
        
        # Get position details
        positions_data = self.get_open_positions()
        if not positions_data.get('success', False):
            return {
                'success': False,
                'error': 'Failed to get positions data',
                'details': positions_data
            }
        
        # Find the position
        position = None
        for pos in positions_data.get('positions', []):
            if pos.get('position_id') == position_id:
                position = pos
                break
                
        if not position:
            return {
                'success': False,
                'error': f'Position {position_id} not found',
                'details': None
            }
        
        # Get position details
        instrument = position.get('instrument')
        direction = position.get('direction')
        current_units = position.get('units')
        unrealized_pl = position.get('unrealized_pl')
        
        # Check if position is winning money
        if unrealized_pl <= 0:
            return {
                'success': True,
                'action': 'none',
                'message': 'Position is not winning money, anti-martingale not applied',
                'position_id': position_id,
                'unrealized_pl': unrealized_pl
            }
        
        # Calculate additional units based on anti-martingale factor
        additional_units = current_units * (factor - 1.0)  # We already have current_units, so add (factor-1) times more
        
        # Determine units based on direction
        units = additional_units
        if direction == 'short':
            units = -additional_units
        
        # Add to the position
        result = self.place_market_order(instrument, units)
        
        if result.get('success', False):
            return {
                'success': True,
                'action': 'anti_martingale_applied',
                'position_id': position_id,
                'original_units': current_units,
                'additional_units': additional_units,
                'total_units': current_units + additional_units,
                'factor': factor,
                'instrument': instrument,
                'direction': direction,
                'details': result
            }
        else:
            return {
                'success': False,
                'error': 'Failed to add units to position',
                'details': result
            }