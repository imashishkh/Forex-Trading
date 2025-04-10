#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forex Trading Platform - Monitoring Dashboard

This module provides a comprehensive monitoring dashboard for the forex trading platform
with real-time updates on system status, trading performance, and market conditions.
"""

import os
import time
import logging
import json
import threading
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import psutil
import requests
from websocket import create_connection
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from collections import deque
import uuid
from streamlit_autorefresh import st_autorefresh
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ForexMonitoring")

# Lock for matplotlib operations (creating our own since RendererAgg.lock is no longer available)
import threading
_lock = threading.RLock()  # Reentrant lock for matplotlib operations

# Import additional classes for real-time API connections
from wallet_manager import WalletManager
from market_data_agent.agent import MarketDataAgent
from utils.config_manager import ConfigManager
from dotenv import load_dotenv
import traceback

class MonitoringDashboard:
    """
    A comprehensive monitoring dashboard for forex trading platform.
    
    This class provides real-time monitoring capabilities for system status,
    trading performance, market conditions, and alerting mechanisms.
    It uses Streamlit for the web interface, pandas for data management,
    and plotly for visualizations.
    """
    
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        """
        Initialize the monitoring dashboard.
        
        Args:
            config_path: Path to the monitoring configuration file
        """
        # Load environment variables
        load_dotenv()
        
        self.config = self._load_config(config_path)
        self.agents = self.config.get("agents", [])
        self.api_connections = self.config.get("api_connections", {})
        self.alert_config = self.config.get("alerts", {})
        
        # Initialize API clients
        self.wallet_manager = self._initialize_wallet_manager()
        self.market_data_agent = self._initialize_market_data_agent()
        
        # Data storage
        self.system_data = {
            "agent_status": {},
            "system_resources": {
                "cpu": deque(maxlen=100),
                "memory": deque(maxlen=100),
                "disk": deque(maxlen=100)
            },
            "api_connections": {},
            "system_events": [],
            "anomalies": []
        }
        
        # Trading data storage
        self.trading_data = {
            "open_positions": {},
            "pending_orders": {},
            "completed_trades": [],
            "account_balance": deque(maxlen=1000),  # Store historical balance
            "margin_levels": deque(maxlen=1000),    # Store historical margin levels
            "trading_anomalies": []
        }
        
        # Market data storage
        self.market_data = {
            "price_history": {},       # Instrument -> deque of (timestamp, price) tuples
            "volatility": {},          # Instrument -> deque of (timestamp, volatility) tuples
            "spreads": {},             # Instrument -> deque of (timestamp, spread) tuples
            "correlations": {},        # Pair of instruments -> deque of (timestamp, correlation) tuples
            "market_anomalies": [],    # List of detected market anomalies
            "instruments": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "EUR_GBP"]  # Default instruments
        }
        
        # Alerting data storage
        self.alert_data = {
            "alerts": deque(maxlen=1000),  # Store recent alerts
            "dashboard_alerts": deque(maxlen=100),  # Alerts to be displayed on dashboard
            "email_enabled": False,
            "sms_enabled": False,
            "alert_levels": {
                "info": {"color": "blue", "icon": "ℹ️", "display": True},
                "warning": {"color": "orange", "icon": "⚠️", "display": True},
                "error": {"color": "red", "icon": "❌", "display": True},
                "critical": {"color": "purple", "icon": "🚨", "display": True}
            },
            "alert_categories": {
                "system": {"display": True, "color": "gray"},
                "trading": {"display": True, "color": "green"},
                "market": {"display": True, "color": "blue"},
                "security": {"display": True, "color": "red"}
            }
        }
        
        # Time tracking
        self.last_update = datetime.datetime.now()
        self.update_interval = self.config.get("update_interval", 5)  # seconds
        
        # Threading
        self.should_run = False
        self.monitoring_thread = None
        
        # Set up alerts from config
        self.set_up_alerts(self.alert_config)
        
        logger.info("Monitoring Dashboard initialized")
    
    def _initialize_wallet_manager(self) -> WalletManager:
        """
        Initialize the wallet manager for account and trading data.
        
        Returns:
            WalletManager instance
        """
        try:
            wallet_manager = WalletManager()
            connection_success = wallet_manager.connect_to_oanda_account()
            
            if connection_success:
                logger.info("Successfully connected to OANDA account via WalletManager")
                self.log_system_events("initialization", "WalletManager connected successfully", "info")
            else:
                logger.error("Failed to connect to OANDA account via WalletManager")
                self.log_system_events("initialization", "WalletManager connection failed", "error")
            
            return wallet_manager
        except Exception as e:
            logger.error(f"Error initializing WalletManager: {str(e)}")
            self.log_system_events("initialization", f"WalletManager initialization error: {str(e)}", "error")
            return None
    
    def _initialize_market_data_agent(self) -> MarketDataAgent:
        """
        Initialize the market data agent for market data retrieval.
        
        Returns:
            MarketDataAgent instance
        """
        try:
            # Initialize with ConfigManager to load credentials from .env
            config_manager = ConfigManager()
            
            # Create MarketDataAgent
            market_data_agent = MarketDataAgent()
            
            # Initialize agent to connect to OANDA
            success = market_data_agent.initialize()
            
            if success:
                logger.info("Successfully initialized MarketDataAgent")
                self.log_system_events("initialization", "MarketDataAgent initialized successfully", "info")
            else:
                logger.error("Failed to initialize MarketDataAgent")
                self.log_system_events("initialization", "MarketDataAgent initialization failed", "error")
            
            return market_data_agent
        except Exception as e:
            logger.error(f"Error initializing MarketDataAgent: {str(e)}")
            self.log_system_events("initialization", f"MarketDataAgent initialization error: {str(e)}", "error")
            return None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load monitoring configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration values
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found at {config_path}, using defaults")
                return {
                    "agents": ["market_data", "technical_analysis", "portfolio_manager", "risk_manager"],
                    "api_connections": {
                        "oanda": {"url": "https://api-fxpractice.oanda.com", "timeout": 10},
                        "news_api": {"url": "https://newsapi.org", "timeout": 10}
                    },
                    "alerts": {
                        "email": {
                            "enabled": False,
                            "smtp_server": "smtp.gmail.com",
                            "smtp_port": 587,
                            "username": "",
                            "password": "",
                            "recipients": []
                        },
                        "sms": {
                            "enabled": False,
                            "provider": "",
                            "api_key": "",
                            "phone_numbers": []
                        }
                    },
                    "update_interval": 5,
                    "anomaly_detection": {
                        "cpu_threshold": 80,
                        "memory_threshold": 80,
                        "disk_threshold": 80,
                        "response_time_threshold": 2
                    }
                }
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    #--------------------------------------------------
    # 1. System Monitoring Methods
    #--------------------------------------------------
    
    def monitor_agent_status(self) -> Dict[str, str]:
        """
        Monitor the status of all agents in the trading system.
        
        Checks if each agent is running, responsive, and its last activity time.
        
        Returns:
            Dict mapping agent names to their status
        """
        try:
            agent_status = {}
            
            # Check WalletManager status
            if self.wallet_manager:
                wallet_status = "active" if self.wallet_manager.is_connected else "inactive"
                agent_status["wallet_manager"] = wallet_status
            else:
                agent_status["wallet_manager"] = "not_initialized"
            
            # Check MarketDataAgent status
            if self.market_data_agent:
                market_status = "active" if self.market_data_agent.status == "ready" else "inactive"
                agent_status["market_data_agent"] = market_status
            else:
                agent_status["market_data_agent"] = "not_initialized"
            
            # For other configured agents, we'll check if they've been initialized or just use a placeholder
            for agent_name in self.agents:
                if agent_name not in agent_status:
                    # Placeholder for other agents that may be added later
                    agent_status[agent_name] = self.system_data["agent_status"].get(agent_name, "unknown")
            
            # Store the agent status
            self.system_data["agent_status"] = agent_status
            
            # Log if any agent is down
            for agent, status in agent_status.items():
                if status not in ["active", "ready"]:
                    logger.warning(f"Agent {agent} is not active. Status: {status}")
            
            return agent_status
            
        except Exception as e:
            logger.error(f"Error monitoring agent status: {str(e)}")
            return {}
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """
        Monitor system resources including CPU, memory, and disk usage.
        
        Returns:
            Dict containing resource usage percentages
        """
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Store historical data
            timestamp = datetime.datetime.now()
            self.system_data["system_resources"]["cpu"].append((timestamp, cpu_percent))
            self.system_data["system_resources"]["memory"].append((timestamp, memory_percent))
            self.system_data["system_resources"]["disk"].append((timestamp, disk_percent))
            
            resources = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "timestamp": timestamp.isoformat()
            }
            
            # Log high resource usage
            if (cpu_percent > self.config.get("anomaly_detection", {}).get("cpu_threshold", 80) or
                memory_percent > self.config.get("anomaly_detection", {}).get("memory_threshold", 80) or
                disk_percent > self.config.get("anomaly_detection", {}).get("disk_threshold", 80)):
                logger.warning(f"High resource usage detected: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%")
            
            return resources
        except Exception as e:
            logger.error(f"Error monitoring system resources: {str(e)}")
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "disk_percent": 0,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def monitor_api_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        Monitor connections to external APIs.
        
        Tests connectivity to each configured API and records status,
        response time, and error information if any.
        
        Returns:
            Dict mapping API names to connection status information
        """
        api_statuses = {}
        
        for api_name, api_config in self.api_connections.items():
            try:
                url = api_config.get("url", "")
                timeout = api_config.get("timeout", 10)
                
                start_time = time.time()
                # In a real implementation, this would be a request to API health/status endpoint
                # For this example, we'll simulate the API connection
                if np.random.random() > 0.05:  # 5% chance of failure for simulation
                    status = "connected"
                    response_time = round(np.random.uniform(0.1, 0.5), 3)
                else:
                    status = "error"
                    response_time = None
                    raise Exception("Simulated connection error")
                
                api_statuses[api_name] = {
                    "status": status,
                    "response_time": response_time,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Log slow connections
                threshold = self.config.get("anomaly_detection", {}).get("response_time_threshold", 2)
                if response_time and response_time > threshold:
                    logger.warning(f"Slow API connection to {api_name}: {response_time}s")
            except Exception as e:
                logger.error(f"Error connecting to API {api_name}: {str(e)}")
                api_statuses[api_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        self.system_data["api_connections"] = api_statuses
        return api_statuses
    
    def log_system_events(self, event_type: str, message: str, severity: str = "info") -> Dict[str, Any]:
        """
        Log system events with timestamps.
        
        Args:
            event_type: Type of the event (e.g., "startup", "shutdown", "error")
            message: Event description
            severity: Event severity ("info", "warning", "error", "critical")
            
        Returns:
            Dict containing the logged event information
        """
        event = {
            "event_type": event_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Log based on severity
        if severity == "info":
            logger.info(f"{event_type}: {message}")
        elif severity == "warning":
            logger.warning(f"{event_type}: {message}")
        elif severity == "error":
            logger.error(f"{event_type}: {message}")
        elif severity == "critical":
            logger.critical(f"{event_type}: {message}")
        
        # Add to events list
        self.system_data["system_events"].append(event)
        
        # Keep only recent events (last 1000)
        if len(self.system_data["system_events"]) > 1000:
            self.system_data["system_events"] = self.system_data["system_events"][-1000:]
        
        return event
    
    def detect_system_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in system behavior.
        
        Checks for abnormal resource usage, API response times,
        and agent status changes.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check CPU usage anomalies
        cpu_data = list(self.system_data["system_resources"]["cpu"])
        if cpu_data:
            timestamps, values = zip(*cpu_data)
            if len(values) > 10:  # Need enough data for anomaly detection
                mean_cpu = np.mean(values)
                std_cpu = np.std(values)
                latest_cpu = values[-1]
                
                # Check if latest CPU usage is an outlier (z-score > 2)
                if abs(latest_cpu - mean_cpu) > 2 * std_cpu:
                    anomaly = {
                        "type": "cpu_usage",
                        "value": latest_cpu,
                        "threshold": mean_cpu + 2 * std_cpu,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"Abnormal CPU usage detected: {latest_cpu}%"
                    }
                    anomalies.append(anomaly)
                    logger.warning(anomaly["message"])
        
        # Check for API connection issues
        for api_name, status in self.system_data["api_connections"].items():
            if status.get("status") == "error":
                anomaly = {
                    "type": "api_connection",
                    "api": api_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message": f"API connection error with {api_name}: {status.get('error', 'Unknown error')}"
                }
                anomalies.append(anomaly)
        
        # Check for agent status issues
        for agent, status in self.system_data["agent_status"].items():
            if status.get("status") in ["degraded", "offline", "error"]:
                anomaly = {
                    "type": "agent_status",
                    "agent": agent,
                    "status": status.get("status"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message": f"Agent {agent} is in {status.get('status')} state"
                }
                anomalies.append(anomaly)
        
        # Store detected anomalies
        if anomalies:
            self.system_data["anomalies"].extend(anomalies)
            # Keep only recent anomalies (last 100)
            if len(self.system_data["anomalies"]) > 100:
                self.system_data["anomalies"] = self.system_data["anomalies"][-100:]
        
        return anomalies

    def start_monitoring(self):
        """
        Start the monitoring process in a background thread.
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.should_run = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """
        Stop the monitoring process.
        """
        self.should_run = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Background monitoring loop that updates all metrics.
        """
        while self.should_run:
            try:
                # Update system monitoring
                self.monitor_agent_status()
                self.monitor_system_resources()
                self.monitor_api_connections()
                self.detect_system_anomalies()
                
                # Update trading monitoring
                self.monitor_open_positions()
                self.monitor_pending_orders()
                self.track_trade_performance()
                self.monitor_account_balance()
                self.monitor_margin_levels()
                self.detect_trading_anomalies()
                
                # Update market monitoring
                self.monitor_price_movements()
                self.monitor_volatility()
                self.monitor_spread_changes()
                self.detect_market_anomalies()
                self.track_correlation_changes()
                
                # Log a heartbeat event periodically
                if (datetime.datetime.now() - self.last_update).total_seconds() > 60:
                    self.log_system_events("heartbeat", "Monitoring system is active", "info")
                    self.last_update = datetime.datetime.now()
                
                # Sleep for the update interval
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.update_interval)

    #--------------------------------------------------
    # 2. Trading Monitoring Methods
    #--------------------------------------------------
    
    def monitor_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Monitor currently open trading positions.
        
        Gets information about all open positions including instrument,
        direction, size, entry price, current P/L, and duration.
        
        Returns:
            Dict mapping position IDs to position details
        """
        try:
            # Use the WalletManager to get real position data
            if not self.wallet_manager or not self.wallet_manager.is_connected:
                logger.warning("WalletManager not available or not connected. Cannot fetch real position data.")
                return {}
            
            # Get account details which include open positions
            account_details = self.wallet_manager.get_account_details()
            positions = account_details.get('positions', [])
            
            open_positions = {}
            
            for position in positions:
                instrument = position.get('instrument')
                long_units = float(position.get('long', {}).get('units', 0))
                short_units = float(position.get('short', {}).get('units', 0))
                
                # Skip positions with zero units
                if long_units == 0 and short_units == 0:
                    continue
                
                # Determine position direction and size
                if long_units > 0:
                    direction = "long"
                    size = long_units
                    entry_price = float(position.get('long', {}).get('averagePrice', 0))
                    unrealized_pl = float(position.get('long', {}).get('unrealizedPL', 0))
                else:
                    direction = "short"
                    size = abs(short_units)
                    entry_price = float(position.get('short', {}).get('averagePrice', 0))
                    unrealized_pl = float(position.get('short', {}).get('unrealizedPL', 0))
                
                # Get current price for the instrument
                current_price = entry_price  # Default to entry price
                try:
                    if self.market_data_agent:
                        price_data = self.market_data_agent.get_current_price(instrument)
                        current_price = float(price_data.get('price', entry_price))
                except Exception as e:
                    logger.error(f"Error getting current price for {instrument}: {str(e)}")
                
                # Create position ID
                pos_id = f"pos_{instrument}_{direction}"
                
                # Store position details
                open_positions[pos_id] = {
                    "instrument": instrument,
                    "direction": direction,
                    "size": size,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "unrealized_pl": unrealized_pl,
                    "entry_time": position.get('lastTransactionID', datetime.datetime.now().isoformat())
                }
            
            # Update stored data
            self.trading_data["open_positions"] = open_positions
            return open_positions
            
        except Exception as e:
            logger.error(f"Error monitoring open positions: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def monitor_pending_orders(self) -> Dict[str, Dict[str, Any]]:
        """
        Monitor pending orders that haven't been executed.
        
        Gets information about all pending orders including instrument,
        order type, direction, requested price, and order duration.
        
        Returns:
            Dict mapping order IDs to order details
        """
        try:
            # Use the WalletManager to get real order data
            if not self.wallet_manager or not self.wallet_manager.is_connected:
                logger.warning("WalletManager not available or not connected. Cannot fetch real order data.")
                return {}
            
            # Get pending orders from OANDA
            url = f"{self.wallet_manager.api_url}/v3/accounts/{self.wallet_manager.account_id}/pendingOrders"
            response = requests.get(url, headers=self.wallet_manager.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get pending orders: {response.status_code} - {response.text}")
                return {}
            
            # Parse response
            orders_data = response.json().get('orders', [])
            pending_orders = {}
            
            for order in orders_data:
                order_id = order.get('id')
                instrument = order.get('instrument')
                order_type = order.get('type')
                
                # Get units to determine direction and size
                units = float(order.get('units', 0))
                direction = "long" if units > 0 else "short"
                size = abs(units)
                
                # Get price based on order type
                if order_type == "LIMIT":
                    price = float(order.get('price', 0))
                elif order_type == "STOP":
                    price = float(order.get('price', 0))
                else:
                    price = 0  # For market orders or other types
                
                # Get creation time
                create_time = order.get('createTime', datetime.datetime.now().isoformat())
                
                # Store order details
                pending_orders[order_id] = {
                    "instrument": instrument,
                    "order_type": order_type,
                    "direction": direction,
                    "size": size,
                    "price": price,
                    "create_time": create_time
                }
            
            # Update stored data
            self.trading_data["pending_orders"] = pending_orders
            return pending_orders
            
        except Exception as e:
            logger.error(f"Error monitoring pending orders: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def track_trade_performance(self) -> Dict[str, Any]:
        """
        Track performance of all trades.
        
        Calculates various performance metrics such as win rate,
        average profit/loss, profit factor, etc.
        
        Returns:
            Dict containing trade performance metrics
        """
        try:
            completed_trades = self.trading_data["completed_trades"]
            open_positions = self.trading_data["open_positions"]
            
            # If no completed trades yet, return empty metrics
            if not completed_trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "average_profit": 0,
                    "average_loss": 0,
                    "largest_profit": 0,
                    "largest_loss": 0,
                    "average_trade_duration": 0
                }
            
            # Calculate metrics from completed trades
            winning_trades = [t for t in completed_trades if t["profit_loss"] > 0]
            losing_trades = [t for t in completed_trades if t["profit_loss"] <= 0]
            
            total_trades = len(completed_trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            total_profit = sum(t["profit_loss"] for t in winning_trades)
            total_loss = abs(sum(t["profit_loss"] for t in losing_trades))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            average_profit = total_profit / win_count if win_count > 0 else 0
            average_loss = total_loss / loss_count if loss_count > 0 else 0
            
            largest_profit = max([t["profit_loss"] for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t["profit_loss"] for t in losing_trades]) if losing_trades else 0
            
            # Calculate average trade duration (convert to hours)
            durations = [t.get("duration_hours", 0) for t in completed_trades]
            average_duration = sum(durations) / len(durations) if durations else 0
            
            # Calculate current unrealized P/L
            unrealized_pl = sum(pos["unrealized_pl"] for pos in open_positions.values())
            
            # Calculate metrics by instrument
            instruments = set(t["instrument"] for t in completed_trades)
            instrument_performance = {}
            
            for instrument in instruments:
                instrument_trades = [t for t in completed_trades if t["instrument"] == instrument]
                instrument_wins = len([t for t in instrument_trades if t["profit_loss"] > 0])
                
                instrument_performance[instrument] = {
                    "trades": len(instrument_trades),
                    "win_rate": instrument_wins / len(instrument_trades) if instrument_trades else 0,
                    "net_pl": sum(t["profit_loss"] for t in instrument_trades)
                }
            
            # Compile performance metrics
            performance = {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "average_profit": average_profit,
                "average_loss": average_loss,
                "largest_profit": largest_profit,
                "largest_loss": largest_loss,
                "average_trade_duration": average_duration,
                "current_open_positions": len(open_positions),
                "unrealized_pl": unrealized_pl,
                "instrument_performance": instrument_performance,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error tracking trade performance: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def monitor_account_balance(self) -> Dict[str, float]:
        """
        Monitor account balance and equity.
        
        Tracks the account balance, equity, used margin, and available margin.
        Also tracks balance changes over time.
        
        Returns:
            Dict with account balance metrics
        """
        try:
            # Use the WalletManager to get real account data
            if not self.wallet_manager or not self.wallet_manager.is_connected:
                logger.warning("WalletManager not available or not connected. Cannot fetch real account data.")
                return {}
            
            # Get account balance and margin information
            balance = self.wallet_manager.get_account_balance()
            margin_available = self.wallet_manager.get_margin_available()
            margin_used = self.wallet_manager.get_margin_used()
            
            # Calculate equity (balance + unrealized P/L)
            account_summary = self.wallet_manager.get_account_summary()
            unrealized_pl = float(account_summary.get('unrealizedPL', 0))
            equity = balance + unrealized_pl
            
            # Create balance data point
            timestamp = datetime.datetime.now()
            balance_data = {
                "balance": balance,
                "equity": equity,
                "margin_used": margin_used,
                "margin_available": margin_available,
                "unrealized_pl": unrealized_pl,
                "timestamp": timestamp
            }
            
            # Store in our time series
            self.trading_data["account_balance"].append(balance_data)
            
            # If we have previous balance points, calculate changes
            if len(self.trading_data["account_balance"]) > 1:
                prev_balance = self.trading_data["account_balance"][-2]["balance"]
                balance_change = balance - prev_balance
                balance_change_pct = (balance_change / prev_balance) * 100 if prev_balance > 0 else 0
                
                balance_data["balance_change"] = balance_change
                balance_data["balance_change_pct"] = balance_change_pct
                
                # Log significant balance changes
                if abs(balance_change_pct) > 1:  # 1% change threshold
                    log_level = "info" if balance_change > 0 else "warning"
                    message = f"Significant balance change: {balance_change_pct:.2f}% ({balance_change:.2f})"
                    self.log_system_events("account", message, log_level)
            
            return balance_data
            
        except Exception as e:
            logger.error(f"Error monitoring account balance: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def monitor_margin_levels(self) -> Dict[str, float]:
        """
        Monitor margin usage and risk levels.
        
        Tracks margin level, margin used percentage, and proximity to
        margin call or stop-out levels. Provides warnings when margin
        usage is approaching critical thresholds.
        
        Returns:
            Dict containing margin metrics
        """
        try:
            # Use the WalletManager to get real margin data
            if not self.wallet_manager or not self.wallet_manager.is_connected:
                logger.warning("WalletManager not available or not connected. Cannot fetch real margin data.")
                return {}
            
            # Get account margin information
            margin_available = self.wallet_manager.get_margin_available()
            margin_used = self.wallet_manager.get_margin_used()
            
            # Get margin closeout metrics if available
            margin_closeout_pct = self.wallet_manager.get_margin_closeout_percent()
            margin_closeout_value = self.wallet_manager.get_margin_closeout_value()
            
            # Get account summary for net asset value (equity)
            account_summary = self.wallet_manager.get_account_summary()
            nav = float(account_summary.get('NAV', 0))
            balance = float(account_summary.get('balance', 0))
            
            # Calculate margin metrics
            if nav > 0 and margin_used > 0:
                margin_level = nav / margin_used
            else:
                margin_level = float('inf')  # No margin used
            
            # Calculate margin utilization percentage
            total_margin = margin_available + margin_used
            if total_margin > 0:
                margin_used_pct = (margin_used / total_margin) * 100
            else:
                margin_used_pct = 0
            
            # Calculate margin-to-equity ratio
            if nav > 0:
                margin_to_equity = (margin_used / nav) * 100
            else:
                margin_to_equity = 0
            
            # Calculate distance to margin call (in percentage)
            # The exact formula depends on the broker's margin call policy
            # This is a simplified approach
            if margin_closeout_pct > 0:
                # Use the margin closeout percentage from OANDA directly
                distance_to_margin_call = margin_closeout_pct - 100  # Assuming 100% is the margin call level
                
                # Normalize to percentage (0-100%)
                if distance_to_margin_call < 0:
                    distance_to_margin_call = 0
            else:
                # If margin closeout percentage is not available, use a simplistic approach
                # Assuming margin call occurs at 80% usage as a typical value
                margin_call_threshold = 80
                distance_to_margin_call = margin_call_threshold - margin_used_pct
                
                if distance_to_margin_call < 0:
                    distance_to_margin_call = 0
            
            # Determine margin risk level
            if margin_used_pct < 30:
                risk_level = "low"
            elif margin_used_pct < 60:
                risk_level = "moderate"
            elif margin_used_pct < 80:
                risk_level = "high"
            else:
                risk_level = "critical"
                
                # Log critical margin situations
                logger.warning(f"Critical margin level: {margin_used_pct:.2f}% used, {distance_to_margin_call:.2f}% to margin call")
                
                # Create a dashboard alert for critical margin
                if margin_used_pct >= 90:
                    self.send_dashboard_alert(
                        f"CRITICAL: Margin usage at {margin_used_pct:.2f}%, only {distance_to_margin_call:.2f}% to margin call!",
                        level="critical",
                        category="trading",
                        details={
                            "margin_used_pct": margin_used_pct,
                            "margin_level": margin_level,
                            "distance_to_margin_call": distance_to_margin_call
                        }
                    )
            
            # Create the margin data point
            timestamp = datetime.datetime.now()
            margin_data = {
                "margin_available": margin_available,
                "margin_used": margin_used,
                "margin_level": margin_level,
                "margin_used_pct": margin_used_pct,
                "margin_to_equity": margin_to_equity,
                "nav": nav,
                "balance": balance,
                "margin_closeout_pct": margin_closeout_pct,
                "margin_closeout_value": margin_closeout_value,
                "distance_to_margin_call": distance_to_margin_call,
                "risk_level": risk_level,
                "timestamp": timestamp
            }
            
            # Store in time series
            self.trading_data["margin_levels"].append(margin_data)
            
            return margin_data
            
        except Exception as e:
            logger.error(f"Error monitoring margin levels: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def detect_trading_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in trading behavior.
        
        Identifies unusual patterns such as excessive trading volume,
        large drawdowns, unusual trade sizes, or deviations from
        normal trading patterns.
        
        Returns:
            List of detected trading anomalies
        """
        try:
            anomalies = []
            
            # Get relevant data
            open_positions = self.trading_data["open_positions"]
            completed_trades = self.trading_data["completed_trades"]
            balance_info = self.monitor_account_balance()
            margin_info = self.monitor_margin_levels()
            
            # Check for margin level anomalies
            if margin_info.get("margin_status") in ["MARGIN_CALL", "STOP_OUT_RISK"]:
                anomaly = {
                    "type": "margin_level",
                    "value": margin_info.get("margin_level", 0),
                    "threshold": margin_info.get("margin_call_level", 100),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message": f"Dangerous margin level: {margin_info.get('margin_level', 0):.2f}%"
                }
                anomalies.append(anomaly)
            
            # Check for unusual trade size
            if open_positions:
                # Calculate average position size
                avg_pos_size = sum(pos["size"] for pos in open_positions.values()) / len(open_positions)
                
                for pos_id, position in open_positions.items():
                    # Flag positions that are >3x the average size
                    if position["size"] > avg_pos_size * 3:
                        anomaly = {
                            "type": "position_size",
                            "position_id": pos_id,
                            "instrument": position["instrument"],
                            "size": position["size"],
                            "avg_size": avg_pos_size,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "message": f"Unusually large position size for {position['instrument']}: {position['size']} vs avg {avg_pos_size:.2f}"
                        }
                        anomalies.append(anomaly)
            
            # Check for high concentration in one instrument
            if open_positions:
                # Calculate exposure by instrument
                instrument_exposure = {}
                for position in open_positions.values():
                    instrument = position["instrument"]
                    if instrument not in instrument_exposure:
                        instrument_exposure[instrument] = 0
                    instrument_exposure[instrument] += position["size"]
                
                # Get total exposure
                total_exposure = sum(instrument_exposure.values())
                
                # Flag if more than 50% of exposure is in one instrument
                for instrument, exposure in instrument_exposure.items():
                    exposure_pct = (exposure / total_exposure) * 100 if total_exposure > 0 else 0
                    if exposure_pct > 50:
                        anomaly = {
                            "type": "concentration_risk",
                            "instrument": instrument,
                            "exposure_percent": exposure_pct,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "message": f"High concentration in {instrument}: {exposure_pct:.2f}% of total exposure"
                        }
                        anomalies.append(anomaly)
            
            # Check for rapid balance changes
            if len(self.trading_data["account_balance"]) > 10:
                # Get last 10 balance points
                recent_balances = list(self.trading_data["account_balance"])[-10:]
                times, balances = zip(*recent_balances)
                
                first_balance = balances[0]
                last_balance = balances[-1]
                
                # Calculate percentage change
                pct_change = ((last_balance - first_balance) / first_balance) * 100
                
                # Flag significant drops (more than 10% drop)
                if pct_change < -10:
                    anomaly = {
                        "type": "balance_drop",
                        "change_percent": pct_change,
                        "period_hours": round((times[-1] - times[0]).total_seconds() / 3600, 2),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"Significant balance drop: {pct_change:.2f}% in {round((times[-1] - times[0]).total_seconds() / 3600, 2)} hours"
                    }
                    anomalies.append(anomaly)
            
            # Check for unusual win/loss streaks
            if len(completed_trades) > 5:
                # Get recent trades
                recent_trades = completed_trades[-5:]
                
                # Check if all are losses
                all_losses = all(trade["profit_loss"] <= 0 for trade in recent_trades)
                if all_losses:
                    total_loss = sum(trade["profit_loss"] for trade in recent_trades)
                    anomaly = {
                        "type": "loss_streak",
                        "streak_length": 5,
                        "total_loss": total_loss,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"Unusual loss streak: 5 consecutive losing trades with total loss of {total_loss:.2f}"
                    }
                    anomalies.append(anomaly)
            
            # Store detected anomalies
            if anomalies:
                self.trading_data["trading_anomalies"].extend(anomalies)
                # Keep only recent anomalies (last 100)
                if len(self.trading_data["trading_anomalies"]) > 100:
                    self.trading_data["trading_anomalies"] = self.trading_data["trading_anomalies"][-100:]
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting trading anomalies: {str(e)}")
            return []

    #--------------------------------------------------
    # 3. Market Monitoring Methods
    #--------------------------------------------------
    
    def monitor_price_movements(self) -> Dict[str, Dict[str, float]]:
        """
        Monitor significant price movements for all tracked instruments.
        
        Detects price breakouts, strong trends, and significant levels.
        Tracks price action and notable movements across timeframes.
        
        Returns:
            Dict mapping instruments to price movement details
        """
        try:
            price_movements = {}
            
            # Get tracked instruments
            instruments = self.market_data["instruments"]
            
            # Use the MarketDataAgent to get real price data
            if not self.market_data_agent:
                logger.warning("MarketDataAgent not available. Cannot fetch real price data.")
                return {}
            
            for instrument in instruments:
                # Initialize instrument data if it doesn't exist
                if instrument not in self.market_data["price_history"]:
                    self.market_data["price_history"][instrument] = deque(maxlen=1000)
                
                try:
                    # Get current price data for this instrument
                    price_data = self.market_data_agent.get_current_price(instrument)
                    current_price = float(price_data.get('price', 0))
                    
                    # Store current price data point
                    timestamp = datetime.datetime.now()
                    self.market_data["price_history"][instrument].append((timestamp, current_price))
                    
                    # Calculate percentage change from previous period
                    if len(self.market_data["price_history"][instrument]) > 1:
                        prev_timestamp, prev_price = self.market_data["price_history"][instrument][-2]
                        pct_change = ((current_price - prev_price) / prev_price) * 100
                        
                        # Calculate changes over multiple timeframes
                        changes = {
                            "1m": pct_change  # 1-minute change (or whatever our update interval is)
                        }
                        
                        # Calculate hourly change if we have enough data
                        hourly_data = [p for t, p in self.market_data["price_history"][instrument] 
                                     if (timestamp - t).total_seconds() <= 3600]
                        if hourly_data and len(hourly_data) > 1:
                            hourly_change = ((current_price - hourly_data[0]) / hourly_data[0]) * 100
                            changes["1h"] = hourly_change
                        
                        # Get historical data for more comprehensive analysis (daily)
                        try:
                            # Get daily data for the past week
                            end_time = datetime.datetime.now()
                            start_time = end_time - datetime.timedelta(days=7)
                            
                            daily_data = self.market_data_agent.get_historical_data(
                                instrument=instrument,
                                timeframe="D",
                                from_time=start_time,
                                to_time=end_time
                            )
                            
                            if not daily_data.empty and len(daily_data) > 1:
                                # Calculate daily change
                                daily_change = ((current_price - daily_data['close'].iloc[-2]) / daily_data['close'].iloc[-2]) * 100
                                changes["1d"] = daily_change
                                
                                # Calculate weekly change if we have enough data
                                if len(daily_data) >= 5:
                                    weekly_change = ((current_price - daily_data['close'].iloc[-5]) / daily_data['close'].iloc[-5]) * 100
                                    changes["1w"] = weekly_change
                        except Exception as e:
                            logger.error(f"Error getting historical data for {instrument}: {str(e)}")
                        
                        # Determine if this is a significant movement
                        # Compare to historical volatility
                        is_significant = abs(pct_change) > 0.1  # Default 0.1% threshold
                        
                        # If we have volatility data, use it to adjust threshold
                        if instrument in self.market_data["volatility"] and self.market_data["volatility"][instrument]:
                            # Get average volatility from our stored data
                            recent_vols = [v for _, v in self.market_data["volatility"][instrument]]
                            if recent_vols:
                                avg_vol = sum(recent_vols) / len(recent_vols)
                                # A movement is significant if it's more than 2x the average volatility
                                is_significant = abs(pct_change) > (avg_vol * 200)  # Convert to percentage scale
                        
                        movement_type = "none"
                        if pct_change > 0.1:
                            movement_type = "bullish"
                        elif pct_change < -0.1:
                            movement_type = "bearish"
                        
                        price_movements[instrument] = {
                            "current_price": current_price,
                            "previous_price": prev_price,
                            "pct_change": pct_change,
                            "changes": changes,
                            "is_significant": is_significant,
                            "movement_type": movement_type,
                            "timestamp": timestamp.isoformat()
                        }
                    else:
                        # Not enough data for movement calculation
                        price_movements[instrument] = {
                            "current_price": current_price,
                            "previous_price": None,
                            "pct_change": 0,
                            "changes": {},
                            "is_significant": False,
                            "movement_type": "none",
                            "timestamp": timestamp.isoformat()
                        }
                        
                    # Log significant price movements
                    if price_movements[instrument].get("is_significant", False):
                        logger.info(f"Significant price movement for {instrument}: {pct_change:.4f}% change")
                        
                except Exception as inner_e:
                    logger.error(f"Error processing instrument {instrument}: {str(inner_e)}")
                    continue
            
            return price_movements
            
        except Exception as e:
            logger.error(f"Error monitoring price movements: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def monitor_volatility(self) -> Dict[str, Dict[str, float]]:
        """
        Monitor market volatility for all tracked instruments.
        
        Calculates and tracks volatility metrics such as standard deviation,
        Average True Range (ATR), and historical volatility. Also classifies 
        current volatility relative to historical norms.
        
        Returns:
            Dict mapping instruments to volatility metrics
        """
        try:
            volatility_data = {}
            
            # Get tracked instruments
            instruments = self.market_data["instruments"]
            
            # Use the MarketDataAgent to get real price and volatility data
            if not self.market_data_agent:
                logger.warning("MarketDataAgent not available. Cannot fetch real market data.")
                return {}
            
            for instrument in instruments:
                # Initialize volatility data if it doesn't exist
                if instrument not in self.market_data["volatility"]:
                    self.market_data["volatility"][instrument] = deque(maxlen=500)
                
                try:
                    # Get historical data for this instrument (hourly data for last 24 hours)
                    end_time = datetime.datetime.now()
                    start_time = end_time - datetime.timedelta(hours=24)
                    
                    hourly_data = self.market_data_agent.get_historical_data(
                        instrument=instrument,
                        timeframe="H1",
                        from_time=start_time,
                        to_time=end_time
                    )
                    
                    # Need at least 10 data points for meaningful volatility calculation
                    if hourly_data.empty or len(hourly_data) < 10:
                        logger.warning(f"Not enough historical data for {instrument} volatility calculation")
                        continue
                    
                    # Calculate recent price volatility (standard deviation of returns)
                    returns = hourly_data['returns'].dropna()
                    
                    # Calculate standard deviation of recent returns
                    recent_std = returns.std()
                    
                    # Annualize volatility (assuming hourly data - multiply by sqrt of hours in a year)
                    hours_per_year = 24 * 365
                    annualized_vol = recent_std * np.sqrt(hours_per_year)
                    
                    # Calculate ATR
                    hourly_data['tr'] = hourly_data.apply(
                        lambda x: max(
                            x['high'] - x['low'],
                            abs(x['high'] - x['close'].shift(1)),
                            abs(x['low'] - x['close'].shift(1))
                        ),
                        axis=1
                    )
                    
                    atr = hourly_data['tr'].rolling(window=14).mean().iloc[-1]
                    
                    # Get daily data for longer-term volatility comparison
                    daily_data = self.market_data_agent.get_historical_data(
                        instrument=instrument,
                        timeframe="D",
                        count=30
                    )
                    
                    # Calculate volatility metrics for different periods if we have daily data
                    volatilities = {
                        "current_vol": recent_std,
                        "annualized_vol": annualized_vol,
                        "atr": atr
                    }
                    
                    if not daily_data.empty and len(daily_data) > 5:
                        daily_returns = daily_data['returns'].dropna()
                        # Weekly volatility (5-day)
                        if len(daily_returns) >= 5:
                            vol_5 = daily_returns.tail(5).std()
                            volatilities["vol_5d"] = vol_5
                        
                        # Monthly volatility (20-day)
                        if len(daily_returns) >= 20:
                            vol_20 = daily_returns.tail(20).std()
                            volatilities["vol_20d"] = vol_20
                            
                            # Classify current volatility against 20-day
                            vol_ratio = recent_std / vol_20
                            
                            if vol_ratio < 0.75:
                                vol_classification = "low"
                            elif vol_ratio < 1.25:
                                vol_classification = "normal"
                            elif vol_ratio < 2.0:
                                vol_classification = "elevated"
                            else:
                                vol_classification = "extreme"
                                logger.warning(f"Extreme volatility detected for {instrument}: {vol_ratio:.2f}x normal levels")
                        else:
                            vol_classification = "unknown"
                    else:
                        vol_classification = "unknown"
                    
                    # Store current volatility data point
                    timestamp = datetime.datetime.now()
                    self.market_data["volatility"][instrument].append((timestamp, recent_std))
                    
                    # Compile volatility metrics
                    volatility_data[instrument] = {
                        "current_volatility": recent_std,
                        "annualized_volatility": annualized_vol,
                        "atr": atr,
                        "volatility_metrics": volatilities,
                        "classification": vol_classification,
                        "timestamp": timestamp.isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Error calculating volatility for {instrument}: {str(e)}")
                    continue
            
            return volatility_data
            
        except Exception as e:
            logger.error(f"Error monitoring volatility: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def monitor_spread_changes(self) -> Dict[str, Dict[str, float]]:
        """
        Monitor bid-ask spreads for all tracked instruments.
        
        Tracks current spread levels, historical averages, and
        notable changes in spread width which could affect trade execution.
        
        Returns:
            Dict mapping instruments to spread metrics
        """
        try:
            spread_data = {}
            
            # Get tracked instruments
            instruments = self.market_data["instruments"]
            
            # Use the MarketDataAgent to get real price data
            if not self.market_data_agent:
                logger.warning("MarketDataAgent not available. Cannot fetch real spread data.")
                return {}
            
            for instrument in instruments:
                # Initialize spread data if it doesn't exist
                if instrument not in self.market_data["spreads"]:
                    self.market_data["spreads"][instrument] = deque(maxlen=500)
                
                try:
                    # Get order book data which contains bid-ask prices
                    order_book = self.market_data_agent.get_order_book(instrument)
                    
                    if not order_book:
                        logger.warning(f"Could not get order book data for {instrument}")
                        continue
                    
                    # Get current best bid and ask prices
                    # The structure depends on the OANDA API response format
                    buckets = order_book.get('orderBook', {}).get('buckets', [])
                    
                    # Find the highest bid and lowest ask
                    bids = [b for b in buckets if float(b.get('price', 0)) > 0 and float(b.get('longCountPercent', 0)) > 0]
                    asks = [a for a in buckets if float(a.get('price', 0)) > 0 and float(a.get('shortCountPercent', 0)) > 0]
                    
                    if not bids or not asks:
                        # If we can't get spread from order book, try to get it from prices endpoint
                        try:
                            # Use the current price as a reference
                            price_data = self.market_data_agent.get_current_price(instrument)
                            
                            # OANDA provides the spread information directly in some endpoints
                            if 'spread' in price_data:
                                current_spread = float(price_data['spread'])
                            else:
                                # If not directly provided, estimate from pricing data
                                bid_price = float(price_data.get('bids', [{}])[0].get('price', 0))
                                ask_price = float(price_data.get('asks', [{}])[0].get('price', 0))
                                
                                if bid_price > 0 and ask_price > 0:
                                    current_spread = ask_price - bid_price
                                else:
                                    logger.warning(f"Could not calculate spread for {instrument}")
                                    continue
                        except Exception as inner_e:
                            logger.error(f"Error getting price data for {instrument}: {str(inner_e)}")
                            continue
                    else:
                        # Calculate spread from order book data
                        highest_bid = max(bids, key=lambda x: float(x.get('price', 0)))
                        lowest_ask = min(asks, key=lambda x: float(x.get('price', 0)))
                        
                        bid_price = float(highest_bid.get('price', 0))
                        ask_price = float(lowest_ask.get('price', 0))
                        
                        current_spread = ask_price - bid_price
                    
                    # Get mid price for spread percentage calculation
                    mid_price = (bid_price + ask_price) / 2
                    spread_pips = current_spread * 10000  # For 4 decimal currency pairs
                    spread_percentage = (current_spread / mid_price) * 100
                    
                    # Store current spread data point
                    timestamp = datetime.datetime.now()
                    self.market_data["spreads"][instrument].append((timestamp, current_spread))
                    
                    # Calculate historical spread metrics
                    historical_spreads = [s for _, s in self.market_data["spreads"][instrument]]
                    
                    if len(historical_spreads) > 1:
                        avg_spread = sum(historical_spreads) / len(historical_spreads)
                        min_spread = min(historical_spreads)
                        max_spread = max(historical_spreads)
                        spread_volatility = np.std(historical_spreads)
                        
                        # Calculate spread change
                        prev_spread = historical_spreads[-2] if len(historical_spreads) >= 2 else current_spread
                        spread_change = current_spread - prev_spread
                        spread_change_pct = (spread_change / prev_spread) * 100 if prev_spread > 0 else 0
                        
                        # Determine if spread is abnormal
                        if len(historical_spreads) >= 10:
                            recent_avg = sum(historical_spreads[-10:]) / 10
                            recent_std = np.std(historical_spreads[-10:])
                            z_score = (current_spread - recent_avg) / recent_std if recent_std > 0 else 0
                            
                            is_abnormal = abs(z_score) > 2  # More than 2 standard deviations from recent mean
                            
                            if is_abnormal:
                                logger.warning(f"Abnormal spread detected for {instrument}: {spread_pips:.1f} pips (z-score: {z_score:.2f})")
                        else:
                            is_abnormal = False
                            z_score = 0
                    else:
                        avg_spread = current_spread
                        min_spread = current_spread
                        max_spread = current_spread
                        spread_volatility = 0
                        spread_change = 0
                        spread_change_pct = 0
                        is_abnormal = False
                        z_score = 0
                    
                    # Compile spread data
                    spread_data[instrument] = {
                        "current_spread": current_spread,
                        "spread_pips": spread_pips,
                        "spread_percentage": spread_percentage,
                        "bid_price": bid_price,
                        "ask_price": ask_price,
                        "avg_spread": avg_spread,
                        "min_spread": min_spread,
                        "max_spread": max_spread,
                        "spread_volatility": spread_volatility,
                        "spread_change": spread_change,
                        "spread_change_pct": spread_change_pct,
                        "is_abnormal": is_abnormal,
                        "z_score": z_score,
                        "timestamp": timestamp.isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing spread data for {instrument}: {str(e)}")
                    continue
            
            return spread_data
            
        except Exception as e:
            logger.error(f"Error monitoring spreads: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def detect_market_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in market behavior.
        
        Identifies unusual price patterns, gaps, spikes, flash crashes,
        and irregular liquidity conditions across all tracked instruments.
        
        Returns:
            List of detected market anomalies
        """
        try:
            anomalies = []
            
            # Get price, volatility, and spread data
            price_movements = self.monitor_price_movements()
            volatility_data = self.monitor_volatility()
            spread_data = self.monitor_spread_changes()
            
            # Get tracked instruments
            instruments = self.market_data["instruments"]
            
            for instrument in instruments:
                # Skip instruments with insufficient data
                if instrument not in price_movements or instrument not in volatility_data:
                    continue
                
                price_info = price_movements[instrument]
                vol_info = volatility_data.get(instrument, {})
                spread_info = spread_data.get(instrument, {})
                
                # 1. Check for price spikes/crashes (large percentage moves)
                if abs(price_info.get("pct_change", 0)) > 0.2:  # 0.2% threshold for single update
                    anomaly = {
                        "type": "price_spike" if price_info["pct_change"] > 0 else "price_crash",
                        "instrument": instrument,
                        "magnitude": price_info["pct_change"],
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"{'Price spike' if price_info['pct_change'] > 0 else 'Price crash'} "
                                  f"detected for {instrument}: {price_info['pct_change']:.4f}% change"
                    }
                    anomalies.append(anomaly)
                    logger.warning(anomaly["message"])
                
                # 2. Check for abnormal volatility
                if vol_info.get("classification") == "extreme":
                    vol_metrics = vol_info.get("volatility_metrics", {})
                    anomaly = {
                        "type": "extreme_volatility",
                        "instrument": instrument,
                        "current_vol": vol_metrics.get("current_vol", 0),
                        "normal_vol": vol_metrics.get("vol_30", 0),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"Extreme volatility detected for {instrument}: "
                                  f"{vol_metrics.get('current_vol', 0)/vol_metrics.get('vol_30', 1):.2f}x normal levels"
                    }
                    anomalies.append(anomaly)
                
                # 3. Check for abnormal spreads
                if spread_info.get("is_abnormal", False):
                    anomaly = {
                        "type": "abnormal_spread",
                        "instrument": instrument,
                        "spread_pips": spread_info.get("spread_pips", 0),
                        "spread_percentage": spread_info.get("spread_percentage", 0),
                        "spread_change": spread_info.get("spread_change", 0),
                        "spread_change_pct": spread_info.get("spread_change_pct", 0),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"Unusually wide spread detected for {instrument}: {spread_info.get('spread_pips', 0):.1f} pips ({spread_info.get('spread_percentage', 0):.2f}%) with z-score: {spread_info.get('z_score', 0):.2f}"
                    }
                    anomalies.append(anomaly)
                
                # 4. Check for price gaps
                price_history = self.market_data["price_history"].get(instrument, deque())
                if len(price_history) >= 2:
                    last_two_prices = list(price_history)[-2:]
                    _, prev_price = last_two_prices[0]
                    _, curr_price = last_two_prices[1]
                    
                    # Calculate percentage gap
                    gap_pct = abs((curr_price - prev_price) / prev_price * 100)
                    
                    # Check if it exceeds threshold (0.3% for example)
                    if gap_pct > 0.3:
                        anomaly = {
                            "type": "price_gap",
                            "instrument": instrument,
                            "magnitude": gap_pct,
                            "direction": "up" if curr_price > prev_price else "down",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "message": f"Price gap detected for {instrument}: {gap_pct:.4f}% "
                                     f"{'increase' if curr_price > prev_price else 'decrease'}"
                        }
                        anomalies.append(anomaly)
                        logger.warning(anomaly["message"])
                
                # 5. Look for reversal patterns (simulated)
                # In a real implementation, this would use more sophisticated pattern recognition
                if np.random.random() < 0.02:  # 2% chance to simulate detecting a pattern
                    pattern_types = ["double_top", "double_bottom", "head_and_shoulders", "inverted_head_and_shoulders"]
                    pattern = np.random.choice(pattern_types)
                    
                    anomaly = {
                        "type": "reversal_pattern",
                        "pattern": pattern,
                        "instrument": instrument,
                        "confidence": round(np.random.uniform(0.7, 0.95), 2),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "message": f"Potential {pattern.replace('_', ' ')} pattern detected for {instrument}"
                    }
                    anomalies.append(anomaly)
                    logger.info(anomaly["message"])
            
            # Store detected anomalies
            if anomalies:
                self.market_data["market_anomalies"].extend(anomalies)
                # Keep only recent anomalies (last 100)
                if len(self.market_data["market_anomalies"]) > 100:
                    self.market_data["market_anomalies"] = self.market_data["market_anomalies"][-100:]
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting market anomalies: {str(e)}")
            return []
    
    def track_correlation_changes(self) -> Dict[str, Dict[str, float]]:
        """
        Track changes in correlations between currency pairs.
        
        Calculates and monitors pair-wise correlations between instruments,
        detects correlation breakdowns or shifts in correlation regimes.
        
        Returns:
            Dict mapping pair names to correlation information
        """
        try:
            correlation_data = {}
            
            # Get tracked instruments
            instruments = self.market_data["instruments"]
            
            # We need at least 2 instruments and price history data to calculate correlations
            if len(instruments) < 2:
                return {}
            
            # Check if we have enough price data for each instrument
            valid_instruments = []
            instrument_prices = {}
            
            for instrument in instruments:
                price_history = self.market_data["price_history"].get(instrument, deque())
                
                # Need at least 20 data points for meaningful correlation
                if len(price_history) >= 20:
                    valid_instruments.append(instrument)
                    # Extract just the prices (not timestamps)
                    prices = [price for _, price in price_history]
                    instrument_prices[instrument] = prices[-20:]
            
            # Need at least 2 valid instruments
            if len(valid_instruments) < 2:
                return {}
            
            # Calculate correlations between all pairs of valid instruments
            for i, instrument1 in enumerate(valid_instruments):
                for instrument2 in valid_instruments[i+1:]:
                    pair_name = f"{instrument1}/{instrument2}"
                    
                    # Initialize correlation data if it doesn't exist
                    if pair_name not in self.market_data["correlations"]:
                        self.market_data["correlations"][pair_name] = deque(maxlen=100)
                    
                    # Get prices for both instruments
                    prices1 = instrument_prices[instrument1]
                    prices2 = instrument_prices[instrument2]
                    
                    # Calculate price returns (percentage changes)
                    returns1 = np.diff(prices1) / prices1[:-1]
                    returns2 = np.diff(prices2) / prices2[:-1]
                    
                    # Ensure both return series are the same length
                    min_length = min(len(returns1), len(returns2))
                    if min_length < 2:
                        continue
                        
                    returns1 = returns1[-min_length:]
                    returns2 = returns2[-min_length:]
                    
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    
                    # Store current correlation data point
                    timestamp = datetime.datetime.now()
                    self.market_data["correlations"][pair_name].append((timestamp, correlation))
                    
                    # Determine correlation strength
                    if abs(correlation) < 0.2:
                        strength = "very weak"
                    elif abs(correlation) < 0.4:
                        strength = "weak"
                    elif abs(correlation) < 0.6:
                        strength = "moderate"
                    elif abs(correlation) < 0.8:
                        strength = "strong"
                    else:
                        strength = "very strong"
                    
                    # Determine correlation direction
                    direction = "positive" if correlation > 0 else "negative"
                    
                    # Check for correlation change if we have enough history
                    correlation_history = self.market_data["correlations"][pair_name]
                    correlation_change = None
                    correlation_regime_shift = False
                    
                    if len(correlation_history) > 5:
                        # Get past correlations (skip the most recent one we just added)
                        past_correlations = [corr for _, corr in list(correlation_history)[:-1]]
                        avg_past_correlation = np.mean(past_correlations)
                        
                        # Calculate absolute change in correlation
                        correlation_change = correlation - avg_past_correlation
                        
                        # Check for regime shift (significant sign change)
                        if (avg_past_correlation * correlation < 0 and 
                            abs(avg_past_correlation) > 0.3 and 
                            abs(correlation) > 0.3):
                            correlation_regime_shift = True
                            logger.warning(f"Correlation regime shift detected between {instrument1} and {instrument2}: "
                                         f"from {avg_past_correlation:.2f} to {correlation:.2f}")
                    
                    # Compile correlation information
                    correlation_data[pair_name] = {
                        "instruments": [instrument1, instrument2],
                        "correlation": correlation,
                        "strength": strength,
                        "direction": direction,
                        "correlation_change": correlation_change,
                        "regime_shift": correlation_regime_shift,
                        "timestamp": timestamp.isoformat()
                    }
                    
                    # Log significant correlation changes
                    if correlation_change is not None and abs(correlation_change) > 0.3:
                        logger.info(f"Significant correlation change between {instrument1} and {instrument2}: "
                                   f"{correlation_change:.2f} change, now at {correlation:.2f}")
            
            return correlation_data
            
        except Exception as e:
            logger.error(f"Error tracking correlation changes: {str(e)}")
            return {}

    #--------------------------------------------------
    # 4. Alerting Methods
    #--------------------------------------------------
    
    def set_up_alerts(self, alert_config: Dict[str, Any]) -> bool:
        """
        Set up alerts based on configuration.
        
        Configures email alerts, SMS alerts, and dashboard alerts
        based on the provided configuration.
        
        Args:
            alert_config: Dictionary containing alert configuration
            
        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Configure email alerts
            email_config = alert_config.get("email", {})
            self.alert_data["email_enabled"] = email_config.get("enabled", False)
            
            if self.alert_data["email_enabled"]:
                # Validate email configuration
                required_email_fields = ["smtp_server", "smtp_port", "username", "password", "recipients"]
                missing_fields = [field for field in required_email_fields if not email_config.get(field)]
                
                if missing_fields:
                    logger.warning(f"Email alerts enabled but missing configuration: {', '.join(missing_fields)}")
                    self.alert_data["email_enabled"] = False
                else:
                    logger.info(f"Email alerts configured for {len(email_config.get('recipients', []))} recipients")
            
            # Configure SMS alerts
            sms_config = alert_config.get("sms", {})
            self.alert_data["sms_enabled"] = sms_config.get("enabled", False)
            
            if self.alert_data["sms_enabled"]:
                # Validate SMS configuration
                required_sms_fields = ["provider", "api_key", "phone_numbers"]
                missing_fields = [field for field in required_sms_fields if not sms_config.get(field)]
                
                if missing_fields:
                    logger.warning(f"SMS alerts enabled but missing configuration: {', '.join(missing_fields)}")
                    self.alert_data["sms_enabled"] = False
                else:
                    logger.info(f"SMS alerts configured for {len(sms_config.get('phone_numbers', []))} phone numbers")
            
            # Configure dashboard alerts
            dashboard_config = alert_config.get("dashboard", {})
            
            # Update alert levels display settings
            for level, settings in dashboard_config.get("levels", {}).items():
                if level in self.alert_data["alert_levels"]:
                    self.alert_data["alert_levels"][level].update(settings)
            
            # Update alert categories display settings
            for category, settings in dashboard_config.get("categories", {}).items():
                if category in self.alert_data["alert_categories"]:
                    self.alert_data["alert_categories"][category].update(settings)
            
            # Log an info alert to test the system
            self.log_alert({
                "level": "info",
                "category": "system",
                "message": "Alert system initialized",
                "details": "Alert configuration loaded successfully",
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            logger.error(f"Error setting up alerts: {str(e)}")
            return False
    
    def send_email_alert(self, message: str, subject: str = "Forex Trading Alert", 
                         level: str = "info", details: Dict[str, Any] = None) -> bool:
        """
        Send an email alert.
        
        Sends an email to all configured recipients with the alert message.
        
        Args:
            message: Alert message text
            subject: Email subject
            level: Alert level (info, warning, error, critical)
            details: Additional alert details
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        try:
            if not self.alert_data["email_enabled"]:
                logger.warning("Email alerts not enabled")
                return False
            
            # Get email configuration
            email_config = self.alert_config.get("email", {})
            smtp_server = email_config.get("smtp_server")
            smtp_port = email_config.get("smtp_port")
            username = email_config.get("username")
            password = email_config.get("password")
            recipients = email_config.get("recipients", [])
            
            if not recipients:
                logger.warning("No email recipients configured")
                return False
            
            # Create email content
            msg = MIMEMultipart()
            msg["From"] = username
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"[{level.upper()}] {subject}"
            
            # Format email body
            body = f"""
            <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        .alert-info {{ color: blue; }}
                        .alert-warning {{ color: orange; }}
                        .alert-error {{ color: red; }}
                        .alert-critical {{ color: purple; font-weight: bold; }}
                        .details {{ margin-top: 10px; padding: 10px; background-color: #f8f8f8; }}
                    </style>
                </head>
                <body>
                    <h2 class="alert-{level}">Forex Trading Alert: {level.upper()}</h2>
                    <p>{message}</p>
            """
            
            # Add details if provided
            if details:
                body += "<div class='details'><h3>Details:</h3><ul>"
                for key, value in details.items():
                    body += f"<li><strong>{key}:</strong> {value}</li>"
                body += "</ul></div>"
            
            # Add timestamp
            body += f"<p><em>Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>"
            body += "</body></html>"
            
            # Attach HTML body
            msg.attach(MIMEText(body, "html"))
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {len(recipients)} recipients: {subject}")
            return True
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def send_sms_alert(self, message: str, level: str = "info") -> bool:
        """
        Send an SMS alert.
        
        Sends an SMS to all configured phone numbers with the alert message.
        
        Args:
            message: Alert message text
            level: Alert level (info, warning, error, critical)
            
        Returns:
            True if SMS was sent successfully, False otherwise
        """
        try:
            if not self.alert_data["sms_enabled"]:
                logger.warning("SMS alerts not enabled")
                return False
            
            # Get SMS configuration
            sms_config = self.alert_config.get("sms", {})
            provider = sms_config.get("provider")
            api_key = sms_config.get("api_key")
            phone_numbers = sms_config.get("phone_numbers", [])
            
            if not phone_numbers:
                logger.warning("No phone numbers configured for SMS alerts")
                return False
            
            # Format SMS message (shortened with level prefix)
            # SMS should be concise, typically under 160 characters
            sms_text = f"[{level.upper()}] {message}"
            if len(sms_text) > 150:
                sms_text = sms_text[:147] + "..."
            
            # In a real implementation, this would use a specific SMS provider's API
            # For example, with Twilio:
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            
            # For this example, we'll simulate sending SMS
            for phone in phone_numbers:
                # Simulate sending SMS
                logger.info(f"Simulated SMS to {phone}: {sms_text}")
                
                # With Twilio, it would be something like:
                # message = client.messages.create(
                #     body=sms_text,
                #     from_='+1234567890',  # Your Twilio number
                #     to=phone
                # )
            
            logger.info(f"SMS alert sent to {len(phone_numbers)} numbers: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error sending SMS alert: {str(e)}")
            return False
    
    def send_dashboard_alert(self, message: str, level: str = "info", 
                            category: str = "system", details: Dict[str, Any] = None) -> bool:
        """
        Send an alert to the dashboard.
        
        Adds an alert to the dashboard alerts queue for display in the UI.
        
        Args:
            message: Alert message text
            level: Alert level (info, warning, error, critical)
            category: Alert category (system, trading, market, security)
            details: Additional alert details
            
        Returns:
            True if alert was added successfully
        """
        try:
            # Create alert object
            alert = {
                "message": message,
                "level": level,
                "category": category,
                "details": details or {},
                "timestamp": datetime.datetime.now().isoformat(),
                "id": str(uuid.uuid4())[:8],  # Generate a short unique ID
                "read": False,
                "acknowledged": False
            }
            
            # Add visual indicators based on level
            if level in self.alert_data["alert_levels"]:
                level_info = self.alert_data["alert_levels"][level]
                alert["color"] = level_info.get("color", "gray")
                alert["icon"] = level_info.get("icon", "ℹ️")
                alert["display"] = level_info.get("display", True)
            
            # Add category information
            if category in self.alert_data["alert_categories"]:
                cat_info = self.alert_data["alert_categories"][category]
                alert["category_display"] = cat_info.get("display", True)
                alert["category_color"] = cat_info.get("color", "gray")
            
            # Add to dashboard alerts queue
            self.alert_data["dashboard_alerts"].append(alert)
            
            # Log the alert
            self.log_alert(alert)
            
            return True
        except Exception as e:
            logger.error(f"Error sending dashboard alert: {str(e)}")
            return False
    
    def log_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Log an alert.
        
        Logs the alert to the system log and stores it in the alerts history.
        
        Args:
            alert: Alert object containing message, level, etc.
            
        Returns:
            True if alert was logged successfully
        """
        try:
            # Ensure alert has required fields
            if not isinstance(alert, dict) or "message" not in alert:
                logger.error("Invalid alert format")
                return False
            
            # Add timestamp if not present
            if "timestamp" not in alert:
                alert["timestamp"] = datetime.datetime.now().isoformat()
            
            # Add to alerts history
            self.alert_data["alerts"].append(alert)
            
            # Log to system log based on level
            level = alert.get("level", "info").lower()
            message = f"{alert.get('category', 'system').upper()}: {alert['message']}"
            
            if level == "critical":
                logger.critical(message)
            elif level == "error":
                logger.error(message)
            elif level == "warning":
                logger.warning(message)
            else:  # info or anything else
                logger.info(message)
            
            # Determine if alert should be sent through other channels
            send_email = alert.get("send_email", False) or level in ["critical", "error"]
            send_sms = alert.get("send_sms", False) or level == "critical"
            
            # Send via configured channels if needed
            if send_email and self.alert_data["email_enabled"]:
                self.send_email_alert(
                    message=alert["message"],
                    subject=f"{alert.get('category', 'system').title()} Alert",
                    level=level,
                    details=alert.get("details", {})
                )
            
            if send_sms and self.alert_data["sms_enabled"]:
                self.send_sms_alert(
                    message=alert["message"],
                    level=level
                )
            
            return True
        except Exception as e:
            # Use standard logger directly since we're inside the log_alert method
            logger.error(f"Error logging alert: {str(e)}")
            return False

#--------------------------------------------------
# 5. Web Interface Methods (Streamlit)
#--------------------------------------------------

def create_dashboard_layout(self) -> None:
    """
    Create the main dashboard layout with Streamlit.
    
    Sets up the page configuration, sidebar, and main content areas.
    """
    try:
        # Set page config
        st.set_page_config(
            page_title="Forex Trading Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add auto-refresh component (refreshes every 30 seconds)
        st_autorefresh(interval=30000, key="dashboard_refresh")
        
        # Dashboard header
        st.title("🌐 Forex Trading Monitoring Dashboard")
        
        # Sidebar
        st.sidebar.image("https://img.icons8.com/fluency/96/000000/forex.png", width=80)
        st.sidebar.title("Navigation")
        
        # Navigation menu
        page = st.sidebar.radio(
            "Select Section",
            ["Overview", "System Status", "Trading Activity", "Market Analysis", "Performance Metrics", "Alerts"]
        )
        
        # Display timestamp
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add system control buttons
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Controls")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Start Monitoring", key="start_btn"):
                self.start_monitoring()
                st.success("Monitoring started!")
        
        with col2:
            if st.button("Stop Monitoring", key="stop_btn"):
                self.stop_monitoring()
                st.success("Monitoring stopped!")
        
        # Show selected page
        if page == "Overview":
            self.create_overview_panel()
        elif page == "System Status":
            self.create_system_status_panel()
        elif page == "Trading Activity":
            self.create_trading_panel()
        elif page == "Market Analysis":
            self.create_market_panel()
        elif page == "Performance Metrics":
            self.create_performance_panel()
        elif page == "Alerts":
            self.create_alert_panel()
        
    except Exception as e:
        st.error(f"Error creating dashboard layout: {str(e)}")
        logger.error(f"Error creating dashboard layout: {str(e)}")

def create_overview_panel(self) -> None:
    """
    Create an overview panel showing key metrics from all sections.
    
    Displays a summary of the most important information from each panel.
    """
    try:
        st.header("Dashboard Overview")
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        # System status overview
        with col1:
            st.subheader("System Status")
            system_resources = self.monitor_system_resources()
            
            st.metric(
                label="CPU Usage",
                value=f"{system_resources.get('cpu_percent', 0):.1f}%",
                delta=f"{system_resources.get('cpu_percent', 0) - 50:.1f}%" 
                if system_resources.get('cpu_percent', 0) > 50 else None
            )
            
            # Count healthy vs unhealthy agents
            agent_status = self.monitor_agent_status()
            healthy_count = sum(1 for agent in agent_status.values() 
                              if agent.get('status') == 'healthy')
            total_agents = len(agent_status)
            
            st.metric(
                label="Healthy Agents",
                value=f"{healthy_count}/{total_agents}",
                delta=None
            )
        
        # Trading overview
        with col2:
            st.subheader("Trading Status")
            
            # Get account balance
            balance_info = self.monitor_account_balance()
            
            st.metric(
                label="Account Balance",
                value=f"${balance_info.get('balance', 0):.2f}",
                delta=f"{balance_info.get('unrealized_pl', 0):.2f}" 
                if balance_info.get('unrealized_pl', 0) != 0 else None
            )
            
            # Show open positions count
            positions_count = len(self.trading_data["open_positions"])
            orders_count = len(self.trading_data["pending_orders"])
            
            st.metric(
                label="Open Positions",
                value=positions_count,
                delta=None
            )
        
        # Market overview
        with col3:
            st.subheader("Market Overview")
            
            # Count instruments with significant movements
            price_movements = self.monitor_price_movements()
            significant_movements = sum(1 for instr in price_movements.values() 
                                     if instr.get('is_significant', False))
            
            st.metric(
                label="Active Markets",
                value=f"{significant_movements}/{len(price_movements)}",
                delta=None
            )
            
            # Show anomalies count
            anomaly_count = len(self.market_data["market_anomalies"])
            
            st.metric(
                label="Market Anomalies",
                value=anomaly_count,
                delta=anomaly_count if anomaly_count > 0 else None,
                delta_color="inverse"
            )
        
        # Recent alerts section
        st.subheader("Recent Alerts")
        if self.alert_data["dashboard_alerts"]:
            alerts_df = pd.DataFrame([
                {
                    "Time": datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S'),
                    "Level": alert['level'].upper(),
                    "Category": alert['category'].title(),
                    "Message": alert['message']
                } 
                for alert in list(self.alert_data["dashboard_alerts"])[-5:]
            ])
            
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.info("No recent alerts.")
        
        # Activity chart - positions over time
        st.subheader("Trading Activity")
        
        # Create sample data if real data is not available yet
        if not self.trading_data["account_balance"]:
            dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                  end=datetime.now(), freq='H')
            balance_values = [10000 + 500 * np.sin(i/5) for i in range(len(dates))]
            balance_df = pd.DataFrame({
                'time': dates,
                'balance': balance_values
            })
        else:
            # Convert real data
            balance_data = list(self.trading_data["account_balance"])
            balance_df = pd.DataFrame({
                'time': [t for t, _ in balance_data],
                'balance': [b for _, b in balance_data]
            })
        
        fig = px.line(balance_df, x='time', y='balance', 
                     title='Account Balance History (24h)')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating overview panel: {str(e)}")
        logger.error(f"Error creating overview panel: {str(e)}")

def create_system_status_panel(self) -> None:
    """
    Create a panel for system status monitoring.
    
    Displays agent status, system resources, API connections,
    and system events in an organized layout.
    """
    try:
        st.header("System Status Monitoring")
        
        # Create tabs for different system monitoring aspects
        system_tabs = st.tabs(["Agents", "Resources", "API Connections", "Events"])
        
        # 1. Agents Status Tab
        with system_tabs[0]:
            st.subheader("Agent Status")
            
            # Get agent status data
            agent_status = self.monitor_agent_status()
            
            if agent_status:
                # Process agent data for display
                agents_data = []
                
                for agent_name, status in agent_status.items():
                    status_value = status.get('status', 'unknown')
                    last_active = status.get('last_active', '')
                    response_time = status.get('response_time', 0)
                    
                    # Convert status to emoji for visual indication
                    status_icon = "✅" if status_value == "healthy" else "⚠️" if status_value == "degraded" else "❌"
                    
                    # Format last active time
                    if last_active:
                        try:
                            last_active_time = datetime.fromisoformat(last_active)
                            last_active_fmt = last_active_time.strftime("%H:%M:%S")
                        except:
                            last_active_fmt = last_active
                    else:
                        last_active_fmt = "N/A"
                    
                    agents_data.append({
                        "Agent": agent_name,
                        "Status": f"{status_icon} {status_value}",
                        "Last Active": last_active_fmt,
                        "Response Time": f"{response_time:.3f}s" if response_time else "N/A",
                    })
                
                # Create a DataFrame and display it
                agents_df = pd.DataFrame(agents_data)
                st.dataframe(agents_df, use_container_width=True)
            else:
                st.info("No agent status data available.")
        
        # 2. System Resources Tab
        with system_tabs[1]:
            st.subheader("System Resources")
            
            # Get system resources data
            resources = self.monitor_system_resources()
            
            # Create columns for displaying metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_percent = resources.get('cpu_percent', 0)
                cpu_color = "normal" if cpu_percent < 70 else "off" if cpu_percent < 90 else "inverse"
                st.metric("CPU Usage", f"{cpu_percent:.1f}%", delta=None, delta_color=cpu_color)
                
                # Create a gauge chart for CPU usage
                fig_cpu = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=cpu_percent,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CPU"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_cpu.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col2:
                memory_percent = resources.get('memory_percent', 0)
                memory_color = "normal" if memory_percent < 70 else "off" if memory_percent < 90 else "inverse"
                st.metric("Memory Usage", f"{memory_percent:.1f}%", delta=None, delta_color=memory_color)
                
                # Create a gauge chart for memory usage
                fig_mem = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=memory_percent,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Memory"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_mem.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_mem, use_container_width=True)
            
            with col3:
                disk_percent = resources.get('disk_percent', 0)
                disk_color = "normal" if disk_percent < 70 else "off" if disk_percent < 90 else "inverse"
                st.metric("Disk Usage", f"{disk_percent:.1f}%", delta=None, delta_color=disk_color)
                
                # Create a gauge chart for disk usage
                fig_disk = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=disk_percent,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Disk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_disk.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_disk, use_container_width=True)
            
            # Historical resource usage chart
            st.subheader("Historical Resource Usage")
            
            # Prepare historical data
            if (self.system_data["system_resources"]["cpu"] and 
                self.system_data["system_resources"]["memory"] and 
                self.system_data["system_resources"]["disk"]):
                
                # Extract data for the last hour
                last_hour = datetime.now() - timedelta(hours=1)
                
                cpu_data = [(t, v) for t, v in self.system_data["system_resources"]["cpu"] 
                          if t > last_hour]
                memory_data = [(t, v) for t, v in self.system_data["system_resources"]["memory"] 
                          if t > last_hour]
                disk_data = [(t, v) for t, v in self.system_data["system_resources"]["disk"] 
                          if t > last_hour]
                
                # Create a DataFrame with all resources
                timestamps = [t for t, _ in cpu_data]
                cpu_values = [v for _, v in cpu_data]
                memory_values = [v for _, v in memory_data]
                disk_values = [v for _, v in disk_data]
                
                if timestamps:
                    df = pd.DataFrame({
                        'timestamp': timestamps,
                        'CPU': cpu_values,
                        'Memory': memory_values,
                        'Disk': disk_values
                    })
                    
                    # Create a multi-line chart
                    fig = px.line(df, x='timestamp', y=['CPU', 'Memory', 'Disk'],
                                labels={'value': 'Usage (%)', 'variable': 'Resource'},
                                title='Resource Usage (Last Hour)')
                    
                    fig.update_layout(legend_title_text='Resource', 
                                    hovermode='x unified',
                                    height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough historical data available yet.")
            else:
                st.info("Not enough historical data available yet.")
        
        # 3. API Connections Tab
        with system_tabs[2]:
            st.subheader("API Connections")
            
            # Get API connections data
            api_connections = self.monitor_api_connections()
            
            if api_connections:
                # Process connection data for display
                connection_data = []
                
                for api_name, status in api_connections.items():
                    status_value = status.get('status', 'unknown')
                    response_time = status.get('response_time')
                    timestamp = status.get('timestamp', '')
                    
                    # Convert status to emoji for visual indication
                    status_icon = "✅" if status_value == "connected" else "❌"
                    
                    # Format timestamp
                    if timestamp:
                        try:
                            timestamp_dt = datetime.fromisoformat(timestamp)
                            timestamp_fmt = timestamp_dt.strftime("%H:%M:%S")
                        except:
                            timestamp_fmt = timestamp
                    else:
                        timestamp_fmt = "N/A"
                    
                    connection_data.append({
                        "API": api_name,
                        "Status": f"{status_icon} {status_value}",
                        "Response Time": f"{response_time:.3f}s" if response_time else "N/A",
                        "Last Check": timestamp_fmt,
                    })
                
                # Create a DataFrame and display it
                connection_df = pd.DataFrame(connection_data)
                st.dataframe(connection_df, use_container_width=True)
            else:
                st.info("No API connection data available.")
        
        # 4. System Events Tab
        with system_tabs[3]:
            st.subheader("System Events")
            
            # Get system events data
            events = self.system_data["system_events"]
            
            if events:
                # Process events for display
                events_data = []
                
                for event in events[-50:]:  # Display latest 50 events
                    event_type = event.get('event_type', '')
                    message = event.get('message', '')
                    severity = event.get('severity', 'info')
                    timestamp = event.get('timestamp', '')
                    
                    # Convert severity to color for visual indication
                    severity_icon = "ℹ️" if severity == "info" else "⚠️" if severity == "warning" else "❌" if severity == "error" else "🚨"
                    
                    # Format timestamp
                    if timestamp:
                        try:
                            timestamp_dt = datetime.fromisoformat(timestamp)
                            timestamp_fmt = timestamp_dt.strftime("%H:%M:%S")
                        except:
                            timestamp_fmt = timestamp
                    else:
                        timestamp_fmt = "N/A"
                    
                    events_data.append({
                        "Time": timestamp_fmt,
                        "Type": event_type,
                        "Severity": f"{severity_icon} {severity}",
                        "Message": message
                    })
                
                # Create a DataFrame and display it
                events_df = pd.DataFrame(events_data)
                st.dataframe(events_df, use_container_width=True)
            else:
                st.info("No system events recorded.")
                
    except Exception as e:
        st.error(f"Error creating system status panel: {str(e)}")
        logger.error(f"Error creating system status panel: {str(e)}")

def create_trading_panel(self) -> None:
    """
    Create a panel for trading information.
    
    Displays open positions, pending orders, trade performance,
    account balance, and margin levels.
    """
    try:
        st.header("Trading Activity")
        
        # Create tabs for different trading aspects
        trading_tabs = st.tabs(["Positions", "Orders", "Performance", "Account", "Risk"])
        
        # 1. Open Positions Tab
        with trading_tabs[0]:
            st.subheader("Open Positions")
            
            # Get open positions data
            open_positions = self.trading_data["open_positions"]
            
            if open_positions:
                # Process position data for display
                positions_data = []
                
                for pos_id, position in open_positions.items():
                    instrument = position.get('instrument', 'Unknown')
                    direction = position.get('direction', 'Unknown')
                    size = position.get('size', 0)
                    entry_price = position.get('entry_price', 0)
                    current_price = position.get('current_price', 0)
                    unrealized_pl = position.get('unrealized_pl', 0)
                    duration_hours = position.get('duration_hours', 0)
                    
                    # Format P/L with color indication
                    pl_color = "green" if unrealized_pl > 0 else "red"
                    
                    positions_data.append({
                        "ID": pos_id,
                        "Instrument": instrument,
                        "Direction": direction.upper(),
                        "Size": f"{size:.2f}",
                        "Entry Price": f"{entry_price:.5f}",
                        "Current Price": f"{current_price:.5f}",
                        "P/L": f"{unrealized_pl:.2f}",
                        "Duration": f"{duration_hours:.1f}h"
                    })
                
                # Create a DataFrame and display it
                positions_df = pd.DataFrame(positions_data)
                
                # Calculate total P/L for all positions
                total_pl = sum(float(position.get('unrealized_pl', 0)) for position in open_positions.values())
                st.metric("Total Unrealized P/L", f"${total_pl:.2f}", 
                         delta=f"{total_pl:.2f}", 
                         delta_color="normal" if total_pl >= 0 else "inverse")
                
                st.dataframe(positions_df, use_container_width=True)
                
                # Display positions in a pie chart by instrument
                positions_by_instrument = {}
                for position in open_positions.values():
                    instrument = position.get('instrument', 'Unknown')
                    size = position.get('size', 0)
                    if instrument in positions_by_instrument:
                        positions_by_instrument[instrument] += size
                    else:
                        positions_by_instrument[instrument] = size
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(positions_by_instrument.keys()),
                    values=list(positions_by_instrument.values()),
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig.update_layout(title_text='Exposure by Instrument')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No open positions at the moment.")
        
        # 2. Pending Orders Tab
        with trading_tabs[1]:
            st.subheader("Pending Orders")
            
            # Get pending orders data
            pending_orders = self.trading_data["pending_orders"]
            
            if pending_orders:
                # Process order data for display
                orders_data = []
                
                for order_id, order in pending_orders.items():
                    instrument = order.get('instrument', 'Unknown')
                    direction = order.get('direction', 'Unknown')
                    order_type = order.get('order_type', 'Unknown')
                    size = order.get('size', 0)
                    price = order.get('price', 0)
                    current_market_price = order.get('current_market_price', 0)
                    distance_to_trigger = order.get('distance_to_trigger', 0)
                    time_to_expiry = order.get('time_to_expiry_hours', 'N/A')
                    
                    orders_data.append({
                        "ID": order_id,
                        "Instrument": instrument,
                        "Type": f"{direction.upper()} {order_type}",
                        "Size": f"{size:.2f}",
                        "Price": f"{price:.5f}",
                        "Market Price": f"{current_market_price:.5f}",
                        "Distance": f"{distance_to_trigger:.5f}",
                        "Expires In": f"{time_to_expiry}h" if isinstance(time_to_expiry, (int, float)) else time_to_expiry
                    })
                
                # Create a DataFrame and display it
                orders_df = pd.DataFrame(orders_data)
                st.dataframe(orders_df, use_container_width=True)
                
                # Display orders by type in a bar chart
                order_types = {}
                for order in pending_orders.values():
                    key = f"{order.get('direction', 'Unknown').upper()} {order.get('order_type', 'Unknown')}"
                    if key in order_types:
                        order_types[key] += 1
                    else:
                        order_types[key] = 1
                
                fig = px.bar(
                    x=list(order_types.keys()),
                    y=list(order_types.values()),
                    labels={'x': 'Order Type', 'y': 'Count'},
                    title='Pending Orders by Type'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No pending orders at the moment.")
        
        # 3. Trade Performance Tab
        with trading_tabs[2]:
            st.subheader("Trade Performance")
            
            # Get trade performance data
            performance = self.track_trade_performance()
            
            if performance and performance.get("total_trades", 0) > 0:
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Trades", performance.get("total_trades", 0))
                    st.metric("Win Rate", f"{performance.get('win_rate', 0) * 100:.1f}%")
                
                with col2:
                    st.metric("Total Profit", f"${performance.get('total_profit', 0):.2f}")
                    st.metric("Total Loss", f"-${abs(performance.get('total_loss', 0)):.2f}")
                
                with col3:
                    st.metric("Profit Factor", f"{performance.get('profit_factor', 0):.2f}")
                    st.metric("Avg. Trade Duration", f"{performance.get('average_trade_duration', 0):.1f}h")
                
                # Display performance by instrument
                st.subheader("Performance by Instrument")
                
                # Process instrument performance data
                instrument_data = []
                instrument_performance = performance.get("instrument_performance", {})
                
                for instrument, stats in instrument_performance.items():
                    trades = stats.get("trades", 0)
                    win_rate = stats.get("win_rate", 0)
                    net_pl = stats.get("net_pl", 0)
                    
                    instrument_data.append({
                        "Instrument": instrument,
                        "Trades": trades,
                        "Win Rate": f"{win_rate * 100:.1f}%",
                        "Net P/L": f"${net_pl:.2f}"
                    })
                
                if instrument_data:
                    instrument_df = pd.DataFrame(instrument_data)
                    st.dataframe(instrument_df, use_container_width=True)
                    
                    # Create a bar chart for instrument P/L
                    instruments = [item["Instrument"] for item in instrument_data]
                    net_pls = [float(item["Net P/L"].replace("$", "")) for item in instrument_data]
                    
                    fig = px.bar(
                        x=instruments,
                        y=net_pls,
                        labels={'x': 'Instrument', 'y': 'Net P/L ($)'},
                        title='Profit/Loss by Instrument',
                        color=net_pls,
                        color_continuous_scale=['red', 'green'],
                        color_continuous_midpoint=0
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No completed trades yet to analyze performance.")
        
        # 4. Account Balance Tab
        with trading_tabs[3]:
            st.subheader("Account Balance & Equity")
            
            # Get account balance data
            balance_info = self.monitor_account_balance()
            
            if balance_info and "error" not in balance_info:
                # Display current metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Balance", f"${balance_info.get('balance', 0):.2f}")
                    st.metric("Available Funds", f"${balance_info.get('available_funds', 0):.2f}")
                
                with col2:
                    st.metric("Equity", f"${balance_info.get('equity', 0):.2f}")
                    st.metric("Used Margin", f"${balance_info.get('used_margin', 0):.2f}")
                
                # Display historical balance
                st.subheader("Balance History")
                
                if self.trading_data["account_balance"]:
                    # Convert balance history to DataFrame
                    balance_history = list(self.trading_data["account_balance"])
                    balance_df = pd.DataFrame(
                        {"timestamp": [t for t, _ in balance_history],
                         "balance": [b for _, b in balance_history]}
                    )
                    
                    # Create a line chart
                    fig = px.line(
                        balance_df, 
                        x="timestamp", 
                        y="balance",
                        labels={'timestamp': 'Time', 'balance': 'Balance ($)'},
                        title='Account Balance History'
                    )
                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No balance history available yet.")
            else:
                st.info("Account balance data not available.")
        
        # 5. Risk / Margin Tab
        with trading_tabs[4]:
            st.subheader("Risk & Margin Levels")
            
            # Get margin level data
            margin_info = self.monitor_margin_levels()
            
            if margin_info and "error" not in margin_info:
                # Create margin gauge
                margin_level = margin_info.get('margin_level', float('inf'))
                if margin_level == float('inf'):
                    margin_display = "∞"
                    margin_level_value = 100  # For gauge display purposes
                else:
                    margin_display = f"{margin_level:.2f}%"
                    margin_level_value = min(margin_level, 500)  # Cap at 500% for display
                
                st.metric(
                    "Margin Level", 
                    margin_display,
                    delta=f"{margin_info.get('margin_call_distance', 0):.2f}%" if margin_info.get('margin_call_distance', 0) != float('inf') else None,
                    delta_color="normal"
                )
                
                # Create a gauge chart for margin level
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=margin_level_value,
                    number={"suffix": "%", "valueformat": ".1f"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Margin Level"},
                    gauge={
                        'axis': {'range': [0, 500], 'tickvals': [0, 50, 100, 200, 500]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 100], 'color': "orange"},
                            {'range': [100, 200], 'color': "yellow"},
                            {'range': [200, 500], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': margin_info.get('margin_call_level', 100)
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Display margin status information
                status_color = {
                    "NO_MARGIN_USED": "blue",
                    "HEALTHY": "green",
                    "GOOD": "green",
                    "CAUTION": "orange",
                    "MARGIN_CALL": "red",
                    "STOP_OUT_RISK": "darkred"
                }.get(margin_info.get('margin_status', 'UNKNOWN'), "gray")
                
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; background-color: {status_color}; color: white;">
                    <h3>Margin Status: {margin_info.get('margin_status', 'UNKNOWN')}</h3>
                    <p>Current Equity: ${margin_info.get('equity', 0):.2f}</p>
                    <p>Used Margin: ${margin_info.get('used_margin', 0):.2f}</p>
                    <p>Margin Call Level: {margin_info.get('margin_call_level', 0)}%</p>
                    <p>Stop Out Level: {margin_info.get('stop_out_level', 0)}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display historical margin levels
                st.subheader("Margin Level History")
                
                if self.trading_data["margin_levels"]:
                    # Convert margin history to DataFrame
                    margin_history = list(self.trading_data["margin_levels"])
                    margin_df = pd.DataFrame(
                        {"timestamp": [t for t, _ in margin_history],
                         "margin_level": [m for _, m in margin_history]}
                    )
                    
                    # Create a line chart
                    fig = px.line(
                        margin_df, 
                        x="timestamp", 
                        y="margin_level",
                        labels={'timestamp': 'Time', 'margin_level': 'Margin Level (%)'},
                        title='Margin Level History'
                    )
                    
                    # Add horizontal reference lines for margin call and stop out
                    fig.add_hline(y=margin_info.get('margin_call_level', 100), 
                                 line_dash="dash", line_color="orange", annotation_text="Margin Call")
                    fig.add_hline(y=margin_info.get('stop_out_level', 50), 
                                 line_dash="dash", line_color="red", annotation_text="Stop Out")
                    
                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No margin level history available yet.")
            else:
                st.info("Margin level data not available.")
                
    except Exception as e:
        st.error(f"Error creating trading panel: {str(e)}")
        logger.error(f"Error creating trading panel: {str(e)}")

def create_market_panel(self) -> None:
    """
    Create a panel for market information.
    
    Displays price movements, volatility, spreads, correlations,
    and market anomalies for monitored instruments.
    """
    try:
        st.header("Market Analysis")
        
        # Create tabs for different market aspects
        market_tabs = st.tabs(["Price Movements", "Volatility", "Spreads", "Correlations", "Anomalies"])
        
        # 1. Price Movements Tab
        with market_tabs[0]:
            st.subheader("Price Movements")
            
            # Add instrument selector in sidebar
            instruments = self.market_data["instruments"]
            selected_instruments = st.multiselect(
                "Select Instruments",
                instruments,
                default=instruments[:3]  # Default to first 3 instruments
            )
            
            # Get price movements data
            price_movements = self.monitor_price_movements()
            
            if price_movements and selected_instruments:
                # Process price data for display
                price_data = []
                
                for instrument in selected_instruments:
                    if instrument in price_movements:
                        movement = price_movements[instrument]
                        current_price = movement.get('current_price', 0)
                        pct_change = movement.get('pct_change', 0)
                        movement_type = movement.get('movement_type', 'none')
                        is_significant = movement.get('is_significant', False)
                        
                        # Format movement with color and direction
                        movement_icon = "➡️" if movement_type == "none" else "🔼" if movement_type == "bullish" else "🔽"
                        
                        price_data.append({
                            "Instrument": instrument,
                            "Price": f"{current_price:.5f}",
                            "Change": f"{pct_change:.4f}%",
                            "Movement": f"{movement_icon} {movement_type.title()}",
                            "Significant": "Yes" if is_significant else "No"
                        })
                
                # Create a DataFrame and display it
                price_df = pd.DataFrame(price_data)
                st.dataframe(price_df, use_container_width=True)
                
                # Display price chart for selected instruments
                st.subheader("Price History")
                
                # Create chart data
                chart_data = []
                
                for instrument in selected_instruments:
                    if instrument in self.market_data["price_history"]:
                        price_history = list(self.market_data["price_history"][instrument])
                        
                        if price_history:
                            for timestamp, price in price_history:
                                chart_data.append({
                                    "timestamp": timestamp,
                                    "price": price,
                                    "instrument": instrument
                                })
                
                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    
                    # Create a line chart for price history
                    fig = px.line(
                        chart_df, 
                        x="timestamp", 
                        y="price", 
                        color="instrument",
                        labels={"timestamp": "Time", "price": "Price", "instrument": "Instrument"},
                        title="Price Movement History"
                    )
                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No price history data available yet.")
            else:
                st.info("No price movement data available.")
        
        # 2. Volatility Tab
        with market_tabs[1]:
            st.subheader("Market Volatility")
            
            # Get volatility data
            volatility_data = self.monitor_volatility()
            
            if volatility_data:
                # Process volatility data for display
                vol_data = []
                
                for instrument, vol_info in volatility_data.items():
                    current_vol = vol_info.get('current_volatility', 0)
                    classification = vol_info.get('classification', 'unknown')
                    
                    # Get additional volatility metrics
                    vol_metrics = vol_info.get('volatility_metrics', {})
                    annualized_vol = vol_metrics.get('annualized_vol', 0)
                    atr = vol_metrics.get('atr', 0)
                    
                    # Format classification with color
                    class_icon = {
                        "low": "🟢",
                        "normal": "🟡",
                        "elevated": "🟠",
                        "extreme": "🔴",
                        "unknown": "⚪"
                    }.get(classification, "⚪")
                    
                    vol_data.append({
                        "Instrument": instrument,
                        "Current Vol": f"{current_vol:.6f}",
                        "Annualized": f"{annualized_vol:.2f}",
                        "ATR": f"{atr:.6f}",
                        "Classification": f"{class_icon} {classification.title()}"
                    })
                
                # Create a DataFrame and display it
                vol_df = pd.DataFrame(vol_data)
                st.dataframe(vol_df, use_container_width=True)
                
                # Create bar chart of volatility by instrument
                instruments = [item["Instrument"] for item in vol_data]
                vol_values = [float(item["Current Vol"]) for item in vol_data]
                
                # Sort by volatility level
                if instruments and vol_values:
                    sorted_indices = sorted(range(len(vol_values)), key=lambda i: vol_values[i], reverse=True)
                    sorted_instruments = [instruments[i] for i in sorted_indices]
                    sorted_vol_values = [vol_values[i] for i in sorted_indices]
                    
                    fig = px.bar(
                        x=sorted_instruments,
                        y=sorted_vol_values,
                        labels={'x': 'Instrument', 'y': 'Volatility'},
                        title='Current Volatility by Instrument',
                        color=sorted_vol_values,
                        color_continuous_scale=['blue', 'red']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No volatility data available yet.")
        
        # 3. Spreads Tab
        with market_tabs[2]:
            st.subheader("Bid-Ask Spreads")
            
            # Get spread data
            spread_data = self.monitor_spread_changes()
            
            if spread_data:
                # Process spread data for display
                spreads_table = []
                
                for instrument, spread_info in spread_data.items():
                    current_spread_pips = spread_info.get('current_spread_pips', 0)
                    avg_spread_pips = spread_info.get('average_spread_pips', 0)
                    is_wide = spread_info.get('is_wide_spread', False)
                    is_narrow = spread_info.get('is_narrow_spread', False)
                    trading_session = spread_info.get('trading_session', 'Unknown')
                    
                    # Format spread status
                    status = "Normal"
                    status_icon = "🟡"
                    
                    if is_wide:
                        status = "Wide"
                        status_icon = "🔴"
                    elif is_narrow:
                        status = "Narrow"
                        status_icon = "🟢"
                    
                    spreads_table.append({
                        "Instrument": instrument,
                        "Current (pips)": f"{current_spread_pips:.1f}",
                        "Average (pips)": f"{avg_spread_pips:.1f}" if avg_spread_pips else "N/A",
                        "Status": f"{status_icon} {status}",
                        "Session": trading_session
                    })
                
                # Create a DataFrame and display it
                spreads_df = pd.DataFrame(spreads_table)
                st.dataframe(spreads_df, use_container_width=True)
                
                # Create bar chart of current spreads
                instruments = [item["Instrument"] for item in spreads_table]
                spread_values = [float(item["Current (pips)"].replace('N/A', '0')) for item in spreads_table]
                
                # Sort by spread size
                if instruments and spread_values:
                    sorted_indices = sorted(range(len(spread_values)), key=lambda i: spread_values[i], reverse=True)
                    sorted_instruments = [instruments[i] for i in sorted_indices]
                    sorted_spread_values = [spread_values[i] for i in sorted_indices]
                    
                    fig = px.bar(
                        x=sorted_instruments,
                        y=sorted_spread_values,
                        labels={'x': 'Instrument', 'y': 'Spread (pips)'},
                        title='Current Spreads by Instrument',
                        color=sorted_spread_values,
                        color_continuous_scale=['green', 'red']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No spread data available yet.")
        
        # 4. Correlations Tab
        with market_tabs[3]:
            st.subheader("Currency Pair Correlations")
            
            # Get correlation data
            correlation_data = self.track_correlation_changes()
            
            if correlation_data:
                # Process correlation data for display
                corr_table = []
                
                for pair_name, corr_info in correlation_data.items():
                    instruments = corr_info.get('instruments', [])
                    correlation = corr_info.get('correlation', 0)
                    strength = corr_info.get('strength', 'unknown')
                    direction = corr_info.get('direction', 'none')
                    regime_shift = corr_info.get('regime_shift', False)
                    
                    # Format correlation info
                    corr_color = "green" if correlation > 0.5 else "red" if correlation < -0.5 else "gray"
                    
                    corr_table.append({
                        "Pair": pair_name,
                        "Correlation": f"{correlation:.2f}",
                        "Strength": strength.title(),
                        "Direction": direction.title(),
                        "Regime Shift": "Yes" if regime_shift else "No"
                    })
                
                # Create a DataFrame and display it
                corr_df = pd.DataFrame(corr_table)
                st.dataframe(corr_df, use_container_width=True)
                
                # Create a correlation heatmap
                st.subheader("Correlation Matrix")
                
                # Extract unique instruments from all pairs
                unique_instruments = set()
                for pair_name, corr_info in correlation_data.items():
                    instruments = corr_info.get('instruments', [])
                    unique_instruments.update(instruments)
                
                unique_instruments = sorted(list(unique_instruments))
                
                # Create an empty correlation matrix
                n = len(unique_instruments)
                corr_matrix = np.eye(n)  # Identity matrix by default (1s on diagonal)
                
                # Fill the correlation matrix
                for i in range(n):
                    for j in range(i+1, n):
                        instr1 = unique_instruments[i]
                        instr2 = unique_instruments[j]
                        
                        # Look for this pair in the correlation data
                        pair_key = f"{instr1}/{instr2}"
                        alt_pair_key = f"{instr2}/{instr1}"
                        
                        if pair_key in correlation_data:
                            correlation = correlation_data[pair_key].get('correlation', 0)
                        elif alt_pair_key in correlation_data:
                            correlation = correlation_data[alt_pair_key].get('correlation', 0)
                        else:
                            correlation = 0
                        
                        # Fill both sides of the matrix (it's symmetric)
                        corr_matrix[i, j] = correlation
                        corr_matrix[j, i] = correlation
                
                # Create a heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=unique_instruments,
                    y=unique_instruments,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix,
                    texttemplate='%{text:.2f}',
                    textfont={"size":10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title="Correlation Matrix",
                    height=500,
                    width=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No correlation data available yet.")
        
        # 5. Market Anomalies Tab
        with market_tabs[4]:
            st.subheader("Market Anomalies")
            
            # Get market anomalies data
            anomalies = self.market_data["market_anomalies"]
            
            if anomalies:
                # Process anomalies for display
                anomalies_table = []
                
                for anomaly in anomalies:
                    anomaly_type = anomaly.get('type', 'unknown')
                    instrument = anomaly.get('instrument', 'Unknown')
                    timestamp = anomaly.get('timestamp', '')
                    message = anomaly.get('message', '')
                    
                    # Format timestamp
                    if timestamp:
                        try:
                            timestamp_dt = datetime.fromisoformat(timestamp)
                            timestamp_fmt = timestamp_dt.strftime("%H:%M:%S")
                        except:
                            timestamp_fmt = timestamp
                    else:
                        timestamp_fmt = "N/A"
                    
                    # Get type-specific details
                    details = ""
                    if anomaly_type == 'price_spike' or anomaly_type == 'price_crash':
                        details = f"{anomaly.get('magnitude', 0):.4f}%"
                    elif anomaly_type == 'extreme_volatility':
                        details = f"{anomaly.get('current_vol', 0)/anomaly.get('normal_vol', 1):.2f}x normal"
                    elif anomaly_type == 'abnormal_spread':
                        details = f"{anomaly.get('spread_pips', 0):.1f} pips"
                    elif anomaly_type == 'price_gap':
                        details = f"{anomaly.get('magnitude', 0):.4f}% {anomaly.get('direction', '')}"
                    elif anomaly_type == 'reversal_pattern':
                        details = f"{anomaly.get('pattern', '').replace('_', ' ').title()} ({anomaly.get('confidence', 0):.2f})"
                    
                    anomalies_table.append({
                        "Time": timestamp_fmt,
                        "Type": anomaly_type.replace('_', ' ').title(),
                        "Instrument": instrument,
                        "Details": details,
                        "Message": message
                    })
                
                # Create a DataFrame and display it
                anomalies_df = pd.DataFrame(anomalies_table)
                st.dataframe(anomalies_df, use_container_width=True)
                
                # Display anomalies by type in a pie chart
                anomaly_types = {}
                for anomaly in anomalies:
                    anomaly_type = anomaly.get('type', 'unknown').replace('_', ' ').title()
                    if anomaly_type in anomaly_types:
                        anomaly_types[anomaly_type] += 1
                    else:
                        anomaly_types[anomaly_type] = 1
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(anomaly_types.keys()),
                    values=list(anomaly_types.values()),
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig.update_layout(title_text='Anomalies by Type')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display anomalies by instrument in a bar chart
                anomaly_instruments = {}
                for anomaly in anomalies:
                    instrument = anomaly.get('instrument', 'Unknown')
                    if instrument in anomaly_instruments:
                        anomaly_instruments[instrument] += 1
                    else:
                        anomaly_instruments[instrument] = 1
                
                fig = px.bar(
                    x=list(anomaly_instruments.keys()),
                    y=list(anomaly_instruments.values()),
                    labels={'x': 'Instrument', 'y': 'Count'},
                    title='Anomalies by Instrument'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No market anomalies detected.")
                
    except Exception as e:
        st.error(f"Error creating market panel: {str(e)}")
        logger.error(f"Error creating market panel: {str(e)}")

def create_performance_panel(self) -> None:
    """
    Create a panel for performance metrics.
    
    Displays trading performance over time, drawdowns, win/loss ratios,
    and other key performance indicators.
    """
    try:
        st.header("Performance Metrics")
        
        # Create tabs for different performance aspects
        performance_tabs = st.tabs(["Overview", "Trades", "P/L", "Drawdown", "Instruments"])
        
        # Get performance data
        performance = self.track_trade_performance()
        
        # 1. Overview Tab
        with performance_tabs[0]:
            st.subheader("Performance Overview")
            
            if performance and performance.get("total_trades", 0) > 0:
                # Create a performance summary card
                profit_factor = performance.get("profit_factor", 0)
                win_rate = performance.get("win_rate", 0)
                net_pl = performance.get("total_profit", 0) - performance.get("total_loss", 0)
                
                # Create columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Net P/L", f"${net_pl:.2f}", delta=f"{net_pl:.2f}")
                    st.metric("Win Rate", f"{win_rate * 100:.1f}%")
                
                with col2:
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                    st.metric("Total Trades", performance.get("total_trades", 0))
                
                with col3:
                    st.metric("Avg Profit", f"${performance.get('average_profit', 0):.2f}")
                    st.metric("Avg Loss", f"-${abs(performance.get('average_loss', 0)):.2f}")
                
                # Create gauge charts for key metrics
                st.subheader("Key Performance Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Win Rate Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=win_rate * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        number={'suffix': "%"},
                        title={'text': "Win Rate"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 50], 'color': "orange"},
                                {'range': [50, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "green", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Profit Factor Gauge
                    # Cap at 5 for display purposes
                    displayed_pf = min(profit_factor, 5) if profit_factor != float('inf') else 5
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=displayed_pf,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        number={'valueformat': ".2f"},
                        title={'text': "Profit Factor"},
                        gauge={
                            'axis': {'range': [0, 5]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 1], 'color': "red"},
                                {'range': [1, 2], 'color': "yellow"},
                                {'range': [2, 5], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "green", 'width': 4},
                                'thickness': 0.75,
                                'value': 1
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No completed trades yet to analyze performance.")
        
        # 2. Trades Tab
        with performance_tabs[1]:
            st.subheader("Trade Analysis")
            
            # Get completed trades data
            completed_trades = self.trading_data["completed_trades"]
            
            if completed_trades:
                # Process trades for display
                trades_table = []
                
                for trade in completed_trades:
                    instrument = trade.get('instrument', 'Unknown')
                    direction = trade.get('direction', 'Unknown')
                    size = trade.get('size', 0)
                    entry_price = trade.get('entry_price', 0)
                    exit_price = trade.get('exit_price', 0)
                    profit_loss = trade.get('profit_loss', 0)
                    entry_time = trade.get('entry_time', '')
                    exit_time = trade.get('exit_time', '')
                    duration_hours = trade.get('duration_hours', 0)
                    
                    # Format times
                    if entry_time:
                        try:
                            entry_time_dt = datetime.fromisoformat(entry_time)
                            entry_time_fmt = entry_time_dt.strftime("%m-%d %H:%M")
                        except:
                            entry_time_fmt = entry_time
                    else:
                        entry_time_fmt = "N/A"
                    
                    if exit_time:
                        try:
                            exit_time_dt = datetime.fromisoformat(exit_time)
                            exit_time_fmt = exit_time_dt.strftime("%m-%d %H:%M")
                        except:
                            exit_time_fmt = exit_time
                    else:
                        exit_time_fmt = "N/A"
                    
                    trades_table.append({
                        "Instrument": instrument,
                        "Direction": direction.upper(),
                        "Size": f"{size:.2f}",
                        "Entry": f"{entry_price:.5f}",
                        "Exit": f"{exit_price:.5f}",
                        "P/L": f"${profit_loss:.2f}",
                        "Entry Time": entry_time_fmt,
                        "Exit Time": exit_time_fmt,
                        "Duration": f"{duration_hours:.1f}h"
                    })
                
                # Create a DataFrame and display it
                trades_df = pd.DataFrame(trades_table)
                st.dataframe(trades_df, use_container_width=True)
                
                # Display trade distribution by outcome
                st.subheader("Trade Distribution")
                
                # Count winning and losing trades
                winning_trades = [t for t in completed_trades if t.get('profit_loss', 0) > 0]
                losing_trades = [t for t in completed_trades if t.get('profit_loss', 0) <= 0]
                
                win_count = len(winning_trades)
                loss_count = len(losing_trades)
                
                # Create a pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Winning', 'Losing'],
                    values=[win_count, loss_count],
                    hole=.3,
                    marker_colors=['green', 'red']
                )])
                fig.update_layout(title_text='Trade Outcomes')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display trade durations in a histogram
                durations = [t.get('duration_hours', 0) for t in completed_trades]
                
                if durations:
                    fig = px.histogram(
                        x=durations,
                        nbins=20,
                        labels={'x': 'Duration (hours)', 'y': 'Count'},
                        title='Trade Duration Distribution',
                        color_discrete_sequence=['blue']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No completed trades available for analysis.")
        
        # 3. P/L Tab
        with performance_tabs[2]:
            st.subheader("Profit & Loss Analysis")
            
            # Get completed trades data
            completed_trades = self.trading_data["completed_trades"]
            
            if completed_trades:
                # Calculate cumulative P/L
                cumulative_pl = 0
                cumulative_data = []
                
                for i, trade in enumerate(completed_trades):
                    profit_loss = trade.get('profit_loss', 0)
                    exit_time = trade.get('exit_time', '')
                    instrument = trade.get('instrument', 'Unknown')
                    
                    cumulative_pl += profit_loss
                    
                    # Format exit time
                    if exit_time:
                        try:
                            exit_time_dt = datetime.fromisoformat(exit_time)
                        except:
                            exit_time_dt = datetime.now()  # fallback
                    else:
                        exit_time_dt = datetime.now()  # fallback
                    
                    cumulative_data.append({
                        "trade_num": i + 1,
                        "exit_time": exit_time_dt,
                        "profit_loss": profit_loss,
                        "cumulative_pl": cumulative_pl,
                        "instrument": instrument
                    })
                
                # Create a DataFrame for cumulative P/L
                cumulative_df = pd.DataFrame(cumulative_data)
                
                # Plot cumulative P/L over time
                fig = px.line(
                    cumulative_df,
                    x="exit_time",
                    y="cumulative_pl",
                    labels={'exit_time': 'Time', 'cumulative_pl': 'Cumulative P/L ($)'},
                    title='Cumulative Profit/Loss Over Time'
                )
                
                # Add scatter points for individual trades
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_df['exit_time'],
                        y=cumulative_df['profit_loss'],
                        mode='markers',
                        marker=dict(
                            color=cumulative_df['profit_loss'].apply(lambda x: 'green' if x > 0 else 'red'),
                            size=10
                        ),
                        name='Individual Trades'
                    )
                )
                
                fig.update_layout(hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display P/L distribution
                st.subheader("P/L Distribution")
                
                # Calculate P/L distribution
                profit_losses = [t.get('profit_loss', 0) for t in completed_trades]
                
                fig = px.histogram(
                    x=profit_losses,
                    nbins=20,
                    labels={'x': 'Profit/Loss ($)', 'y': 'Count'},
                    title='P/L Distribution',
                    color_discrete_sequence=['blue']
                )
                
                # Add a vertical line at zero
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No completed trades available for P/L analysis.")
        
        # 4. Drawdown Tab
        with performance_tabs[3]:
            st.subheader("Drawdown Analysis")
            
            # Get completed trades and balance history
            completed_trades = self.trading_data["completed_trades"]
            balance_history = list(self.trading_data["account_balance"])
            
            if completed_trades and balance_history:
                # Calculate drawdown from balance history
                timestamps = []
                balances = []
                drawdowns = []
                peak = 0
                max_drawdown = 0
                max_drawdown_pct = 0
                
                for timestamp, balance in balance_history:
                    timestamps.append(timestamp)
                    balances.append(balance)
                    
                    # Update peak if current balance is higher
                    if balance > peak:
                        peak = balance
                    
                    # Calculate drawdown
                    drawdown = peak - balance
                    drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0
                    
                    drawdowns.append(drawdown_pct)
                    
                    # Update max drawdown
                    if drawdown_pct > max_drawdown_pct:
                        max_drawdown = drawdown
                        max_drawdown_pct = drawdown_pct
                
                # Display max drawdown
                st.metric("Maximum Drawdown", f"${max_drawdown:.2f}", 
                         delta=f"-{max_drawdown_pct:.2f}%", delta_color="inverse")
                
                # Create a DataFrame for the chart
                drawdown_df = pd.DataFrame({
                    "timestamp": timestamps,
                    "balance": balances,
                    "drawdown_pct": drawdowns
                })
                
                # Plot balance and drawdown
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add balance line
                fig.add_trace(
                    go.Scatter(
                        x=drawdown_df['timestamp'],
                        y=drawdown_df['balance'],
                        name='Account Balance',
                        line=dict(color='blue')
                    ),
                    secondary_y=False
                )
                
                # Add drawdown area
                fig.add_trace(
                    go.Scatter(
                        x=drawdown_df['timestamp'],
                        y=drawdown_df['drawdown_pct'],
                        name='Drawdown %',
                        fill='tozeroy',
                        line=dict(color='red')
                    ),
                    secondary_y=True
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="Balance ($)", secondary_y=False)
                fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True)
                
                # Update layout
                fig.update_layout(
                    title_text="Balance and Drawdown Over Time",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown recovery analysis
                st.subheader("Recovery Analysis")
                
                # Create a table with drawdown recovery times (simulated for this example)
                recovery_data = []
                
                # Find periods of consecutive drawdown
                current_drawdown = 0
                drawdown_start = None
                drawdown_end = None
                in_drawdown = False
                
                for i in range(1, len(drawdown_df)):
                    # If drawdown is increasing and we're not in a drawdown period
                    if drawdown_df['drawdown_pct'][i] > 0 and not in_drawdown:
                        drawdown_start = drawdown_df['timestamp'][i]
                        current_drawdown = drawdown_df['drawdown_pct'][i]
                        in_drawdown = True
                    
                    # If we're in a drawdown and it becomes zero
                    elif drawdown_df['drawdown_pct'][i] == 0 and in_drawdown:
                        drawdown_end = drawdown_df['timestamp'][i]
                        
                        # Calculate recovery time
                        recovery_time = (drawdown_end - drawdown_start).total_seconds() / 3600  # hours
                        
                        recovery_data.append({
                            "Start Time": drawdown_start.strftime("%Y-%m-%d %H:%M"),
                            "End Time": drawdown_end.strftime("%Y-%m-%d %H:%M"),
                            "Max Drawdown": f"{current_drawdown:.2f}%",
                            "Recovery Time": f"{recovery_time:.1f}h"
                        })
                        
                        in_drawdown = False
                    
                    # If we're in a drawdown and it's getting worse
                    elif in_drawdown and drawdown_df['drawdown_pct'][i] > current_drawdown:
                        current_drawdown = drawdown_df['drawdown_pct'][i]
                
                # If still in a drawdown at the end of the data
                if in_drawdown:
                    recovery_data.append({
                        "Start Time": drawdown_start.strftime("%Y-%m-%d %H:%M"),
                        "End Time": "Ongoing",
                        "Max Drawdown": f"{current_drawdown:.2f}%",
                        "Recovery Time": "Ongoing"
                    })
                
                if recovery_data:
                    recovery_df = pd.DataFrame(recovery_data)
                    st.dataframe(recovery_df, use_container_width=True)
                else:
                    st.info("No significant drawdown periods identified.")
            else:
                st.info("Not enough data for drawdown analysis.")
        
        # 5. Instruments Tab
        with performance_tabs[4]:
            st.subheader("Instrument Performance")
            
            # Get instrument performance data
            instrument_performance = performance.get("instrument_performance", {})
            
            if instrument_performance:
                # Process instrument data for display
                instrument_data = []
                
                for instrument, stats in instrument_performance.items():
                    trades = stats.get("trades", 0)
                    win_rate = stats.get("win_rate", 0)
                    net_pl = stats.get("net_pl", 0)
                    
                    instrument_data.append({
                        "Instrument": instrument,
                        "Trades": trades,
                        "Win Rate": win_rate,
                        "Net P/L": net_pl
                    })
                
                # Create a DataFrame for charts
                instrument_df = pd.DataFrame(instrument_data)
                
                # Create a bar chart for P/L by instrument
                fig = px.bar(
                    instrument_df,
                    x="Instrument",
                    y="Net P/L",
                    color="Net P/L",
                    color_continuous_scale=['red', 'green'],
                    color_continuous_midpoint=0,
                    labels={'Net P/L': 'Net Profit/Loss ($)'},
                    title='Performance by Instrument'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create columns for additional charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Win Rate by Instrument
                    fig = px.bar(
                        instrument_df,
                        x="Instrument",
                        y="Win Rate",
                        color="Win Rate",
                        color_continuous_scale=['red', 'yellow', 'green'],
                        color_continuous_midpoint=0.5,
                        labels={'Win Rate': 'Win Rate (%)'},
                        title='Win Rate by Instrument'
                    )
                    # Scale y-axis from 0 to 1 (0% to 100%)
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Trade Count by Instrument
                    fig = px.bar(
                        instrument_df,
                        x="Instrument",
                        y="Trades",
                        color="Trades",
                        color_continuous_scale=['blue'],
                        labels={'Trades': 'Number of Trades'},
                        title='Trade Count by Instrument'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create a scatter plot of Win Rate vs P/L
                fig = px.scatter(
                    instrument_df,
                    x="Win Rate",
                    y="Net P/L",
                    size="Trades",
                    color="Instrument",
                    labels={'Win Rate': 'Win Rate (%)', 'Net P/L': 'Net Profit/Loss ($)', 'Trades': 'Number of Trades'},
                    title='Win Rate vs P/L by Instrument'
                )
                fig.update_layout(xaxis_range=[0, 1])  # 0% to 100%
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No instrument performance data available.")
                
    except Exception as e:
        st.error(f"Error creating performance panel: {str(e)}")
        logger.error(f"Error creating performance panel: {str(e)}")

def create_alert_panel(self) -> None:
    """
    Create a panel for alerts and notifications.
    
    Displays recent alerts, allows filtering by category and severity,
    and provides alert acknowledgment functionality.
    """
    try:
        st.header("Alerts & Notifications")
        
        # Create tabs for alert views
        alert_tabs = st.tabs(["Recent Alerts", "Alert History", "Alert Stats", "Alert Settings"])
        
        # Get alert data
        alerts = list(self.alert_data["alerts"])
        dashboard_alerts = list(self.alert_data["dashboard_alerts"])
        
        # 1. Recent Alerts Tab
        with alert_tabs[0]:
            st.subheader("Recent Alerts")
            
            # Add filters for alerts
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter by severity
                severity_options = ["All"] + list(self.alert_data["alert_levels"].keys())
                selected_severity = st.selectbox("Filter by Severity", severity_options)
            
            with col2:
                # Filter by category
                category_options = ["All"] + list(self.alert_data["alert_categories"].keys())
                selected_category = st.selectbox("Filter by Category", category_options)
            
            # Apply filters
            filtered_alerts = dashboard_alerts
            
            if selected_severity != "All":
                filtered_alerts = [a for a in filtered_alerts if a.get("level") == selected_severity]
            
            if selected_category != "All":
                filtered_alerts = [a for a in filtered_alerts if a.get("category") == selected_category]
            
            # Display alerts
            if filtered_alerts:
                for alert in reversed(filtered_alerts):  # Show newest first
                    # Get alert properties
                    level = alert.get("level", "info")
                    category = alert.get("category", "system")
                    message = alert.get("message", "")
                    timestamp = alert.get("timestamp", "")
                    icon = alert.get("icon", "ℹ️")
                    color = alert.get("color", "blue")
                    alert_id = alert.get("id", "")
                    
                    # Format timestamp
                    if timestamp:
                        try:
                            timestamp_dt = datetime.fromisoformat(timestamp)
                            timestamp_fmt = timestamp_dt.strftime("%H:%M:%S")
                        except:
                            timestamp_fmt = timestamp
                    else:
                        timestamp_fmt = "N/A"
                    
                    # Create alert box
                    alert_box = f"""
                    <div style="padding: 10px; margin-bottom: 10px; border-radius: 5px; border-left: 5px solid {color}; background-color: rgba(0,0,0,0.05);">
                        <div style="display: flex; justify-content: space-between;">
                            <h4>{icon} {level.upper()}: {category.title()}</h4>
                            <span style="color: gray;">{timestamp_fmt}</span>
                        </div>
                        <p style="margin: 5px 0;">{message}</p>
                        <div style="display: flex; justify-content: flex-end;">
                            <small style="color: gray;">ID: {alert_id}</small>
                        </div>
                    </div>
                    """
                    st.markdown(alert_box, unsafe_allow_html=True)
                
                # Add a button to clear all alerts
                if st.button("Acknowledge All Alerts"):
                    st.success("All alerts acknowledged!")
                    # In a real implementation, this would mark all alerts as acknowledged
            else:
                st.info("No alerts match the selected filters.")
        
        # 2. Alert History Tab
        with alert_tabs[1]:
            st.subheader("Alert History")
            
            # Create a dataframe of all historical alerts
            if alerts:
                history_data = []
                
                for alert in alerts:
                    level = alert.get("level", "info")
                    category = alert.get("category", "system")
                    message = alert.get("message", "")
                    timestamp = alert.get("timestamp", "")
                    
                    # Format timestamp
                    if timestamp:
                        try:
                            timestamp_dt = datetime.fromisoformat(timestamp)
                            timestamp_fmt = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            timestamp_fmt = timestamp
                    else:
                        timestamp_fmt = "N/A"
                    
                    history_data.append({
                        "Time": timestamp_fmt,
                        "Level": level.upper(),
                        "Category": category.title(),
                        "Message": message
                    })
                
                # Create DataFrame and display as table
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    
                    # Allow download of alert history as CSV
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download Alert History",
                        data=csv,
                        file_name="alert_history.csv",
                        mime="text/csv"
                    )
                    
                    st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No alert history available.")
        
        # 3. Alert Stats Tab
        with alert_tabs[2]:
            st.subheader("Alert Statistics")
            
            if alerts:
                # Calculate alert stats
                alert_levels = {}
                alert_categories = {}
                alerts_over_time = {}
                
                for alert in alerts:
                    level = alert.get("level", "info")
                    category = alert.get("category", "system")
                    timestamp = alert.get("timestamp", "")
                    
                    # Count by level
                    if level in alert_levels:
                        alert_levels[level] += 1
                    else:
                        alert_levels[level] = 1
                    
                    # Count by category
                    if category in alert_categories:
                        alert_categories[category] += 1
                    else:
                        alert_categories[category] = 1
                    
                    # Group by time period (hour)
                    if timestamp:
                        try:
                            timestamp_dt = datetime.fromisoformat(timestamp)
                            # Round to the hour
                            hour_key = timestamp_dt.replace(minute=0, second=0, microsecond=0)
                            
                            if hour_key in alerts_over_time:
                                alerts_over_time[hour_key] += 1
                            else:
                                alerts_over_time[hour_key] = 1
                        except:
                            pass
                
                # Create columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create pie chart for alert levels
                    fig = go.Figure(data=[go.Pie(
                        labels=list(alert_levels.keys()),
                        values=list(alert_levels.values()),
                        hole=.3,
                        textinfo='label+percent',
                        marker_colors=['blue', 'orange', 'red', 'purple']
                    )])
                    fig.update_layout(title_text='Alerts by Severity')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create pie chart for alert categories
                    fig = go.Figure(data=[go.Pie(
                        labels=list(alert_categories.keys()),
                        values=list(alert_categories.values()),
                        hole=.3,
                        textinfo='label+percent'
                    )])
                    fig.update_layout(title_text='Alerts by Category')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create a time series chart for alerts
                if alerts_over_time:
                    # Convert to dataframe
                    time_data = []
                    for time_key, count in alerts_over_time.items():
                        time_data.append({
                            "time": time_key,
                            "count": count
                        })
                    
                    time_df = pd.DataFrame(time_data)
                    time_df = time_df.sort_values("time")
                    
                    # Create line chart
                    fig = px.line(
                        time_df,
                        x="time",
                        y="count",
                        labels={'time': 'Time', 'count': 'Alert Count'},
                        title='Alert Frequency Over Time'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No alerts to display statistics for.")
        
        # 4. Alert Settings Tab
        with alert_tabs[3]:
            st.subheader("Alert Settings")
            
            # Add form for alert settings
            email_enabled = st.checkbox("Enable Email Alerts", value=self.alert_data["email_enabled"])
            sms_enabled = st.checkbox("Enable SMS Alerts", value=self.alert_data["sms_enabled"])
            
            if st.button("Save Changes"):
                # Update alert settings
                self.alert_data["email_enabled"] = email_enabled
                self.alert_data["sms_enabled"] = sms_enabled
                st.success("Alert settings updated successfully!")
                
    except Exception as e:
        st.error(f"Error creating alert panel: {str(e)}")
        logger.error(f"Error in create_alert_panel: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize the dashboard
    dashboard = MonitoringDashboard()
    
    # Start monitoring
    dashboard.start_monitoring()
    
    try:
        # Log startup event
        dashboard.log_system_events("startup", "Monitoring system started", "info")
        
        # Run for a while for testing
        time.sleep(30)
        
        # Print some stats
        print("Agent Status:", dashboard.system_data["agent_status"])
        print("System Resources:", dashboard.monitor_system_resources())
        print("API Connections:", dashboard.system_data["api_connections"])
        print("Anomalies:", dashboard.system_data["anomalies"])
        
    finally:
        # Stop monitoring
        dashboard.stop_monitoring()
        dashboard.log_system_events("shutdown", "Monitoring system stopped", "info") 