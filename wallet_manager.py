#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wallet Manager for Forex Trading Platform

This module provides a comprehensive wallet management system for interacting with 
the OANDA trading platform. It handles account connections, balance management,
transaction tracking, safety features, and reporting.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import requests
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wallet_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WalletManager")

class WalletManager:
    """
    A comprehensive wallet manager for forex trading via OANDA.
    
    This class provides functionality for account connection, balance management,
    transaction tracking, safety features, and reporting using the OANDA API.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the WalletManager.
        
        Args:
            config_path: Path to a configuration file with API credentials.
                         If None, will load from environment variables.
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize from config file or environment variables
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.api_key = config.get('OANDA_API_KEY')
                self.account_id = config.get('OANDA_ACCOUNT_ID')
                self.api_url = config.get('OANDA_API_URL', 'https://api-fxpractice.oanda.com')
        else:
            self.api_key = os.getenv('OANDA_API_KEY')
            self.account_id = os.getenv('OANDA_ACCOUNT_ID')
            self.api_url = os.getenv('OANDA_API_URL', 'https://api-fxpractice.oanda.com')
        
        # Validate credentials
        if not all([self.api_key, self.account_id]):
            raise ValueError("Missing OANDA API credentials. Please provide via config file or environment variables.")
        
        # Set up request headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize account connection status
        self.is_connected = False
        self.account_details = None
        
        # Set up safety features default values
        self.max_drawdown_percentage = 0.10  # 10% default max drawdown
        self.daily_loss_limit = None
        self.margin_alert_threshold = 0.50  # 50% margin used alert
        
        # Create directories for reports if they don't exist
        self.reports_dir = Path('reports')
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info("WalletManager initialized. Ready to connect to OANDA account.")
    
    #--------------------------------------------------
    # 1. Account Connection Methods
    #--------------------------------------------------
    
    def connect_to_oanda_account(self) -> bool:
        """
        Connect to the OANDA account using the provided credentials.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            # Test connection by getting account details
            account_details = self.get_account_details()
            
            if account_details:
                self.is_connected = True
                self.account_details = account_details
                logger.info(f"Successfully connected to OANDA account {self.account_id}")
                return True
            else:
                self.is_connected = False
                logger.error("Failed to connect to OANDA account: No account details received")
                return False
                
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to OANDA account: {str(e)}")
            return False
    
    def get_account_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the OANDA account.
        
        Returns:
            Dict containing comprehensive account details.
            Empty dict if there was an error.
        """
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                account_details = response.json().get('account', {})
                logger.info("Successfully retrieved account details")
                return account_details
            else:
                logger.error(f"Failed to get account details: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error retrieving account details: {str(e)}")
            return {}
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the OANDA account.
        
        Returns:
            Dict containing a summary of the account.
            Empty dict if there was an error.
        """
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}/summary"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                account_summary = response.json().get('account', {})
                logger.info("Successfully retrieved account summary")
                return account_summary
            else:
                logger.error(f"Failed to get account summary: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error retrieving account summary: {str(e)}")
            return {}
    
    def get_account_instruments(self) -> List[Dict[str, Any]]:
        """
        Get information about instruments available for the account.
        
        Returns:
            List of dicts containing instrument details.
            Empty list if there was an error.
        """
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}/instruments"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                instruments = response.json().get('instruments', [])
                logger.info(f"Successfully retrieved {len(instruments)} account instruments")
                return instruments
            else:
                logger.error(f"Failed to get account instruments: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving account instruments: {str(e)}")
            return []
    
    def handle_connection_errors(self, response: requests.Response) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle connection errors from the OANDA API.
        
        Args:
            response: The response object from an API request
            
        Returns:
            Tuple containing (success: bool, data: Dict)
        """
        if response.status_code == 200:
            return True, response.json()
        elif response.status_code == 401:
            logger.error("Authentication failed: Invalid API key or permissions")
            return False, {"error": "Authentication failed"}
        elif response.status_code == 403:
            logger.error("Forbidden: Insufficient permissions")
            return False, {"error": "Forbidden access"}
        elif response.status_code == 404:
            logger.error("Not found: Resource not found")
            return False, {"error": "Resource not found"}
        elif response.status_code == 429:
            logger.error("Rate limiting: Too many requests")
            return False, {"error": "Rate limit exceeded"}
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return False, {"error": f"API error: {response.status_code}"}
    
    #--------------------------------------------------
    # 2. Balance Management Methods
    #--------------------------------------------------
    
    def get_account_balance(self) -> float:
        """
        Get the current account balance.
        
        Returns:
            float: The current account balance.
            0.0 if there was an error.
        """
        try:
            account_summary = self.get_account_summary()
            
            if account_summary:
                balance = float(account_summary.get('balance', 0.0))
                logger.info(f"Current account balance: {balance}")
                return balance
            else:
                logger.error("Failed to get account balance: No account summary available")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error retrieving account balance: {str(e)}")
            return 0.0
    
    def get_margin_available(self) -> float:
        """
        Get the available margin for the account.
        
        Returns:
            float: The available margin.
            0.0 if there was an error.
        """
        try:
            account_summary = self.get_account_summary()
            
            if account_summary:
                margin_available = float(account_summary.get('marginAvailable', 0.0))
                logger.info(f"Available margin: {margin_available}")
                return margin_available
            else:
                logger.error("Failed to get available margin: No account summary available")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error retrieving available margin: {str(e)}")
            return 0.0
    
    def get_margin_used(self) -> float:
        """
        Get the margin currently in use.
        
        Returns:
            float: The margin used.
            0.0 if there was an error.
        """
        try:
            account_summary = self.get_account_summary()
            
            if account_summary:
                margin_used = float(account_summary.get('marginUsed', 0.0))
                logger.info(f"Margin used: {margin_used}")
                return margin_used
            else:
                logger.error("Failed to get margin used: No account summary available")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error retrieving margin used: {str(e)}")
            return 0.0
    
    def get_margin_closeout_percent(self) -> float:
        """
        Get the margin closeout percentage.
        
        Returns:
            float: The margin closeout percentage (0-100).
            0.0 if there was an error.
        """
        try:
            account_summary = self.get_account_summary()
            
            if account_summary:
                margin_closeout_percent = float(account_summary.get('marginCloseoutPercent', 0.0)) * 100
                logger.info(f"Margin closeout percentage: {margin_closeout_percent:.2f}%")
                return margin_closeout_percent
            else:
                logger.error("Failed to get margin closeout percentage: No account summary available")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error retrieving margin closeout percentage: {str(e)}")
            return 0.0
    
    def get_margin_closeout_value(self) -> float:
        """
        Get the margin closeout value.
        
        Returns:
            float: The margin closeout value.
            0.0 if there was an error.
        """
        try:
            account_summary = self.get_account_summary()
            
            if account_summary:
                margin_closeout_value = float(account_summary.get('marginCloseoutValue', 0.0))
                logger.info(f"Margin closeout value: {margin_closeout_value}")
                return margin_closeout_value
            else:
                logger.error("Failed to get margin closeout value: No account summary available")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error retrieving margin closeout value: {str(e)}")
            return 0.0
    
    def calculate_margin_requirements(self, instrument: str, units: float) -> Dict[str, float]:
        """
        Calculate margin requirements for a trade.
        
        Args:
            instrument: The instrument to trade (e.g., "EUR_USD")
            units: The number of units to trade (positive for buy, negative for sell)
            
        Returns:
            Dict containing margin requirements details.
            Empty dict if there was an error.
        """
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}/instruments/{instrument}/position"
            payload = {
                "longUnits": str(abs(units)) if units > 0 else "0",
                "shortUnits": str(abs(units)) if units < 0 else "0"
            }
            
            response = requests.get(
                url, 
                headers=self.headers,
                params=payload
            )
            
            if response.status_code == 200:
                margin_data = response.json().get('positionMargin', {})
                
                # Extract and return relevant margin information
                result = {
                    'marginRequired': float(margin_data.get('marginRequired', 0.0)),
                    'marginUsed': float(margin_data.get('marginUsed', 0.0)),
                    'marginAvailable': float(margin_data.get('marginAvailable', 0.0))
                }
                
                logger.info(f"Margin requirements for {instrument} ({units} units): {result}")
                return result
            else:
                logger.error(f"Failed to calculate margin requirements: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error calculating margin requirements: {str(e)}")
            return {}
    
    #--------------------------------------------------
    # 3. Transaction Tracking Methods
    #--------------------------------------------------
    
    def get_transaction_history(self, from_time: Optional[datetime] = None, 
                              to_time: Optional[datetime] = None, 
                              transaction_type: Optional[str] = None,
                              count: int = 100) -> List[Dict[str, Any]]:
        """
        Get transaction history for the account.
        
        Args:
            from_time: Start time for transactions (defaults to 7 days ago)
            to_time: End time for transactions (defaults to now)
            transaction_type: Filter by transaction type
            count: Maximum number of transactions to return
            
        Returns:
            List of transaction dictionaries.
            Empty list if there was an error.
        """
        try:
            # Set default time range if not provided
            if not from_time:
                from_time = datetime.utcnow() - timedelta(days=7)
            if not to_time:
                to_time = datetime.utcnow()
            
            # Format dates as required by the API
            from_str = from_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            to_str = to_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Build URL and parameters
            url = f"{self.api_url}/v3/accounts/{self.account_id}/transactions"
            params = {
                "from": from_str,
                "to": to_str,
                "count": count
            }
            
            if transaction_type:
                params["type"] = transaction_type
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                transactions = response.json().get('transactions', [])
                logger.info(f"Retrieved {len(transactions)} transactions")
                return transactions
            else:
                logger.error(f"Failed to get transaction history: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving transaction history: {str(e)}")
            return []
    
    def get_transaction_details(self, transaction_id: str) -> Dict[str, Any]:
        """
        Get details of a specific transaction.
        
        Args:
            transaction_id: The ID of the transaction
            
        Returns:
            Dict containing transaction details.
            Empty dict if there was an error.
        """
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}/transactions/{transaction_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                transaction = response.json().get('transaction', {})
                logger.info(f"Retrieved details for transaction {transaction_id}")
                return transaction
            else:
                logger.error(f"Failed to get transaction details: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error retrieving transaction details: {str(e)}")
            return {}
    
    def categorize_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize transactions by type.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dict mapping transaction types to lists of transactions
        """
        try:
            categorized = {}
            
            for transaction in transactions:
                tx_type = transaction.get('type', 'UNKNOWN')
                
                if tx_type not in categorized:
                    categorized[tx_type] = []
                
                categorized[tx_type].append(transaction)
            
            # Log summary of categorization
            summary = {tx_type: len(txs) for tx_type, txs in categorized.items()}
            logger.info(f"Categorized transactions: {summary}")
            
            return categorized
            
        except Exception as e:
            logger.error(f"Error categorizing transactions: {str(e)}")
            return {}
    
    def calculate_realized_pnl(self, timeframe: str = "7d") -> float:
        """
        Calculate realized profit/loss over a given timeframe.
        
        Args:
            timeframe: Time period for calculation ('1d', '7d', '30d', '90d', 'ytd', 'all')
            
        Returns:
            float: Total realized P/L
            0.0 if there was an error
        """
        try:
            # Determine the start date based on timeframe
            end_time = datetime.utcnow()
            
            if timeframe == "1d":
                start_time = end_time - timedelta(days=1)
            elif timeframe == "7d":
                start_time = end_time - timedelta(days=7)
            elif timeframe == "30d":
                start_time = end_time - timedelta(days=30)
            elif timeframe == "90d":
                start_time = end_time - timedelta(days=90)
            elif timeframe == "ytd":
                start_time = datetime(end_time.year, 1, 1)
            elif timeframe == "all":
                start_time = datetime(2000, 1, 1)  # A date far in the past
            else:
                logger.error(f"Invalid timeframe: {timeframe}")
                return 0.0
            
            # Get relevant transaction types for P/L calculation
            transactions = self.get_transaction_history(
                from_time=start_time,
                to_time=end_time,
                count=500  # Increase if needed for longer timeframes
            )
            
            total_pnl = 0.0
            
            for transaction in transactions:
                # Sum up realized P/L from various transaction types
                tx_type = transaction.get('type', '')
                
                if tx_type == 'ORDER_FILL':
                    # Add P/L from closed trades
                    if 'pl' in transaction:
                        total_pnl += float(transaction['pl'])
                    
                    # Add financing costs
                    if 'financing' in transaction:
                        total_pnl += float(transaction['financing'])
                
                elif tx_type == 'TRADE_CLOSE':
                    # Add P/L from manually closed trades
                    if 'pl' in transaction:
                        total_pnl += float(transaction['pl'])
                
                elif tx_type in ['MARGIN_CALL_ENTER', 'MARGIN_CALL_EXIT']:
                    # These may also affect realized P/L
                    if 'pl' in transaction:
                        total_pnl += float(transaction['pl'])
            
            logger.info(f"Realized P/L for {timeframe}: {total_pnl}")
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error calculating realized P/L: {str(e)}")
            return 0.0
    
    def track_deposits_withdrawals(self) -> Dict[str, float]:
        """
        Track deposits and withdrawals to the account.
        
        Returns:
            Dict with total deposits and withdrawals
        """
        try:
            # Get all relevant transactions (past year)
            start_time = datetime.utcnow() - timedelta(days=365)
            end_time = datetime.utcnow()
            
            transactions = self.get_transaction_history(
                from_time=start_time,
                to_time=end_time,
                count=1000  # Adjust as needed
            )
            
            total_deposits = 0.0
            total_withdrawals = 0.0
            
            for transaction in transactions:
                tx_type = transaction.get('type', '')
                
                if tx_type == 'TRANSFER_FUNDS':
                    amount = float(transaction.get('amount', 0.0))
                    
                    if amount > 0:
                        total_deposits += amount
                    else:
                        total_withdrawals += abs(amount)
            
            result = {
                'deposits': total_deposits,
                'withdrawals': total_withdrawals,
                'net': total_deposits - total_withdrawals
            }
            
            logger.info(f"Deposits/withdrawals summary: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error tracking deposits/withdrawals: {str(e)}")
            return {'deposits': 0.0, 'withdrawals': 0.0, 'net': 0.0}
    
    def generate_transaction_report(self, timeframe: str = "30d") -> pd.DataFrame:
        """
        Generate a detailed transaction report.
        
        Args:
            timeframe: Time period for the report ('1d', '7d', '30d', '90d', 'ytd', 'all')
            
        Returns:
            DataFrame with transaction details
            Empty DataFrame if there was an error
        """
        try:
            # Determine the start date based on timeframe
            end_time = datetime.utcnow()
            
            if timeframe == "1d":
                start_time = end_time - timedelta(days=1)
            elif timeframe == "7d":
                start_time = end_time - timedelta(days=7)
            elif timeframe == "30d":
                start_time = end_time - timedelta(days=30)
            elif timeframe == "90d":
                start_time = end_time - timedelta(days=90)
            elif timeframe == "ytd":
                start_time = datetime(end_time.year, 1, 1)
            elif timeframe == "all":
                start_time = datetime(2000, 1, 1)  # A date far in the past
            else:
                logger.error(f"Invalid timeframe: {timeframe}")
                return pd.DataFrame()
            
            # Get transactions
            transactions = self.get_transaction_history(
                from_time=start_time,
                to_time=end_time,
                count=1000  # Adjust as needed
            )
            
            if not transactions:
                return pd.DataFrame()
            
            # Extract relevant fields for each transaction type
            processed_transactions = []
            
            for tx in transactions:
                tx_type = tx.get('type', 'UNKNOWN')
                tx_id = tx.get('id', 'UNKNOWN')
                tx_time = tx.get('time', '')
                
                # Process different transaction types differently
                if tx_type == 'ORDER_FILL':
                    instrument = tx.get('instrument', '')
                    units = float(tx.get('units', 0))
                    price = float(tx.get('price', 0))
                    pl = float(tx.get('pl', 0))
                    financing = float(tx.get('financing', 0))
                    commission = float(tx.get('commission', 0))
                    
                    processed_transactions.append({
                        'id': tx_id,
                        'time': tx_time,
                        'type': tx_type,
                        'instrument': instrument,
                        'units': units,
                        'price': price,
                        'pl': pl,
                        'financing': financing,
                        'commission': commission,
                        'total_impact': pl + financing + commission
                    })
                
                elif tx_type == 'TRADE_CLOSE':
                    instrument = tx.get('instrument', '')
                    units = float(tx.get('units', 0))
                    price = float(tx.get('price', 0))
                    pl = float(tx.get('pl', 0))
                    
                    processed_transactions.append({
                        'id': tx_id,
                        'time': tx_time,
                        'type': tx_type,
                        'instrument': instrument,
                        'units': units,
                        'price': price,
                        'pl': pl,
                        'financing': 0.0,
                        'commission': 0.0,
                        'total_impact': pl
                    })
                
                elif tx_type == 'TRANSFER_FUNDS':
                    amount = float(tx.get('amount', 0))
                    
                    processed_transactions.append({
                        'id': tx_id,
                        'time': tx_time,
                        'type': tx_type,
                        'instrument': 'FUNDS',
                        'units': 0,
                        'price': 0,
                        'pl': 0,
                        'financing': 0,
                        'commission': 0,
                        'total_impact': amount
                    })
                
                else:
                    # Generic handling for other transaction types
                    processed_transactions.append({
                        'id': tx_id,
                        'time': tx_time,
                        'type': tx_type,
                        'instrument': tx.get('instrument', 'N/A'),
                        'units': float(tx.get('units', 0)) if 'units' in tx else 0,
                        'price': float(tx.get('price', 0)) if 'price' in tx else 0,
                        'pl': float(tx.get('pl', 0)) if 'pl' in tx else 0,
                        'financing': float(tx.get('financing', 0)) if 'financing' in tx else 0,
                        'commission': float(tx.get('commission', 0)) if 'commission' in tx else 0,
                        'total_impact': 0.0  # Cannot determine generically
                    })
            
            # Create DataFrame
            df = pd.DataFrame(processed_transactions)
            
            # Convert time strings to datetime objects
            df['time'] = pd.to_datetime(df['time'])
            
            # Sort by time
            df = df.sort_values('time', ascending=False)
            
            # Save report to CSV
            report_path = self.reports_dir / f"transaction_report_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(report_path, index=False)
            logger.info(f"Transaction report saved to {report_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating transaction report: {str(e)}")
            return pd.DataFrame()
    
    #--------------------------------------------------
    # 4. Safety Features Methods
    #--------------------------------------------------
    
    def check_margin_health(self) -> Dict[str, Any]:
        """
        Check the health of margin levels and return a status assessment.
        
        Returns:
            Dict with margin health assessment
        """
        try:
            margin_used = self.get_margin_used()
            margin_available = self.get_margin_available()
            margin_closeout_percent = self.get_margin_closeout_percent()
            account_balance = self.get_account_balance()
            
            # Calculate metrics
            margin_usage_ratio = margin_used / (margin_used + margin_available) if (margin_used + margin_available) > 0 else 0
            
            # Determine health status based on margin usage
            if margin_usage_ratio < 0.25:
                status = "HEALTHY"
                message = "Margin levels are healthy."
                risk_level = "LOW"
            elif margin_usage_ratio < 0.50:
                status = "NORMAL"
                message = "Margin usage is at normal levels."
                risk_level = "MEDIUM"
            elif margin_usage_ratio < 0.75:
                status = "WARNING"
                message = "Warning: Margin usage is elevated."
                risk_level = "HIGH"
            else:
                status = "DANGER"
                message = "DANGER: Margin usage is very high. Risk of margin call."
                risk_level = "EXTREME"
            
            # Compile results
            result = {
                "status": status,
                "message": message,
                "risk_level": risk_level,
                "margin_usage_ratio": margin_usage_ratio,
                "margin_used": margin_used,
                "margin_available": margin_available,
                "margin_closeout_percent": margin_closeout_percent,
                "account_balance": account_balance
            }
            
            logger.info(f"Margin health check: {status}, {message}")
            return result
            
        except Exception as e:
            logger.error(f"Error checking margin health: {str(e)}")
            return {
                "status": "ERROR",
                "message": f"Error checking margin health: {str(e)}",
                "risk_level": "UNKNOWN"
            }
    
    def implement_stop_out_protection(self, emergency_close: bool = False) -> Dict[str, Any]:
        """
        Implement protection against margin stop outs.
        
        This checks margin levels and can automatically close positions
        if margin usage is too high.
        
        Args:
            emergency_close: If True, will forcibly close positions if margin is critical
            
        Returns:
            Dict with results of the stop out protection check/actions
        """
        try:
            # Get margin health assessment
            margin_health = self.check_margin_health()
            
            # Determine if action is needed
            action_taken = False
            positions_closed = []
            
            if margin_health["risk_level"] == "EXTREME" and emergency_close:
                # Get open positions
                url = f"{self.api_url}/v3/accounts/{self.account_id}/openPositions"
                response = requests.get(url, headers=self.headers)
                
                if response.status_code != 200:
                    logger.error(f"Failed to get open positions: {response.status_code} - {response.text}")
                    return {
                        "success": False,
                        "message": "Failed to get open positions",
                        "action_taken": False
                    }
                
                positions = response.json().get('positions', [])
                
                # Close positions to reduce margin
                for position in positions:
                    instrument = position.get('instrument')
                    long_units = abs(float(position.get('long', {}).get('units', '0')))
                    short_units = abs(float(position.get('short', {}).get('units', '0')))
                    
                    # Determine units to close
                    if long_units > 0:
                        units_to_close = -long_units  # Negative to close long
                    elif short_units > 0:
                        units_to_close = short_units  # Positive to close short
                    else:
                        continue
                    
                    # Close the position
                    url = f"{self.api_url}/v3/accounts/{self.account_id}/orders"
                    payload = {
                        "order": {
                            "type": "MARKET",
                            "instrument": instrument,
                            "units": str(units_to_close),
                            "timeInForce": "FOK",
                            "positionFill": "REDUCE_ONLY",
                            "reason": "STOP_OUT_PROTECTION"
                        }
                    }
                    
                    close_response = requests.post(url, headers=self.headers, json=payload)
                    
                    if close_response.status_code == 201:
                        positions_closed.append(instrument)
                        action_taken = True
                        logger.warning(f"Stop out protection closed position: {instrument}")
                    else:
                        logger.error(f"Failed to close position {instrument}: {close_response.status_code} - {close_response.text}")
            
            result = {
                "success": True,
                "margin_health": margin_health,
                "action_taken": action_taken,
                "positions_closed": positions_closed
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error implementing stop out protection: {str(e)}")
            return {
                "success": False,
                "message": f"Error implementing stop out protection: {str(e)}",
                "action_taken": False
            }
    
    def set_max_drawdown_protection(self, percentage: float) -> bool:
        """
        Set protection against excessive drawdown.
        
        Args:
            percentage: Maximum allowable drawdown as a decimal (0.05 = 5%)
            
        Returns:
            bool: True if successfully set, False otherwise
        """
        try:
            if percentage <= 0 or percentage >= 1:
                logger.error(f"Invalid drawdown percentage: {percentage}. Must be between 0 and 1.")
                return False
            
            self.max_drawdown_percentage = percentage
            logger.info(f"Max drawdown protection set to {percentage:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting max drawdown protection: {str(e)}")
            return False
    
    def check_drawdown_protection(self) -> Dict[str, Any]:
        """
        Check if current drawdown exceeds the max drawdown protection.
        
        Returns:
            Dict with drawdown assessment and actions
        """
        try:
            # Calculate current equity curve to find peak and current values
            start_time = datetime.utcnow() - timedelta(days=30)  # Look back 30 days by default
            
            # Get transaction history to calculate equity curve
            transactions = self.get_transaction_history(
                from_time=start_time,
                count=1000
            )
            
            # Get initial balance (earliest in the period)
            account_details = self.get_account_details()
            current_balance = float(account_details.get('balance', 0.0))
            
            # Sort transactions by time
            sorted_txs = sorted(transactions, key=lambda x: x.get('time', ''))
            
            # Build equity curve
            equity_points = []
            running_balance = current_balance
            
            # Work backwards to reconstruct equity curve
            for tx in reversed(sorted_txs):
                tx_type = tx.get('type', '')
                
                if tx_type in ['ORDER_FILL', 'TRADE_CLOSE']:
                    # Subtract P/L to get previous balance
                    pl = float(tx.get('pl', 0.0))
                    financing = float(tx.get('financing', 0.0))
                    commission = float(tx.get('commission', 0.0))
                    
                    # Adjust balance (reverse of what happened)
                    running_balance -= (pl + financing + commission)
                
                elif tx_type == 'TRANSFER_FUNDS':
                    # Handle deposits/withdrawals
                    amount = float(tx.get('amount', 0.0))
                    running_balance -= amount
                
                # Save this point in the equity curve
                equity_points.append({
                    'time': tx.get('time', ''),
                    'balance': running_balance
                })
            
            # Add current balance as the last point
            equity_points.append({
                'time': datetime.utcnow().isoformat(),
                'balance': current_balance
            })
            
            # Find peak equity
            peak_equity = max(equity_points, key=lambda x: x['balance'])['balance']
            
            # Calculate drawdown
            current_drawdown = (peak_equity - current_balance) / peak_equity if peak_equity > 0 else 0
            
            # Determine if action needed
            action_needed = current_drawdown >= self.max_drawdown_percentage
            
            result = {
                "current_drawdown": current_drawdown,
                "max_allowable_drawdown": self.max_drawdown_percentage,
                "peak_equity": peak_equity,
                "current_equity": current_balance,
                "drawdown_exceeded": action_needed,
                "action_recommended": "REDUCE_POSITIONS" if action_needed else "NONE"
            }
            
            if action_needed:
                logger.warning(f"Drawdown protection triggered! Current drawdown: {current_drawdown:.2%}, Max allowed: {self.max_drawdown_percentage:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking drawdown protection: {str(e)}")
            return {
                "error": str(e),
                "drawdown_exceeded": False,
                "action_recommended": "ERROR"
            }
    
    def set_daily_loss_limit(self, amount: float) -> bool:
        """
        Set a daily loss limit.
        
        Args:
            amount: Maximum allowable daily loss (positive number)
            
        Returns:
            bool: True if successfully set, False otherwise
        """
        try:
            if amount <= 0:
                logger.error(f"Invalid daily loss limit: {amount}. Must be positive.")
                return False
            
            self.daily_loss_limit = amount
            logger.info(f"Daily loss limit set to {amount}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting daily loss limit: {str(e)}")
            return False
    
    def check_daily_loss_limit(self) -> Dict[str, Any]:
        """
        Check if the daily loss limit has been reached.
        
        Returns:
            Dict with daily loss assessment and actions
        """
        try:
            if self.daily_loss_limit is None:
                return {
                    "daily_loss_limit_set": False,
                    "message": "No daily loss limit has been set."
                }
            
            # Calculate today's P/L
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get today's transactions
            transactions = self.get_transaction_history(
                from_time=today,
                count=500
            )
            
            # Calculate realized P/L for today
            daily_pnl = 0.0
            
            for tx in transactions:
                tx_type = tx.get('type', '')
                
                if tx_type in ['ORDER_FILL', 'TRADE_CLOSE']:
                    pl = float(tx.get('pl', 0.0))
                    financing = float(tx.get('financing', 0.0))
                    commission = float(tx.get('commission', 0.0))
                    
                    daily_pnl += (pl + financing + commission)
            
            # Check if limit exceeded
            limit_exceeded = daily_pnl <= -self.daily_loss_limit
            
            result = {
                "daily_loss_limit_set": True,
                "daily_loss_limit": self.daily_loss_limit,
                "current_daily_pnl": daily_pnl,
                "limit_exceeded": limit_exceeded,
                "action_recommended": "STOP_TRADING" if limit_exceeded else "CONTINUE"
            }
            
            if limit_exceeded:
                logger.warning(f"Daily loss limit reached! Current P/L: {daily_pnl}, Limit: {self.daily_loss_limit}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {str(e)}")
            return {
                "daily_loss_limit_set": self.daily_loss_limit is not None,
                "error": str(e),
                "action_recommended": "ERROR"
            }
    
    def implement_circuit_breakers(self, volatility_threshold: float = 2.0,
                                consecutive_losses: int = 3) -> Dict[str, Any]:
        """
        Implement circuit breakers for extreme market conditions.
        
        Args:
            volatility_threshold: Volatility multiple above normal to trigger circuit breaker
            consecutive_losses: Number of consecutive losing trades to trigger circuit breaker
            
        Returns:
            Dict with circuit breaker assessment and actions
        """
        try:
            # Get recent transactions for detecting consecutive losses
            recent_txs = self.get_transaction_history(
                from_time=datetime.utcnow() - timedelta(days=1),
                count=100
            )
            
            # Extract recent trades and check for consecutive losses
            recent_trades = [tx for tx in recent_txs if tx.get('type') in ['ORDER_FILL', 'TRADE_CLOSE']]
            
            # Sort by time, most recent first
            recent_trades = sorted(recent_trades, key=lambda x: x.get('time', ''), reverse=True)
            
            # Count consecutive losses
            loss_count = 0
            for trade in recent_trades:
                pl = float(trade.get('pl', 0.0))
                
                if pl < 0:
                    loss_count += 1
                else:
                    break  # Stop counting at first profitable trade
                    
                if loss_count >= consecutive_losses:
                    break  # No need to count more
            
            # Check for high volatility in major pairs
            major_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD']
            high_volatility_pairs = []
            
            for pair in major_pairs:
                try:
                    # Get recent candles
                    url = f"{self.api_url}/v3/instruments/{pair}/candles"
                    params = {
                        "count": 20,
                        "granularity": "M5",
                        "price": "M"
                    }
                    
                    response = requests.get(url, headers=self.headers, params=params)
                    
                    if response.status_code == 200:
                        candles = response.json().get('candles', [])
                        
                        if candles:
                            # Calculate typical ATR over these periods
                            ranges = []
                            for candle in candles:
                                high = float(candle.get('mid', {}).get('h', 0))
                                low = float(candle.get('mid', {}).get('l', 0))
                                ranges.append(high - low)
                            
                            avg_range = sum(ranges) / len(ranges) if ranges else 0
                            
                            # Check the most recent candle
                            latest_candle = candles[-1]
                            latest_high = float(latest_candle.get('mid', {}).get('h', 0))
                            latest_low = float(latest_candle.get('mid', {}).get('l', 0))
                            latest_range = latest_high - latest_low
                            
                            # Check if current range exceeds threshold
                            if latest_range > avg_range * volatility_threshold:
                                high_volatility_pairs.append(pair)
                except Exception as e:
                    logger.error(f"Error checking volatility for {pair}: {str(e)}")
            
            # Determine if circuit breaker should be triggered
            trigger_consecutive_losses = loss_count >= consecutive_losses
            trigger_high_volatility = len(high_volatility_pairs) > 0
            
            circuit_breaker_triggered = trigger_consecutive_losses or trigger_high_volatility
            
            result = {
                "circuit_breaker_triggered": circuit_breaker_triggered,
                "consecutive_losses": {
                    "count": loss_count,
                    "threshold": consecutive_losses,
                    "triggered": trigger_consecutive_losses
                },
                "high_volatility": {
                    "pairs": high_volatility_pairs,
                    "threshold": volatility_threshold,
                    "triggered": trigger_high_volatility
                },
                "action_recommended": "SUSPEND_TRADING" if circuit_breaker_triggered else "CONTINUE"
            }
            
            if circuit_breaker_triggered:
                logger.warning(f"Circuit breaker triggered! Consecutive losses: {loss_count}, High volatility pairs: {high_volatility_pairs}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error implementing circuit breakers: {str(e)}")
            return {
                "circuit_breaker_triggered": False,
                "error": str(e),
                "action_recommended": "ERROR"
            }
    
    def send_margin_alerts(self, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Send alerts when margin reaches a threshold.
        
        Args:
            threshold: Margin usage threshold to trigger alert (0.5 = 50%)
            
        Returns:
            Dict with alert status and details
        """
        try:
            # Save alert threshold
            self.margin_alert_threshold = threshold
            
            # Check current margin usage
            margin_used = self.get_margin_used()
            margin_available = self.get_margin_available()
            
            # Calculate margin usage ratio
            total_margin = margin_used + margin_available
            margin_usage_ratio = margin_used / total_margin if total_margin > 0 else 0
            
            # Determine if alert should be triggered
            alert_triggered = margin_usage_ratio >= threshold
            
            # Format alert message
            alert_message = (
                f"⚠️ MARGIN ALERT ⚠️\n"
                f"Margin usage has reached {margin_usage_ratio:.2%}, threshold: {threshold:.2%}\n"
                f"Margin used: {margin_used:.2f}, Available: {margin_available:.2f}"
            )
            
            # Log the alert
            if alert_triggered:
                logger.warning(alert_message)
                
                # In a real implementation, would send email/SMS/notification here
                # For this implementation, we just log it
            
            result = {
                "alert_triggered": alert_triggered,
                "margin_usage_ratio": margin_usage_ratio,
                "threshold": threshold,
                "alert_message": alert_message if alert_triggered else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in margin alerts: {str(e)}")
            return {
                "alert_triggered": False,
                "error": str(e)
            }
    
    #--------------------------------------------------
    # 5. Reporting Methods
    #--------------------------------------------------
    
    def generate_account_statement(self, from_time: Optional[datetime] = None, 
                                 to_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive account statement.
        
        Args:
            from_time: Start time for the statement (defaults to 30 days ago)
            to_time: End time for the statement (defaults to now)
            
        Returns:
            Dict with account statement details and file path
        """
        try:
            # Set default time range if not provided
            if not from_time:
                from_time = datetime.utcnow() - timedelta(days=30)
            if not to_time:
                to_time = datetime.utcnow()
            
            # Format dates for filenames
            from_str = from_time.strftime("%Y%m%d")
            to_str = to_time.strftime("%Y%m%d")
            
            # Get account summary
            account_summary = self.get_account_summary()
            
            # Get transaction history
            transactions = self.get_transaction_history(
                from_time=from_time,
                to_time=to_time,
                count=1000
            )
            
            # Generate transaction report DataFrame
            tx_df = self.generate_transaction_report(timeframe="custom")
            
            # Calculate summary statistics
            starting_balance = None
            ending_balance = float(account_summary.get('balance', 0.0))
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            total_profit = 0.0
            total_loss = 0.0
            total_commissions = 0.0
            total_financing = 0.0
            
            # Process transactions to build statistics
            categorized_txs = self.categorize_transactions(transactions)
            
            # Handle ORDER_FILL and TRADE_CLOSE transactions
            trade_txs = categorized_txs.get('ORDER_FILL', []) + categorized_txs.get('TRADE_CLOSE', [])
            
            for tx in trade_txs:
                if 'pl' in tx:
                    pl = float(tx['pl'])
                    if pl > 0:
                        winning_trades += 1
                        total_profit += pl
                    elif pl < 0:
                        losing_trades += 1
                        total_loss += abs(pl)
                
                if 'commission' in tx:
                    total_commissions += float(tx['commission'])
                
                if 'financing' in tx:
                    total_financing += float(tx['financing'])
            
            total_trades = winning_trades + losing_trades
            
            # Try to determine starting balance from earliest transaction or balance history
            if transactions:
                # Sort by time
                sorted_txs = sorted(transactions, key=lambda x: x.get('time', ''))
                
                # Estimate starting balance by subtracting all transaction impacts
                starting_balance = ending_balance
                
                for tx in sorted_txs:
                    tx_type = tx.get('type', '')
                    
                    if tx_type in ['ORDER_FILL', 'TRADE_CLOSE']:
                        pl = float(tx.get('pl', 0.0)) if 'pl' in tx else 0.0
                        financing = float(tx.get('financing', 0.0)) if 'financing' in tx else 0.0
                        commission = float(tx.get('commission', 0.0)) if 'commission' in tx else 0.0
                        
                        starting_balance -= (pl + financing + commission)
                    
                    elif tx_type == 'TRANSFER_FUNDS':
                        amount = float(tx.get('amount', 0.0)) if 'amount' in tx else 0.0
                        starting_balance -= amount
            else:
                starting_balance = ending_balance
            
            # Generate equity curve
            equity_curve = []
            running_balance = starting_balance
            
            for tx in sorted_txs:
                tx_type = tx.get('type', '')
                tx_time = tx.get('time', '')
                
                if tx_type in ['ORDER_FILL', 'TRADE_CLOSE']:
                    pl = float(tx.get('pl', 0.0)) if 'pl' in tx else 0.0
                    financing = float(tx.get('financing', 0.0)) if 'financing' in tx else 0.0
                    commission = float(tx.get('commission', 0.0)) if 'commission' in tx else 0.0
                    
                    running_balance += (pl + financing + commission)
                
                elif tx_type == 'TRANSFER_FUNDS':
                    amount = float(tx.get('amount', 0.0)) if 'amount' in tx else 0.0
                    running_balance += amount
                
                equity_curve.append({
                    'time': tx_time,
                    'balance': running_balance
                })
            
            # Add ending balance point
            equity_curve.append({
                'time': to_time.isoformat(),
                'balance': ending_balance
            })
            
            # Generate plots
            # 1. Equity curve
            if equity_curve:
                try:
                    eq_df = pd.DataFrame(equity_curve)
                    eq_df['time'] = pd.to_datetime(eq_df['time'])
                    eq_df = eq_df.sort_values('time')
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(eq_df['time'], eq_df['balance'])
                    plt.title(f'Account Equity Curve ({from_str} to {to_str})')
                    plt.xlabel('Date')
                    plt.ylabel('Equity')
                    plt.grid(True)
                    plt.tight_layout()
                    
                    equity_plot_path = self.reports_dir / f"equity_curve_{from_str}_to_{to_str}.png"
                    plt.savefig(equity_plot_path)
                    plt.close()
                except Exception as plot_error:
                    logger.error(f"Error generating equity curve plot: {str(plot_error)}")
                    equity_plot_path = None
            else:
                equity_plot_path = None
            
            # 2. Winning vs Losing Trades
            if total_trades > 0:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.bar(['Winning', 'Losing'], [winning_trades, losing_trades])
                    plt.title(f'Trade Performance ({from_str} to {to_str})')
                    plt.ylabel('Number of Trades')
                    plt.grid(axis='y')
                    plt.tight_layout()
                    
                    trades_plot_path = self.reports_dir / f"trade_performance_{from_str}_to_{to_str}.png"
                    plt.savefig(trades_plot_path)
                    plt.close()
                except Exception as plot_error:
                    logger.error(f"Error generating trade performance plot: {str(plot_error)}")
                    trades_plot_path = None
            else:
                trades_plot_path = None
            
            # Create HTML report
            win_rate_str = f"{(winning_trades / total_trades) * 100:.2f}%" if total_trades > 0 else "N/A"
            
            html_content = f"""
            <html>
            <head>
                <title>Account Statement: {from_str} to {to_str}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333366; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Account Statement</h1>
                <p>Period: {from_time.strftime('%Y-%m-%d')} to {to_time.strftime('%Y-%m-%d')}</p>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <table>
                        <tr><td>Starting Balance:</td><td>${starting_balance:.2f}</td></tr>
                        <tr><td>Ending Balance:</td><td>${ending_balance:.2f}</td></tr>
                        <tr><td>Net Profit/Loss:</td><td class="{'positive' if ending_balance > starting_balance else 'negative'}">${ending_balance - starting_balance:.2f}</td></tr>
                        <tr><td>Return:</td><td class="{'positive' if ending_balance > starting_balance else 'negative'}">{((ending_balance / starting_balance) - 1) * 100:.2f}%</td></tr>
                    </table>
                </div>
                
                <h2>Trading Activity</h2>
                <table>
                    <tr><td>Total Trades:</td><td>{total_trades}</td></tr>
                    <tr><td>Winning Trades:</td><td>{winning_trades}</td></tr>
                    <tr><td>Losing Trades:</td><td>{losing_trades}</td></tr>
                    <tr><td>Win Rate:</td><td>{win_rate_str}</td></tr>
                    <tr><td>Total Profit:</td><td class="positive">${total_profit:.2f}</td></tr>
                    <tr><td>Total Loss:</td><td class="negative">${total_loss:.2f}</td></tr>
                    <tr><td>Profit Factor:</td><td>{total_profit / total_loss if total_loss > 0 else 'N/A'}</td></tr>
                    <tr><td>Total Commissions:</td><td class="negative">${total_commissions:.2f}</td></tr>
                    <tr><td>Total Financing:</td><td class="{'positive' if total_financing > 0 else 'negative'}">${total_financing:.2f}</td></tr>
                </table>
                
                <h2>Charts</h2>
                <div>
                    <h3>Equity Curve</h3>
                    <img src="{equity_plot_path.name if equity_plot_path else 'No data available'}" alt="Equity Curve" style="max-width: 100%;">
                </div>
                
                <div>
                    <h3>Trade Performance</h3>
                    <img src="{trades_plot_path.name if trades_plot_path else 'No data available'}" alt="Trade Performance" style="max-width: 100%;">
                </div>
                
                <h2>Transactions</h2>
                <p>Total transactions: {len(transactions)}</p>
                <p>For detailed transaction information, please refer to the CSV report.</p>
            </body>
            </html>
            """
            
            # Save HTML report
            html_path = self.reports_dir / f"account_statement_{from_str}_to_{to_str}.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            # Save transaction data as CSV
            if not tx_df.empty:
                csv_path = self.reports_dir / f"transactions_{from_str}_to_{to_str}.csv"
                tx_df.to_csv(csv_path, index=False)
            else:
                csv_path = None
            
            # Return summary and file paths
            result = {
                "success": True,
                "period": {
                    "from": from_time.isoformat(),
                    "to": to_time.isoformat()
                },
                "summary": {
                    "starting_balance": starting_balance,
                    "ending_balance": ending_balance,
                    "net_profit_loss": ending_balance - starting_balance,
                    "return_percentage": ((ending_balance / starting_balance) - 1) * 100 if starting_balance > 0 else 0.0,
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": (winning_trades / total_trades) if total_trades > 0 else 0.0,
                    "profit_factor": (total_profit / total_loss) if total_loss > 0 else float('inf')
                },
                "files": {
                    "html_report": str(html_path),
                    "csv_transactions": str(csv_path) if csv_path else None,
                    "equity_plot": str(equity_plot_path) if equity_plot_path else None,
                    "trades_plot": str(trades_plot_path) if trades_plot_path else None
                }
            }
            
            logger.info(f"Generated account statement for period {from_str} to {to_str}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating account statement: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_tax_report(self, year: int) -> Dict[str, Any]:
        """
        Generate a tax report for a specific year.
        
        Args:
            year: The year to generate the report for
            
        Returns:
            Dict with tax report details and file path
        """
        try:
            # Define time period for the year
            start_time = datetime(year, 1, 1)
            end_time = datetime(year, 12, 31, 23, 59, 59)
            
            # Get transactions for the year
            transactions = self.get_transaction_history(
                from_time=start_time,
                to_time=end_time,
                count=5000  # Increase for active accounts
            )
            
            if not transactions:
                logger.warning(f"No transactions found for tax year {year}")
                return {
                    "success": True,
                    "message": f"No transactions found for tax year {year}",
                    "taxable_events": 0
                }
            
            # Process transactions for tax reporting
            realized_gains = []
            financing_charges = []
            commissions = []
            
            for tx in transactions:
                tx_type = tx.get('type', '')
                tx_time = tx.get('time', '')
                
                if tx_type in ['ORDER_FILL', 'TRADE_CLOSE']:
                    # Handle P/L
                    if 'pl' in tx and float(tx['pl']) != 0:
                        realized_gains.append({
                            'time': tx_time,
                            'instrument': tx.get('instrument', ''),
                            'units': float(tx.get('units', 0)),
                            'price': float(tx.get('price', 0)) if 'price' in tx else 0,
                            'pl': float(tx['pl']),
                            'trade_id': tx.get('tradeId', ''),
                            'order_id': tx.get('orderId', '')
                        })
                    
                    # Handle financing
                    if 'financing' in tx and float(tx['financing']) != 0:
                        financing_charges.append({
                            'time': tx_time,
                            'instrument': tx.get('instrument', ''),
                            'amount': float(tx['financing']),
                            'trade_id': tx.get('tradeId', '')
                        })
                    
                    # Handle commissions
                    if 'commission' in tx and float(tx['commission']) != 0:
                        commissions.append({
                            'time': tx_time,
                            'instrument': tx.get('instrument', ''),
                            'amount': float(tx['commission']),
                            'trade_id': tx.get('tradeId', '')
                        })
            
            # Create DataFrames
            gains_df = pd.DataFrame(realized_gains) if realized_gains else pd.DataFrame()
            financing_df = pd.DataFrame(financing_charges) if financing_charges else pd.DataFrame()
            commissions_df = pd.DataFrame(commissions) if commissions else pd.DataFrame()
            
            # Convert time columns to datetime
            for df in [gains_df, financing_df, commissions_df]:
                if not df.empty and 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
            
            # Summary calculations
            total_realized_gains = gains_df['pl'].sum() if not gains_df.empty else 0.0
            total_financing = financing_df['amount'].sum() if not financing_df.empty else 0.0
            total_commissions = commissions_df['amount'].sum() if not commissions_df.empty else 0.0
            
            # Net taxable profit/loss
            net_taxable = total_realized_gains + total_financing - abs(total_commissions)
            
            # Create tax report CSV
            csv_path = self.reports_dir / f"tax_report_{year}.csv"
            
            # Combine all taxable events into one report
            if not gains_df.empty:
                gains_df.to_csv(csv_path, index=False)
                
                # Add financing and commissions as separate sections if they exist
                if not financing_df.empty or not commissions_df.empty:
                    with open(csv_path, 'a') as f:
                        f.write("\n\nFinancing Charges\n")
                    
                    if not financing_df.empty:
                        financing_df.to_csv(csv_path, mode='a', index=False)
                    
                    with open(csv_path, 'a') as f:
                        f.write("\n\nCommissions\n")
                    
                    if not commissions_df.empty:
                        commissions_df.to_csv(csv_path, mode='a', index=False)
            
            # Generate summary report
            summary_path = self.reports_dir / f"tax_summary_{year}.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Tax Year: {year}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Realized Gains/Losses: ${total_realized_gains:.2f}\n")
                f.write(f"Total Financing Charges: ${total_financing:.2f}\n")
                f.write(f"Total Commissions: ${abs(total_commissions):.2f}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Net Taxable Profit/Loss: ${net_taxable:.2f}\n\n")
                f.write("Note: This summary is for informational purposes only.\n")
                f.write("Please consult with a tax professional for accurate tax filing.\n")
            
            # Return summary and file paths
            result = {
                "success": True,
                "tax_year": year,
                "summary": {
                    "total_realized_gains": total_realized_gains,
                    "total_financing": total_financing,
                    "total_commissions": total_commissions,
                    "net_taxable": net_taxable
                },
                "taxable_events": len(realized_gains) + len(financing_charges) + len(commissions),
                "files": {
                    "csv_report": str(csv_path),
                    "summary_report": str(summary_path)
                }
            }
            
            logger.info(f"Generated tax report for {year} with {result['taxable_events']} taxable events")
            return result
            
        except Exception as e:
            logger.error(f"Error generating tax report: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def calculate_account_growth(self, timeframe: str = "YTD") -> Dict[str, Any]:
        """
        Calculate account growth over time.
        
        Args:
            timeframe: Time period for calculation ('1W', '1M', '3M', '6M', 'YTD', '1Y', 'ALL')
            
        Returns:
            Dict with account growth metrics
        """
        try:
            end_time = datetime.utcnow()
            
            # Determine start time based on timeframe
            if timeframe == "1W":
                start_time = end_time - timedelta(weeks=1)
            elif timeframe == "1M":
                start_time = end_time - timedelta(days=30)
            elif timeframe == "3M":
                start_time = end_time - timedelta(days=90)
            elif timeframe == "6M":
                start_time = end_time - timedelta(days=180)
            elif timeframe == "YTD":
                start_time = datetime(end_time.year, 1, 1)
            elif timeframe == "1Y":
                start_time = end_time - timedelta(days=365)
            elif timeframe == "ALL":
                start_time = datetime(2000, 1, 1)  # Far in the past
            else:
                logger.error(f"Invalid timeframe: {timeframe}")
                return {
                    "success": False,
                    "error": f"Invalid timeframe: {timeframe}"
                }
            
            # Get transactions for the period
            transactions = self.get_transaction_history(
                from_time=start_time,
                to_time=end_time,
                count=2000  # Adjust as needed
            )
            
            if not transactions:
                logger.warning(f"No transactions found for timeframe {timeframe}")
                return {
                    "success": True,
                    "message": f"No transactions found for timeframe {timeframe}",
                    "growth_metrics": {}
                }
            
            # Get current account summary
            account_summary = self.get_account_summary()
            current_balance = float(account_summary.get('balance', 0.0))
            
            # Sort transactions by time
            sorted_txs = sorted(transactions, key=lambda x: x.get('time', ''))
            
            # Reconstruct equity curve
            equity_points = []
            
            # Start with current balance and work backwards
            running_balance = current_balance
            
            # Process transactions in reverse chronological order
            for tx in reversed(sorted_txs):
                tx_type = tx.get('type', '')
                tx_time = tx.get('time', '')
                
                if tx_type in ['ORDER_FILL', 'TRADE_CLOSE']:
                    # Subtract P/L (since we're going backwards)
                    pl = float(tx.get('pl', 0.0)) if 'pl' in tx else 0.0
                    financing = float(tx.get('financing', 0.0)) if 'financing' in tx else 0.0
                    commission = float(tx.get('commission', 0.0)) if 'commission' in tx else 0.0
                    
                    running_balance -= (pl + financing + commission)
                
                elif tx_type == 'TRANSFER_FUNDS':
                    # Subtract funds transfers (since we're going backwards)
                    amount = float(tx.get('amount', 0.0)) if 'amount' in tx else 0.0
                    running_balance -= amount
                
                # Add point to equity curve
                equity_points.append({
                    'time': tx_time,
                    'balance': running_balance
                })
            
            # Add current balance as the first point (most recent)
            equity_points.insert(0, {
                'time': end_time.isoformat(),
                'balance': current_balance
            })
            
            # Create DataFrame
            eq_df = pd.DataFrame(equity_points)
            eq_df['time'] = pd.to_datetime(eq_df['time'])
            eq_df = eq_df.sort_values('time')
            
            # Get initial and final balances
            initial_balance = eq_df['balance'].iloc[0]
            final_balance = eq_df['balance'].iloc[-1]
            
            # Calculate growth metrics
            absolute_growth = final_balance - initial_balance
            percentage_growth = (final_balance / initial_balance - 1) * 100 if initial_balance > 0 else 0.0
            
            # Calculate CAGR if timeframe is long enough
            days_in_period = (end_time - start_time).days
            
            if days_in_period > 30:  # Only calculate for periods > 30 days
                years = days_in_period / 365.0
                cagr = ((final_balance / initial_balance) ** (1 / years) - 1) * 100 if initial_balance > 0 and years > 0 else 0.0
            else:
                cagr = None
            
            # Calculate drawdown
            eq_df['peak'] = eq_df['balance'].cummax()
            eq_df['drawdown'] = (eq_df['balance'] - eq_df['peak']) / eq_df['peak'] * 100
            max_drawdown = abs(eq_df['drawdown'].min())
            
            # Calculate volatility (standard deviation of daily returns)
            if len(eq_df) > 1:
                # Resample to daily and calculate returns
                eq_df.set_index('time', inplace=True)
                daily_eq = eq_df['balance'].resample('D').last().dropna()
                
                if len(daily_eq) > 1:
                    daily_returns = daily_eq.pct_change().dropna()
                    volatility = daily_returns.std() * 100  # As percentage
                else:
                    volatility = 0.0
            else:
                volatility = 0.0
            
            # Prepare result
            result = {
                "success": True,
                "timeframe": timeframe,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "days": days_in_period
                },
                "growth_metrics": {
                    "initial_balance": initial_balance,
                    "final_balance": final_balance,
                    "absolute_growth": absolute_growth,
                    "percentage_growth": percentage_growth,
                    "cagr": cagr,
                    "max_drawdown": max_drawdown,
                    "volatility": volatility
                }
            }
            
            logger.info(f"Calculated account growth for {timeframe}: {percentage_growth:.2f}% growth")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating account growth: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def visualize_account_performance(self, timeframe: str = "YTD") -> Dict[str, Any]:
        """
        Create visualizations of account performance.
        
        Args:
            timeframe: Time period for visualization ('1W', '1M', '3M', '6M', 'YTD', '1Y', 'ALL')
            
        Returns:
            Dict with paths to generated visualizations
        """
        try:
            # Calculate account growth to get the data
            growth_data = self.calculate_account_growth(timeframe)
            
            if not growth_data.get('success', False):
                return {
                    "success": False,
                    "error": growth_data.get('error', 'Failed to calculate account growth')
                }
            
            # Extract time period
            start_time_str = growth_data['period']['start']
            end_time_str = growth_data['period']['end']
            
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            
            # Get transactions for the period
            transactions = self.get_transaction_history(
                from_time=start_time,
                to_time=end_time,
                count=2000
            )
            
            if not transactions:
                logger.warning(f"No transactions found for timeframe {timeframe}")
                return {
                    "success": True,
                    "message": f"No transactions found for timeframe {timeframe}",
                    "visualizations": {}
                }
            
            # Sort transactions by time
            sorted_txs = sorted(transactions, key=lambda x: x.get('time', ''))
            
            # Get current account balance
            account_summary = self.get_account_summary()
            current_balance = float(account_summary.get('balance', 0.0))
            
            # Reconstruct equity curve
            equity_points = []
            running_balance = current_balance
            
            # Work backwards from current balance
            for tx in reversed(sorted_txs):
                tx_type = tx.get('type', '')
                tx_time = tx.get('time', '')
                
                if tx_type in ['ORDER_FILL', 'TRADE_CLOSE']:
                    pl = float(tx.get('pl', 0.0)) if 'pl' in tx else 0.0
                    financing = float(tx.get('financing', 0.0)) if 'financing' in tx else 0.0
                    commission = float(tx.get('commission', 0.0)) if 'commission' in tx else 0.0
                    
                    running_balance -= (pl + financing + commission)
                
                elif tx_type == 'TRANSFER_FUNDS':
                    amount = float(tx.get('amount', 0.0)) if 'amount' in tx else 0.0
                    running_balance -= amount
                
                equity_points.append({
                    'time': tx_time,
                    'balance': running_balance
                })
            
            # Add current balance
            equity_points.insert(0, {
                'time': end_time.isoformat(),
                'balance': current_balance
            })
            
            # Create DataFrame
            eq_df = pd.DataFrame(equity_points)
            eq_df['time'] = pd.to_datetime(eq_df['time'])
            eq_df = eq_df.sort_values('time')
            
            # Generate file prefix
            time_str = datetime.now().strftime('%Y%m%d')
            prefix = f"{timeframe}_{time_str}"
            
            # List to store paths of generated visualizations
            visualization_paths = {}
            
            # 1. Equity Curve
            plt.figure(figsize=(12, 6))
            plt.plot(eq_df['time'], eq_df['balance'])
            plt.title(f'Account Equity Curve ({timeframe})')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.tight_layout()
            
            equity_plot_path = self.reports_dir / f"equity_curve_{prefix}.png"
            plt.savefig(equity_plot_path)
            plt.close()
            
            visualization_paths['equity_curve'] = str(equity_plot_path)
            
            # 2. Drawdown Chart
            eq_df['peak'] = eq_df['balance'].cummax()
            eq_df['drawdown'] = (eq_df['balance'] - eq_df['peak']) / eq_df['peak'] * 100
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(eq_df['time'], eq_df['drawdown'], 0, color='red', alpha=0.3)
            plt.title(f'Account Drawdown ({timeframe})')
            plt.xlabel('Date')
            plt.ylabel('Drawdown %')
            plt.grid(True)
            plt.tight_layout()
            
            drawdown_plot_path = self.reports_dir / f"drawdown_{prefix}.png"
            plt.savefig(drawdown_plot_path)
            plt.close()
            
            visualization_paths['drawdown'] = str(drawdown_plot_path)
            
            # 3. Trade Distribution by Instrument
            trade_txs = [tx for tx in transactions if tx.get('type') in ['ORDER_FILL', 'TRADE_CLOSE']]
            
            if trade_txs:
                # Count trades by instrument
                instruments = {}
                for tx in trade_txs:
                    instrument = tx.get('instrument', 'Unknown')
                    if instrument not in instruments:
                        instruments[instrument] = 0
                    
                    instruments[instrument] += 1
                
                # Create pie chart
                plt.figure(figsize=(10, 8))
                plt.pie(instruments.values(), labels=instruments.keys(), autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                plt.title(f'Trade Distribution by Instrument ({timeframe})')
                
                instrument_plot_path = self.reports_dir / f"instrument_distribution_{prefix}.png"
                plt.savefig(instrument_plot_path)
                plt.close()
                
                visualization_paths['instrument_distribution'] = str(instrument_plot_path)
            
            # 4. Profit/Loss by Month
            if len(eq_df) > 30:  # Only if we have enough data
                # Add month column
                eq_df['month'] = eq_df['time'].dt.strftime('%Y-%m')
                
                # Get monthly start and end values
                monthly_data = []
                
                for month, group in eq_df.groupby('month'):
                    start_value = group['balance'].iloc[0]
                    end_value = group['balance'].iloc[-1]
                    monthly_change = end_value - start_value
                    
                    monthly_data.append({
                        'month': month,
                        'change': monthly_change
                    })
                
                # Create monthly change DataFrame
                monthly_df = pd.DataFrame(monthly_data)
                
                # Plot monthly changes
                plt.figure(figsize=(12, 6))
                colors = ['green' if x >= 0 else 'red' for x in monthly_df['change']]
                plt.bar(monthly_df['month'], monthly_df['change'], color=colors)
                plt.title(f'Monthly Profit/Loss ({timeframe})')
                plt.xlabel('Month')
                plt.ylabel('Profit/Loss')
                plt.xticks(rotation=45)
                plt.grid(axis='y')
                plt.tight_layout()
                
                monthly_plot_path = self.reports_dir / f"monthly_pnl_{prefix}.png"
                plt.savefig(monthly_plot_path)
                plt.close()
                
                visualization_paths['monthly_pnl'] = str(monthly_plot_path)
            
            # Return results
            result = {
                "success": True,
                "timeframe": timeframe,
                "visualizations": visualization_paths
            }
            
            logger.info(f"Generated {len(visualization_paths)} visualizations for timeframe {timeframe}")
            return result
            
        except Exception as e:
            logger.error(f"Error visualizing account performance: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_account_data(self, filepath: str) -> Dict[str, Any]:
        """
        Export account data to a file.
        
        Args:
            filepath: Path where the data should be saved
            
        Returns:
            Dict with export status and details
        """
        try:
            export_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            # Default filename if only directory provided
            if os.path.isdir(filepath):
                filepath = os.path.join(filepath, f"account_data_export_{export_time}.json")
            
            # Get account details
            account_details = self.get_account_details()
            
            # Get account summary
            account_summary = self.get_account_summary()
            
            # Get open positions
            url = f"{self.api_url}/v3/accounts/{self.account_id}/openPositions"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get open positions: {response.status_code} - {response.text}")
                open_positions = []
            else:
                open_positions = response.json().get('positions', [])
            
            # Get open orders
            url = f"{self.api_url}/v3/accounts/{self.account_id}/orders"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get open orders: {response.status_code} - {response.text}")
                open_orders = []
            else:
                open_orders = response.json().get('orders', [])
            
            # Get recent transactions (last 7 days)
            start_time = datetime.utcnow() - timedelta(days=7)
            recent_transactions = self.get_transaction_history(
                from_time=start_time,
                count=500
            )
            
            # Compile export data
            export_data = {
                "export_info": {
                    "time": export_time,
                    "account_id": self.account_id
                },
                "account_details": account_details,
                "account_summary": account_summary,
                "open_positions": open_positions,
                "open_orders": open_orders,
                "recent_transactions": recent_transactions
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported account data to {filepath}")
            
            return {
                "success": True,
                "filepath": filepath,
                "export_time": export_time,
                "data_sections": list(export_data.keys())
            }
            
        except Exception as e:
            logger.error(f"Error exporting account data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 