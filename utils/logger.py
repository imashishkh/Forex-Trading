#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging utility for the Forex Trading Platform
"""

import os
import sys
import logging
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for colored console output
    """
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            record.msg = f"{self.COLORS[levelname]}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

def get_logger(name, log_level=None, log_file=None):
    """
    Get a logger instance with specified settings
    
    Args:
        name (str): Name of the logger
        log_level (str, optional): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file (str, optional): Path to log file. If None, only console logging is used.
    
    Returns:
        logging.Logger: Logger instance
    """
    # Convert string log level to logging level
    if log_level is None:
        level = logging.INFO
    else:
        level = getattr(logging, log_level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_format = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Add timestamp to log file
        if not log_file.endswith('.log'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{log_file}_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def setup_trading_logger(component_name, log_dir=None):
    """
    Set up a logger specifically for trading components
    
    Args:
        component_name (str): Name of the trading component
        log_dir (str, optional): Directory to store log files. If None, uses default location.
    
    Returns:
        logging.Logger: Logger instance
    """
    # Determine log directory
    if log_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, 'logs')
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log file path
    log_file = os.path.join(log_dir, f"{component_name}.log")
    
    # Get logger
    logger = get_logger(component_name, log_level="INFO", log_file=log_file)
    
    return logger

def log_trade(logger, trade_info, status='EXECUTED'):
    """
    Log trade information in a standardized format
    
    Args:
        logger (logging.Logger): Logger instance
        trade_info (dict): Trade information
        status (str): Trade status (EXECUTED, CLOSED, REJECTED, etc.)
    """
    symbol = trade_info.get('symbol', 'UNKNOWN')
    direction = trade_info.get('direction', 'UNKNOWN')
    price = trade_info.get('price', 0)
    size = trade_info.get('size', 0)
    
    if status == 'EXECUTED':
        logger.info(f"TRADE {status}: {direction.upper()} {symbol} @ {price:.5f} (Size: {size:.2f})")
    elif status == 'CLOSED':
        pnl = trade_info.get('pnl', 0)
        pnl_pct = trade_info.get('pnl_percentage', 0)
        reason = trade_info.get('exit_reason', 'UNKNOWN')
        logger.info(f"TRADE {status}: {direction.upper()} {symbol} @ {price:.5f} - P&L: {pnl:.2f} ({pnl_pct:.2f}%) - Reason: {reason}")
    elif status == 'REJECTED':
        reason = trade_info.get('reason', 'UNKNOWN')
        logger.warning(f"TRADE {status}: {direction.upper()} {symbol} @ {price:.5f} - Reason: {reason}")
    else:
        logger.info(f"TRADE {status}: {direction.upper()} {symbol} @ {price:.5f} (Size: {size:.2f})")

def log_portfolio_summary(logger, portfolio_summary):
    """
    Log portfolio summary information in a standardized format
    
    Args:
        logger (logging.Logger): Logger instance
        portfolio_summary (dict): Portfolio summary information
    """
    equity = portfolio_summary.get('equity', 0)
    balance = portfolio_summary.get('account_balance', 0)
    open_positions = portfolio_summary.get('open_positions_count', 0)
    unrealized_pnl = portfolio_summary.get('unrealized_pnl', 0)
    realized_pnl = portfolio_summary.get('realized_pnl', 0)
    margin_level = portfolio_summary.get('margin_level', 0)
    
    logger.info(f"PORTFOLIO SUMMARY: Equity: {equity:.2f} | Balance: {balance:.2f} | "
                f"Open Positions: {open_positions} | Unrealized P&L: {unrealized_pnl:.2f} | "
                f"Realized P&L: {realized_pnl:.2f} | Margin Level: {margin_level:.2f}%") 