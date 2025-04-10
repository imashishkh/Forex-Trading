#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Financial metrics utilities for the Forex Trading Platform
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a series of returns
    
    Args:
        returns (pd.Series): Series of percentage returns
        risk_free_rate (float): Risk-free rate (default: 0.0)
        periods_per_year (int): Number of periods in a year (252 for daily, 12 for monthly, etc.)
    
    Returns:
        float: Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert risk-free rate to same periodicity as returns
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate annualized return and volatility
    mean_excess_return = excess_returns.mean()
    volatility = returns.std()
    
    if volatility == 0:
        return 0.0
    
    # Sharpe ratio
    sharpe = mean_excess_return / volatility
    
    # Annualize
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)
    
    return sharpe_annualized


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino ratio for a series of returns
    
    Args:
        returns (pd.Series): Series of percentage returns
        risk_free_rate (float): Risk-free rate (default: 0.0)
        periods_per_year (int): Number of periods in a year (252 for daily, 12 for monthly, etc.)
    
    Returns:
        float: Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert risk-free rate to same periodicity as returns
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate annualized return and downside deviation
    mean_excess_return = excess_returns.mean()
    
    # Calculate downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return float('inf')  # No negative returns
    
    downside_deviation = negative_returns.std()
    
    if downside_deviation == 0:
        return 0.0
    
    # Sortino ratio
    sortino = mean_excess_return / downside_deviation
    
    # Annualize
    sortino_annualized = sortino * np.sqrt(periods_per_year)
    
    return sortino_annualized


def calculate_drawdown(equity_curve: pd.Series) -> Dict[str, Union[float, int, pd.Timestamp]]:
    """
    Calculate the maximum drawdown for an equity curve
    
    Args:
        equity_curve (pd.Series): Series of equity values
    
    Returns:
        dict: Dictionary with drawdown metrics
    """
    if len(equity_curve) < 2:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'peak': equity_curve.iloc[0] if len(equity_curve) > 0 else 0,
            'trough': equity_curve.iloc[0] if len(equity_curve) > 0 else 0,
            'peak_date': equity_curve.index[0] if len(equity_curve) > 0 else None,
            'trough_date': equity_curve.index[0] if len(equity_curve) > 0 else None,
            'recovery_date': equity_curve.index[0] if len(equity_curve) > 0 else None,
            'drawdown_length': 0,
            'recovery_length': 0
        }
    
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown in currency value
    drawdown = running_max - equity_curve
    
    # Calculate drawdown in percentage
    drawdown_pct = drawdown / running_max
    
    # Find maximum drawdown
    max_drawdown = drawdown.max()
    max_drawdown_pct = drawdown_pct.max()
    
    # Find peak and trough dates
    peak_idx = drawdown[drawdown == max_drawdown].index[0]
    peak_date = running_max[peak_idx:].idxmax()
    trough_date = equity_curve[peak_date:].idxmin()
    
    # Find recovery date (might not have recovered yet)
    recovery_mask = (equity_curve[trough_date:] >= equity_curve[peak_date])
    recovery_date = recovery_mask.idxmax() if recovery_mask.any() else None
    
    # Calculate drawdown length in periods
    try:
        drawdown_length = (trough_date - peak_date).days
    except:
        drawdown_length = 0
    
    # Calculate recovery length in periods
    try:
        recovery_length = (recovery_date - trough_date).days if recovery_date else None
    except:
        recovery_length = None
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'peak': equity_curve[peak_date],
        'trough': equity_curve[trough_date],
        'peak_date': peak_date,
        'trough_date': trough_date,
        'recovery_date': recovery_date,
        'drawdown_length': drawdown_length,
        'recovery_length': recovery_length
    }


def calculate_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR)
    
    Args:
        equity_curve (pd.Series): Series of equity values
        periods_per_year (int): Number of periods in a year (252 for daily, 12 for monthly, etc.)
    
    Returns:
        float: CAGR as a percentage
    """
    if len(equity_curve) < 2:
        return 0.0
    
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    
    # Number of years
    try:
        num_periods = len(equity_curve)
        num_years = num_periods / periods_per_year
    except:
        # If can't determine periods from index, use start and end date
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        num_years = (end_date - start_date).days / 365.25
    
    if num_years == 0 or start_value == 0:
        return 0.0
    
    # Calculate CAGR
    cagr = (end_value / start_value) ** (1 / num_years) - 1
    
    return cagr * 100  # Convert to percentage


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate the win rate from a list of trades
    
    Args:
        trades (list): List of trade dictionaries with 'pnl' or 'profit' keys
    
    Returns:
        float: Win rate as a percentage
    """
    if not trades:
        return 0.0
    
    # Count profitable trades
    profitable_trades = 0
    
    for trade in trades:
        # Check if 'pnl' or 'profit' key exists
        if 'pnl' in trade:
            profit = trade['pnl']
        elif 'profit' in trade:
            profit = trade['profit']
        else:
            continue
        
        if profit > 0:
            profitable_trades += 1
    
    # Calculate win rate
    win_rate = profitable_trades / len(trades) * 100
    
    return win_rate


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate the profit factor from a list of trades
    
    Args:
        trades (list): List of trade dictionaries with 'pnl' or 'profit' keys
    
    Returns:
        float: Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = 0
    gross_loss = 0
    
    for trade in trades:
        # Check if 'pnl' or 'profit' key exists
        if 'pnl' in trade:
            profit = trade['pnl']
        elif 'profit' in trade:
            profit = trade['profit']
        else:
            continue
        
        if profit > 0:
            gross_profit += profit
        else:
            gross_loss += abs(profit)
    
    # Calculate profit factor
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    profit_factor = gross_profit / gross_loss
    
    return profit_factor


def calculate_expectancy(trades: List[Dict]) -> float:
    """
    Calculate the expectancy from a list of trades
    
    Args:
        trades (list): List of trade dictionaries with 'pnl' or 'profit' keys
    
    Returns:
        float: Expectancy
    """
    if not trades:
        return 0.0
    
    total_profit = 0
    
    for trade in trades:
        # Check if 'pnl' or 'profit' key exists
        if 'pnl' in trade:
            profit = trade['pnl']
        elif 'profit' in trade:
            profit = trade['profit']
        else:
            continue
        
        total_profit += profit
    
    # Calculate expectancy
    expectancy = total_profit / len(trades)
    
    return expectancy


def calculate_average_trade(trades: List[Dict]) -> Dict[str, float]:
    """
    Calculate average trade metrics from a list of trades
    
    Args:
        trades (list): List of trade dictionaries with 'pnl' or 'profit' keys
    
    Returns:
        dict: Dictionary with average trade metrics
    """
    if not trades:
        return {
            'average_profit': 0.0,
            'average_loss': 0.0,
            'average_trade': 0.0,
            'largest_profit': 0.0,
            'largest_loss': 0.0
        }
    
    profits = []
    losses = []
    
    for trade in trades:
        # Check if 'pnl' or 'profit' key exists
        if 'pnl' in trade:
            profit = trade['pnl']
        elif 'profit' in trade:
            profit = trade['profit']
        else:
            continue
        
        if profit > 0:
            profits.append(profit)
        else:
            losses.append(profit)
    
    # Calculate metrics
    average_profit = np.mean(profits) if profits else 0.0
    average_loss = np.mean(losses) if losses else 0.0
    average_trade = (sum(profits) + sum(losses)) / len(trades)
    largest_profit = max(profits) if profits else 0.0
    largest_loss = min(losses) if losses else 0.0
    
    return {
        'average_profit': average_profit,
        'average_loss': average_loss,
        'average_trade': average_trade,
        'largest_profit': largest_profit,
        'largest_loss': largest_loss
    }


def calculate_risk_reward_ratio(trades: List[Dict]) -> float:
    """
    Calculate the average risk-reward ratio from a list of trades
    
    Args:
        trades (list): List of trade dictionaries with 'exit_price', 'entry_price', 'stop_loss' keys
    
    Returns:
        float: Average risk-reward ratio
    """
    if not trades:
        return 0.0
    
    ratios = []
    
    for trade in trades:
        # Skip trades without required keys
        if not all(k in trade for k in ['entry_price', 'stop_loss']):
            continue
        
        # Calculate reward
        if 'take_profit' in trade:
            reward = abs(trade['take_profit'] - trade['entry_price'])
        elif 'exit_price' in trade and trade.get('pnl', 0) > 0:
            reward = abs(trade['exit_price'] - trade['entry_price'])
        else:
            continue
        
        # Calculate risk
        risk = abs(trade['entry_price'] - trade['stop_loss'])
        
        if risk == 0:
            continue
        
        # Calculate ratio
        ratio = reward / risk
        ratios.append(ratio)
    
    # Calculate average ratio
    if not ratios:
        return 0.0
    
    average_ratio = np.mean(ratios)
    
    return average_ratio


def calculate_volatility(equity_curve: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate volatility metrics for an equity curve
    
    Args:
        equity_curve (pd.Series): Series of equity values
        periods_per_year (int): Number of periods in a year (252 for daily, 12 for monthly, etc.)
    
    Returns:
        dict: Dictionary with volatility metrics
    """
    if len(equity_curve) < 2:
        return {
            'daily_volatility': 0.0,
            'annualized_volatility': 0.0
        }
    
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Calculate daily volatility
    daily_volatility = returns.std()
    
    # Calculate annualized volatility
    annualized_volatility = daily_volatility * np.sqrt(periods_per_year)
    
    return {
        'daily_volatility': daily_volatility,
        'annualized_volatility': annualized_volatility
    }


def calculate_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate the Value at Risk (VaR) for a series of returns
    
    Args:
        returns (pd.Series): Series of percentage returns
        confidence_level (float): Confidence level for VaR (default: 0.95)
    
    Returns:
        float: Value at Risk
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate VaR
    var = np.percentile(returns, 100 * (1 - confidence_level))
    
    return -var  # Return as a positive number


def calculate_calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar ratio for an equity curve
    
    Args:
        equity_curve (pd.Series): Series of equity values
        periods_per_year (int): Number of periods in a year (252 for daily, 12 for monthly, etc.)
    
    Returns:
        float: Calmar ratio
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate CAGR
    cagr = calculate_cagr(equity_curve, periods_per_year) / 100  # Convert from percentage
    
    # Calculate maximum drawdown
    drawdown_info = calculate_drawdown(equity_curve)
    max_drawdown_pct = drawdown_info['max_drawdown_pct']
    
    if max_drawdown_pct == 0:
        return float('inf') if cagr > 0 else 0.0
    
    # Calculate Calmar ratio
    calmar_ratio = cagr / max_drawdown_pct
    
    return calmar_ratio


def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate the Omega ratio for a series of returns
    
    Args:
        returns (pd.Series): Series of percentage returns
        threshold (float): Threshold for considering returns as gains or losses (default: 0.0)
    
    Returns:
        float: Omega ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate gains and losses
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    
    sum_gains = gains.sum()
    sum_losses = losses.sum()
    
    if sum_losses == 0:
        return float('inf') if sum_gains > 0 else 0.0
    
    # Calculate Omega ratio
    omega_ratio = sum_gains / sum_losses
    
    return omega_ratio


def generate_performance_report(equity_curve: pd.Series, trades: List[Dict], periods_per_year: int = 252) -> Dict:
    """
    Generate a comprehensive performance report for a trading strategy
    
    Args:
        equity_curve (pd.Series): Series of equity values
        trades (list): List of trade dictionaries
        periods_per_year (int): Number of periods in a year (252 for daily, 12 for monthly, etc.)
    
    Returns:
        dict: Dictionary with performance metrics
    """
    if len(equity_curve) < 2:
        return {
            'total_return': 0.0,
            'cagr': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_trade': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'largest_profit': 0.0,
            'largest_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'value_at_risk': 0.0,
            'omega_ratio': 0.0
        }
    
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Calculate total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    
    # Calculate volatility
    volatility_info = calculate_volatility(equity_curve, periods_per_year)
    
    # Calculate drawdown
    drawdown_info = calculate_drawdown(equity_curve)
    
    # Calculate trade metrics
    trade_metrics = calculate_average_trade(trades)
    
    # Generate report
    report = {
        'total_return': total_return,
        'cagr': calculate_cagr(equity_curve, periods_per_year),
        'volatility': volatility_info['annualized_volatility'] * 100,  # Convert to percentage
        'sharpe_ratio': calculate_sharpe_ratio(returns, 0.0, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, 0.0, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(equity_curve, periods_per_year),
        'max_drawdown': drawdown_info['max_drawdown'],
        'max_drawdown_pct': drawdown_info['max_drawdown_pct'] * 100,  # Convert to percentage
        'win_rate': calculate_win_rate(trades),
        'profit_factor': calculate_profit_factor(trades),
        'expectancy': calculate_expectancy(trades),
        'avg_trade': trade_metrics['average_trade'],
        'avg_profit': trade_metrics['average_profit'],
        'avg_loss': trade_metrics['average_loss'],
        'largest_profit': trade_metrics['largest_profit'],
        'largest_loss': trade_metrics['largest_loss'],
        'risk_reward_ratio': calculate_risk_reward_ratio(trades),
        'value_at_risk': calculate_value_at_risk(returns) * 100,  # Convert to percentage
        'omega_ratio': calculate_omega_ratio(returns)
    }
    
    return report 