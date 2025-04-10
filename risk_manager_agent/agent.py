#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Risk Manager Agent implementation for the Forex Trading Platform
"""

import os
import json
import abc
import logging
import traceback
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta

from utils.base_agent import BaseAgent
from utils.logger import get_logger
from utils.metrics import calculate_drawdown, calculate_sharpe_ratio

class RiskManagerAgent(BaseAgent):
    """
    Agent responsible for risk assessment and position sizing.
    
    Handles risk management tasks for the forex trading platform including:
    - Position sizing calculations
    - Risk assessment for trades and portfolios
    - Risk controls and limits
    - Trade approval and modification
    - Risk reporting and monitoring
    """
    
    def __init__(
        self,
        agent_name: str,
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Risk Manager Agent
        
        Args:
            agent_name: A unique identifier for the agent
            llm: Language model to use (defaults to OpenAI if None)
            config: Configuration parameters for the agent
            logger: Logger instance for the agent
        """
        super().__init__(agent_name, llm, config, logger)
        
        # Extract configuration
        self.config = config or {}
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2% of account
        self.max_open_positions = self.config.get('max_open_positions', 5)
        self.max_risk_per_currency = self.config.get('max_risk_per_currency', 0.05)  # 5% of account
        self.max_daily_drawdown = self.config.get('max_daily_drawdown', 0.05)  # 5% of account
        self.max_portfolio_var = self.config.get('max_portfolio_var', 0.03)  # 3% of account
        self.stop_loss_atr_multiplier = self.config.get('stop_loss_atr_multiplier', 2.0)
        
        # Initialize risk metrics
        self.risk_metrics = {
            'portfolio_var': 0.0,
            'expected_shortfall': 0.0,
            'portfolio_correlation': 0.0,
            'current_drawdown': 0.0,
        }
        
        # Initialize positions tracker
        self.positions_file = os.path.join(self._get_data_dir(), 'positions.json')
        if os.path.exists(self.positions_file):
            with open(self.positions_file, 'r') as f:
                self.positions = json.load(f)
        else:
            self.positions = {
                'open_positions': [],
                'closed_positions': [],
                'account_history': []
            }
        
        self.log_action("init", f"Initialized Risk Manager Agent")
        
    def _get_data_dir(self) -> str:
        """Get the data directory path, creating it if it doesn't exist"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'risk_manager')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir
        
    def initialize(self) -> bool:
        """
        Set up the agent and its resources.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Set up any additional resources needed
            self.status = "ready"
            self.state["status"] = "ready"
            self.log_action("initialize", "Risk Manager Agent successfully initialized")
            return True
        except Exception as e:
            self.handle_error(e)
            return False

    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        This method routes the task to the appropriate risk management function
        based on the task type.
        
        Args:
            task: Task description and parameters

        Returns:
            Dict[str, Any]: Task execution results
        """
        self.log_action("run_task", f"Running task: {task.get('type', 'Unknown')}")
        
        # Update state
        self.state["tasks"].append(task)
        self.last_active = datetime.now()
        self.state["last_active"] = self.last_active
        
        try:
            task_type = task.get('type', '')
            params = task.get('parameters', {})
            
            if task_type == 'calculate_position_size':
                result = self.calculate_position_size(
                    params.get('account_balance', 0),
                    params.get('risk_percentage', 0.01),
                    params.get('stop_loss_pips', 0),
                    params.get('currency_pair', '')
                )
            elif task_type == 'calculate_portfolio_var':
                result = self.calculate_portfolio_var(
                    params.get('positions', []),
                    params.get('confidence_level', 0.95)
                )
            elif task_type == 'approve_trade':
                result = self.approve_trade(params.get('trade_parameters', {}))
            elif task_type == 'generate_risk_report':
                result = self.generate_risk_report(params.get('positions', []))
            # Add more task types as needed
            else:
                result = {
                    'status': 'error',
                    'error': f'Unknown task type: {task_type}'
                }
            
            return {
                'status': 'success',
                'task_type': task_type,
                'result': result
            }
            
        except Exception as e:
            self.handle_error(e)
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def assess_risk(self, market_data, technical_signals, fundamental_insights, sentiment_insights):
        """
        Assess risk for potential trades based on multiple inputs
        
        Args:
            market_data (dict): Market data by symbol
            technical_signals (dict): Technical analysis signals
            fundamental_insights (dict): Fundamental analysis insights
            sentiment_insights (dict): Sentiment analysis insights
        
        Returns:
            dict: Risk assessment results by symbol
        """
        self.log_action("assess_risk", "Assessing risk for potential trades")
        
        results = {}
        
        # Process each symbol
        for symbol in market_data:
            # If symbol is a dictionary with data for multiple symbols, process each
            if isinstance(market_data[symbol], dict) and 'open' not in market_data[symbol]:
                symbol_results = {}
                for sub_symbol in market_data[symbol]:
                    # Get data for this specific sub_symbol
                    symbol_data = market_data[symbol][sub_symbol]
                    
                    # Get relevant signals for this sub_symbol
                    tech_signal = technical_signals.get(sub_symbol, {})
                    fund_insight = fundamental_insights.get(sub_symbol, {})
                    sent_insight = sentiment_insights.get(sub_symbol, {})
                    
                    # Assess risk for this sub_symbol
                    symbol_results[sub_symbol] = self._assess_symbol_risk(
                        sub_symbol, symbol_data, tech_signal, fund_insight, sent_insight
                    )
                
                results[symbol] = symbol_results
            else:
                # Process single symbol
                tech_signal = technical_signals.get(symbol, {})
                fund_insight = fundamental_insights.get(symbol, {})
                sent_insight = sentiment_insights.get(symbol, {})
                
                results[symbol] = self._assess_symbol_risk(
                    symbol, market_data[symbol], tech_signal, fund_insight, sent_insight
                )
        
        # Update risk metrics based on portfolio
        self._update_portfolio_risk_metrics(results)
        
        return results
    
    def _assess_symbol_risk(self, symbol, price_data, technical_signals, fundamental_insights, sentiment_insights):
        """
        Assess risk for a single symbol
        
        Args:
            symbol (str): Symbol to assess
            price_data (pd.DataFrame): Price data for the symbol
            technical_signals (dict): Technical analysis signals for the symbol
            fundamental_insights (dict): Fundamental analysis insights for the symbol
            sentiment_insights (dict): Sentiment analysis insights for the symbol
        
        Returns:
            dict: Risk assessment for the symbol
        """
        self.log_action(f"assess_symbol_risk", f"Assessing risk for {symbol}")
        
        # Initialize risk assessment
        risk_assessment = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'position_sizing': {},
            'signals': {
                'technical': technical_signals.get('signals', {}).get('overall', 'NEUTRAL'),
                'fundamental': fundamental_insights.get('recommendation', {}).get('direction', 'neutral'),
                'sentiment': sentiment_insights.get('combined', {}).get('signal', 'neutral')
            }
        }
        
        # Calculate volatility metrics
        volatility = self._calculate_volatility_metrics(price_data)
        risk_assessment['metrics']['volatility'] = volatility
        
        # Calculate correlation metrics
        correlation = self._calculate_correlation(symbol, price_data)
        risk_assessment['metrics']['correlation'] = correlation
        
        # Calculate drawdown metrics
        drawdown = self._calculate_drawdown_metrics(price_data)
        risk_assessment['metrics']['drawdown'] = drawdown
        
        # Determine overall risk level
        risk_level = self._determine_risk_level(
            volatility, correlation, drawdown,
            technical_signals, fundamental_insights, sentiment_insights
        )
        risk_assessment['risk_level'] = risk_level
        
        # Calculate position sizing
        position_sizing = self._calculate_position_sizing(
            symbol, price_data, risk_level,
            technical_signals, fundamental_insights, sentiment_insights
        )
        risk_assessment['position_sizing'] = position_sizing
        
        # Assess current market conditions
        market_conditions = self._assess_market_conditions(
            price_data, technical_signals, fundamental_insights, sentiment_insights
        )
        risk_assessment['market_conditions'] = market_conditions
        
        # Calculate stop loss and take profit levels
        risk_management = self._calculate_risk_management_levels(
            symbol, price_data, position_sizing,
            technical_signals, risk_level
        )
        risk_assessment['risk_management'] = risk_management
        
        return risk_assessment
    
    def _calculate_volatility_metrics(self, price_data):
        """
        Calculate volatility metrics for a symbol
        
        Args:
            price_data (pd.DataFrame): Price data for the symbol
        
        Returns:
            dict: Volatility metrics
        """
        # Check if price_data is valid
        if price_data is None or len(price_data) == 0:
            return {
                'daily_volatility': 0,
                'atr_14': 0,
                'atr_percentage': 0,
                'bollinger_width': 0,
                'volatility_rank': 0
            }
        
        # Calculate daily returns
        if 'close' in price_data.columns:
            returns = price_data['close'].pct_change().dropna()
            daily_volatility = returns.std()
            
            # Calculate ATR (Average True Range)
            high_low = price_data['high'] - price_data['low']
            high_close = (price_data['high'] - price_data['close'].shift()).abs()
            low_close = (price_data['low'] - price_data['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculate ATR as percentage of price
            current_price = price_data['close'].iloc[-1]
            atr_percentage = (atr_14 / current_price) * 100
            
            # Calculate Bollinger Band width
            sma_20 = price_data['close'].rolling(window=20).mean()
            std_20 = price_data['close'].rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            bollinger_width = ((upper_band - lower_band) / sma_20).iloc[-1]
            
            # Calculate volatility rank (percentile over the last year)
            rolling_vol = returns.rolling(window=20).std()
            vol_rank = pd.Series(rolling_vol).rank(pct=True).iloc[-1]
            
            return {
                'daily_volatility': daily_volatility,
                'atr_14': atr_14,
                'atr_percentage': atr_percentage,
                'bollinger_width': bollinger_width,
                'volatility_rank': vol_rank
            }
        else:
            return {
                'daily_volatility': 0,
                'atr_14': 0,
                'atr_percentage': 0,
                'bollinger_width': 0,
                'volatility_rank': 0
            }
    
    def _calculate_correlation(self, symbol, price_data):
        """
        Calculate correlation with other instruments
        
        Args:
            symbol (str): Symbol to calculate correlation for
            price_data (pd.DataFrame): Price data for the symbol
        
        Returns:
            dict: Correlation metrics
        """
        # In a real implementation, we would calculate correlation with other instruments
        # For demonstration, we'll return dummy values
        return {
            'average_correlation': 0.3,
            'max_correlation': 0.7,
            'diversification_score': 0.7
        }
    
    def _calculate_drawdown_metrics(self, price_data):
        """
        Calculate drawdown metrics
        
        Args:
            price_data (pd.DataFrame): Price data for the symbol
        
        Returns:
            dict: Drawdown metrics
        """
        # Check if price_data is valid
        if price_data is None or len(price_data) == 0 or 'close' not in price_data.columns:
            return {
                'max_drawdown': 0,
                'current_drawdown': 0,
                'avg_recovery_time': 0
            }
        
        # Calculate equity curve based on close prices
        equity_curve = price_data['close']
        
        # Calculate drawdown metrics
        drawdown_result = calculate_drawdown(equity_curve)
        
        # Get current drawdown
        current_drawdown = drawdown_result['drawdown'].iloc[-1] if not drawdown_result['drawdown'].empty else 0
        
        return {
            'max_drawdown': drawdown_result['max_drawdown'],
            'current_drawdown': current_drawdown,
            'max_drawdown_duration': drawdown_result['max_drawdown_duration']
        }
    
    def _determine_risk_level(self, volatility, correlation, drawdown, technical_signals, fundamental_insights, sentiment_insights):
        """
        Determine overall risk level based on multiple factors
        
        Args:
            volatility (dict): Volatility metrics
            correlation (dict): Correlation metrics
            drawdown (dict): Drawdown metrics
            technical_signals (dict): Technical analysis signals
            fundamental_insights (dict): Fundamental analysis insights
            sentiment_insights (dict): Sentiment analysis insights
        
        Returns:
            dict: Risk level assessment
        """
        # Initialize risk factors
        risk_factors = {
            'volatility_risk': 0,
            'correlation_risk': 0,
            'drawdown_risk': 0,
            'technical_risk': 0,
            'fundamental_risk': 0,
            'sentiment_risk': 0
        }
        
        # Assess volatility risk (0 to 1, where 1 is highest risk)
        if volatility['atr_percentage'] > 2.0:
            risk_factors['volatility_risk'] = 1.0
        elif volatility['atr_percentage'] > 1.0:
            risk_factors['volatility_risk'] = 0.7
        elif volatility['atr_percentage'] > 0.5:
            risk_factors['volatility_risk'] = 0.4
        else:
            risk_factors['volatility_risk'] = 0.2
        
        # Assess correlation risk (lower correlation is better for diversification)
        if correlation['average_correlation'] < 0.3:
            risk_factors['correlation_risk'] = 0.2
        elif correlation['average_correlation'] < 0.5:
            risk_factors['correlation_risk'] = 0.4
        elif correlation['average_correlation'] < 0.7:
            risk_factors['correlation_risk'] = 0.7
        else:
            risk_factors['correlation_risk'] = 1.0
        
        # Assess drawdown risk
        if drawdown['max_drawdown'] < -0.1:
            risk_factors['drawdown_risk'] = 0.3
        elif drawdown['max_drawdown'] < -0.2:
            risk_factors['drawdown_risk'] = 0.6
        else:
            risk_factors['drawdown_risk'] = 1.0
        
        # Assess technical risk
        technical_signal = technical_signals.get('signals', {}).get('overall', 'NEUTRAL')
        if technical_signal == 'BUY':
            risk_factors['technical_risk'] = 0.3
        elif technical_signal == 'SELL':
            risk_factors['technical_risk'] = 0.3
        else:
            risk_factors['technical_risk'] = 0.7  # Neutral signals indicate uncertainty
        
        # Assess fundamental risk
        fund_direction = fundamental_insights.get('recommendation', {}).get('direction', 'neutral')
        fund_strength = fundamental_insights.get('recommendation', {}).get('strength', 'low')
        
        if fund_strength == 'high':
            risk_factors['fundamental_risk'] = 0.2
        elif fund_strength == 'medium':
            risk_factors['fundamental_risk'] = 0.5
        else:
            risk_factors['fundamental_risk'] = 0.8
        
        # Assess sentiment risk
        sentiment_signal = sentiment_insights.get('combined', {}).get('signal', 'neutral')
        sentiment_strength = sentiment_insights.get('combined', {}).get('strength', 'low')
        
        if sentiment_strength == 'high':
            risk_factors['sentiment_risk'] = 0.3
        elif sentiment_strength == 'medium':
            risk_factors['sentiment_risk'] = 0.5
        else:
            risk_factors['sentiment_risk'] = 0.7
        
        # Calculate weighted average of risk factors
        weights = {
            'volatility_risk': 0.25,
            'correlation_risk': 0.15,
            'drawdown_risk': 0.20,
            'technical_risk': 0.20,
            'fundamental_risk': 0.10,
            'sentiment_risk': 0.10
        }
        
        weighted_risk = sum(risk_factors[k] * weights[k] for k in risk_factors)
        
        # Determine risk level category
        if weighted_risk < 0.3:
            risk_category = 'low'
        elif weighted_risk < 0.6:
            risk_category = 'medium'
        else:
            risk_category = 'high'
        
        return {
            'value': weighted_risk,
            'category': risk_category,
            'factors': risk_factors
        }
    
    def _calculate_position_sizing(self, symbol, price_data, risk_level, technical_signals, fundamental_insights, sentiment_insights):
        """
        Calculate appropriate position size based on risk assessment
        
        Args:
            symbol (str): Symbol to calculate position size for
            price_data (pd.DataFrame): Price data for the symbol
            risk_level (dict): Risk level assessment
            technical_signals (dict): Technical analysis signals
            fundamental_insights (dict): Fundamental analysis insights
            sentiment_insights (dict): Sentiment analysis insights
        
        Returns:
            dict: Position sizing recommendation
        """
        # Get account size from config
        account_size = self.config.get('account_size', 10000)
        max_risk_amount = account_size * self.max_risk_per_trade
        
        # Get position sizing method from config
        position_sizing_method = self.config.get('position_sizing_method', 'risk_based')
        
        # Get current price
        current_price = price_data['close'].iloc[-1] if not price_data.empty and 'close' in price_data.columns else 1.0
        
        # Get stop loss distance based on ATR
        atr = risk_level.get('metrics', {}).get('volatility', {}).get('atr_14', 0)
        if atr == 0:
            atr = current_price * 0.01  # Default to 1% of price if ATR is not available
        
        stop_loss_distance = atr * self.stop_loss_atr_multiplier
        
        # Risk percentage based on risk level
        risk_category = risk_level.get('category', 'medium')
        if risk_category == 'low':
            risk_percentage = self.max_risk_per_trade
        elif risk_category == 'medium':
            risk_percentage = self.max_risk_per_trade * 0.7
        else:  # high risk
            risk_percentage = self.max_risk_per_trade * 0.5
        
        risk_amount = account_size * risk_percentage
        
        # Calculate position size
        if position_sizing_method == 'risk_based':
            # Risk-based position sizing
            if stop_loss_distance > 0:
                position_size = risk_amount / stop_loss_distance
                position_value = position_size * current_price
            else:
                position_size = 0
                position_value = 0
                
        elif position_sizing_method == 'percent_based':
            # Percent-based position sizing
            position_value = account_size * risk_percentage
            position_size = position_value / current_price
            
        else:  # fixed_lot
            # Fixed lot sizing (e.g., 1 standard lot = 100,000 units)
            standard_lot = 100000
            mini_lot = 10000
            micro_lot = 1000
            
            if risk_category == 'low':
                position_size = mini_lot
            elif risk_category == 'medium':
                position_size = micro_lot
            else:  # high risk
                position_size = micro_lot / 2
                
            position_value = position_size * current_price
        
        # Account for signal alignment
        technical_signal = technical_signals.get('signals', {}).get('overall', 'NEUTRAL')
        fundamental_signal = fundamental_insights.get('recommendation', {}).get('direction', 'neutral')
        sentiment_signal = sentiment_insights.get('combined', {}).get('signal', 'neutral')
        
        # Count aligned signals
        aligned_signals = sum([
            1 if technical_signal == 'BUY' or technical_signal == 'SELL' else 0,
            1 if fundamental_signal == 'buy' or fundamental_signal == 'sell' else 0,
            1 if sentiment_signal == 'buy' or sentiment_signal == 'sell' else 0
        ])
        
        # Adjust position size based on signal alignment
        if aligned_signals == 3:
            # All signals aligned, increase position size
            position_size *= 1.2
            position_value *= 1.2
        elif aligned_signals == 1:
            # Only one signal, reduce position size
            position_size *= 0.8
            position_value *= 0.8
        elif aligned_signals == 0:
            # No clear signals, minimal position
            position_size *= 0.5
            position_value *= 0.5
        
        # Check if position value exceeds max risk amount
        if position_value > max_risk_amount:
            position_size = (max_risk_amount / position_value) * position_size
            position_value = max_risk_amount
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'stop_loss_distance': stop_loss_distance,
            'method': position_sizing_method
        }
    
    def _assess_market_conditions(self, price_data, technical_signals, fundamental_insights, sentiment_insights):
        """
        Assess current market conditions
        
        Args:
            price_data (pd.DataFrame): Price data for the symbol
            technical_signals (dict): Technical analysis signals
            fundamental_insights (dict): Fundamental analysis insights
            sentiment_insights (dict): Sentiment analysis insights
        
        Returns:
            dict: Market condition assessment
        """
        # Determine trend direction
        trend = 'sideways'
        if 'moving_averages' in technical_signals:
            ma_signals = technical_signals['moving_averages']
            if 'SMA_50' in ma_signals and 'SMA_200' in ma_signals:
                sma_50 = ma_signals['SMA_50'].iloc[-1] if not ma_signals['SMA_50'].empty else 0
                sma_200 = ma_signals['SMA_200'].iloc[-1] if not ma_signals['SMA_200'].empty else 0
                
                if sma_50 > sma_200:
                    trend = 'uptrend'
                elif sma_50 < sma_200:
                    trend = 'downtrend'
        
        # Determine market regime
        volatility = 'normal'
        if 'volatility' in technical_signals:
            vol_signals = technical_signals['volatility']
            if 'BB_width' in vol_signals:
                bb_width = vol_signals['BB_width'].iloc[-1] if not vol_signals['BB_width'].empty else 0
                
                if bb_width > 0.05:
                    volatility = 'high'
                elif bb_width < 0.02:
                    volatility = 'low'
        
        # Market regime
        if trend == 'sideways' and volatility == 'low':
            regime = 'range_bound'
        elif trend != 'sideways' and volatility == 'high':
            regime = 'trending_volatile'
        elif trend != 'sideways':
            regime = 'trending'
        else:
            regime = 'choppy'
        
        # Signal conflict assessment
        technical_signal = technical_signals.get('signals', {}).get('overall', 'NEUTRAL')
        fundamental_signal = fundamental_insights.get('recommendation', {}).get('direction', 'neutral')
        sentiment_signal = sentiment_insights.get('combined', {}).get('signal', 'neutral')
        
        signals = [
            technical_signal.lower() if technical_signal.lower() in ['buy', 'sell'] else 'neutral',
            fundamental_signal,
            sentiment_signal
        ]
        
        unique_signals = set(signals)
        if len(unique_signals) == 1:
            signal_alignment = 'aligned'
        elif len(unique_signals) == 3:
            signal_alignment = 'conflicting'
        else:
            signal_alignment = 'partially_aligned'
        
        return {
            'trend': trend,
            'volatility': volatility,
            'regime': regime,
            'signal_alignment': signal_alignment,
            'signals': {
                'technical': technical_signal,
                'fundamental': fundamental_signal,
                'sentiment': sentiment_signal
            }
        }
    
    def _calculate_risk_management_levels(self, symbol, price_data, position_sizing, technical_signals, risk_level):
        """
        Calculate risk management levels (stop loss, take profit)
        
        Args:
            symbol (str): Symbol
            price_data (pd.DataFrame): Price data for the symbol
            position_sizing (dict): Position sizing data
            technical_signals (dict): Technical analysis signals
            risk_level (dict): Risk level assessment
        
        Returns:
            dict: Risk management levels
        """
        # Get current price
        current_price = price_data['close'].iloc[-1] if not price_data.empty and 'close' in price_data.columns else 1.0
        
        # Get signals
        signal = technical_signals.get('signals', {}).get('overall', 'NEUTRAL')
        
        # Get stop loss distance
        stop_loss_distance = position_sizing.get('stop_loss_distance', current_price * 0.01)
        
        # Calculate stop loss level
        if signal == 'BUY':
            stop_loss = current_price - stop_loss_distance
        elif signal == 'SELL':
            stop_loss = current_price + stop_loss_distance
        else:
            stop_loss = current_price - stop_loss_distance  # Default to long position
        
        # Calculate risk-reward ratios based on risk level
        risk_category = risk_level.get('category', 'medium')
        if risk_category == 'low':
            risk_reward_ratio = 1.5
        elif risk_category == 'medium':
            risk_reward_ratio = 2.0
        else:  # high risk
            risk_reward_ratio = 3.0
        
        # Calculate take profit level
        take_profit_distance = stop_loss_distance * risk_reward_ratio
        
        if signal == 'BUY':
            take_profit = current_price + take_profit_distance
        elif signal == 'SELL':
            take_profit = current_price - take_profit_distance
        else:
            take_profit = current_price + take_profit_distance  # Default to long position
        
        # Calculate trailing stop parameters
        trailing_stop_activation = take_profit_distance * 0.5
        trailing_stop_distance = stop_loss_distance * 0.8
        
        return {
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'trailing_stop': {
                'activation': trailing_stop_activation,
                'distance': trailing_stop_distance
            }
        }
    
    def _update_portfolio_risk_metrics(self, risk_assessments):
        """
        Update portfolio-wide risk metrics
        
        Args:
            risk_assessments (dict): Risk assessments by symbol
        
        Returns:
            None (updates risk_assessments in place)
        """
        # Aggregate risk metrics across all symbols
        total_risk = 0
        risk_count = 0
        symbols = []
        
        # Flatten nested dictionaries if needed
        flat_assessments = {}
        for key, value in risk_assessments.items():
            if isinstance(value, dict) and 'risk_level' not in value:
                for sub_key, sub_value in value.items():
                    flat_assessments[f"{key}_{sub_key}"] = sub_value
            else:
                flat_assessments[key] = value
        
        # Process each symbol's risk assessment
        for symbol, assessment in flat_assessments.items():
            if 'risk_level' in assessment:
                total_risk += assessment['risk_level']['value']
                risk_count += 1
                symbols.append(symbol)
        
        # Calculate average portfolio risk
        avg_portfolio_risk = total_risk / risk_count if risk_count > 0 else 0
        
        # Determine portfolio risk category
        if avg_portfolio_risk < 0.3:
            portfolio_risk_category = 'low'
        elif avg_portfolio_risk < 0.6:
            portfolio_risk_category = 'medium'
        else:
            portfolio_risk_category = 'high'
        
        # Add portfolio risk metrics to each assessment
        portfolio_risk = {
            'average_risk': avg_portfolio_risk,
            'category': portfolio_risk_category,
            'diversification': len(symbols)
        }
        
        for symbol, assessment in flat_assessments.items():
            if 'risk_level' in assessment:
                assessment['portfolio_risk'] = portfolio_risk
        
        # Update nested dictionaries if necessary
        for key, value in risk_assessments.items():
            if isinstance(value, dict) and 'risk_level' not in value:
                for sub_key, sub_value in value.items():
                    if 'risk_level' in sub_value:
                        sub_value['portfolio_risk'] = portfolio_risk

    # === Position Sizing Methods ===
    
    def calculate_position_size(
        self, 
        account_balance: float, 
        risk_percentage: float, 
        stop_loss_pips: float, 
        currency_pair: str
    ) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on account balance, risk tolerance, and stop loss.
        
        Args:
            account_balance: Current account balance in base currency
            risk_percentage: Percentage of account willing to risk on this trade (e.g., 0.01 for 1%)
            stop_loss_pips: Distance to stop loss in pips
            currency_pair: Currency pair being traded (e.g., 'EUR/USD')
        
        Returns:
            Dict containing position size in units/lots and monetary risk
        """
        self.log_action("calculate_position_size", f"Calculating position size for {currency_pair}")
        
        if stop_loss_pips <= 0:
            return {
                'position_size': 0.0,
                'lots': 0.0,
                'risk_amount': 0.0,
                'error': 'Stop loss must be greater than 0'
            }
            
        # Calculate risk amount in account currency
        risk_amount = account_balance * risk_percentage
        
        # Get pip value for the currency pair
        pip_value = self._calculate_pip_value(currency_pair, 1.0)  # Value per 1.0 lot
        
        # Calculate risk per pip
        risk_per_pip = risk_amount / stop_loss_pips
        
        # Calculate position size in lots
        position_size_lots = risk_per_pip / pip_value
        
        # Calculate position size in units (standard lot = 100,000 units)
        position_size_units = position_size_lots * 100000
        
        return {
            'position_size': position_size_units,
            'lots': position_size_lots,
            'risk_amount': risk_amount,
            'risk_per_pip': risk_per_pip,
            'currency_pair': currency_pair
        }
    
    def adjust_for_correlation(
        self, 
        positions: List[Dict[str, Any]], 
        new_position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adjust position size based on correlation with existing positions.
        
        Reduces position size when there is high correlation between the proposed
        trade and existing positions to avoid overexposure to correlated risks.
        
        Args:
            positions: List of current open positions
            new_position: Proposed new position to evaluate
        
        Returns:
            Dict containing adjusted position size and correlation metrics
        """
        self.log_action("adjust_for_correlation", f"Adjusting position for correlation")
        
        if not positions or len(positions) == 0:
            return new_position  # No adjustment needed
            
        # Extract the currency pair from new position
        new_pair = new_position.get('currency_pair', '')
        if not new_pair:
            return new_position
            
        # Calculate correlation factor
        correlation_factor = self._calculate_correlation_factor(positions, new_pair)
        
        # Apply correlation adjustment to position size
        if correlation_factor > 0.2:  # Only adjust if meaningful correlation exists
            # Higher correlation = greater reduction
            adjustment_multiplier = 1 - (correlation_factor * 0.5)  # Max 50% reduction at perfect correlation
            
            # Apply adjustment to position size
            original_size = new_position.get('position_size', 0.0)
            original_lots = new_position.get('lots', 0.0)
            
            adjusted_size = original_size * adjustment_multiplier
            adjusted_lots = original_lots * adjustment_multiplier
            
            # Update the position with adjusted sizes
            adjusted_position = new_position.copy()
            adjusted_position['position_size'] = adjusted_size
            adjusted_position['lots'] = adjusted_lots
            adjusted_position['correlation_factor'] = correlation_factor
            adjusted_position['correlation_adjustment'] = adjustment_multiplier
            
            return adjusted_position
        
        # No significant correlation, return original position with correlation info
        new_position['correlation_factor'] = correlation_factor
        new_position['correlation_adjustment'] = 1.0
        return new_position
    
    def calculate_max_positions(self, account_balance: float) -> Dict[str, Any]:
        """
        Determine maximum number of concurrent positions based on account balance.
        
        Adjusts position limits based on account size to maintain appropriate 
        diversification without excessive exposure.
        
        Args:
            account_balance: Current account balance in base currency
        
        Returns:
            Dict containing maximum positions and related metrics
        """
        self.log_action("calculate_max_positions", f"Calculating max positions for account balance: {account_balance}")
        
        # Base calculation on tiers of account balance
        if account_balance < 1000:
            base_max = 2  # Micro account - limit positions
        elif account_balance < 5000:
            base_max = 4  # Small account
        elif account_balance < 20000:
            base_max = 7  # Medium account
        elif account_balance < 50000:
            base_max = 10  # Large account
        elif account_balance < 100000:
            base_max = 15  # Very large account
        else:
            base_max = 20  # Professional account
            
        # Calculate max based on risk profile in config
        config_max = self.max_open_positions
        
        # Use the more conservative of the two limits
        max_positions = min(base_max, config_max)
        
        # Calculate maximum positions per asset class (e.g., currency groups)
        max_per_group = max(1, int(max_positions / 3))
        
        return {
            'max_positions': max_positions,
            'max_per_currency_group': max_per_group,
            'account_balance': account_balance,
            'account_tier': self._get_account_tier(account_balance)
        }
    
    def calculate_tiered_position_sizes(
        self, 
        account_balance: float, 
        risk_levels: List[Dict[str, float]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate position sizes for different risk levels.
        
        Creates a tiered approach to position sizing for scaling in/out of trades
        or for implementing a portfolio with varying risk levels.
        
        Args:
            account_balance: Current account balance in base currency
            risk_levels: List of dicts containing risk level configurations
                         Each dict should have 'name', 'risk_percentage', 'stop_loss_pips'
        
        Returns:
            Dict containing tiered position sizes for specified risk levels
        """
        self.log_action("calculate_tiered_position_sizes", f"Calculating tiered position sizes")
        
        if not risk_levels or len(risk_levels) == 0:
            # Default risk levels if none provided
            risk_levels = [
                {'name': 'conservative', 'risk_percentage': 0.01, 'stop_loss_pips': 50},
                {'name': 'moderate', 'risk_percentage': 0.02, 'stop_loss_pips': 40},
                {'name': 'aggressive', 'risk_percentage': 0.03, 'stop_loss_pips': 30}
            ]
            
        # Get list of common currency pairs
        currency_pairs = self._get_common_currency_pairs()
        
        results = {}
        
        # Calculate position sizes for each currency pair at each risk level
        for pair in currency_pairs:
            pair_results = []
            
            for level in risk_levels:
                name = level.get('name', 'unknown')
                risk_pct = level.get('risk_percentage', 0.01)
                sl_pips = level.get('stop_loss_pips', 50)
                
                # Calculate position size for this level
                position_size = self.calculate_position_size(
                    account_balance, risk_pct, sl_pips, pair
                )
                
                # Add risk level info
                position_size['risk_level'] = name
                
                pair_results.append(position_size)
                
            results[pair] = pair_results
            
        return {
            'account_balance': account_balance,
            'tiered_position_sizes': results
        }
    
    def _calculate_pip_value(self, currency_pair: str, lot_size: float) -> float:
        """
        Calculate the pip value for a given currency pair and lot size.
        
        Args:
            currency_pair: The currency pair (e.g., EUR/USD)
            lot_size: Size in standard lots (1.0 = 100,000 units)
            
        Returns:
            float: Value of one pip in the account's base currency
        """
        # For simplicity, using approximations for common pairs
        # In a real system, this would query current rates and do proper calculations
        base_pip_values = {
            'EUR/USD': 10.0,
            'GBP/USD': 10.0,
            'USD/JPY': 9.4,
            'AUD/USD': 10.0,
            'USD/CHF': 10.6,
            'USD/CAD': 7.6,
            'NZD/USD': 10.0,
            'EUR/GBP': 13.1,
            'EUR/JPY': 9.4,
            'GBP/JPY': 9.4
        }
        
        # Get base pip value for the pair, default to 10.0 if unknown
        base_value = base_pip_values.get(currency_pair, 10.0)
        
        # Scale by lot size
        return base_value * lot_size
    
    def _calculate_correlation_factor(self, positions: List[Dict[str, Any]], new_pair: str) -> float:
        """
        Calculate correlation factor between a new pair and existing positions.
        
        Args:
            positions: List of existing positions
            new_pair: New currency pair to evaluate
            
        Returns:
            float: Correlation factor (0-1)
        """
        if not positions:
            return 0.0
            
        correlation_sum = 0.0
        count = 0
        
        # Extract base and quote currencies
        if '/' in new_pair:
            new_base, new_quote = new_pair.split('/')
        else:
            # Handle alternative format like EURUSD
            new_base = new_pair[:3]
            new_quote = new_pair[3:] if len(new_pair) >= 6 else ""
        
        for position in positions:
            pos_pair = position.get('currency_pair', '')
            if not pos_pair:
                continue
                
            # Extract currencies from position pair
            if '/' in pos_pair:
                pos_base, pos_quote = pos_pair.split('/')
            else:
                pos_base = pos_pair[:3]
                pos_quote = pos_pair[3:] if len(pos_pair) >= 6 else ""
            
            # Calculate correlation based on currency overlap
            # Exact same pair = 1.0
            if pos_pair == new_pair:
                correlation_sum += 1.0
                count += 1
                continue
                
            correlation = 0.0
            
            # Check for shared currencies
            if new_base == pos_base:
                correlation += 0.5
            if new_quote == pos_quote:
                correlation += 0.5
            if new_base == pos_quote:
                correlation += 0.3
            if new_quote == pos_base:
                correlation += 0.3
                
            # Adjust for direction (long/short)
            new_dir = 1  # Assuming long for new position
            pos_dir = 1 if position.get('direction', 'long').lower() == 'long' else -1
            
            if new_dir != pos_dir:
                # Opposite directions reduce correlation
                correlation *= 0.5
                
            if correlation > 0:
                correlation_sum += correlation
                count += 1
                
        # Calculate average correlation
        return correlation_sum / max(1, count)
    
    def _get_account_tier(self, account_balance: float) -> str:
        """Determine account tier based on balance"""
        if account_balance < 1000:
            return "micro"
        elif account_balance < 5000:
            return "mini"
        elif account_balance < 20000:
            return "standard"
        elif account_balance < 50000:
            return "premium"
        elif account_balance < 100000:
            return "professional"
        else:
            return "institutional"
            
    def _get_common_currency_pairs(self) -> List[str]:
        """Return a list of common forex currency pairs"""
        return [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD',
            'USD/CHF', 'USD/CAD', 'NZD/USD', 'EUR/GBP'
        ]

    # === Risk Assessment Methods ===
    
    def calculate_portfolio_var(
        self, 
        positions: List[Dict[str, Any]], 
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Uses historical simulation method to determine potential losses at 
        the specified confidence level.
        
        Args:
            positions: List of open positions with exposure details
            confidence_level: Statistical confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Dict containing VaR and related risk metrics
        """
        self.log_action("calculate_portfolio_var", f"Calculating portfolio VaR at {confidence_level*100}% confidence")
        
        if not positions or len(positions) == 0:
            return {
                'var': 0.0,
                'confidence_level': confidence_level,
                'positions_count': 0,
                'method': 'historical_simulation'
            }
            
        # Extract position values and create return series
        position_values = []
        weights = []
        returns_data = {}
        
        total_exposure = sum(position.get('exposure', 0) for position in positions)
        
        for position in positions:
            exposure = position.get('exposure', 0)
            currency_pair = position.get('currency_pair', '')
            direction = 1 if position.get('direction', 'long').lower() == 'long' else -1
            
            if exposure <= 0 or not currency_pair:
                continue
                
            # Weight by relative exposure
            weight = exposure / total_exposure if total_exposure > 0 else 0
            weights.append(weight)
            position_values.append(exposure)
            
            # Get or generate historical returns for this currency pair
            pair_returns = self._get_historical_returns(currency_pair)
            
            # Adjust direction based on long/short
            if direction == -1:
                pair_returns = [-r for r in pair_returns]
                
            returns_data[currency_pair] = pair_returns
            
        # If no valid positions, return zero VAR
        if not position_values or not weights:
            return {
                'var': 0.0,
                'confidence_level': confidence_level,
                'positions_count': 0,
                'method': 'historical_simulation'
            }
            
        # Create correlation-adjusted portfolio returns
        portfolio_returns = self._simulate_portfolio_returns(returns_data, weights)
        
        # Calculate VaR using historical simulation
        var = self._calculate_var_from_returns(portfolio_returns, confidence_level, sum(position_values))
        
        # Store VaR in risk metrics
        self.risk_metrics['portfolio_var'] = var['var_amount']
        
        return var
    
    def calculate_expected_shortfall(
        self, 
        positions: List[Dict[str, Any]], 
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate Expected Shortfall (CVaR) for the portfolio.
        
        Expected Shortfall is the expected loss given that the loss exceeds VaR,
        providing a more conservative risk measure than VaR.
        
        Args:
            positions: List of open positions with exposure details
            confidence_level: Statistical confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Dict containing Expected Shortfall and related metrics
        """
        self.log_action("calculate_expected_shortfall", f"Calculating expected shortfall at {confidence_level*100}% confidence")
        
        # First calculate VaR to get portfolio returns
        var_result = self.calculate_portfolio_var(positions, confidence_level)
        
        if var_result.get('positions_count', 0) == 0:
            return {
                'expected_shortfall': 0.0,
                'var': 0.0,
                'confidence_level': confidence_level,
                'positions_count': 0
            }
            
        # Extract portfolio returns and total value from VaR calculation
        portfolio_returns = var_result.get('portfolio_returns', [])
        total_value = var_result.get('portfolio_value', 0)
        
        if not portfolio_returns or total_value == 0:
            # If we don't have returns data, calculate it
            if not positions or len(positions) == 0:
                return {
                    'expected_shortfall': 0.0,
                    'var': var_result.get('var_amount', 0.0),
                    'confidence_level': confidence_level,
                    'positions_count': 0
                }
                
            # Extract position values and create return series
            weights = []
            returns_data = {}
            
            total_exposure = sum(position.get('exposure', 0) for position in positions)
            
            for position in positions:
                exposure = position.get('exposure', 0)
                currency_pair = position.get('currency_pair', '')
                direction = 1 if position.get('direction', 'long').lower() == 'long' else -1
                
                if exposure <= 0 or not currency_pair:
                    continue
                    
                # Weight by relative exposure
                weight = exposure / total_exposure if total_exposure > 0 else 0
                weights.append(weight)
                
                # Get or generate historical returns for this currency pair
                pair_returns = self._get_historical_returns(currency_pair)
                
                # Adjust direction based on long/short
                if direction == -1:
                    pair_returns = [-r for r in pair_returns]
                    
                returns_data[currency_pair] = pair_returns
                
            # Create correlation-adjusted portfolio returns
            portfolio_returns = self._simulate_portfolio_returns(returns_data, weights)
            total_value = total_exposure
            
        # Sort returns from worst to best
        sorted_returns = sorted(portfolio_returns)
        
        # Calculate cutoff index based on confidence level
        cutoff_index = int(len(sorted_returns) * (1 - confidence_level))
        
        # Get the tail returns (the worst outcomes)
        tail_returns = sorted_returns[:cutoff_index]
        
        if not tail_returns:
            # If no tail returns (can happen with small datasets), use VaR
            expected_shortfall_pct = var_result.get('var_percentage', 0.0)
        else:
            # Expected shortfall is the average of the tail returns
            expected_shortfall_pct = np.mean(tail_returns)
            
        # Convert to amount
        expected_shortfall_amount = abs(expected_shortfall_pct * total_value)
        
        # Store in risk metrics
        self.risk_metrics['expected_shortfall'] = expected_shortfall_amount
        
        return {
            'expected_shortfall': expected_shortfall_amount,
            'expected_shortfall_percentage': expected_shortfall_pct,
            'var': var_result.get('var_amount', 0.0),
            'var_percentage': var_result.get('var_percentage', 0.0),
            'confidence_level': confidence_level,
            'tail_size': len(tail_returns),
            'positions_count': var_result.get('positions_count', 0)
        }
    
    def assess_drawdown_risk(self, strategy_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess potential drawdown risk based on historical performance.
        
        Analyzes historical drawdowns to project potential future drawdowns
        and evaluates the strategy's resilience to market downturns.
        
        Args:
            strategy_performance: Dict containing historical performance data
        
        Returns:
            Dict containing drawdown risk assessment
        """
        self.log_action("assess_drawdown_risk", "Assessing drawdown risk")
        
        # Extract performance data
        equity_curve = strategy_performance.get('equity_curve', [])
        returns = strategy_performance.get('returns', [])
        
        if not equity_curve or len(equity_curve) < 10:
            return {
                'max_drawdown': 0.0,
                'average_drawdown': 0.0,
                'drawdown_risk': 'unknown',
                'recovery_potential': 'unknown',
                'error': 'Insufficient historical data'
            }
            
        # Calculate historical drawdowns
        drawdowns = self._calculate_historical_drawdowns(equity_curve)
        
        if not drawdowns:
            return {
                'max_drawdown': 0.0,
                'average_drawdown': 0.0,
                'drawdown_risk': 'low',
                'recovery_potential': 'high'
            }
            
        # Calculate key drawdown metrics
        max_drawdown = max(drawdowns) if drawdowns else 0
        average_drawdown = np.mean(drawdowns) if drawdowns else 0
        drawdown_frequency = len(drawdowns) / len(equity_curve)
        
        # Calculate recovery metrics
        recovery_times = self._calculate_recovery_times(equity_curve, drawdowns)
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        # Calculate drawdown to return ratio
        avg_return = np.mean(returns) if returns else 0
        drawdown_return_ratio = average_drawdown / max(0.0001, avg_return)
        
        # Assess drawdown risk level
        if max_drawdown < 0.05:
            risk_level = 'very low'
        elif max_drawdown < 0.10:
            risk_level = 'low'
        elif max_drawdown < 0.20:
            risk_level = 'moderate'
        elif max_drawdown < 0.30:
            risk_level = 'high'
        else:
            risk_level = 'very high'
            
        # Assess recovery potential
        if avg_recovery_time < 5:
            recovery_potential = 'very high'
        elif avg_recovery_time < 15:
            recovery_potential = 'high'
        elif avg_recovery_time < 30:
            recovery_potential = 'moderate'
        elif avg_recovery_time < 60:
            recovery_potential = 'low'
        else:
            recovery_potential = 'very low'
            
        # Store current drawdown in risk metrics
        current_dd = drawdowns[-1] if drawdowns else 0
        self.risk_metrics['current_drawdown'] = current_dd
        
        return {
            'max_drawdown': max_drawdown,
            'average_drawdown': average_drawdown,
            'current_drawdown': current_dd,
            'drawdown_frequency': drawdown_frequency,
            'avg_recovery_time': avg_recovery_time,
            'drawdown_return_ratio': drawdown_return_ratio,
            'drawdown_risk': risk_level,
            'recovery_potential': recovery_potential
        }
    
    def calculate_risk_reward_ratio(
        self, 
        entry: float, 
        stop_loss: float, 
        take_profit: float
    ) -> Dict[str, Any]:
        """
        Calculate risk-reward ratio for a trade.
        
        Determines the potential reward compared to the risk being taken
        based on entry, stop loss, and take profit levels.
        
        Args:
            entry: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Dict containing risk-reward ratio and related metrics
        """
        self.log_action("calculate_risk_reward_ratio", "Calculating risk-reward ratio")
        
        # Validate inputs
        if not all(isinstance(x, (int, float)) for x in [entry, stop_loss, take_profit]):
            return {
                'risk_reward_ratio': 0.0,
                'risk': 0.0,
                'reward': 0.0,
                'error': 'Invalid input prices'
            }
            
        # Determine direction
        is_long = take_profit > entry
        
        # Calculate risk and reward
        if is_long:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
        else:
            risk = abs(stop_loss - entry)
            reward = abs(entry - take_profit)
            
        # Calculate risk-reward ratio
        if risk == 0:
            risk_reward_ratio = 0.0
            evaluation = 'invalid'
        else:
            risk_reward_ratio = reward / risk
            
            # Evaluate the quality of the risk-reward ratio
            if risk_reward_ratio < 1.0:
                evaluation = 'poor'
            elif risk_reward_ratio < 1.5:
                evaluation = 'suboptimal'
            elif risk_reward_ratio < 2.0:
                evaluation = 'acceptable'
            elif risk_reward_ratio < 3.0:
                evaluation = 'good'
            else:
                evaluation = 'excellent'
                
        return {
            'risk_reward_ratio': risk_reward_ratio,
            'risk': risk,
            'reward': reward,
            'direction': 'long' if is_long else 'short',
            'evaluation': evaluation
        }
    
    def evaluate_trade_risk(self, trade_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate overall risk of a proposed trade.
        
        Comprehensive assessment of a trade's risk considering multiple factors
        such as risk-reward, market conditions, and portfolio impact.
        
        Args:
            trade_parameters: Dict containing trade details including entry, stop loss, 
                             take profit, currency pair, position size, etc.
        
        Returns:
            Dict containing risk evaluation results
        """
        self.log_action("evaluate_trade_risk", f"Evaluating trade risk for {trade_parameters.get('currency_pair', 'unknown')}")
        
        # Extract trade parameters
        currency_pair = trade_parameters.get('currency_pair', '')
        entry = trade_parameters.get('entry_price', 0)
        stop_loss = trade_parameters.get('stop_loss', 0)
        take_profit = trade_parameters.get('take_profit', 0)
        position_size = trade_parameters.get('position_size', 0)
        direction = trade_parameters.get('direction', 'long')
        account_balance = trade_parameters.get('account_balance', 0)
        
        if not currency_pair or not entry or not stop_loss or not take_profit:
            return {
                'risk_level': 'unknown',
                'risk_score': 0,
                'error': 'Incomplete trade parameters'
            }
            
        # Calculate risk-reward ratio
        risk_reward = self.calculate_risk_reward_ratio(entry, stop_loss, take_profit)
        
        # Calculate monetary risk
        if direction.lower() == 'long':
            pip_risk = abs(entry - stop_loss) * 10000
        else:
            pip_risk = abs(stop_loss - entry) * 10000
            
        pip_value = self._calculate_pip_value(currency_pair, position_size / 100000)
        monetary_risk = pip_risk * pip_value
        
        # Calculate risk as percentage of account
        risk_percentage = monetary_risk / account_balance if account_balance > 0 else 0
        
        # Calculate win probability based on historical performance for this pair
        win_rate = self._get_historical_win_rate(currency_pair, direction)
        
        # Calculate expectancy
        expectancy = (risk_reward['risk_reward_ratio'] * win_rate) - (1 - win_rate)
        
        # Assess market conditions
        market_conditions = self._assess_market_conditions_for_trade(trade_parameters)
        
        # Assess correlation with existing positions
        correlation_risk = self._assess_correlation_risk_for_trade(trade_parameters)
        
        # Calculate overall risk score (0-100)
        risk_factors = [
            # Poor risk-reward reduces score
            100 - max(0, min(100, (3 - risk_reward['risk_reward_ratio']) * 20)),
            
            # High risk percentage reduces score
            100 - max(0, min(100, risk_percentage * 1000)),
            
            # Poor expectancy reduces score
            max(0, min(100, expectancy * 50 + 50)),
            
            # Poor market conditions reduce score
            market_conditions['favorability_score'],
            
            # High correlation reduces score
            100 - correlation_risk['correlation_score']
        ]
        
        risk_score = sum(risk_factors) / len(risk_factors)
        
        # Determine risk level
        if risk_score < 30:
            risk_level = 'very high'
        elif risk_score < 45:
            risk_level = 'high'
        elif risk_score < 60:
            risk_level = 'moderate'
        elif risk_score < 75:
            risk_level = 'low'
        else:
            risk_level = 'very low'
            
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_reward_ratio': risk_reward['risk_reward_ratio'],
            'monetary_risk': monetary_risk,
            'risk_percentage': risk_percentage,
            'win_probability': win_rate,
            'expectancy': expectancy,
            'correlation_risk': correlation_risk,
            'market_conditions': market_conditions,
            'max_risk_exceeded': risk_percentage > self.max_risk_per_trade
        }
        
    def _get_historical_returns(self, currency_pair: str) -> List[float]:
        """
        Get historical returns for a currency pair.
        
        In a real implementation, this would retrieve actual historical data.
        For this example, we generate synthetic returns.
        
        Args:
            currency_pair: The currency pair to get returns for
            
        Returns:
            List of historical returns
        """
        # Generate synthetic historical returns
        np.random.seed(hash(currency_pair) % 10000)  # Seed based on currency pair
        
        # Generate 1000 return samples
        # Different pairs have different characteristics
        if 'JPY' in currency_pair:
            # JPY pairs typically have higher volatility
            return np.random.normal(0.0001, 0.007, 1000).tolist()
        elif 'USD' in currency_pair:
            # USD pairs are typically less volatile
            return np.random.normal(0.0001, 0.005, 1000).tolist()
        else:
            # Other pairs
            return np.random.normal(0.0001, 0.006, 1000).tolist()
    
    def _simulate_portfolio_returns(
        self, 
        returns_data: Dict[str, List[float]], 
        weights: List[float]
    ) -> List[float]:
        """
        Simulate portfolio returns based on historical returns of components.
        
        Args:
            returns_data: Dict mapping currency pairs to their historical returns
            weights: Weight of each currency pair in the portfolio
            
        Returns:
            List of simulated portfolio returns
        """
        if not returns_data or not weights:
            return []
            
        # Determine the length of simulation
        min_length = min(len(returns) for returns in returns_data.values())
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else weights
        
        # Combine returns based on weights
        portfolio_returns = []
        pairs = list(returns_data.keys())
        
        for i in range(min_length):
            period_return = sum(
                returns_data[pair][i] * normalized_weights[j]
                for j, pair in enumerate(pairs)
                if j < len(normalized_weights)
            )
            portfolio_returns.append(period_return)
            
        return portfolio_returns
    
    def _calculate_var_from_returns(
        self, 
        returns: List[float], 
        confidence_level: float, 
        portfolio_value: float
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk from a list of returns.
        
        Args:
            returns: List of historical or simulated returns
            confidence_level: Confidence level for VaR (e.g., 0.95)
            portfolio_value: Total value of the portfolio
            
        Returns:
            Dict containing VaR metrics
        """
        if not returns:
            return {
                'var_percentage': 0.0,
                'var_amount': 0.0,
                'confidence_level': confidence_level,
                'portfolio_value': portfolio_value
            }
            
        # Sort returns from worst to best
        sorted_returns = sorted(returns)
        
        # Find the return at the specified confidence level
        index = int(len(sorted_returns) * (1 - confidence_level))
        var_return = sorted_returns[max(0, index)]
        
        # Calculate VaR as a percentage and amount
        var_percentage = abs(var_return)
        var_amount = var_percentage * portfolio_value
        
        return {
            'var_percentage': var_percentage,
            'var_amount': var_amount,
            'confidence_level': confidence_level,
            'portfolio_value': portfolio_value,
            'portfolio_returns': returns
        }
        
    def _calculate_historical_drawdowns(self, equity_curve: List[float]) -> List[float]:
        """
        Calculate historical drawdowns from an equity curve.
        
        Args:
            equity_curve: List of equity values over time
            
        Returns:
            List of drawdown percentages
        """
        if not equity_curve or len(equity_curve) < 2:
            return []
            
        # Initialize variables
        peak = equity_curve[0]
        drawdowns = []
        
        for equity in equity_curve:
            # Update peak if we have a new high
            if equity > peak:
                peak = equity
                
            # Calculate drawdown if we're below peak
            if peak > 0:
                drawdown = (peak - equity) / peak
                drawdowns.append(drawdown)
                
        return drawdowns
        
    def _calculate_recovery_times(
        self, 
        equity_curve: List[float], 
        drawdowns: List[float]
    ) -> List[int]:
        """
        Calculate recovery times from drawdowns.
        
        Args:
            equity_curve: List of equity values over time
            drawdowns: List of drawdown percentages
            
        Returns:
            List of recovery times in periods
        """
        if not equity_curve or len(equity_curve) < 2 or not drawdowns:
            return []
            
        # Find periods of recovery
        in_drawdown = False
        peak = equity_curve[0]
        drawdown_start = 0
        recovery_times = []
        
        for i, equity in enumerate(equity_curve):
            # Update peak if we have a new high
            if equity > peak:
                # If we were in a drawdown, we've recovered
                if in_drawdown:
                    recovery_time = i - drawdown_start
                    recovery_times.append(recovery_time)
                    in_drawdown = False
                    
                peak = equity
                
            # Check if we're entering a drawdown
            elif equity < peak and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
                
        return recovery_times
        
    def _get_historical_win_rate(self, currency_pair: str, direction: str) -> float:
        """
        Get historical win rate for a currency pair and direction.
        
        In a real implementation, this would retrieve actual historical data.
        For this example, we generate synthetic win rates.
        
        Args:
            currency_pair: The currency pair
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Win rate as a decimal (0.0-1.0)
        """
        # In a real system, this would be based on historical data
        # For this example, we'll use some reasonable defaults with randomness
        base_win_rate = 0.55  # Average win rate
        
        # Add some variation based on the currency pair
        pair_seed = hash(currency_pair) % 10000
        np.random.seed(pair_seed)
        pair_adjustment = np.random.uniform(-0.05, 0.05)
        
        # Add variation based on direction
        direction_adjustment = 0.02 if direction.lower() == 'long' else -0.02
        
        # Calculate final win rate
        win_rate = base_win_rate + pair_adjustment + direction_adjustment
        
        # Ensure it's within valid range
        return max(0.4, min(0.7, win_rate))
        
    def _assess_market_conditions_for_trade(self, trade_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess market conditions for a specific trade.
        
        Args:
            trade_parameters: Dict containing trade parameters
            
        Returns:
            Dict with market condition assessment
        """
        # In a real system, this would analyze actual market data
        # For this example, we'll return a simplified assessment
        
        direction = trade_parameters.get('direction', 'long').lower()
        currency_pair = trade_parameters.get('currency_pair', '')
        
        # Generate a favorability score (0-100)
        np.random.seed(hash(currency_pair + direction) % 10000)
        favorability_score = np.random.randint(40, 80)
        
        if favorability_score < 50:
            assessment = 'unfavorable'
        elif favorability_score < 70:
            assessment = 'neutral'
        else:
            assessment = 'favorable'
            
        return {
            'assessment': assessment,
            'favorability_score': favorability_score,
            'volatility': 'moderate',
            'trend_alignment': direction
        }
        
    def _assess_correlation_risk_for_trade(self, trade_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess correlation risk for a specific trade against existing positions.
        
        Args:
            trade_parameters: Dict containing trade parameters
            
        Returns:
            Dict with correlation risk assessment
        """
        currency_pair = trade_parameters.get('currency_pair', '')
        
        # Extract current positions from the positions tracker
        open_positions = self.positions.get('open_positions', [])
        
        # Calculate correlation factor
        correlation_factor = self._calculate_correlation_factor(open_positions, currency_pair)
        
        # Convert to correlation score (0-100, where 0 is high correlation risk)
        correlation_score = (1 - correlation_factor) * 100
        
        if correlation_score < 50:
            assessment = 'high correlation'
        elif correlation_score < 75:
            assessment = 'moderate correlation'
        else:
            assessment = 'low correlation'
            
        return {
            'assessment': assessment,
            'correlation_factor': correlation_factor,
            'correlation_score': correlation_score,
            'currency_pair': currency_pair,
            'open_positions_count': len(open_positions)
        }

    # === Risk Controls Methods ===
    
    def check_max_drawdown(
        self, 
        current_drawdown: float, 
        max_allowed: float = None
    ) -> Dict[str, Any]:
        """
        Check if drawdown exceeds maximum allowed.
        
        Verifies if the current drawdown is within acceptable limits and
        recommends actions if limits are breached.
        
        Args:
            current_drawdown: Current drawdown as a decimal (e.g., 0.05 for 5%)
            max_allowed: Maximum allowed drawdown, defaults to config value if None
            
        Returns:
            Dict containing check results and recommended actions
        """
        self.log_action("check_max_drawdown", f"Checking maximum drawdown: {current_drawdown:.2%}")
        
        # Use config value if no max_allowed specified
        if max_allowed is None:
            max_allowed = self.max_daily_drawdown
            
        # Check if drawdown exceeds maximum
        limit_exceeded = current_drawdown > max_allowed
        
        # Calculate how close we are to the limit (0-100%)
        proximity_percentage = (current_drawdown / max_allowed) * 100 if max_allowed > 0 else 0
        
        # Determine status
        if proximity_percentage < 50:
            status = 'safe'
        elif proximity_percentage < 75:
            status = 'caution'
        elif proximity_percentage < 100:
            status = 'warning'
        else:
            status = 'critical'
            
        # Recommend actions based on status
        actions = []
        
        if status == 'warning':
            actions.append('Reduce position sizes by 50%')
            actions.append('Close underperforming positions')
        elif status == 'critical':
            actions.append('Close all positions')
            actions.append('Pause trading for the day')
            actions.append('Review strategy performance')
            
        return {
            'limit_exceeded': limit_exceeded,
            'current_drawdown': current_drawdown,
            'max_allowed': max_allowed,
            'proximity_percentage': proximity_percentage,
            'status': status,
            'recommended_actions': actions
        }
    
    def check_exposure_limits(
        self, 
        current_exposure: Dict[str, float], 
        limits: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Check if exposure exceeds limits.
        
        Verifies if exposure to various currencies, pairs, or asset classes
        is within acceptable limits.
        
        Args:
            current_exposure: Dict mapping categories to exposure amounts 
                             (e.g., {'EUR': 0.3, 'USD': 0.4})
            limits: Dict mapping categories to maximum allowed exposure,
                   defaults to config values if None
                   
        Returns:
            Dict containing check results and recommended actions
        """
        self.log_action("check_exposure_limits", "Checking exposure limits")
        
        # Use config values if no limits specified
        if limits is None:
            limits = {
                'single_currency': self.max_risk_per_currency,
                'single_pair': self.max_risk_per_trade * 2,
                'total': 0.5  # Default 50% max total exposure
            }
            
        # Check each exposure against its limit
        results = {
            'exceeded_limits': [],
            'warnings': [],
            'status': 'safe',
            'recommended_actions': []
        }
        
        for category, exposure in current_exposure.items():
            # Determine the applicable limit
            if category in limits:
                limit = limits[category]
            elif category.startswith('currency_'):
                # Currency exposure (e.g., 'currency_EUR')
                limit = limits.get('single_currency', 0.05)
            elif category.startswith('pair_'):
                # Currency pair exposure (e.g., 'pair_EURUSD')
                limit = limits.get('single_pair', 0.04)
            elif category == 'total':
                limit = limits.get('total', 0.5)
            else:
                # Default limit
                limit = 0.1
                
            # Check if exposure exceeds limit
            if exposure > limit:
                results['exceeded_limits'].append({
                    'category': category,
                    'exposure': exposure,
                    'limit': limit,
                    'excess': exposure - limit
                })
                
            # Check if exposure is close to limit (within 85%)
            elif exposure > limit * 0.85:
                results['warnings'].append({
                    'category': category,
                    'exposure': exposure,
                    'limit': limit,
                    'proximity': (exposure / limit) * 100
                })
        
        # Determine overall status
        if results['exceeded_limits']:
            results['status'] = 'critical'
            
            # Recommend actions
            for exceeded in results['exceeded_limits']:
                category = exceeded['category']
                excess = exceeded['excess']
                
                if category == 'total':
                    results['recommended_actions'].append(f"Reduce total exposure by {excess:.2%}")
                else:
                    results['recommended_actions'].append(f"Reduce {category} exposure by {excess:.2%}")
        elif results['warnings']:
            results['status'] = 'warning'
            results['recommended_actions'].append("Consider reducing exposure in categories approaching limits")
            
        return results
    
    def check_correlation_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for excessive correlation between positions.
        
        Analyzes the correlation between open positions to identify
        potential concentration risks.
        
        Args:
            positions: List of open positions with currency pair information
            
        Returns:
            Dict containing correlation risk assessment and recommendations
        """
        self.log_action("check_correlation_risk", "Checking correlation risk")
        
        if not positions or len(positions) < 2:
            return {
                'correlation_risk': 'none',
                'average_correlation': 0.0,
                'highly_correlated_pairs': [],
                'status': 'safe',
                'recommended_actions': []
            }
            
        # Calculate correlation between all pairs of positions
        correlations = []
        highly_correlated = []
        
        for i, pos1 in enumerate(positions):
            pair1 = pos1.get('currency_pair', '')
            dir1 = 1 if pos1.get('direction', 'long').lower() == 'long' else -1
            
            for j in range(i+1, len(positions)):
                pos2 = positions[j]
                pair2 = pos2.get('currency_pair', '')
                dir2 = 1 if pos2.get('direction', 'long').lower() == 'long' else -1
                
                if not pair1 or not pair2:
                    continue
                    
                # Calculate correlation between the two positions
                pair_correlation = self._calculate_pair_correlation(pair1, pair2)
                
                # Adjust for direction
                if dir1 != dir2:
                    pair_correlation *= -1
                    
                correlations.append(pair_correlation)
                
                # Check if highly correlated
                if abs(pair_correlation) > 0.7:
                    highly_correlated.append({
                        'pair1': pair1,
                        'pair2': pair2,
                        'correlation': pair_correlation,
                        'directions': ('same' if dir1 == dir2 else 'opposite')
                    })
                    
        # Calculate average correlation
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Store in risk metrics
        self.risk_metrics['portfolio_correlation'] = avg_correlation
        
        # Determine risk level
        if avg_correlation > 0.7:
            risk_level = 'very high'
            status = 'critical'
        elif avg_correlation > 0.5:
            risk_level = 'high'
            status = 'warning'
        elif avg_correlation > 0.3:
            risk_level = 'moderate'
            status = 'caution'
        else:
            risk_level = 'low'
            status = 'safe'
            
        # Generate recommendations
        recommendations = []
        
        if status == 'warning' or status == 'critical':
            recommendations.append("Reduce position sizes in correlated pairs")
            
            if len(highly_correlated) > 0:
                recommendations.append(f"Consider closing one position from each highly correlated pair")
                
            recommendations.append("Add positions in uncorrelated or negatively correlated pairs")
            
        return {
            'correlation_risk': risk_level,
            'average_correlation': avg_correlation,
            'highly_correlated_pairs': highly_correlated,
            'status': status,
            'recommended_actions': recommendations
        }
    
    def implement_circuit_breakers(
        self, 
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implement circuit breakers for adverse conditions.
        
        Evaluates performance metrics to determine if trading should be
        paused or restricted due to adverse market conditions.
        
        Args:
            performance_metrics: Dict containing various performance metrics
            
        Returns:
            Dict containing circuit breaker decisions and recommendations
        """
        self.log_action("implement_circuit_breakers", "Implementing circuit breakers")
        
        # Extract key metrics
        daily_pnl = performance_metrics.get('daily_pnl', 0.0)
        daily_pnl_percentage = performance_metrics.get('daily_pnl_percentage', 0.0)
        drawdown = performance_metrics.get('current_drawdown', 0.0)
        win_rate = performance_metrics.get('recent_win_rate', 0.5)
        consecutive_losses = performance_metrics.get('consecutive_losses', 0)
        
        # Initialize circuit breaker results
        circuit_breakers = {
            'trading_paused': False,
            'size_reduction': 0.0,
            'triggered_breakers': [],
            'status': 'normal',
            'recommended_actions': []
        }
        
        # Check daily loss breaker
        daily_loss_threshold = -0.05  # 5% daily loss
        if daily_pnl_percentage < daily_loss_threshold:
            circuit_breakers['triggered_breakers'].append({
                'name': 'daily_loss',
                'threshold': daily_loss_threshold,
                'actual': daily_pnl_percentage,
                'severity': 'high'
            })
            circuit_breakers['trading_paused'] = True
            circuit_breakers['status'] = 'halted'
            circuit_breakers['recommended_actions'].append("Pause trading for the remainder of the day")
            circuit_breakers['recommended_actions'].append("Review all open positions and close if necessary")
            
        # Check drawdown breaker
        drawdown_threshold = self.max_daily_drawdown
        if drawdown > drawdown_threshold:
            circuit_breakers['triggered_breakers'].append({
                'name': 'drawdown',
                'threshold': drawdown_threshold,
                'actual': drawdown,
                'severity': 'high'
            })
            circuit_breakers['trading_paused'] = True
            circuit_breakers['status'] = 'halted'
            circuit_breakers['recommended_actions'].append("Pause trading until drawdown improves")
            
        # Check consecutive losses breaker
        consecutive_loss_threshold = 5
        if consecutive_losses >= consecutive_loss_threshold:
            circuit_breakers['triggered_breakers'].append({
                'name': 'consecutive_losses',
                'threshold': consecutive_loss_threshold,
                'actual': consecutive_losses,
                'severity': 'medium'
            })
            circuit_breakers['size_reduction'] = 0.5  # Reduce size by 50%
            circuit_breakers['status'] = 'reduced'
            circuit_breakers['recommended_actions'].append("Reduce position size by 50%")
            circuit_breakers['recommended_actions'].append("Review strategy performance")
            
        # Check win rate breaker
        win_rate_threshold = 0.35
        if win_rate < win_rate_threshold:
            circuit_breakers['triggered_breakers'].append({
                'name': 'win_rate',
                'threshold': win_rate_threshold,
                'actual': win_rate,
                'severity': 'medium'
            })
            if circuit_breakers['size_reduction'] < 0.3:
                circuit_breakers['size_reduction'] = 0.3  # Reduce size by at least 30%
            circuit_breakers['status'] = 'reduced'
            circuit_breakers['recommended_actions'].append("Reduce position size by 30%")
            circuit_breakers['recommended_actions'].append("Focus on highest probability setups only")
            
        return circuit_breakers
    
    def adjust_risk_for_volatility(
        self, 
        position_size: float, 
        volatility: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Adjust risk based on current market volatility.
        
        Modifies position sizing based on current volatility conditions to
        maintain consistent risk exposure across different market environments.
        
        Args:
            position_size: Original position size
            volatility: Dict containing volatility metrics
            
        Returns:
            Dict containing adjusted position size and volatility assessment
        """
        self.log_action("adjust_risk_for_volatility", "Adjusting risk for volatility")
        
        # Extract volatility metrics
        current_volatility = volatility.get('current', 0.0)
        average_volatility = volatility.get('average', 0.0)
        volatility_ratio = current_volatility / average_volatility if average_volatility > 0 else 1.0
        
        # Determine adjustment factor based on volatility ratio
        if volatility_ratio < 0.7:
            # Lower volatility than normal - can increase size
            adjustment_factor = 1.2
            volatility_assessment = 'low'
        elif volatility_ratio < 0.9:
            # Slightly lower volatility
            adjustment_factor = 1.1
            volatility_assessment = 'below_average'
        elif volatility_ratio < 1.1:
            # Normal volatility
            adjustment_factor = 1.0
            volatility_assessment = 'average'
        elif volatility_ratio < 1.5:
            # Higher than normal volatility
            adjustment_factor = 0.8
            volatility_assessment = 'above_average'
        else:
            # Much higher volatility
            adjustment_factor = 0.5
            volatility_assessment = 'high'
            
        # Calculate adjusted position size
        adjusted_position_size = position_size * adjustment_factor
        
        return {
            'original_position_size': position_size,
            'adjusted_position_size': adjusted_position_size,
            'adjustment_factor': adjustment_factor,
            'volatility_ratio': volatility_ratio,
            'volatility_assessment': volatility_assessment
        }
    
    def _calculate_pair_correlation(self, pair1: str, pair2: str) -> float:
        """
        Calculate correlation between two currency pairs.
        
        In a real system, this would use actual price data to calculate correlation.
        For this example, we use currency composition to estimate correlation.
        
        Args:
            pair1: First currency pair
            pair2: Second currency pair
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Parse the currency pairs
        if '/' in pair1:
            base1, quote1 = pair1.split('/')
        else:
            base1 = pair1[:3]
            quote1 = pair1[3:] if len(pair1) >= 6 else ""
            
        if '/' in pair2:
            base2, quote2 = pair2.split('/')
        else:
            base2 = pair2[:3]
            quote2 = pair2[3:] if len(pair2) >= 6 else ""
            
        # Check for direct relationship
        if pair1 == pair2:
            return 1.0
            
        # Calculate correlation based on shared currencies
        correlation = 0.0
        
        # Same base currency
        if base1 == base2:
            correlation += 0.5
            
        # Same quote currency
        if quote1 == quote2:
            correlation += 0.5
            
        # Opposite pairing (e.g., EUR/USD and USD/CHF)
        if base1 == quote2 or base2 == quote1:
            correlation -= 0.3
            
        # Same currencies but in different positions
        if (base1 == quote2 and quote1 == base2):
            correlation -= 0.8  # Strong negative correlation
            
        # Add some known correlations for common pairs
        known_correlations = {
            ('EUR/USD', 'GBP/USD'): 0.85,
            ('EUR/USD', 'USD/CHF'): -0.9,
            ('EUR/USD', 'AUD/USD'): 0.65,
            ('USD/CAD', 'USD/JPY'): 0.55,
            ('GBP/USD', 'USD/CHF'): -0.8
        }
        
        # Check if we have a known correlation
        if (pair1, pair2) in known_correlations:
            return known_correlations[(pair1, pair2)]
        elif (pair2, pair1) in known_correlations:
            return known_correlations[(pair2, pair1)]
            
        # Ensure correlation is in [-1, 1]
        return max(-1.0, min(1.0, correlation)) 

    # === Decision Making Methods ===
    
    def approve_trade(self, trade_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Approve or reject a proposed trade.
        
        Evaluates trade parameters against risk management rules to determine
        if the trade should be executed, modified, or rejected.
        
        Args:
            trade_parameters: Dict containing trade details including entry, stop loss, 
                             take profit, currency pair, position size, etc.
        
        Returns:
            Dict containing approval decision and recommendations
        """
        self.log_action("approve_trade", f"Evaluating trade approval for {trade_parameters.get('currency_pair', 'unknown')}")
        
        # Evaluate trade risk
        risk_evaluation = self.evaluate_trade_risk(trade_parameters)
        
        # Check if max risk per trade is exceeded
        max_risk_exceeded = risk_evaluation.get('max_risk_exceeded', False)
        
        # Check risk-reward ratio
        risk_reward_ratio = risk_evaluation.get('risk_reward_ratio', 0.0)
        poor_risk_reward = risk_reward_ratio < 1.5
        
        # Check correlation with existing positions
        correlation_risk = risk_evaluation.get('correlation_risk', {})
        high_correlation = correlation_risk.get('assessment', '') == 'high correlation'
        
        # Check market conditions
        market_conditions = risk_evaluation.get('market_conditions', {})
        unfavorable_market = market_conditions.get('assessment', '') == 'unfavorable'
        
        # Check portfolio limits
        currency_pair = trade_parameters.get('currency_pair', '')
        position_size = trade_parameters.get('position_size', 0.0)
        account_balance = trade_parameters.get('account_balance', 0.0)
        
        # Get current portfolio exposure
        portfolio_exposure = self._calculate_current_exposure(currency_pair)
        
        # Initialize decision
        decision = {
            'approved': True,
            'status': 'approved',
            'risk_evaluation': risk_evaluation,
            'modifications_required': [],
            'warnings': [],
            'rejection_reasons': []
        }
        
        # Check for rejection reasons
        if max_risk_exceeded:
            decision['approved'] = False
            decision['status'] = 'rejected'
            decision['rejection_reasons'].append("Maximum risk per trade exceeded")
            
        if poor_risk_reward and risk_reward_ratio < 1.0:
            decision['approved'] = False
            decision['status'] = 'rejected'
            decision['rejection_reasons'].append(f"Risk-reward ratio too low: {risk_reward_ratio:.2f}")
            
        if len(portfolio_exposure.get('positions', [])) >= self.max_open_positions:
            decision['approved'] = False
            decision['status'] = 'rejected'
            decision['rejection_reasons'].append("Maximum number of open positions reached")
            
        # Check for required modifications
        if poor_risk_reward and risk_reward_ratio >= 1.0:
            decision['status'] = 'needs_modification'
            decision['modifications_required'].append("Improve risk-reward ratio to at least 1.5")
            
        if high_correlation:
            decision['status'] = 'needs_modification'
            decision['modifications_required'].append("Reduce position size due to high correlation with existing positions")
            
        if unfavorable_market:
            decision['status'] = 'needs_modification'
            decision['modifications_required'].append("Reduce position size due to unfavorable market conditions")
            
        # Add warnings
        if risk_evaluation.get('risk_level', '') in ['high', 'very high']:
            decision['warnings'].append("High risk trade - consider reducing position size")
            
        # If needs modifications but not rejected, suggest modified parameters
        if decision['status'] == 'needs_modification' and decision['approved']:
            modified_params = self.modify_trade_parameters(trade_parameters)
            decision['modified_parameters'] = modified_params
            
        return decision
    
    def modify_trade_parameters(self, trade_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify parameters to reduce risk.
        
        Adjusts trade parameters (position size, stop loss, take profit) to
        improve risk profile while maintaining the trade's core thesis.
        
        Args:
            trade_parameters: Dict containing original trade parameters
            
        Returns:
            Dict containing modified parameters and explanation
        """
        self.log_action("modify_trade_parameters", "Modifying trade parameters to reduce risk")
        
        # Create a copy of the original parameters
        modified = trade_parameters.copy()
        modifications = []
        
        # Evaluate original risk
        risk_evaluation = self.evaluate_trade_risk(trade_parameters)
        
        # Extract key parameters
        currency_pair = trade_parameters.get('currency_pair', '')
        original_position_size = trade_parameters.get('position_size', 0.0)
        account_balance = trade_parameters.get('account_balance', 0.0)
        entry = trade_parameters.get('entry_price', 0.0)
        stop_loss = trade_parameters.get('stop_loss', 0.0)
        take_profit = trade_parameters.get('take_profit', 0.0)
        
        # Check risk-reward ratio
        risk_reward_ratio = risk_evaluation.get('risk_reward_ratio', 0.0)
        if risk_reward_ratio < 1.5:
            # Improve risk-reward by adjusting take profit
            original_tp = take_profit
            if trade_parameters.get('direction', 'long').lower() == 'long':
                # For long positions, increase take profit
                price_range = abs(entry - stop_loss)
                adjusted_tp = entry + (price_range * 1.6)  # Aim for RR of at least 1.6
                modified['take_profit'] = adjusted_tp
            else:
                # For short positions, decrease take profit
                price_range = abs(entry - stop_loss)
                adjusted_tp = entry - (price_range * 1.6)  # Aim for RR of at least 1.6
                modified['take_profit'] = adjusted_tp
                
            modifications.append(f"Adjusted take profit from {original_tp:.5f} to {modified['take_profit']:.5f}")
            
        # Check position size against risk percentage
        risk_percentage = risk_evaluation.get('risk_percentage', 0.0)
        if risk_percentage > self.max_risk_per_trade:
            # Reduce position size to meet max risk per trade
            original_size = original_position_size
            adjustment_factor = self.max_risk_per_trade / risk_percentage
            adjusted_size = original_size * adjustment_factor
            modified['position_size'] = adjusted_size
            
            modifications.append(f"Reduced position size from {original_size:.2f} to {adjusted_size:.2f} units")
            
        # Check for correlation risk
        correlation_risk = risk_evaluation.get('correlation_risk', {})
        if correlation_risk.get('assessment', '') == 'high correlation':
            # Reduce position size for correlation risk
            if 'position_size' not in modifications:  # Only if not already adjusted
                original_size = modified.get('position_size', original_position_size)
                correlation_factor = correlation_risk.get('correlation_factor', 0.5)
                adjustment_factor = 1 - (correlation_factor * 0.5)  # Max 50% reduction
                adjusted_size = original_size * adjustment_factor
                modified['position_size'] = adjusted_size
                
                modifications.append(f"Reduced position size to {adjusted_size:.2f} units due to correlation risk")
                
        # Check market conditions
        market_conditions = risk_evaluation.get('market_conditions', {})
        if market_conditions.get('assessment', '') == 'unfavorable':
            # Reduce position size for unfavorable market
            if 'position_size' not in modifications:  # Only if not already adjusted
                original_size = modified.get('position_size', original_position_size)
                adjusted_size = original_size * 0.7  # 30% reduction
                modified['position_size'] = adjusted_size
                
                modifications.append(f"Reduced position size to {adjusted_size:.2f} units due to unfavorable market")
                
        # Re-evaluate risk with modified parameters
        modified_risk = self.evaluate_trade_risk(modified)
        
        return {
            'original_parameters': trade_parameters,
            'modified_parameters': modified,
            'modifications': modifications,
            'original_risk': risk_evaluation,
            'modified_risk': modified_risk
        }
    
    def prioritize_trades(self, proposed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prioritize trades based on risk-reward profiles.
        
        Ranks multiple proposed trades according to their risk-adjusted
        expected value and allocates capital efficiently.
        
        Args:
            proposed_trades: List of dicts containing trade parameters
            
        Returns:
            Dict containing prioritized trades with risk-adjusted capital allocation
        """
        self.log_action("prioritize_trades", f"Prioritizing {len(proposed_trades)} proposed trades")
        
        if not proposed_trades:
            return {
                'prioritized_trades': [],
                'rejected_trades': [],
                'total_approved': 0
            }
            
        # Evaluate each trade
        evaluated_trades = []
        for trade in proposed_trades:
            # Evaluate trade risk
            risk_evaluation = self.evaluate_trade_risk(trade)
            
            # Calculate score for prioritization
            expectancy = risk_evaluation.get('expectancy', 0.0)
            risk_reward = risk_evaluation.get('risk_reward_ratio', 0.0)
            correlation_factor = risk_evaluation.get('correlation_risk', {}).get('correlation_factor', 0.0)
            
            # Combine factors for score (higher is better)
            score = (expectancy * 2) + (risk_reward * 0.5) - (correlation_factor * 0.3)
            
            evaluated_trades.append({
                'trade': trade,
                'risk_evaluation': risk_evaluation,
                'score': score,
                'approved': False  # Will be set during approval process
            })
            
        # Sort by score (highest first)
        prioritized = sorted(evaluated_trades, key=lambda x: x['score'], reverse=True)
        
        # Filter out negative expectancy trades
        filtered = [t for t in prioritized if t['risk_evaluation'].get('expectancy', 0.0) > 0]
        
        # Prioritize and allocate capital
        approved_trades = []
        rejected_trades = []
        
        # Calculate total available risk
        account_balance = proposed_trades[0].get('account_balance', 0.0) if proposed_trades else 0.0
        max_total_risk = account_balance * 0.1  # Max 10% total risk
        allocated_risk = 0.0
        
        # Track used currency pairs for correlation management
        used_pairs = set()
        
        for eval_trade in filtered:
            trade = eval_trade['trade']
            risk = eval_trade['risk_evaluation']
            
            # Extract parameters
            currency_pair = trade.get('currency_pair', '')
            monetary_risk = risk.get('monetary_risk', 0.0)
            
            # Check if we'd exceed max risk
            if allocated_risk + monetary_risk > max_total_risk:
                # Reduce position size to fit within limits
                adjustment_factor = (max_total_risk - allocated_risk) / monetary_risk
                if adjustment_factor > 0.2:  # Only approve if we can keep at least 20% of size
                    modified_trade = trade.copy()
                    modified_trade['position_size'] = trade.get('position_size', 0.0) * adjustment_factor
                    modified_risk = self.evaluate_trade_risk(modified_trade)
                    
                    approved_trades.append({
                        'trade': modified_trade,
                        'risk_evaluation': modified_risk,
                        'score': eval_trade['score'],
                        'adjustment': f"Reduced to {adjustment_factor:.1%} of original size"
                    })
                    
                    allocated_risk += modified_risk.get('monetary_risk', 0.0)
                    used_pairs.add(currency_pair)
                else:
                    # Reject if adjustment would be too small
                    rejected_trades.append({
                        'trade': trade,
                        'risk_evaluation': risk,
                        'rejection_reason': "Insufficient risk budget remaining"
                    })
            else:
                # Check correlation with already approved trades
                if any(self._calculate_pair_correlation(currency_pair, p) > 0.7 for p in used_pairs):
                    # High correlation with already approved trade
                    adjustment_factor = 0.5  # 50% reduction for correlated trades
                    modified_trade = trade.copy()
                    modified_trade['position_size'] = trade.get('position_size', 0.0) * adjustment_factor
                    modified_risk = self.evaluate_trade_risk(modified_trade)
                    
                    approved_trades.append({
                        'trade': modified_trade,
                        'risk_evaluation': modified_risk,
                        'score': eval_trade['score'],
                        'adjustment': "Reduced size due to correlation with existing positions"
                    })
                    
                    allocated_risk += modified_risk.get('monetary_risk', 0.0)
                else:
                    # Approve at full size
                    approved_trades.append({
                        'trade': trade,
                        'risk_evaluation': risk,
                        'score': eval_trade['score'],
                        'adjustment': None
                    })
                    
                    allocated_risk += monetary_risk
                    
                used_pairs.add(currency_pair)
                
        # Add remaining trades to rejected list
        for eval_trade in prioritized:
            if eval_trade['score'] <= 0 or eval_trade['risk_evaluation'].get('expectancy', 0.0) <= 0:
                rejected_trades.append({
                    'trade': eval_trade['trade'],
                    'risk_evaluation': eval_trade['risk_evaluation'],
                    'rejection_reason': "Negative expected value"
                })
                
        return {
            'prioritized_trades': approved_trades,
            'rejected_trades': rejected_trades,
            'total_approved': len(approved_trades),
            'total_risk_allocated': allocated_risk,
            'max_total_risk': max_total_risk,
            'account_balance': account_balance
        }
    
    def generate_risk_adjusted_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adjust trading signals based on risk assessment.
        
        Modifies raw trading signals from other agents to incorporate
        risk management considerations.
        
        Args:
            signals: List of dicts containing trading signals
            
        Returns:
            Dict containing risk-adjusted signals and explanation
        """
        self.log_action("generate_risk_adjusted_signals", f"Adjusting {len(signals)} trading signals")
        
        if not signals:
            return {
                'adjusted_signals': [],
                'rejected_signals': [],
                'adjustments_made': []
            }
            
        # Initialize results
        adjusted_signals = []
        rejected_signals = []
        adjustments_made = []
        
        # Get account balance (assume all signals have the same account balance)
        account_balance = signals[0].get('account_balance', 0.0) if signals else 0.0
        
        # Get current market volatility
        market_volatility = self._get_current_market_volatility()
        
        # Get current exposure
        current_exposure = self._calculate_current_exposure()
        
        # Process each signal
        for signal in signals:
            # Extract signal details
            currency_pair = signal.get('currency_pair', '')
            direction = signal.get('direction', 'long')
            confidence = signal.get('confidence', 0.5)
            source = signal.get('source', 'unknown')
            
            # Skip if missing critical information
            if not currency_pair:
                rejected_signals.append({
                    'signal': signal,
                    'reason': "Missing currency pair"
                })
                continue
                
            # Convert signal to trade parameters
            trade_params = self._convert_signal_to_trade_parameters(signal, account_balance)
            
            # Evaluate trade risk
            risk_evaluation = self.evaluate_trade_risk(trade_params)
            
            # Check if signal should be rejected
            reject_signal = False
            rejection_reason = ""
            
            if risk_evaluation.get('max_risk_exceeded', False):
                reject_signal = True
                rejection_reason = "Maximum risk per trade exceeded"
                
            elif risk_evaluation.get('risk_reward_ratio', 0.0) < 1.0:
                reject_signal = True
                rejection_reason = "Risk-reward ratio too low"
                
            elif current_exposure.get('max_positions_reached', False):
                reject_signal = True
                rejection_reason = "Maximum number of open positions reached"
                
            # If rejected, add to rejected list
            if reject_signal:
                rejected_signals.append({
                    'signal': signal,
                    'risk_evaluation': risk_evaluation,
                    'reason': rejection_reason
                })
                continue
                
            # If not rejected, adjust the signal
            adjusted_signal = signal.copy()
            
            # Adjust for correlation
            correlation_risk = risk_evaluation.get('correlation_risk', {})
            if correlation_risk.get('assessment', '') == 'high correlation':
                original_size = trade_params.get('position_size', 0.0)
                adjusted_size = original_size * 0.7  # 30% reduction
                adjusted_signal['position_size'] = adjusted_size
                adjustments_made.append(f"Reduced {currency_pair} position size due to correlation")
                
            # Adjust for market conditions
            market_conditions = risk_evaluation.get('market_conditions', {})
            if market_conditions.get('assessment', '') == 'unfavorable':
                # If already adjusted, reduce further
                original_size = adjusted_signal.get('position_size', trade_params.get('position_size', 0.0))
                adjusted_size = original_size * 0.8  # 20% reduction
                adjusted_signal['position_size'] = adjusted_size
                adjustments_made.append(f"Reduced {currency_pair} position size due to unfavorable market")
                
            # Adjust for volatility
            volatility_adjustment = self.adjust_risk_for_volatility(
                adjusted_signal.get('position_size', 0.0),
                market_volatility
            )
            
            adjusted_signal['position_size'] = volatility_adjustment.get('adjusted_position_size', 0.0)
            adjustments_made.append(
                f"Adjusted {currency_pair} position size for {volatility_adjustment.get('volatility_assessment', 'normal')} volatility"
            )
            
            # Add risk metrics to signal
            adjusted_signal['risk_evaluation'] = risk_evaluation
            adjusted_signal['risk_adjusted_expectancy'] = risk_evaluation.get('expectancy', 0.0)
            
            # Add to adjusted signals
            adjusted_signals.append(adjusted_signal)
            
        # Sort by expectancy (highest first)
        sorted_signals = sorted(
            adjusted_signals, 
            key=lambda x: x.get('risk_adjusted_expectancy', 0.0),
            reverse=True
        )
        
        return {
            'adjusted_signals': sorted_signals,
            'rejected_signals': rejected_signals,
            'adjustments_made': adjustments_made,
            'total_signals': len(signals),
            'accepted_signals': len(adjusted_signals),
            'rejected_count': len(rejected_signals)
        }
    
    def _calculate_current_exposure(self, new_pair: str = None) -> Dict[str, Any]:
        """
        Calculate current portfolio exposure.
        
        Args:
            new_pair: Optional new pair to include in calculation
            
        Returns:
            Dict containing exposure metrics
        """
        # Get open positions
        open_positions = self.positions.get('open_positions', [])
        
        # Calculate exposure by currency and pair
        exposure = {
            'total': 0.0,
            'currencies': {},
            'pairs': {},
            'positions': open_positions,
            'count': len(open_positions),
            'max_positions_reached': len(open_positions) >= self.max_open_positions
        }
        
        # Process each position
        for position in open_positions:
            position_size = position.get('position_size', 0.0)
            pair = position.get('currency_pair', '')
            
            if not pair:
                continue
                
            # Parse the currency pair
            if '/' in pair:
                base, quote = pair.split('/')
            else:
                base = pair[:3]
                quote = pair[3:] if len(pair) >= 6 else ""
                
            # Add to total exposure
            exposure['total'] += position_size
            
            # Add to currency exposure
            exposure['currencies'][base] = exposure['currencies'].get(base, 0.0) + position_size
            exposure['currencies'][quote] = exposure['currencies'].get(quote, 0.0) + position_size
            
            # Add to pair exposure
            exposure['pairs'][pair] = exposure['pairs'].get(pair, 0.0) + position_size
            
        return exposure
    
    def _get_current_market_volatility(self) -> Dict[str, float]:
        """
        Get current market volatility metrics.
        
        In a real system, this would calculate from market data.
        For this example, we return mock data.
        
        Returns:
            Dict containing volatility metrics
        """
        return {
            'current': 0.008,  # Current daily volatility
            'average': 0.007,  # Average daily volatility
            'percentile': 65,   # Current volatility percentile
            'trend': 'increasing'  # Volatility trend
        }
        
    def _convert_signal_to_trade_parameters(
        self, 
        signal: Dict[str, Any], 
        account_balance: float
    ) -> Dict[str, Any]:
        """
        Convert trading signal to trade parameters.
        
        Args:
            signal: Trading signal
            account_balance: Account balance
            
        Returns:
            Dict containing trade parameters
        """
        # Extract signal details
        currency_pair = signal.get('currency_pair', '')
        direction = signal.get('direction', 'long')
        
        # Get current price (mock)
        current_price = 1.1000  # Mock price
        
        # Calculate stop loss and take profit
        pip_size = 0.0001  # For most pairs
        if 'JPY' in currency_pair:
            pip_size = 0.01
            
        # Default to 50 pip stop loss and 1.5 risk-reward ratio
        stop_loss_pips = signal.get('stop_loss_pips', 50)
        target_pips = signal.get('target_pips', stop_loss_pips * 1.5)
        
        if direction.lower() == 'long':
            entry_price = current_price
            stop_loss_price = entry_price - (stop_loss_pips * pip_size)
            take_profit_price = entry_price + (target_pips * pip_size)
        else:
            entry_price = current_price
            stop_loss_price = entry_price + (stop_loss_pips * pip_size)
            take_profit_price = entry_price - (target_pips * pip_size)
            
        # Calculate position size
        risk_percentage = 0.01  # 1% risk per trade
        
        return {
            'currency_pair': currency_pair,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'account_balance': account_balance,
            'risk_percentage': risk_percentage,
            'stop_loss_pips': stop_loss_pips,
            'position_size': signal.get('position_size', 0.0),
            'signal_source': signal.get('source', 'unknown'),
            'signal_confidence': signal.get('confidence', 0.5)
        }
    
    # === Reporting Methods ===
    
    def generate_risk_report(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report.
        
        Creates a detailed risk assessment of the current portfolio,
        including VaR, exposure, correlation, and drawdown metrics.
        
        Args:
            positions: List of open positions with exposure details
            
        Returns:
            Dict containing comprehensive risk report
        """
        self.log_action("generate_risk_report", "Generating comprehensive risk report")
        
        # Calculate basic portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(positions)
        
        # Calculate Value at Risk
        var_results = self.calculate_portfolio_var(positions)
        
        # Calculate Expected Shortfall
        es_results = self.calculate_expected_shortfall(positions)
        
        # Check correlation risk
        correlation_results = self.check_correlation_risk(positions)
        
        # Calculate exposure by currency
        exposure_by_currency = self._calculate_exposure_by_currency(positions)
        
        # Check exposure limits
        exposure_limits_check = self.check_exposure_limits(exposure_by_currency)
        
        # Generate report summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'positions_count': len(positions),
            'total_exposure': portfolio_metrics.get('total_exposure', 0.0),
            'var_95': var_results.get('var_amount', 0.0),
            'expected_shortfall_95': es_results.get('expected_shortfall', 0.0),
            'correlation_risk': correlation_results.get('correlation_risk', 'low'),
            'exposure_status': exposure_limits_check.get('status', 'safe'),
            'risk_level': self._determine_overall_risk_level(
                var_results.get('var_percentage', 0.0),
                correlation_results.get('average_correlation', 0.0),
                exposure_limits_check.get('status', 'safe')
            )
        }
        
        # Generate warnings
        warnings = []
        if summary['risk_level'] in ['high', 'very high']:
            warnings.append("Overall portfolio risk level is elevated")
            
        if correlation_results.get('status', 'safe') in ['warning', 'critical']:
            warnings.append("High correlation detected between positions")
            
        if exposure_limits_check.get('status', 'safe') in ['warning', 'critical']:
            warnings.append("Exposure limits approached or exceeded")
            
        if var_results.get('var_percentage', 0.0) > 0.05:
            warnings.append("Value at Risk (95%) exceeds 5% of portfolio")
            
        # Combine everything into final report
        report = {
            'summary': summary,
            'var_analysis': var_results,
            'expected_shortfall': es_results,
            'correlation_analysis': correlation_results,
            'exposure_analysis': {
                'by_currency': exposure_by_currency,
                'limits_check': exposure_limits_check
            },
            'portfolio_metrics': portfolio_metrics,
            'warnings': warnings,
            'recommendations': exposure_limits_check.get('recommended_actions', []) + 
                               correlation_results.get('recommended_actions', [])
        }
        
        # Log the report generation
        self.log_risk_events({
            'event': 'risk_report_generated',
            'risk_level': summary['risk_level'],
            'warnings_count': len(warnings)
        })
        
        return report
    
    def log_risk_events(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log risk-related events.
        
        Records significant risk-related events for monitoring,
        compliance, and historical analysis.
        
        Args:
            event: Dict containing event details
            
        Returns:
            Dict containing logged event with additional metadata
        """
        self.log_action("log_risk_events", f"Logging risk event: {event.get('event', 'unknown')}")
        
        # Add timestamp and metadata
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name,
            **event
        }
        
        # Get risk events file path
        log_file = os.path.join(self._get_data_dir(), 'risk_events.json')
        
        try:
            # Load existing events
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    events = json.load(f)
            else:
                events = []
                
            # Add new event
            events.append(log_entry)
            
            # Save updated events
            with open(log_file, 'w') as f:
                json.dump(events, f, indent=2)
                
            return {
                'status': 'success',
                'logged_event': log_entry,
                'total_events': len(events)
            }
        except Exception as e:
            self.handle_error(e)
            return {
                'status': 'error',
                'error': str(e),
                'logged_event': log_entry
            }
    
    def track_risk_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track key risk metrics over time.
        
        Monitors and stores important risk indicators to identify
        trends and changes in the risk profile.
        
        Args:
            metrics: Dict containing risk metrics to track
            
        Returns:
            Dict containing tracking status and historical data
        """
        self.log_action("track_risk_metrics", "Tracking risk metrics")
        
        # Add timestamp
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Get metrics file path
        metrics_file = os.path.join(self._get_data_dir(), 'risk_metrics.json')
        
        try:
            # Load existing metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    historical_metrics = json.load(f)
            else:
                historical_metrics = []
                
            # Add new metrics
            historical_metrics.append(metrics_entry)
            
            # If we have too many entries, remove oldest
            max_entries = 1000
            if len(historical_metrics) > max_entries:
                historical_metrics = historical_metrics[-max_entries:]
                
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(historical_metrics, f, indent=2)
                
            # Calculate trends
            trends = self._calculate_metric_trends(historical_metrics)
                
            return {
                'status': 'success',
                'current_metrics': metrics_entry,
                'trends': trends,
                'historical_count': len(historical_metrics)
            }
        except Exception as e:
            self.handle_error(e)
            return {
                'status': 'error',
                'error': str(e),
                'current_metrics': metrics_entry
            }
    
    def alert_risk_threshold_breach(
        self, 
        metric: str, 
        threshold: float, 
        value: float, 
        severity: str = 'warning'
    ) -> Dict[str, Any]:
        """
        Generate alerts for risk threshold breaches.
        
        Creates notifications when risk metrics exceed predefined
        thresholds, allowing for timely intervention.
        
        Args:
            metric: Name of the risk metric that exceeded threshold
            threshold: Threshold value that was breached
            value: Current value of the metric
            severity: Severity level ('info', 'warning', 'critical')
            
        Returns:
            Dict containing alert details
        """
        self.log_action("alert_risk_threshold_breach", f"Alert: {metric} exceeded threshold ({value} > {threshold})")
        
        # Create alert object
        alert = {
            'metric': metric,
            'threshold': threshold,
            'current_value': value,
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'breach_percentage': (value / threshold - 1) * 100 if threshold != 0 else float('inf')
        }
        
        # Save alert to log
        self.log_risk_events({
            'event': 'threshold_breach',
            'metric': metric,
            'threshold': threshold,
            'value': value,
            'severity': severity
        })
        
        # Recommend actions based on severity and metric
        actions = self._get_alert_actions(metric, severity, alert['breach_percentage'])
        alert['recommended_actions'] = actions
        
        return alert
    
    def _calculate_portfolio_metrics(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate basic portfolio metrics.
        
        Args:
            positions: List of positions
            
        Returns:
            Dict containing portfolio metrics
        """
        if not positions:
            return {
                'total_exposure': 0.0,
                'long_exposure': 0.0,
                'short_exposure': 0.0,
                'net_exposure': 0.0,
                'positions_count': 0
            }
            
        total_exposure = 0.0
        long_exposure = 0.0
        short_exposure = 0.0
        
        for position in positions:
            exposure = position.get('exposure', 0.0)
            direction = position.get('direction', 'long').lower()
            
            total_exposure += abs(exposure)
            
            if direction == 'long':
                long_exposure += exposure
            else:
                short_exposure += exposure
                
        return {
            'total_exposure': total_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'positions_count': len(positions)
        }
    
    def _calculate_exposure_by_currency(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate exposure by currency.
        
        Args:
            positions: List of positions
            
        Returns:
            Dict mapping currencies to exposure amounts
        """
        exposure = {}
        
        for position in positions:
            pair = position.get('currency_pair', '')
            exposure_amount = position.get('exposure', 0.0)
            
            if not pair or exposure_amount == 0:
                continue
                
            # Parse the currency pair
            if '/' in pair:
                base, quote = pair.split('/')
            else:
                base = pair[:3]
                quote = pair[3:] if len(pair) >= 6 else ""
                
            # Add to currency exposure
            exposure[base] = exposure.get(base, 0.0) + exposure_amount
            exposure[quote] = exposure.get(quote, 0.0) + exposure_amount
            
            # Add to pair exposure
            exposure[f"pair_{pair}"] = exposure.get(f"pair_{pair}", 0.0) + exposure_amount
            
        # Add total exposure
        exposure['total'] = sum(exposure_amount for currency, exposure_amount in exposure.items() 
                               if not currency.startswith('pair_'))
            
        return exposure
    
    def _determine_overall_risk_level(
        self, 
        var_percentage: float, 
        correlation: float, 
        exposure_status: str
    ) -> str:
        """
        Determine overall portfolio risk level.
        
        Args:
            var_percentage: Value at Risk as percentage
            correlation: Average correlation between positions
            exposure_status: Status of exposure limits check
            
        Returns:
            Risk level as string
        """
        # Assign scores to each factor (0-10, higher = more risky)
        var_score = min(10, var_percentage * 200)  # 5% VaR = max score
        
        correlation_score = correlation * 10  # 1.0 correlation = max score
        
        exposure_scores = {
            'safe': 0,
            'caution': 3,
            'warning': 7,
            'critical': 10
        }
        exposure_score = exposure_scores.get(exposure_status, 0)
        
        # Calculate weighted average score
        total_score = (var_score * 0.4) + (correlation_score * 0.3) + (exposure_score * 0.3)
        
        # Determine risk level
        if total_score < 2:
            return 'very low'
        elif total_score < 4:
            return 'low'
        elif total_score < 6:
            return 'moderate'
        elif total_score < 8:
            return 'high'
        else:
            return 'very high'
    
    def _calculate_metric_trends(self, historical_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trends in risk metrics.
        
        Args:
            historical_metrics: List of historical metric readings
            
        Returns:
            Dict containing trend analysis
        """
        if not historical_metrics or len(historical_metrics) < 2:
            return {}
            
        # Get common metrics
        common_metrics = set(historical_metrics[0].keys())
        for metrics in historical_metrics[1:]:
            common_metrics &= set(metrics.keys())
            
        # Exclude non-numeric and timestamp fields
        numeric_metrics = []
        for metric in common_metrics:
            if metric == 'timestamp':
                continue
                
            try:
                float(historical_metrics[0][metric])
                numeric_metrics.append(metric)
            except (ValueError, TypeError):
                continue
                
        # Calculate trends
        trends = {}
        
        for metric in numeric_metrics:
            # Extract values
            values = [entry[metric] for entry in historical_metrics if metric in entry]
            
            if not values or len(values) < 2:
                continue
                
            # Calculate changes
            changes = [values[i] - values[i-1] for i in range(1, len(values))]
            
            # Calculate trend metrics
            last_value = values[-1]
            last_change = changes[-1] if changes else 0
            avg_change = sum(changes) / len(changes) if changes else 0
            
            # Determine trend direction
            if avg_change > 0.01 * abs(last_value):
                direction = 'increasing'
            elif avg_change < -0.01 * abs(last_value):
                direction = 'decreasing'
            else:
                direction = 'stable'
                
            # Add to trends
            trends[metric] = {
                'current': last_value,
                'previous': values[-2] if len(values) > 1 else last_value,
                'change': last_change,
                'avg_change': avg_change,
                'direction': direction,
                'samples': len(values)
            }
            
        return trends
    
    def _get_alert_actions(self, metric: str, severity: str, breach_percentage: float) -> List[str]:
        """
        Get recommended actions for an alert.
        
        Args:
            metric: Alert metric
            severity: Alert severity
            breach_percentage: Percentage by which threshold was breached
            
        Returns:
            List of recommended actions
        """
        actions = []
        
        # Add actions based on severity
        if severity == 'critical':
            actions.append("Pause all trading immediately")
            actions.append("Close highest risk positions")
            
        elif severity == 'warning':
            actions.append("Reduce position sizes for new trades")
            actions.append("Review highest risk positions")
            
        # Add metric-specific actions
        if metric == 'var' or metric == 'portfolio_var':
            if severity == 'critical':
                actions.append("Reduce total exposure by 50%")
            elif severity == 'warning':
                actions.append("Reduce total exposure by 25%")
                
        elif metric == 'correlation' or metric == 'portfolio_correlation':
            actions.append("Add diversifying positions")
            actions.append("Reduce size of correlated positions")
            
        elif metric == 'drawdown' or metric == 'current_drawdown':
            actions.append("Review and adjust strategy parameters")
            
        elif 'exposure' in metric.lower():
            currency = metric.split('_')[-1] if '_' in metric else ''
            if currency:
                actions.append(f"Reduce {currency} exposure")
                
        return actions
        
    # === LangGraph Node Implementation ===
    
    def setup_node(self) -> Callable:
        """
        Set up the LangGraph node for this agent.
        
        Returns:
            Callable: Risk manager node function
        """
        def risk_manager_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """LangGraph node implementation for the risk manager agent."""
            # Extract the task from the state
            task = state.get('task', {})
            
            # Extract parameters
            action = task.get('action', '')
            params = task.get('parameters', {})
            
            result = None
            
            try:
                # Route to the appropriate method based on the action
                if action == 'calculate_position_size':
                    result = self.calculate_position_size(
                        params.get('account_balance', 0),
                        params.get('risk_percentage', 0.01),
                        params.get('stop_loss_pips', 0),
                        params.get('currency_pair', '')
                    )
                    
                elif action == 'adjust_for_correlation':
                    result = self.adjust_for_correlation(
                        params.get('positions', []),
                        params.get('new_position', {})
                    )
                    
                elif action == 'calculate_portfolio_var':
                    result = self.calculate_portfolio_var(
                        params.get('positions', []),
                        params.get('confidence_level', 0.95)
                    )
                    
                elif action == 'calculate_expected_shortfall':
                    result = self.calculate_expected_shortfall(
                        params.get('positions', []),
                        params.get('confidence_level', 0.95)
                    )
                    
                elif action == 'check_max_drawdown':
                    result = self.check_max_drawdown(
                        params.get('current_drawdown', 0.0),
                        params.get('max_allowed', None)
                    )
                    
                elif action == 'approve_trade':
                    result = self.approve_trade(params.get('trade_parameters', {}))
                    
                elif action == 'prioritize_trades':
                    result = self.prioritize_trades(params.get('proposed_trades', []))
                    
                elif action == 'generate_risk_report':
                    result = self.generate_risk_report(params.get('positions', []))
                    
                elif action == 'generate_risk_adjusted_signals':
                    result = self.generate_risk_adjusted_signals(params.get('signals', []))
                    
                elif action == 'alert_risk_threshold_breach':
                    result = self.alert_risk_threshold_breach(
                        params.get('metric', ''),
                        params.get('threshold', 0.0),
                        params.get('value', 0.0),
                        params.get('severity', 'warning')
                    )
                    
                else:
                    result = {
                        'status': 'error',
                        'message': f"Unknown action: {action}"
                    }
                    
            except Exception as e:
                result = {
                    'status': 'error',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
                
            # Update the state with the result
            state['result'] = result
            
            return state
            
        return risk_manager_node