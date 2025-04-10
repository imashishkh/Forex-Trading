#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the PortfolioManagerAgent
"""

import os
from pprint import pprint
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Mock LLM class to avoid OpenAI dependencies
class MockLLM:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "mock-llm")
        self.temperature = kwargs.get("temperature", 0.0)
        
    def predict(self, prompt: str) -> str:
        return f"Response to: {prompt[:30]}..."
        
    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {
            "content": "This is a mock response from the LLM",
            "role": "assistant"
        }

# Import the PortfolioManagerAgent
from portfolio_manager_agent import PortfolioManagerAgent

# Create a test subclass that implements the required abstract method
class TestPortfolioManager(PortfolioManagerAgent):
    """
    Test implementation of the PortfolioManagerAgent with the required abstract methods implemented.
    """
    
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement the required run_task abstract method.
        
        Args:
            task: Task description and parameters
            
        Returns:
            Dict[str, Any]: Task execution results
        """
        self.log_action("run_task", f"Running task: {task.get('type', 'Unknown')}")
        
        # Update state tracking
        self.state["tasks"].append(task)
        self.last_active = datetime.now()
        self.state["last_active"] = self.last_active
        
        # Task dispatcher
        task_type = task.get('type', '')
        params = task.get('parameters', {})
        
        if task_type == 'execute_trade':
            result = self.execute_trade(
                instrument=params.get('instrument', ''),
                units=params.get('units', 0),
                side=params.get('side', 'buy'),
                order_type=params.get('order_type', 'market'),
                price=params.get('price'),
                stop_loss=params.get('stop_loss'),
                take_profit=params.get('take_profit')
            )
        elif task_type == 'get_open_positions':
            result = self.get_open_positions()
        elif task_type == 'get_account_summary':
            result = self.get_account_summary()
        elif task_type == 'close_position':
            result = self.close_position(params.get('position_id', ''))
        else:
            result = {
                'success': False,
                'error': f'Unknown task type: {task_type}'
            }
        
        return {
            'status': 'success' if result.get('success', False) else 'error',
            'task_type': task_type,
            'result': result
        }
    
    def log_trade(self, trade_details: Dict[str, Any]) -> None:
        """
        Mock implementation of log_trade for testing.
        
        Args:
            trade_details: Dict containing trade details to log
        """
        self.log_action("log_trade", f"Mock logging trade for {trade_details.get('instrument')}")
        
        # Just print the trade details instead of saving to a file
        print(f"\nTRADE LOGGED: {trade_details}")
        
        # We don't need to actually save to CSV in our test implementation

def main():
    """Main test function for PortfolioManagerAgent"""
    print("\n=== PORTFOLIO MANAGER AGENT TEST ===\n")
    
    # Load environment variables
    load_dotenv()
    
    # Determine OANDA environment
    oanda_url = os.getenv('OANDA_API_URL', 'https://api-fxpractice.oanda.com')
    # OANDA API requires 'practice' or 'live' as the environment parameter, not a URL
    oanda_env = 'practice' if 'practice' in oanda_url else 'live'
    
    # Create agent configuration
    config = {
        'api_credentials': {
            'oanda': {
                'api_key': os.getenv('OANDA_API_KEY'),
                'account_id': os.getenv('OANDA_ACCOUNT_ID'),
                'api_url': oanda_env  # Use 'practice' or 'live' instead of URL
            }
        },
        'system': {
            'data_storage_path': 'data',
            'cache_enabled': True
        },
        'trading': {
            'default_stop_loss': 0.5,  # 0.5% stop loss
            'default_take_profit': 1.0  # 1.0% take profit
        }
    }
    
    # Check if we have the required environment variables
    if not all([os.getenv('OANDA_API_KEY'), os.getenv('OANDA_ACCOUNT_ID')]):
        print("ERROR: Missing required environment variables OANDA_API_KEY and/or OANDA_ACCOUNT_ID")
        print("Please set these in your .env file")
        return
    
    # Create a mock LLM
    mock_llm = MockLLM(model="mock-gpt-3.5", temperature=0.1)
    
    # Initialize the agent
    print("Initializing Portfolio Manager Agent...")
    agent = TestPortfolioManager(
        agent_name="test_portfolio_manager",
        llm=mock_llm,
        config=config
    )
    
    # Initialize the agent
    success = agent.initialize()
    if not success:
        print("Failed to initialize Portfolio Manager Agent")
        return
    print("Agent initialized successfully")
    
    # Get account summary
    print("\n--- ACCOUNT SUMMARY ---\n")
    account_summary = agent.get_account_summary()
    if account_summary.get('success', False):
        summary = account_summary.get('summary', {})
        print(f"Account Balance: ${summary.get('balance', 0):.2f}")
        print(f"Unrealized P/L: ${summary.get('unrealized_pl', 0):.2f}")
        print(f"Open Trades: {summary.get('open_trade_count', 0)}")
        print(f"Margin Available: ${summary.get('margin_available', 0):.2f}")
    else:
        print("Failed to get account summary:", account_summary.get('error', 'Unknown error'))
    
    # Get open positions
    print("\n--- OPEN POSITIONS ---\n")
    positions = agent.get_open_positions()
    if positions.get('success', False):
        if positions.get('count', 0) > 0:
            print(f"Found {positions.get('count')} open positions:")
            for position in positions.get('positions', []):
                direction = position.get('direction', 'unknown')
                units = position.get('units', 0)
                instrument = position.get('instrument', 'unknown')
                pl = position.get('unrealized_pl', 0)
                print(f"  {direction.upper()} {units} units of {instrument} (P/L: ${pl:.2f})")
        else:
            print("No open positions found")
    else:
        print("Failed to get open positions:", positions.get('error', 'Unknown error'))
    
    # Execute a market order
    print("\n--- EXECUTE MARKET ORDER ---\n")
    
    # Prompt for confirmation
    response = input("Do you want to execute a real market order? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Market order execution skipped")
    else:
        # Execute a small market order for EUR/USD
        instrument = "EUR_USD"
        units = 100  # Small position size
        side = "buy"
        order_type = "market"
        
        print(f"Executing {side} {order_type} order for {instrument}: {units} units")
        
        result = agent.execute_trade(
            instrument=instrument,
            units=units,
            side=side,
            order_type=order_type
        )
        
        if result.get('success', False):
            print("Order executed successfully:")
            print(f"Trade ID: {result.get('trade_id')}")
            print(f"Order ID: {result.get('order_id')}")
        else:
            print("Failed to execute order:")
            print(result.get('error', 'Unknown error'))
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 