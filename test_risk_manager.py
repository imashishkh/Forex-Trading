#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the RiskManagerAgent class
"""

import os
from dotenv import load_dotenv
from pprint import pprint
from typing import Any, Dict, List, Optional

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

# Import the RiskManagerAgent
from risk_manager_agent.agent import RiskManagerAgent

# Load environment variables
load_dotenv()

def main():
    """Main test function for RiskManagerAgent"""
    print("\n=== RISK MANAGER AGENT TEST ===\n")
    
    # Initialize the agent
    config = {
        'max_risk_per_trade': 0.02,      # 2% of account 
        'max_open_positions': 5,
        'max_risk_per_currency': 0.05,   # 5% of account
        'max_daily_drawdown': 0.05,      # 5% of account
        'max_portfolio_var': 0.03        # 3% Value at Risk
    }
    
    # Create a mock LLM
    mock_llm = MockLLM(model="mock-gpt-3.5", temperature=0.1)
    
    agent = RiskManagerAgent(
        agent_name="test_risk_manager",
        llm=mock_llm,
        config=config
    )
    
    # Initialize the agent
    print("\nInitializing RiskManagerAgent...")
    success = agent.initialize()
    if not success:
        print("Failed to initialize RiskManagerAgent")
        return
    print("Agent initialized successfully")
    
    # Test position sizing
    print("\n--- POSITION SIZING TEST ---\n")
    
    account_balance = 10000.0
    risk_percentage = 0.01  # 1% risk
    stop_loss_pips = 50
    currency_pair = "EUR/USD"
    
    position_size = agent.calculate_position_size(
        account_balance,
        risk_percentage,
        stop_loss_pips,
        currency_pair
    )
    
    print(f"Position sizing for {currency_pair} with {risk_percentage:.1%} risk on ${account_balance}:")
    pprint(position_size)
    
    # Test tiered position sizing
    print("\n--- TIERED POSITION SIZING TEST ---\n")
    
    risk_levels = [
        {'name': 'conservative', 'risk_percentage': 0.005, 'stop_loss_pips': 50},
        {'name': 'moderate', 'risk_percentage': 0.01, 'stop_loss_pips': 40},
        {'name': 'aggressive', 'risk_percentage': 0.02, 'stop_loss_pips': 30}
    ]
    
    tiered_sizes = agent.calculate_tiered_position_sizes(account_balance, risk_levels)
    
    print(f"Tiered position sizes for ${account_balance}:")
    print(f"\nPosition sizes for EUR/USD:")
    for position in tiered_sizes['tiered_position_sizes']['EUR/USD']:
        print(f"  {position['risk_level']}: {position['lots']:.2f} lots ({position['position_size']:.0f} units)")
    
    # Test risk assessment
    print("\n--- RISK ASSESSMENT TEST ---\n")
    
    # Create mock positions for portfolio risk assessment
    positions = [
        {
            'currency_pair': 'EUR/USD',
            'direction': 'long',
            'exposure': 3000.0,
            'position_size': 0.3  # 0.3 lots
        },
        {
            'currency_pair': 'GBP/USD',
            'direction': 'long',
            'exposure': 2000.0,
            'position_size': 0.2  # 0.2 lots
        },
        {
            'currency_pair': 'USD/JPY',
            'direction': 'short',
            'exposure': 1500.0,
            'position_size': 0.15  # 0.15 lots
        }
    ]
    
    # Calculate portfolio VaR
    var_results = agent.calculate_portfolio_var(positions)
    print("\nPortfolio Value at Risk (95% confidence):")
    print(f"VaR: ${var_results['var_amount']:.2f} ({var_results['var_percentage']:.2%} of portfolio)")
    
    # Calculate Expected Shortfall
    es_results = agent.calculate_expected_shortfall(positions)
    print("\nPortfolio Expected Shortfall (95% confidence):")
    print(f"ES: ${es_results['expected_shortfall']:.2f}")
    if 'expected_shortfall_percentage' in es_results:
        print(f"ES Percentage: {es_results['expected_shortfall_percentage']:.2%} of portfolio")
    
    # Check correlation risk
    correlation_results = agent.check_correlation_risk(positions)
    print("\nPortfolio Correlation Risk:")
    print(f"Risk Level: {correlation_results['correlation_risk']}")
    print(f"Average Correlation: {correlation_results['average_correlation']:.2f}")
    if correlation_results['highly_correlated_pairs']:
        print("\nHighly Correlated Pairs:")
        for pair in correlation_results['highly_correlated_pairs']:
            print(f"  {pair['pair1']} and {pair['pair2']}: {pair['correlation']:.2f}")
    
    # Test trade evaluation
    print("\n--- TRADE EVALUATION TEST ---\n")
    
    # Create a trade to evaluate
    trade = {
        'currency_pair': 'USD/JPY',
        'direction': 'long',
        'entry_price': 110.50,
        'stop_loss': 110.00,
        'take_profit': 111.50,
        'position_size': 0.2,  # 0.2 lots
        'account_balance': account_balance
    }
    
    # Calculate risk-reward ratio
    rr_ratio = agent.calculate_risk_reward_ratio(
        trade['entry_price'],
        trade['stop_loss'],
        trade['take_profit']
    )
    
    print("\nRisk-Reward Analysis:")
    print(f"Risk-Reward Ratio: {rr_ratio['risk_reward_ratio']:.2f}")
    print(f"Evaluation: {rr_ratio['evaluation']}")
    
    # Evaluate trade risk
    risk_evaluation = agent.evaluate_trade_risk(trade)
    
    print("\nTrade Risk Evaluation:")
    print(f"Risk Level: {risk_evaluation['risk_level']}")
    print(f"Risk Score: {risk_evaluation['risk_score']:.1f}/100")
    print(f"Monetary Risk: ${risk_evaluation['monetary_risk']:.2f}")
    print(f"Risk Percentage: {risk_evaluation['risk_percentage']:.2%}")
    print(f"Win Probability: {risk_evaluation['win_probability']:.2%}")
    print(f"Expectancy: {risk_evaluation['expectancy']:.2f}")
    
    # Approve trade
    approval = agent.approve_trade(trade)
    
    print("\nTrade Approval Decision:")
    print(f"Approved: {approval['approved']}")
    print(f"Status: {approval['status']}")
    
    if approval.get('warnings', []):
        print("\nWarnings:")
        for warning in approval['warnings']:
            print(f"  - {warning}")
            
    if approval.get('rejection_reasons', []):
        print("\nRejection Reasons:")
        for reason in approval['rejection_reasons']:
            print(f"  - {reason}")
            
    if approval.get('modified_parameters'):
        print("\nModified Parameters:")
        for mod in approval['modified_parameters'].get('modifications', []):
            print(f"  - {mod}")
    
    # Generate risk report
    print("\n--- RISK REPORT TEST ---\n")
    
    risk_report = agent.generate_risk_report(positions)
    
    print("\nRisk Report Summary:")
    print(f"Total Positions: {risk_report['summary']['positions_count']}")
    print(f"Total Exposure: ${risk_report['summary']['total_exposure']:.2f}")
    print(f"Value at Risk (95%): ${risk_report['summary']['var_95']:.2f}")
    print(f"Expected Shortfall (95%): ${risk_report['summary']['expected_shortfall_95']:.2f}")
    print(f"Correlation Risk: {risk_report['summary']['correlation_risk']}")
    print(f"Overall Risk Level: {risk_report['summary']['risk_level']}")
    
    if risk_report.get('warnings', []):
        print("\nWarnings:")
        for warning in risk_report['warnings']:
            print(f"  - {warning}")
            
    if risk_report.get('recommendations', []):
        print("\nRecommendations:")
        for rec in risk_report['recommendations']:
            print(f"  - {rec}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 