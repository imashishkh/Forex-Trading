#!/usr/bin/env python
import os
from dotenv import load_dotenv
import pandas as pd
from pprint import pprint
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Mock implementation of MarketDataAgent
class MockMarketDataAgent:
    def __init__(self):
        print("Initializing Market Data Agent")
        
    def get_historical_prices(self, currency_pair, timeframe='1h', periods=100):
        print(f"Fetching historical prices for {currency_pair}, timeframe: {timeframe}, periods: {periods}")
        
        # Generate mock price data
        end_date = datetime.now()
        dates = [(end_date - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:00') for i in range(periods)]
        dates.reverse()  # Put in ascending order
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducibility
        base_price = 1.10  # Starting price for EUR/USD
        price_changes = np.random.normal(0, 0.0015, periods)  # Small random changes
        
        # Create cumulative price changes to simulate a price series
        cumulative_changes = np.cumsum(price_changes)
        prices = base_price + cumulative_changes
        
        # Create volume data
        volumes = np.random.randint(50, 500, periods)
        
        # Create DataFrame
        data = {
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0.0001, 0.0020, periods),
            'low': prices - np.random.uniform(0.0001, 0.0020, periods),
            'close': prices + np.random.uniform(-0.0015, 0.0015, periods),
            'volume': volumes,
            'currency_pair': currency_pair
        }
        
        return pd.DataFrame(data)
    
    def get_latest_price(self, currency_pair):
        print(f"Fetching latest price for {currency_pair}")
        return {
            'currency_pair': currency_pair,
            'bid': 1.0921,
            'ask': 1.0923,
            'spread': 0.0002,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_economic_calendar(self, days=7):
        print(f"Fetching economic calendar for next {days} days")
        events = [
            {
                'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'time': '10:00',
                'currency': 'EUR',
                'event': 'ECB Interest Rate Decision',
                'importance': 'high',
                'forecast': '3.75%',
                'previous': '3.75%'
            },
            {
                'date': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
                'time': '13:30',
                'currency': 'USD',
                'event': 'Non-Farm Payrolls',
                'importance': 'high',
                'forecast': '180K',
                'previous': '175K'
            },
            {
                'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                'time': '15:00',
                'currency': 'EUR',
                'event': 'German Manufacturing PMI',
                'importance': 'medium',
                'forecast': '45.5',
                'previous': '44.8'
            }
        ]
        return pd.DataFrame(events)

# Mock implementation of TechnicalAnalystAgent
class MockTechnicalAnalystAgent:
    def __init__(self):
        print("Initializing Technical Analyst Agent")
    
    def analyze_price_data(self, price_data, indicators=None):
        print(f"Analyzing price data with {len(price_data)} periods")
        
        if indicators is None:
            indicators = ['sma', 'rsi', 'macd']
            
        print(f"Calculating indicators: {', '.join(indicators)}")
        
        # Generate mock technical analysis results
        analysis_results = {
            'indicators': {},
            'signals': [],
            'summary': {}
        }
        
        # Simulate indicator calculations
        price_series = price_data['close'].values
        
        if 'sma' in indicators:
            # Simple moving averages
            analysis_results['indicators']['sma'] = {
                'sma20': np.mean(price_series[-20:]),
                'sma50': np.mean(price_series[-50:]),
                'sma200': np.mean(price_series[-100:]) if len(price_series) >= 100 else None
            }
            
        if 'rsi' in indicators:
            # Simplified RSI calculation
            rsi_value = 50 + np.random.uniform(-20, 20)
            analysis_results['indicators']['rsi'] = {
                'value': rsi_value,
                'is_overbought': rsi_value > 70,
                'is_oversold': rsi_value < 30
            }
            
        if 'macd' in indicators:
            # Mock MACD
            macd_line = np.random.uniform(-0.002, 0.002)
            signal_line = np.random.uniform(-0.002, 0.002)
            
            analysis_results['indicators']['macd'] = {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': macd_line - signal_line,
                'crossover': abs(macd_line - signal_line) < 0.0003 and macd_line > signal_line
            }
        
        # Generate signals based on indicators
        signals = []
        
        # SMA signals
        if 'sma' in indicators:
            sma_data = analysis_results['indicators']['sma']
            if sma_data['sma20'] > sma_data['sma50']:
                signals.append({
                    'indicator': 'sma',
                    'signal': 'buy',
                    'strength': 'medium',
                    'description': 'SMA20 above SMA50 indicates bullish momentum'
                })
            else:
                signals.append({
                    'indicator': 'sma',
                    'signal': 'sell',
                    'strength': 'medium',
                    'description': 'SMA20 below SMA50 indicates bearish momentum'
                })
                
        # RSI signals
        if 'rsi' in indicators:
            rsi_data = analysis_results['indicators']['rsi']
            if rsi_data['is_oversold']:
                signals.append({
                    'indicator': 'rsi',
                    'signal': 'buy',
                    'strength': 'strong',
                    'description': 'RSI in oversold territory'
                })
            elif rsi_data['is_overbought']:
                signals.append({
                    'indicator': 'rsi',
                    'signal': 'sell',
                    'strength': 'strong',
                    'description': 'RSI in overbought territory'
                })
                
        # MACD signals
        if 'macd' in indicators:
            macd_data = analysis_results['indicators']['macd']
            if macd_data['crossover']:
                signals.append({
                    'indicator': 'macd',
                    'signal': 'buy' if macd_data['macd_line'] > macd_data['signal_line'] else 'sell',
                    'strength': 'strong',
                    'description': 'MACD crossed signal line'
                })
        
        analysis_results['signals'] = signals
        
        # Generate overall summary
        buy_signals = [s for s in signals if s['signal'] == 'buy']
        sell_signals = [s for s in signals if s['signal'] == 'sell']
        
        if len(buy_signals) > len(sell_signals):
            overall_signal = 'buy'
            signal_strength = len(buy_signals) / len(signals)
        elif len(sell_signals) > len(buy_signals):
            overall_signal = 'sell'
            signal_strength = len(sell_signals) / len(signals)
        else:
            overall_signal = 'neutral'
            signal_strength = 0.5
            
        analysis_results['summary'] = {
            'signal': overall_signal,
            'strength': signal_strength,
            'confidence': 'medium',
            'timeframe': '1h',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analysis_results

# Mock implementation of FundamentalsAgent
class MockFundamentalsAgent:
    def __init__(self):
        print("Initializing Fundamentals Agent")
    
    def analyze_economic_events(self, calendar_data, currency_pair):
        print(f"Analyzing economic events impact on {currency_pair}")
        
        # Parse currency pair
        base_currency, quote_currency = currency_pair.split('/')
        
        # Filter events relevant to the currency pair
        base_events = calendar_data[calendar_data['currency'] == base_currency]
        quote_events = calendar_data[calendar_data['currency'] == quote_currency]
        
        # Generate mock analysis
        analysis = {
            'currency_pair': currency_pair,
            'base_currency': {
                'events_count': len(base_events),
                'high_impact_count': len(base_events[base_events['importance'] == 'high']),
                'potential_impact': self._calculate_potential_impact(base_events)
            },
            'quote_currency': {
                'events_count': len(quote_events),
                'high_impact_count': len(quote_events[quote_events['importance'] == 'high']),
                'potential_impact': self._calculate_potential_impact(quote_events)
            },
            'overall_bias': '',
            'confidence': 'medium',
            'key_events': []
        }
        
        # Determine overall bias
        base_impact = analysis['base_currency']['potential_impact']
        quote_impact = analysis['quote_currency']['potential_impact']
        
        if base_impact > 0 and quote_impact < 0:
            analysis['overall_bias'] = 'strongly_bullish'
        elif base_impact > 0 and quote_impact > 0:
            if base_impact > quote_impact:
                analysis['overall_bias'] = 'bullish'
            else:
                analysis['overall_bias'] = 'bearish'
        elif base_impact < 0 and quote_impact < 0:
            if base_impact < quote_impact:
                analysis['overall_bias'] = 'bearish'
            else:
                analysis['overall_bias'] = 'bullish'
        elif base_impact < 0 and quote_impact > 0:
            analysis['overall_bias'] = 'strongly_bearish'
        else:
            analysis['overall_bias'] = 'neutral'
            
        # Extract key events
        if not base_events.empty:
            for _, event in base_events.iterrows():
                if event['importance'] == 'high':
                    analysis['key_events'].append({
                        'currency': event['currency'],
                        'event': event['event'],
                        'date': event['date'],
                        'time': event['time'],
                        'forecast': event['forecast'],
                        'previous': event['previous'],
                        'potential_impact': 'positive' if 'rate' in event['event'].lower() else 'variable'
                    })
                    
        if not quote_events.empty:
            for _, event in quote_events.iterrows():
                if event['importance'] == 'high':
                    analysis['key_events'].append({
                        'currency': event['currency'],
                        'event': event['event'],
                        'date': event['date'],
                        'time': event['time'],
                        'forecast': event['forecast'],
                        'previous': event['previous'],
                        'potential_impact': 'negative' if 'payrolls' in event['event'].lower() else 'variable'
                    })
        
        return analysis
    
    def _calculate_potential_impact(self, events):
        """Calculate potential impact score for economic events"""
        if events.empty:
            return 0
            
        impact_score = 0
        for _, event in events.iterrows():
            # Weight by importance
            importance_weight = 3 if event['importance'] == 'high' else 1 if event['importance'] == 'medium' else 0.2
            
            # Specific event types impact
            event_name = event['event'].lower()
            if 'interest rate' in event_name or 'rate decision' in event_name:
                event_weight = 5
            elif 'gdp' in event_name:
                event_weight = 4
            elif 'inflation' in event_name or 'cpi' in event_name:
                event_weight = 3.5
            elif 'employment' in event_name or 'payroll' in event_name or 'job' in event_name:
                event_weight = 3
            elif 'pmi' in event_name or 'manufacturing' in event_name:
                event_weight = 2.5
            else:
                event_weight = 1
                
            # Random factor for variability
            random_factor = np.random.uniform(0.8, 1.2)
            
            # Add to score (positive or negative randomly for this mock)
            impact_score += importance_weight * event_weight * random_factor * np.random.choice([-1, 1])
            
        return impact_score

def main():
    print("\n=== FOREX TRADING AGENT INTEGRATION TEST ===\n")
    
    # Initialize the agents
    market_agent = MockMarketDataAgent()
    tech_agent = MockTechnicalAnalystAgent()
    fundamentals_agent = MockFundamentalsAgent()
    
    currency_pair = "EUR/USD"
    print(f"\nTesting integration with {currency_pair}\n")
    
    # Step 1: Get market data
    print("\n--- STEP 1: MARKET DATA RETRIEVAL ---\n")
    price_data = market_agent.get_historical_prices(currency_pair, timeframe='1h', periods=100)
    print(f"Retrieved {len(price_data)} price data points")
    print("\nSample of price data:")
    print(price_data.tail(3).to_string(index=False))
    
    # Get economic calendar
    calendar_data = market_agent.get_economic_calendar(days=7)
    print("\nEconomic Calendar:")
    print(calendar_data.to_string(index=False))
    
    # Step 2: Analyze with Technical Analyst
    print("\n\n--- STEP 2: TECHNICAL ANALYSIS ---\n")
    technical_analysis = tech_agent.analyze_price_data(price_data, indicators=['sma', 'rsi', 'macd'])
    
    print("\nTechnical Analysis Results:")
    print("\nIndicators:")
    pprint(technical_analysis['indicators'])
    
    print("\nTechnical Signals:")
    for signal in technical_analysis['signals']:
        print(f"- {signal['signal'].upper()} signal from {signal['indicator']} ({signal['strength']}): {signal['description']}")
    
    print("\nTechnical Summary:")
    pprint(technical_analysis['summary'])
    
    # Step 3: Analyze with Fundamentals Agent
    print("\n\n--- STEP 3: FUNDAMENTAL ANALYSIS ---\n")
    fundamental_analysis = fundamentals_agent.analyze_economic_events(calendar_data, currency_pair)
    
    print("\nFundamental Analysis Results:")
    print(f"\nOverall Bias: {fundamental_analysis['overall_bias']} (Confidence: {fundamental_analysis['confidence']})")
    
    print(f"\nBase Currency ({fundamental_analysis['base_currency']['events_count']} events):")
    print(f"High Impact Events: {fundamental_analysis['base_currency']['high_impact_count']}")
    print(f"Potential Impact Score: {fundamental_analysis['base_currency']['potential_impact']:.2f}")
    
    print(f"\nQuote Currency ({fundamental_analysis['quote_currency']['events_count']} events):")
    print(f"High Impact Events: {fundamental_analysis['quote_currency']['high_impact_count']}")
    print(f"Potential Impact Score: {fundamental_analysis['quote_currency']['potential_impact']:.2f}")
    
    print("\nKey Economic Events:")
    for event in fundamental_analysis['key_events']:
        print(f"- {event['date']} {event['time']}: {event['currency']} {event['event']} " +
              f"(Forecast: {event['forecast']}, Previous: {event['previous']})")
    
    # Step 4: Integration Summary
    print("\n\n--- STEP 4: INTEGRATED ANALYSIS SUMMARY ---\n")
    
    # Combine technical and fundamental signals
    technical_signal = technical_analysis['summary']['signal']
    technical_strength = technical_analysis['summary']['strength']
    fundamental_bias = fundamental_analysis['overall_bias']
    
    # Simplified integration logic
    if technical_signal == 'buy' and fundamental_bias in ['bullish', 'strongly_bullish']:
        integrated_signal = 'strong_buy'
        confidence = 'high'
    elif technical_signal == 'sell' and fundamental_bias in ['bearish', 'strongly_bearish']:
        integrated_signal = 'strong_sell'
        confidence = 'high'
    elif technical_signal == 'buy' and fundamental_bias in ['bearish', 'strongly_bearish']:
        integrated_signal = 'conflicting_signals'
        confidence = 'low'
    elif technical_signal == 'sell' and fundamental_bias in ['bullish', 'strongly_bullish']:
        integrated_signal = 'conflicting_signals'
        confidence = 'low'
    elif technical_signal == 'buy':
        integrated_signal = 'weak_buy'
        confidence = 'medium'
    elif technical_signal == 'sell':
        integrated_signal = 'weak_sell'
        confidence = 'medium'
    else:
        integrated_signal = 'neutral'
        confidence = 'medium'
    
    print(f"Technical Signal: {technical_signal.upper()} (Strength: {technical_strength:.2f})")
    print(f"Fundamental Bias: {fundamental_bias.upper()}")
    print(f"Integrated Signal: {integrated_signal.upper()}")
    print(f"Confidence: {confidence.upper()}")
    
    latest_price = market_agent.get_latest_price(currency_pair)
    print(f"\nLatest {currency_pair} Price: {latest_price['bid']}/{latest_price['ask']} (Spread: {latest_price['spread']})")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 