#!/usr/bin/env python
import os
from dotenv import load_dotenv
import pandas as pd
from pprint import pprint
from typing import Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load environment variables from .env file
load_dotenv()

# Simple mock implementation for testing
class MockSentiment:
    def __init__(self, currency_pair="EUR/USD"):
        self.currency_pair = currency_pair
        print(f"Initializing mock sentiment analysis for {currency_pair}")
        
    def get_news_sentiment(self, currency_pair, days_back=7):
        print(f"Analyzing news sentiment for {currency_pair} for past {days_back} days")
        # Create mock data
        data = {
            'date': ['2023-11-20', '2023-11-19', '2023-11-18', '2023-11-17', '2023-11-16'],
            'title': [
                'ECB signals potential rate cut amid slowing inflation',
                'EUR/USD rises on positive Eurozone economic data',
                'Fed minutes suggest cautious approach to future rate hikes',
                'Euro strengthens as German manufacturing rebounds',
                'Dollar weakens following lower than expected US inflation data'
            ],
            'sentiment_score': [0.65, 0.72, 0.45, 0.58, -0.32]
        }
        return pd.DataFrame(data)
    
    def get_combined_sentiment(self, currency_pair, days_back=7):
        print(f"Getting combined sentiment for {currency_pair}")
        return {
            'overall_score': 0.57,
            'sentiment_label': 'bullish',
            'news_weight': 0.4,
            'social_weight': 0.3,
            'analyst_weight': 0.2,
            'market_weight': 0.1,
            'news_score': 0.62,
            'social_score': 0.53,
            'analyst_score': 0.48,
            'market_score': 0.59,
            'confidence': 'medium'
        }
    
    def generate_sentiment_signals(self, currency_pair):
        print(f"Generating sentiment signals for {currency_pair}")
        return {
            'currency_pair': currency_pair,
            'signal': 'buy',
            'strength': 0.65,
            'confidence': 'medium',
            'time_frame': '1d',
            'factors': [
                {'factor': 'positive news coverage', 'impact': 'high'},
                {'factor': 'bullish analyst consensus', 'impact': 'medium'}
            ],
            'is_contrarian': False,
            'timestamp': '2023-11-20T15:30:00'
        }
    
    def generate_sentiment_report(self, currency_pair):
        print(f"Generating sentiment report for {currency_pair}")
        return {
            'currency_pair': currency_pair,
            'timestamp': '2023-11-20T15:30:00',
            'combined_sentiment': self.get_combined_sentiment(currency_pair),
            'signal': self.generate_sentiment_signals(currency_pair),
            'narrative': f"Market sentiment for {currency_pair} is currently bullish, with a positive sentiment score of 0.57. The sentiment has been improving over the recent period. Based on the current sentiment analysis, a medium confidence BUY signal is generated. This analysis is based on 15 news articles and 42 social media posts. Key factors influencing sentiment include: positive ECB outlook (positive), strong Eurozone economic data (positive), improved German manufacturing (positive)."
        }

def main():
    print("Starting sentiment analysis test...")
    
    # Create a mock sentiment analyzer
    sentiment = MockSentiment("EUR/USD")
    
    currency_pair = "EUR/USD"
    print(f"\nAnalyzing sentiment for {currency_pair}...")
    
    # Get news sentiment
    print("\n1. Recent News Headlines and Sentiment:")
    news = sentiment.get_news_sentiment(currency_pair, days_back=7)
    pd.set_option('display.max_colwidth', 100)
    display_cols = ['date', 'title', 'sentiment_score']
    print(news[display_cols].sort_values('date', ascending=False).head(5).to_string(index=False))
    
    # Get combined sentiment
    print("\n2. Combined Sentiment Analysis:")
    combined = sentiment.get_combined_sentiment(currency_pair)
    pprint(combined)
    
    # Generate trading signals
    print("\n3. Trading Signals Based on Sentiment:")
    signals = sentiment.generate_sentiment_signals(currency_pair)
    pprint(signals)
    
    # Generate comprehensive report
    print("\n4. Generating Comprehensive Sentiment Report...")
    report = sentiment.generate_sentiment_report(currency_pair)
    
    # Print the narrative summary
    print("\nSentiment Summary:")
    print(report.get('narrative', 'No narrative available'))
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 