#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the FundamentalsAgent's mock economic calendar data

This script demonstrates how to generate mock economic event data
without requiring API keys or full agent initialization.
"""

import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_mock_economic_calendar(
    start_date: str, 
    end_date: str, 
    countries: list
) -> pd.DataFrame:
    """Generate mock economic calendar data for testing."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days = (end - start).days + 1
    
    # Map of currencies to countries/regions
    country_map = {
        'USD': 'United States',
        'EUR': 'Euro Zone',
        'GBP': 'United Kingdom',
        'JPY': 'Japan',
        'AUD': 'Australia',
        'CAD': 'Canada',
        'CHF': 'Switzerland',
        'NZD': 'New Zealand',
        'CNY': 'China',
        'SGD': 'Singapore',
        'MXN': 'Mexico',
        'SEK': 'Sweden',
        'NOK': 'Norway'
    }
    
    # Common economic events
    event_types = [
        ("Interest Rate Decision", "high"),
        ("GDP", "high"),
        ("CPI", "high"),
        ("Unemployment Rate", "high"),
        ("Retail Sales", "medium"),
        ("PMI", "medium"),
        ("Trade Balance", "medium"),
        ("Industrial Production", "medium"),
        ("Consumer Confidence", "medium"),
        ("Building Permits", "low"),
        ("Current Account", "low"),
        ("Housing Starts", "low")
    ]
    
    # Create events list
    events = []
    
    # Generate events
    for _ in range(min(days * 5, 100)):  # Max 5 events per day, 100 total
        currency = np.random.choice(countries)
        country_name = country_map.get(currency, currency)
        
        event_type, importance = event_types[np.random.randint(0, len(event_types))]
        
        # Randomize date within range
        days_offset = np.random.randint(0, days)
        event_date = (start + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        # Randomize time
        hour = np.random.randint(7, 18)
        minute = np.random.choice([0, 15, 30, 45])
        event_time = f"{hour:02d}:{minute:02d}"
        
        # Add to events list
        events.append({
            'date': event_date,
            'time': event_time,
            'currency': currency,
            'country': country_name,
            'event': f"{country_name} {event_type}",
            'importance': importance,
            'previous': f"{np.random.uniform(-1, 5):.1f}%",
            'forecast': f"{np.random.uniform(-1, 5):.1f}%",
            'actual': None  # Will be filled after the event
        })
        
    # Sort by date and time
    events_df = pd.DataFrame(events)
    events_df['datetime'] = pd.to_datetime(events_df['date'] + ' ' + events_df['time'])
    events_df = events_df.sort_values('datetime')
    events_df = events_df.drop('datetime', axis=1)
    
    return events_df


def main():
    """Main function to test mock economic calendar generation"""
    
    logger.info("Testing mock economic calendar data generation")
    
    try:
        # Define major currencies to analyze
        major_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        
        # Set date range for economic calendar
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        logger.info(f"Generating mock economic calendar from {start_date} to {end_date}")
        
        # Generate mock economic calendar
        calendar = generate_mock_economic_calendar(
            start_date=start_date,
            end_date=end_date,
            countries=major_currencies
        )
        
        # Display the economic calendar
        if not calendar.empty:
            # Print the number of events found
            logger.info(f"Generated {len(calendar)} mock economic events for the next 7 days")
            
            # Display summary by currency
            logger.info("\nEvents by Currency:")
            currency_counts = calendar['currency'].value_counts()
            for currency, count in currency_counts.items():
                logger.info(f"{currency}: {count} events")
            
            # Display high-importance events
            logger.info("\nHigh Importance Events:")
            high_importance = calendar[calendar['importance'] == 'high']
            
            # Display events in a formatted way
            for _, event in high_importance.iterrows():
                date_str = event['date'] 
                time_str = event['time']
                currency = event['currency']
                event_name = event['event']
                impact = event['importance']
                
                logger.info(f"{date_str} {time_str} - {currency} - {event_name} - Importance: {impact}")
            
            # For each major currency, extract key events
            logger.info("\nKey Events by Currency:")
            for currency in major_currencies:
                logger.info(f"\nKey events for {currency}:")
                
                # Filter the mock calendar for this currency
                currency_events = calendar[calendar['currency'] == currency]
                
                # Sort by importance and date
                importance_order = {'high': 0, 'medium': 1, 'low': 2}
                currency_events['importance_order'] = currency_events['importance'].map(importance_order)
                sorted_events = currency_events.sort_values(['importance_order', 'date'])
                
                # Extract top 3 events
                if not sorted_events.empty:
                    for _, event in sorted_events.head(3).iterrows():
                        date_str = event['date']
                        time_str = event['time'] 
                        event_name = event['event']
                        impact = event['importance']
                        
                        logger.info(f"{date_str} {time_str} - {event_name} - Importance: {impact}")
                else:
                    logger.info(f"No key events found for {currency}")
        else:
            logger.info("No mock economic events generated")
        
        logger.info("\nMock fundamentals data testing completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)


if __name__ == "__main__":
    main() 