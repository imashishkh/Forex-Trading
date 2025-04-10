#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time utilities for the Forex Trading Platform
"""

import pytz
from datetime import datetime, timedelta

def get_current_time(timezone='UTC'):
    """
    Get the current time in the specified timezone
    
    Args:
        timezone (str): Timezone name (default: 'UTC')
    
    Returns:
        datetime: Current datetime in the specified timezone
    """
    tz = pytz.timezone(timezone)
    return datetime.now(tz)

def get_forex_sessions():
    """
    Determine the currently active forex trading sessions
    
    The major forex sessions are:
    - Sydney: 9:00 PM - 6:00 AM UTC (summer) / 10:00 PM - 7:00 AM UTC (winter)
    - Tokyo: 12:00 AM - 9:00 AM UTC (summer) / 1:00 AM - 10:00 AM UTC (winter)
    - London: 8:00 AM - 5:00 PM UTC (summer) / 9:00 AM - 6:00 PM UTC (winter)
    - New York: 1:00 PM - 10:00 PM UTC (summer) / 2:00 PM - 11:00 PM UTC (winter)
    
    Returns:
        list: List of currently active sessions ('Sydney', 'Tokyo', 'London', 'New York')
    """
    # Get current time in UTC
    now_utc = datetime.now(pytz.UTC)
    
    # Check day of week (0 = Monday, 6 = Sunday)
    day_of_week = now_utc.weekday()
    
    # If it's weekend (Saturday or Sunday), no sessions are active
    if day_of_week >= 5:
        return []
    
    # Get current UTC hour
    hour_utc = now_utc.hour
    
    # Determine if it's summer time (DST) in the major centers
    # Simplified approach - actual DST dates vary by year and region
    is_summer = 3 <= now_utc.month <= 10  # Rough approximation for Northern Hemisphere DST
    
    active_sessions = []
    
    # Sydney session (9:00 PM - 6:00 AM UTC summer, 10:00 PM - 7:00 AM UTC winter)
    sydney_start = 21 if is_summer else 22
    sydney_end = 6 if is_summer else 7
    if sydney_start <= hour_utc < 24 or 0 <= hour_utc < sydney_end:
        active_sessions.append('Sydney')
    
    # Tokyo session (12:00 AM - 9:00 AM UTC summer, 1:00 AM - 10:00 AM UTC winter)
    tokyo_start = 0 if is_summer else 1
    tokyo_end = 9 if is_summer else 10
    if tokyo_start <= hour_utc < tokyo_end:
        active_sessions.append('Tokyo')
    
    # London session (8:00 AM - 5:00 PM UTC summer, 9:00 AM - 6:00 PM UTC winter)
    london_start = 8 if is_summer else 9
    london_end = 17 if is_summer else 18
    if london_start <= hour_utc < london_end:
        active_sessions.append('London')
    
    # New York session (1:00 PM - 10:00 PM UTC summer, 2:00 PM - 11:00 PM UTC winter)
    ny_start = 13 if is_summer else 14
    ny_end = 22 if is_summer else 23
    if ny_start <= hour_utc < ny_end:
        active_sessions.append('New York')
    
    return active_sessions

def is_market_open(symbol=None):
    """
    Check if the forex market is open for a specific symbol
    
    Args:
        symbol (str): Symbol to check (default: None, checks overall market)
    
    Returns:
        bool: True if market is open, False otherwise
    """
    # Get current time in UTC
    now_utc = datetime.now(pytz.UTC)
    
    # Check day of week (0 = Monday, 6 = Sunday)
    day_of_week = now_utc.weekday()
    
    # Forex market is closed on weekends
    if day_of_week >= 5:
        return False
    
    # Forex market opens around 10 PM UTC Sunday (Sydney open) and closes around 10 PM UTC Friday (New York close)
    if day_of_week == 4:  # Friday
        hour_utc = now_utc.hour
        if hour_utc >= 22:  # After 10 PM UTC on Friday, market is closed
            return False
    
    # If we have a specific symbol, check if any relevant session is active
    if symbol:
        active_sessions = get_forex_sessions()
        
        # Parse currency pairs
        if '/' in symbol:
            base, quote = symbol.split('/')
            
            # Map currencies to their primary trading sessions
            currency_sessions = {
                'USD': 'New York',
                'CAD': 'New York',
                'EUR': 'London',
                'GBP': 'London',
                'CHF': 'London',
                'JPY': 'Tokyo',
                'AUD': 'Sydney',
                'NZD': 'Sydney'
            }
            
            # Check if any session for the currencies in the pair is active
            base_session = currency_sessions.get(base)
            quote_session = currency_sessions.get(quote)
            
            if base_session and base_session in active_sessions:
                return True
            if quote_session and quote_session in active_sessions:
                return True
            
            # Special case: if both London and New York sessions are active,
            # it's generally a good time to trade any major currency pair
            if 'London' in active_sessions and 'New York' in active_sessions:
                return True
    
    # By default, if we reach here, the market is open
    return True

def get_session_overlap_times():
    """
    Get the forex session overlap times in UTC
    
    Returns:
        dict: Dictionary with session overlaps and their UTC times
    """
    # Check if it's during DST (Daylight Saving Time) in the Northern Hemisphere
    # Simplified approach - actual DST dates vary by year and region
    now = datetime.now()
    is_summer = 3 <= now.month <= 10  # Rough approximation for Northern Hemisphere DST
    
    if is_summer:
        # Summer time (DST) overlaps
        return {
            'Tokyo-London': {
                'start': 8,  # 8:00 AM UTC
                'end': 9,    # 9:00 AM UTC
                'duration': 1
            },
            'London-New York': {
                'start': 13,  # 1:00 PM UTC
                'end': 17,    # 5:00 PM UTC
                'duration': 4
            },
            'New York-Sydney': {
                'start': 21,  # 9:00 PM UTC
                'end': 22,    # 10:00 PM UTC
                'duration': 1
            },
            'Sydney-Tokyo': {
                'start': 0,   # 12:00 AM UTC
                'end': 6,     # 6:00 AM UTC
                'duration': 6
            }
        }
    else:
        # Winter time (standard time) overlaps
        return {
            'Tokyo-London': {
                'start': 9,   # 9:00 AM UTC
                'end': 10,    # 10:00 AM UTC
                'duration': 1
            },
            'London-New York': {
                'start': 14,  # 2:00 PM UTC
                'end': 18,    # 6:00 PM UTC
                'duration': 4
            },
            'New York-Sydney': {
                'start': 22,  # 10:00 PM UTC
                'end': 23,    # 11:00 PM UTC
                'duration': 1
            },
            'Sydney-Tokyo': {
                'start': 1,   # 1:00 AM UTC
                'end': 7,     # 7:00 AM UTC
                'duration': 6
            }
        }
    
def is_in_session_overlap():
    """
    Check if current time is in a forex session overlap
    
    Returns:
        bool: True if in session overlap, False otherwise
    """
    # Get current time in UTC
    now_utc = datetime.now(pytz.UTC)
    hour_utc = now_utc.hour
    
    # Check day of week (0 = Monday, 6 = Sunday)
    day_of_week = now_utc.weekday()
    
    # Forex market is closed on weekends
    if day_of_week >= 5:
        return False
    
    # Get session overlaps
    overlaps = get_session_overlap_times()
    
    # Check if current hour is in any overlap
    for overlap_name, overlap_times in overlaps.items():
        start_hour = overlap_times['start']
        end_hour = overlap_times['end']
        
        # Handle overlaps that cross midnight
        if start_hour > end_hour:
            if hour_utc >= start_hour or hour_utc < end_hour:
                return True
        else:
            if start_hour <= hour_utc < end_hour:
                return True
    
    return False

def get_next_session_overlap():
    """
    Get the next forex session overlap from the current time
    
    Returns:
        dict: Information about the next session overlap
    """
    # Get current time in UTC
    now_utc = datetime.now(pytz.UTC)
    hour_utc = now_utc.hour
    
    # Get session overlaps
    overlaps = get_session_overlap_times()
    
    # Sort overlaps by start time
    sorted_overlaps = sorted(overlaps.items(), key=lambda x: x[1]['start'])
    
    # Find the next overlap
    for overlap_name, overlap_times in sorted_overlaps:
        start_hour = overlap_times['start']
        
        # Handle overlaps that cross midnight
        if start_hour > hour_utc or (start_hour < hour_utc and overlap_times['end'] <= start_hour):
            # Calculate time until next overlap
            hours_until = start_hour - hour_utc if start_hour > hour_utc else 24 - hour_utc + start_hour
            
            return {
                'name': overlap_name,
                'start_hour': start_hour,
                'end_hour': overlap_times['end'],
                'duration': overlap_times['duration'],
                'hours_until': hours_until
            }
    
    # If no overlap found, return the first one for the next day
    first_overlap = sorted_overlaps[0]
    hours_until = 24 - hour_utc + first_overlap[1]['start']
    
    return {
        'name': first_overlap[0],
        'start_hour': first_overlap[1]['start'],
        'end_hour': first_overlap[1]['end'],
        'duration': first_overlap[1]['duration'],
        'hours_until': hours_until
    }

def timestamp_to_datetime(timestamp, unit='ms'):
    """
    Convert a timestamp to a datetime object
    
    Args:
        timestamp (int or float): Timestamp to convert
        unit (str): Unit of the timestamp ('s' for seconds, 'ms' for milliseconds, 'us' for microseconds)
    
    Returns:
        datetime: Datetime object
    """
    if unit == 's':
        return datetime.fromtimestamp(timestamp)
    elif unit == 'ms':
        return datetime.fromtimestamp(timestamp / 1000)
    elif unit == 'us':
        return datetime.fromtimestamp(timestamp / 1000000)
    else:
        raise ValueError(f"Invalid unit: {unit}. Use 's', 'ms', or 'us'.")

def datetime_to_timestamp(dt, unit='ms'):
    """
    Convert a datetime object to a timestamp
    
    Args:
        dt (datetime): Datetime object
        unit (str): Unit for the timestamp ('s' for seconds, 'ms' for milliseconds, 'us' for microseconds)
    
    Returns:
        int: Timestamp
    """
    timestamp = dt.timestamp()
    
    if unit == 's':
        return int(timestamp)
    elif unit == 'ms':
        return int(timestamp * 1000)
    elif unit == 'us':
        return int(timestamp * 1000000)
    else:
        raise ValueError(f"Invalid unit: {unit}. Use 's', 'ms', or 'us'.") 