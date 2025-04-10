#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data utility functions for the Forex Trading Platform
"""

import os
import pandas as pd
from datetime import datetime

def load_dataframe(file_path, index_col=None):
    """
    Load a pandas DataFrame from a file
    
    Args:
        file_path (str): Path to the file (csv, parquet, or pickle)
        index_col (str, optional): Column to use as the index
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load DataFrame based on file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
        if index_col and index_col in df.columns:
            df.set_index(index_col, inplace=True)
    elif file_ext == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_ext in ['.pkl', '.pickle']:
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return df

def save_dataframe(df, file_path, index=True):
    """
    Save a pandas DataFrame to a file
    
    Args:
        df (pd.DataFrame): DataFrame to save
        file_path (str): Path to save the file (csv, parquet, or pickle)
        index (bool): Whether to include the index in the saved file
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save DataFrame based on file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        df.to_csv(file_path, index=index)
    elif file_ext == '.parquet':
        df.to_parquet(file_path, index=index)
    elif file_ext in ['.pkl', '.pickle']:
        df.to_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def resample_ohlc(df, timeframe):
    """
    Resample OHLC (Open, High, Low, Close) data to a different timeframe
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        timeframe (str): Pandas-compatible timeframe string (e.g., '1H', '4H', '1D')
    
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    # Ensure DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    # Dictionary to map column names to resampling functions
    resample_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Keep only the columns that are in resample_dict
    columns_to_resample = [col for col in df.columns if col.lower() in resample_dict]
    
    # Create a new resampling dictionary with actual column names
    actual_resample_dict = {col: resample_dict[col.lower()] for col in columns_to_resample}
    
    # Resample the DataFrame
    resampled_df = df.resample(timeframe).agg(actual_resample_dict)
    
    return resampled_df 