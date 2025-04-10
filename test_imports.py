#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to check if all required modules can be imported correctly
"""

print("Testing imports...")

try:
    import os
    import time
    import logging
    import json
    import threading
    import datetime
    from typing import Dict, List, Any, Optional, Tuple, Union, Callable

    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import streamlit as st
    import psutil
    import requests
    from websocket import create_connection
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import RendererAgg
    from collections import deque
    import uuid
    from streamlit_autorefresh import st_autorefresh
    import sys
    
    # Import additional classes for real-time API connections
    from wallet_manager import WalletManager
    from market_data_agent.agent import MarketDataAgent
    from utils.config_manager import ConfigManager
    from dotenv import load_dotenv
    import traceback
    
    print("All modules imported successfully.")
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc() 