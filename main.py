#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forex Trading Platform with AI Agents
Main entry point for the application

This application orchestrates a forex trading platform powered by AI agents,
providing functionality for live trading, paper trading, backtesting, and
strategy optimization.
"""

import os
import sys
import logging
import signal
import argparse
import time
import json
import importlib
import datetime
import traceback
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path

# Third-party imports
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables must be set manually.")
    def load_dotenv(**kwargs):
        pass

try:
    from colorama import Fore, Style, init
    init()
except ImportError:
    print("Warning: colorama not installed. Console output will not be colored.")
    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        RESET = ""
    
    class Style:
        BRIGHT = ""
        RESET_ALL = ""

# Local imports
from orchestrator import Orchestrator
from utils.config_manager import ConfigManager
try:
    from monitoring import MonitoringDashboard
except ImportError:
    print("Warning: MonitoringDashboard not found or monitoring.py has issues. Monitoring disabled.")
    MonitoringDashboard = None

def load_environment_variables() -> bool:
    """
    Load environment variables from .env file.
    
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    try:
        # Load .env file
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            return True
        else:
            print("Warning: .env file not found. Using existing environment variables.")
            return False
    except Exception as e:
        print(f"Error loading environment variables: {str(e)}")
        return False

def initialize_logging(log_level: str = "INFO", log_file: Optional[str] = "forex_trading.log") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, or None to disable file logging
    
    Returns:
        logging.Logger: Configured logger
    """
    # Set up logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {str(e)}")
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers
    )
    
    # Create and return logger for this module
    logger = logging.getLogger("forex_trading")
    logger.info(f"Logging initialized at level: {log_level}")
    
    return logger

def parse_command_line_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = create_argument_parser()
    return parser.parse_args()

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for command-line options.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Forex Trading Platform with AI Agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global options
    parser.add_argument(
        "--config", 
        help="Path to configuration file",
        default="config/settings.json"
    )
    parser.add_argument(
        "--log-level", 
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    parser.add_argument(
        "--log-file", 
        help="Path to log file",
        default="forex_trading.log"
    )
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the trading platform")
    start_parser.add_argument(
        "--mode", 
        help="Trading mode",
        choices=["live", "paper", "backtest", "optimize"],
        default="paper"
    )
    start_parser.add_argument(
        "--instruments", 
        help="Trading instruments (comma-separated)",
        default="EUR_USD,GBP_USD,USD_JPY"
    )
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run in backtest mode")
    backtest_parser.add_argument(
        "--start-date", 
        help="Start date for backtesting (YYYY-MM-DD)",
        required=True
    )
    backtest_parser.add_argument(
        "--end-date", 
        help="End date for backtesting (YYYY-MM-DD)",
        required=True
    )
    backtest_parser.add_argument(
        "--instruments", 
        help="Trading instruments (comma-separated)",
        default="EUR_USD,GBP_USD,USD_JPY"
    )
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Run in strategy optimization mode")
    optimize_parser.add_argument(
        "--parameters", 
        help="Parameters to optimize (comma-separated)",
        required=True
    )
    optimize_parser.add_argument(
        "--start-date", 
        help="Start date for optimization period (YYYY-MM-DD)",
        required=True
    )
    optimize_parser.add_argument(
        "--end-date", 
        help="End date for optimization period (YYYY-MM-DD)",
        required=True
    )
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the trading platform")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get platform status")
    
    return parser

def initialize_config(config_path: str) -> Dict[str, Any]:
    """
    Initialize configuration.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.as_dict()
        
        # Override config with environment variables
        config = override_config_with_env(config)
        
        return config
    except Exception as e:
        print(f"Error initializing configuration: {str(e)}")
        sys.exit(1)

def override_config_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values with environment variables.
    
    Args:
        config: Original configuration dictionary
    
    Returns:
        Dict[str, Any]: Updated configuration dictionary
    """
    # API credentials
    if 'OANDA_API_KEY' in os.environ:
        if 'api_credentials' not in config:
            config['api_credentials'] = {}
        if 'oanda' not in config['api_credentials']:
            config['api_credentials']['oanda'] = {}
        config['api_credentials']['oanda']['api_key'] = os.environ['OANDA_API_KEY']
    
    if 'OANDA_ACCOUNT_ID' in os.environ:
        if 'api_credentials' not in config:
            config['api_credentials'] = {}
        if 'oanda' not in config['api_credentials']:
            config['api_credentials']['oanda'] = {}
        config['api_credentials']['oanda']['account_id'] = os.environ['OANDA_ACCOUNT_ID']
    
    # Other environment variables can be added here
    
    return config

def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are present, False otherwise
    """
    required_packages = ["pandas", "matplotlib", "numpy", "openai"]
    
    missing_packages = []
    for package in required_packages:
        try:
            # Replace pkg_resources with importlib.metadata
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

class ForexTradingPlatform:
    """
    Main application class for the Forex Trading Platform.
    
    This class is responsible for initializing the platform, managing its
    lifecycle, and coordinating the different operational modes.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Forex Trading Platform.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("forex_trading")
        self.orchestrator = None
        self.is_running = False
        self.mode = "paper"  # Default mode
        self.platform_status = "initialized"
        self.start_time = None
        self.trading_instruments = []
        
        # Register signal handlers
        self._register_signal_handlers()
        
        self.logger.info("Forex Trading Platform initialized")
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.handle_termination_signal)
        signal.signal(signal.SIGTERM, self.handle_termination_signal)
    
    def handle_termination_signal(self, signum: int, frame: Any) -> None:
        """
        Handle termination signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.info(f"Received termination signal {signum}")
        self.perform_graceful_shutdown()
    
    def start(self, mode: str = "paper", instruments: Optional[List[str]] = None) -> bool:
        """
        Start the trading platform.
        
        Args:
            mode: Trading mode (live, paper, backtest, optimize)
            instruments: List of trading instruments
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        self.logger.info(f"Starting Forex Trading Platform in {mode} mode")
        
        # Set mode and instruments
        self.mode = mode
        if instruments:
            self.trading_instruments = instruments
        else:
            self.trading_instruments = self.config.get("trading", {}).get("instruments", [])
        
        # Update config with instruments
        if "trading" not in self.config:
            self.config["trading"] = {}
        self.config["trading"]["instruments"] = self.trading_instruments
        
        # Initialize orchestrator
        try:
            self.orchestrator = Orchestrator(config=self.config, logger=self.logger.getChild("orchestrator"))
            
            # Initialize agents
            if not self.orchestrator.initialize_agents():
                self.logger.error("Failed to initialize agents")
                return False
            
            # Define workflow
            self.orchestrator.define_workflow()
            
            # Select and run appropriate mode
            if mode == "live":
                result = self.run_live_trading_mode()
            elif mode == "paper":
                result = self.run_paper_trading_mode()
            elif mode == "backtest":
                # Default to last 30 days if not specified
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=30)
                result = self.run_backtest_mode(start_date, end_date)
            elif mode == "optimize":
                result = self.run_optimization_mode({})
            else:
                self.logger.error(f"Unknown mode: {mode}")
                return False
                
            # Return result
            return result
            
        except Exception as e:
            self.logger.error(f"Error starting platform: {str(e)}", exc_info=True)
            return False
    
    def stop(self) -> bool:
        """
        Stop the trading platform.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        self.logger.info("Stopping Forex Trading Platform")
        
        if not self.orchestrator:
            self.logger.warning("No active orchestrator to stop")
            return False
        
        try:
            # Shutdown orchestrator
            result = self.orchestrator.shutdown()
            
            # Update state
            self.is_running = False
            self.platform_status = "stopped"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error stopping platform: {str(e)}", exc_info=True)
            return False
    
    def restart(self) -> bool:
        """
        Restart the trading platform.
        
        Returns:
            bool: True if restarted successfully, False otherwise
        """
        self.logger.info("Restarting Forex Trading Platform")
        
        # Remember current settings
        current_mode = self.mode
        current_instruments = self.trading_instruments
        
        # Stop the platform
        if not self.stop():
            self.logger.error("Failed to stop platform for restart")
            return False
        
        # Wait a moment
        time.sleep(2)
        
        # Start the platform again
        return self.start(mode=current_mode, instruments=current_instruments)
    
    def status(self) -> Dict[str, Any]:
        """
        Get the current status of the trading platform.
        
        Returns:
            Dict[str, Any]: Status information
        """
        status_info = {
            "platform_status": self.platform_status,
            "mode": self.mode,
            "is_running": self.is_running,
            "trading_instruments": self.trading_instruments,
            "start_time": self.start_time
        }
        
        # Add orchestrator status if available
        if self.orchestrator:
            orchestrator_status = self.orchestrator.monitor_workflow()
            status_info["orchestrator"] = orchestrator_status
            
            # Calculate runtime
            if self.start_time:
                runtime = datetime.datetime.now() - self.start_time
                status_info["runtime_seconds"] = runtime.total_seconds()
        
        return status_info
    
    def run_paper_trading_mode(self) -> bool:
        """
        Run in paper trading mode.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        self.logger.info("Starting paper trading mode")
        
        try:
            # Update configuration for paper trading
            if "trading" not in self.config:
                self.config["trading"] = {}
            self.config["trading"]["mode"] = "paper"
            
            # Start the orchestrator workflow
            if not self.orchestrator.start_workflow():
                self.logger.error("Failed to start workflow")
                return False
            
            # Update state
            self.is_running = True
            self.platform_status = "running"
            self.start_time = datetime.datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in paper trading mode: {str(e)}", exc_info=True)
            return False
    
    def run_live_trading_mode(self) -> bool:
        """
        Run in live trading mode.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        self.logger.info("Starting live trading mode")
        
        try:
            # Check for API credentials
            if not self._validate_api_credentials():
                self.logger.error("Missing API credentials for live trading")
                return False
            
            # Update configuration for live trading
            if "trading" not in self.config:
                self.config["trading"] = {}
            self.config["trading"]["mode"] = "live"
            
            # Start the orchestrator workflow
            if not self.orchestrator.start_workflow():
                self.logger.error("Failed to start workflow")
                return False
            
            # Update state
            self.is_running = True
            self.platform_status = "running"
            self.start_time = datetime.datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in live trading mode: {str(e)}", exc_info=True)
            return False
    
    def _validate_api_credentials(self) -> bool:
        """
        Validate API credentials for live trading.
        
        Returns:
            bool: True if valid credentials are found, False otherwise
        """
        api_credentials = self.config.get("api_credentials", {})
        oanda_credentials = api_credentials.get("oanda", {})
        
        if not oanda_credentials.get("api_key") or not oanda_credentials.get("account_id"):
            self.logger.error("Missing OANDA API credentials")
            return False
            
        return True
    
    def run_backtest_mode(self, start_date: datetime.datetime, end_date: datetime.datetime) -> bool:
        """
        Run in backtest mode.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        self.logger.info(f"Starting backtest mode from {start_date} to {end_date}")
        
        try:
            # Update configuration for backtesting
            if "trading" not in self.config:
                self.config["trading"] = {}
            self.config["trading"]["mode"] = "backtest"
            self.config["trading"]["backtest"] = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
            
            # Start the orchestrator workflow
            if not self.orchestrator.start_workflow():
                self.logger.error("Failed to start workflow")
                return False
            
            # Update state
            self.is_running = True
            self.platform_status = "running"
            self.start_time = datetime.datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in backtest mode: {str(e)}", exc_info=True)
            return False
    
    def run_optimization_mode(self, parameters: Dict[str, Any]) -> bool:
        """
        Run in strategy optimization mode.
        
        Args:
            parameters: Parameters to optimize
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        self.logger.info(f"Starting optimization mode with parameters: {parameters}")
        
        try:
            # Update configuration for optimization
            if "trading" not in self.config:
                self.config["trading"] = {}
            self.config["trading"]["mode"] = "optimize"
            self.config["trading"]["optimization"] = {
                "parameters": parameters
            }
            
            # Start the orchestrator workflow
            if not self.orchestrator.start_workflow():
                self.logger.error("Failed to start workflow")
                return False
            
            # Update state
            self.is_running = True
            self.platform_status = "running"
            self.start_time = datetime.datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in optimization mode: {str(e)}", exc_info=True)
            return False
    
    def perform_graceful_shutdown(self) -> None:
        """Perform a graceful shutdown of the platform."""
        self.logger.info("Performing graceful shutdown")
        
        try:
            # Save application state
            self.save_application_state()
            
            # Stop the platform
            self.stop()
            
            # Generate shutdown report
            self.generate_shutdown_report()
            
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {str(e)}", exc_info=True)
        
        finally:
            self.logger.info("Graceful shutdown completed")
            sys.exit(0)
    
    def save_application_state(self) -> bool:
        """
        Save the application state before shutdown.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        self.logger.info("Saving application state")
        
        try:
            # Create state directory if it doesn't exist
            state_dir = Path("data/states")
            state_dir.mkdir(parents=True, exist_ok=True)
            
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save orchestrator state if available
            if self.orchestrator:
                state_file = f"data/states/platform_state_{timestamp}.json"
                if self.orchestrator.save_state(state_file):
                    self.logger.info(f"Orchestrator state saved to {state_file}")
            
            # Save platform status
            platform_status = self.status()
            status_file = f"data/states/platform_status_{timestamp}.json"
            
            with open(status_file, 'w') as f:
                json.dump(platform_status, f, indent=2, default=str)
                
            self.logger.info(f"Platform status saved to {status_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving application state: {str(e)}", exc_info=True)
            return False
    
    def generate_shutdown_report(self) -> Dict[str, Any]:
        """
        Generate a report on shutdown.
        
        Returns:
            Dict[str, Any]: Shutdown report
        """
        self.logger.info("Generating shutdown report")
        
        # Create report structure
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "runtime": None,
            "platform_status": self.platform_status,
            "mode": self.mode,
            "instruments": self.trading_instruments,
            "metrics": {}
        }
        
        # Add runtime if available
        if self.start_time:
            runtime = datetime.datetime.now() - self.start_time
            report["runtime"] = str(runtime)
            report["runtime_seconds"] = runtime.total_seconds()
        
        # Add orchestrator metrics if available
        if self.orchestrator:
            report["metrics"] = self.orchestrator.generate_workflow_metrics()
        
        # Save report to file
        try:
            # Create reports directory if it doesn't exist
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save report
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"data/reports/shutdown_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            self.logger.info(f"Shutdown report saved to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving shutdown report: {str(e)}", exc_info=True)
        
        return report

def handle_keyboard_interrupt(platform: Optional[ForexTradingPlatform] = None) -> None:
    """
    Handle Ctrl+C interrupt.
    
    Args:
        platform: Forex Trading Platform instance
    """
    print("\nKeyboard interrupt received")
    if platform:
        platform.perform_graceful_shutdown()
    else:
        print("Exiting...")
        sys.exit(0)

def handle_start_command(args: argparse.Namespace, platform: ForexTradingPlatform) -> int:
    """
    Handle the start command.
    
    Args:
        args: Command-line arguments
        platform: Forex Trading Platform instance
    
    Returns:
        int: Exit code
    """
    # Parse instruments
    instruments = args.instruments.split(',') if args.instruments else None
    
    # Start the platform in the specified mode
    result = platform.start(mode=args.mode, instruments=instruments)
    
    if result:
        print(f"Forex Trading Platform started in {args.mode} mode")
        
        # For interactive modes, keep the main thread running
        if args.mode in ["live", "paper"]:
            try:
                while platform.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                handle_keyboard_interrupt(platform)
                
        return 0
    else:
        print("Failed to start Forex Trading Platform")
        return 1

def handle_stop_command(args: argparse.Namespace, platform: ForexTradingPlatform) -> int:
    """
    Handle the stop command.
    
    Args:
        args: Command-line arguments
        platform: Forex Trading Platform instance
    
    Returns:
        int: Exit code
    """
    result = platform.stop()
    
    if result:
        print("Forex Trading Platform stopped")
        return 0
    else:
        print("Failed to stop Forex Trading Platform")
        return 1

def handle_status_command(args: argparse.Namespace, platform: ForexTradingPlatform) -> int:
    """
    Handle the status command.
    
    Args:
        args: Command-line arguments
        platform: Forex Trading Platform instance
    
    Returns:
        int: Exit code
    """
    status = platform.status()
    
    # Display status information
    print("\nForex Trading Platform Status:")
    print(f"Status: {status['platform_status']}")
    print(f"Mode: {status['mode']}")
    print(f"Running: {'Yes' if status['is_running'] else 'No'}")
    print(f"Instruments: {', '.join(status['trading_instruments'])}")
    
    if 'start_time' in status and status['start_time']:
        print(f"Start Time: {status['start_time']}")
        
    if 'runtime_seconds' in status:
        runtime = datetime.timedelta(seconds=status['runtime_seconds'])
        print(f"Runtime: {runtime}")
    
    # If orchestrator is running, display more details
    if 'orchestrator' in status:
        orch_status = status['orchestrator']
        print("\nOrchestrator Status:")
        print(f"Cycle Count: {orch_status.get('cycle_count', 0)}")
        print(f"Workflow ID: {orch_status.get('workflow_id', 'N/A')}")
        
        # Show agent statuses
        if 'agent_status' in orch_status:
            print("\nAgent Status:")
            for agent, agent_status in orch_status['agent_status'].items():
                print(f"  {agent}: {agent_status}")
    
    return 0

def handle_backtest_command(args: argparse.Namespace, platform: ForexTradingPlatform) -> int:
    """
    Handle the backtest command.
    
    Args:
        args: Command-line arguments
        platform: Forex Trading Platform instance
    
    Returns:
        int: Exit code
    """
    try:
        # Parse dates
        start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Parse instruments
        instruments = args.instruments.split(',') if args.instruments else None
        
        # Run backtest
        platform.start(mode="backtest", instruments=instruments)
        result = platform.run_backtest_mode(start_date, end_date)
        
        if result:
            print(f"Backtest started from {args.start_date} to {args.end_date}")
            
            try:
                while platform.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                handle_keyboard_interrupt(platform)
                
            return 0
        else:
            print("Failed to start backtest")
            return 1
            
    except ValueError as e:
        print(f"Error parsing dates: {str(e)}")
        return 1

def handle_optimize_command(args: argparse.Namespace, platform: ForexTradingPlatform) -> int:
    """
    Handle the optimize command.
    
    Args:
        args: Command-line arguments
        platform: Forex Trading Platform instance
    
    Returns:
        int: Exit code
    """
    try:
        # Parse dates
        start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Parse parameters
        parameters = {}
        for param_str in args.parameters.split(','):
            param_parts = param_str.split('=')
            if len(param_parts) == 2:
                param_name, param_value = param_parts
                parameters[param_name.strip()] = param_value.strip()
        
        # Run optimization
        platform.start(mode="optimize", instruments=None)
        result = platform.run_optimization_mode(parameters)
        
        if result:
            print(f"Optimization started with parameters: {parameters}")
            
            try:
                while platform.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                handle_keyboard_interrupt(platform)
                
            return 0
        else:
            print("Failed to start optimization")
            return 1
            
    except ValueError as e:
        print(f"Error parsing inputs: {str(e)}")
        return 1

def main() -> int:
    """
    Main function to run the Forex Trading Platform.
    
    Returns:
        int: Exit code
    """
    try:
        # Load environment variables
        load_environment_variables()
        
        # Parse command-line arguments
        args = parse_command_line_arguments()
        
        # Initialize logging
        logger = initialize_logging(log_level=args.log_level, log_file=args.log_file)
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Missing required dependencies")
            return 1
        
        # Initialize configuration
        config = initialize_config(args.config)
        
        # Create platform instance
        platform = ForexTradingPlatform(config=config, logger=logger)
        
        # Handle commands
        if args.command == "start":
            return handle_start_command(args, platform)
        elif args.command == "stop":
            return handle_stop_command(args, platform)
        elif args.command == "status":
            return handle_status_command(args, platform)
        elif args.command == "backtest":
            return handle_backtest_command(args, platform)
        elif args.command == "optimize":
            return handle_optimize_command(args, platform)
        else:
            # No command specified, show help
            create_argument_parser().print_help()
            return 0
        
    except KeyboardInterrupt:
        handle_keyboard_interrupt()
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 