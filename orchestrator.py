#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orchestrator for Forex Trading Platform

This module provides a comprehensive Orchestrator class that coordinates
all agents in the forex trading platform using LangGraph for workflow
orchestration. It manages agent initialization, workflow definition,
state management, and monitoring.
"""

import os
import time
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from datetime import datetime
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

# Workflow and state management
import langgraph.graph as lg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

# Import agents
from market_data_agent.agent import MarketDataAgent
from technical_analyst_agent.agent import TechnicalAnalystAgent
from fundamentals_agent.agent import FundamentalsAgent
from sentiment_agent.agent import SentimentAgent
from risk_manager_agent.agent import RiskManagerAgent
from portfolio_manager_agent.agent import PortfolioManagerAgent

# Utils
from utils.config_manager import ConfigManager
from utils.base_agent import BaseAgent, AgentState, AgentMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Orchestrator:
    """
    Orchestrator class that coordinates all agents in the forex trading platform.
    
    This class is responsible for:
    1. Agent initialization and management
    2. LangGraph workflow definition and execution
    3. State management
    4. Monitoring and control
    
    It serves as the central component that ensures all agents work together
    coherently within the defined workflow.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            config: Configuration parameters for the orchestrator and agents
            logger: Logger instance for the orchestrator
        """
        # Initialize configuration
        if config is None:
            config_manager = ConfigManager()
            self.config = config_manager.as_dict()
        else:
            self.config = config
            
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger("orchestrator")
        else:
            self.logger = logger
            
        # Agent registry and management
        self.agents = {}
        self.agent_status = {}
        
        # Workflow state
        self.workflow_id = str(uuid.uuid4())
        self.workflow_state = self._initialize_state()
        self.is_running = False
        self.is_paused = False
        
        # Error handling
        self.error_handlers = {}
        self.max_retries = self.config.get("orchestrator", {}).get("max_retries", 3)
        
        # LangGraph components
        self.graph = None
        self.memory_saver = MemorySaver()
        self.compiled_graph = None
        
        # Metrics and monitoring
        self.metrics = {
            "cycle_count": 0,
            "start_time": None,
            "end_time": None,
            "agent_execution_times": {},
            "errors": [],
            "successful_cycles": 0
        }
        
        self.logger.info("Orchestrator initialized with workflow ID: %s", self.workflow_id) 

    def _initialize_state(self) -> Dict[str, Any]:
        """
        Initialize the orchestrator's workflow state.
        
        Returns:
            Dict[str, Any]: Initial state dictionary
        """
        return {
            "workflow_id": self.workflow_id,
            "messages": [],
            "market_data": {},
            "technical_analysis": {},
            "fundamentals": {},
            "sentiment": {},
            "risk_assessment": {},
            "portfolio_decisions": {},
            "execution_results": {},
            "errors": [],
            "status": "initialized",
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "cycle_count": 0
        }
    
    def initialize_agents(self) -> bool:
        """
        Initialize all six agents for the forex trading platform.
        
        This method creates instances of all required agents and initializes
        them with the appropriate configuration.
        
        Returns:
            bool: True if all agents initialized successfully, False otherwise
        """
        self.logger.info("Initializing all agents")
        
        try:
            # Initialize Market Data Agent
            self.agents["market_data"] = MarketDataAgent(
                agent_name="market_data_agent",
                config=self.config.get("market_data", {}),
                logger=self.logger.getChild("market_data_agent")
            )
            
            # Initialize Technical Analyst Agent
            self.agents["technical_analyst"] = TechnicalAnalystAgent(
                agent_name="technical_analyst_agent",
                config=self.config.get("technical_analysis", {}),
                logger=self.logger.getChild("technical_analyst_agent")
            )
            
            # Initialize Fundamentals Agent
            self.agents["fundamentals"] = FundamentalsAgent(
                agent_name="fundamentals_agent",
                config=self.config.get("fundamentals", {}),
                logger=self.logger.getChild("fundamentals_agent")
            )
            
            # Initialize Sentiment Agent
            self.agents["sentiment"] = SentimentAgent(
                agent_name="sentiment_agent",
                config=self.config.get("sentiment", {}),
                logger=self.logger.getChild("sentiment_agent")
            )
            
            # Initialize Risk Manager Agent
            self.agents["risk_manager"] = RiskManagerAgent(
                agent_name="risk_manager_agent",
                config=self.config.get("risk_management", {}),
                logger=self.logger.getChild("risk_manager_agent")
            )
            
            # Initialize Portfolio Manager Agent
            self.agents["portfolio_manager"] = PortfolioManagerAgent(
                agent_name="portfolio_manager_agent",
                config=self.config.get("portfolio_management", {}),
                logger=self.logger.getChild("portfolio_manager_agent")
            )
            
            # Initialize all agents
            for name, agent in self.agents.items():
                success = agent.initialize()
                self.agent_status[name] = "ready" if success else "error"
                if not success:
                    self.logger.error(f"Failed to initialize agent: {name}")
                    return False
            
            self.logger.info("All agents initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {str(e)}", exc_info=True)
            return False
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get a specific agent by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Optional[BaseAgent]: The requested agent or None if not found
        """
        if agent_name not in self.agents:
            self.logger.warning(f"Agent not found: {agent_name}")
            return None
        
        return self.agents.get(agent_name)
    
    def start_all_agents(self) -> bool:
        """
        Start all agents in the platform.
        
        Returns:
            bool: True if all agents started successfully, False otherwise
        """
        self.logger.info("Starting all agents")
        
        success = True
        for name, agent in self.agents.items():
            try:
                # Check if agent is ready
                if self.agent_status.get(name) != "ready":
                    # Try to re-initialize if not ready
                    if not agent.initialize():
                        self.logger.error(f"Could not initialize agent: {name}")
                        success = False
                        continue
                
                # Update agent status
                self.agent_status[name] = "running"
                self.logger.info(f"Agent started: {name}")
                
            except Exception as e:
                self.logger.error(f"Error starting agent {name}: {str(e)}", exc_info=True)
                self.agent_status[name] = "error"
                success = False
        
        return success
    
    def stop_all_agents(self) -> bool:
        """
        Stop all agents in the platform.
        
        Returns:
            bool: True if all agents stopped successfully, False otherwise
        """
        self.logger.info("Stopping all agents")
        
        success = True
        for name, agent in self.agents.items():
            try:
                # Call shutdown method if agent is running
                if self.agent_status.get(name) == "running":
                    if not agent.shutdown():
                        self.logger.error(f"Error shutting down agent: {name}")
                        success = False
                        continue
                
                # Update agent status
                self.agent_status[name] = "stopped"
                self.logger.info(f"Agent stopped: {name}")
                
            except Exception as e:
                self.logger.error(f"Error stopping agent {name}: {str(e)}", exc_info=True)
                success = False
        
        return success
    
    def check_agent_status(self, agent_name: str) -> str:
        """
        Check the status of a specific agent.
        
        Args:
            agent_name: Name of the agent to check
            
        Returns:
            str: The current status of the agent
        """
        if agent_name not in self.agent_status:
            return "not_found"
        
        return self.agent_status.get(agent_name, "unknown")
    
    def handle_agent_failures(self, agent_name: str, error: Exception) -> bool:
        """
        Handle failures in any agent.
        
        This method implements failure recovery mechanisms for agents,
        including retry logic and fallback mechanisms.
        
        Args:
            agent_name: Name of the agent that failed
            error: The exception that caused the failure
            
        Returns:
            bool: True if the failure was recovered, False otherwise
        """
        self.logger.error(f"Agent failure: {agent_name} - {str(error)}", exc_info=True)
        
        # Record the error
        self.workflow_state["errors"].append({
            "agent": agent_name,
            "error": str(error),
            "timestamp": datetime.now(),
            "stack_trace": traceback.format_exc()
        })
        
        # Check for custom error handler
        if agent_name in self.error_handlers:
            try:
                return self.error_handlers[agent_name](error)
            except Exception as e:
                self.logger.error(f"Error in custom error handler for {agent_name}: {str(e)}")
        
        # Default error recovery - retry logic
        agent = self.get_agent(agent_name)
        if agent is None:
            return False
        
        # Try to reinitialize the agent
        retry_count = 0
        while retry_count < self.max_retries:
            retry_count += 1
            self.logger.info(f"Attempting to recover {agent_name} (attempt {retry_count}/{self.max_retries})")
            
            try:
                # Reinitialize the agent
                if agent.initialize():
                    self.agent_status[agent_name] = "ready"
                    self.logger.info(f"Successfully recovered agent: {agent_name}")
                    return True
            except Exception as retry_error:
                self.logger.error(f"Retry failed for {agent_name}: {str(retry_error)}")
            
            # Wait before next retry
            time.sleep(2 ** retry_count)  # Exponential backoff
        
        # Mark agent as failed if recovery unsuccessful
        self.agent_status[agent_name] = "failed"
        return False
    
    def define_workflow(self) -> None:
        """
        Define the workflow between agents using LangGraph.
        
        This method sets up the workflow structure, defining how agents
        interact with each other and the sequence of operations.
        """
        self.logger.info("Defining LangGraph workflow")
        
        try:
            # Create graph
            self.graph = self.create_graph()
            
            # Compile graph with memory saver
            self.compiled_graph = self.graph.compile(checkpointer=self.memory_saver)
            
            self.logger.info("LangGraph workflow defined successfully")
            
        except Exception as e:
            self.logger.error(f"Error defining workflow: {str(e)}", exc_info=True)
            raise
    
    def create_graph(self) -> StateGraph:
        """
        Create the LangGraph graph structure.
        
        This method creates a StateGraph with all the necessary nodes
        and edges to define the workflow between agents.
        
        Returns:
            StateGraph: The configured LangGraph graph
        """
        # Create a state graph
        state_schema = {
            "workflow_id": str,
            "messages": list,
            "market_data": dict,
            "technical_analysis": dict,
            "fundamentals": dict,
            "sentiment": dict,
            "risk_assessment": dict,
            "portfolio_decisions": dict,
            "execution_results": dict,
            "errors": list,
            "status": str,
            "created_at": datetime,
            "last_updated": datetime,
            "cycle_count": int
        }
        
        graph = StateGraph(state_schema)
        
        # Add nodes for each agent
        for name, agent in self.agents.items():
            node_func = self.define_node_for_agent(name, agent)
            graph.add_node(name, node_func)
            
        # Define edges between nodes
        self.define_edges(graph)
        
        return graph
    
    def define_node_for_agent(self, agent_name: str, agent: BaseAgent) -> callable:
        """
        Define a LangGraph node for an agent.
        
        This method creates a node function that can be added to the graph,
        handling the execution of agent tasks and error handling.
        
        Args:
            agent_name: Name of the agent
            agent: The agent instance
            
        Returns:
            callable: A function that processes state for this node
        """
        def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            self.logger.info(f"Executing node for {agent_name}")
            
            try:
                # Record start time for metrics
                start_time = time.time()
                
                # Create task for the agent based on current state
                task = self.create_task_for_agent(agent_name, state)
                
                # Execute agent task
                result = agent.run_task(task)
                
                # Update metrics
                end_time = time.time()
                execution_time = end_time - start_time
                if agent_name not in self.metrics["agent_execution_times"]:
                    self.metrics["agent_execution_times"][agent_name] = []
                self.metrics["agent_execution_times"][agent_name].append(execution_time)
                
                # Update state with results
                updated_state = state.copy()
                updated_state[self.get_result_key_for_agent(agent_name)] = result
                updated_state["last_updated"] = datetime.now()
                
                return updated_state
                
            except Exception as e:
                # Handle agent failure
                success = self.handle_agent_failures(agent_name, e)
                
                if success:
                    # Retry execution
                    task = self.create_task_for_agent(agent_name, state)
                    result = agent.run_task(task)
                    
                    # Update state with results after recovery
                    updated_state = state.copy()
                    updated_state[self.get_result_key_for_agent(agent_name)] = result
                    updated_state["last_updated"] = datetime.now()
                    
                    return updated_state
                else:
                    # Update state with error
                    updated_state = state.copy()
                    updated_state["errors"].append({
                        "agent": agent_name,
                        "error": str(e),
                        "timestamp": datetime.now()
                    })
                    updated_state["status"] = "error"
                    updated_state["last_updated"] = datetime.now()
                    
                    return updated_state
        
        return node_function
    
    def create_task_for_agent(self, agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a task object for a specific agent based on the current state.
        
        Args:
            agent_name: Name of the agent
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Task object for the agent
        """
        # Common task properties
        task = {
            "task_id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "workflow_id": state["workflow_id"],
            "agent": agent_name
        }
        
        # Agent-specific task data
        if agent_name == "market_data":
            # Market data agent doesn't need other agent outputs
            task["action"] = "fetch_market_data"
            task["parameters"] = {
                "instruments": self.config.get("trading", {}).get("instruments", []),
                "timeframes": self.config.get("trading", {}).get("timeframes", ["H1"]),
                "count": 500  # Default number of candles
            }
            
        elif agent_name == "technical_analyst":
            # Technical analyst needs market data
            task["action"] = "analyze_technical"
            task["parameters"] = {
                "market_data": state.get("market_data", {}),
                "indicators": self.config.get("technical_analysis", {}).get("indicators", [])
            }
            
        elif agent_name == "fundamentals":
            # Fundamentals agent works independently
            task["action"] = "analyze_fundamentals"
            task["parameters"] = {
                "currencies": self.get_currencies_from_instruments(
                    self.config.get("trading", {}).get("instruments", [])
                ),
                "time_horizon": self.config.get("fundamentals", {}).get("time_horizon", "medium")
            }
            
        elif agent_name == "sentiment":
            # Sentiment agent works independently
            task["action"] = "analyze_sentiment"
            task["parameters"] = {
                "currencies": self.get_currencies_from_instruments(
                    self.config.get("trading", {}).get("instruments", [])
                ),
                "sources": self.config.get("sentiment", {}).get("sources", ["news", "social"])
            }
            
        elif agent_name == "risk_manager":
            # Risk manager needs all analysis data
            task["action"] = "assess_risk"
            task["parameters"] = {
                "market_data": state.get("market_data", {}),
                "technical_analysis": state.get("technical_analysis", {}),
                "fundamentals": state.get("fundamentals", {}),
                "sentiment": state.get("sentiment", {}),
                "portfolio": self.config.get("portfolio", {})
            }
            
        elif agent_name == "portfolio_manager":
            # Portfolio manager needs everything
            task["action"] = "manage_portfolio"
            task["parameters"] = {
                "market_data": state.get("market_data", {}),
                "technical_analysis": state.get("technical_analysis", {}),
                "fundamentals": state.get("fundamentals", {}),
                "sentiment": state.get("sentiment", {}),
                "risk_assessment": state.get("risk_assessment", {}),
                "portfolio": self.config.get("portfolio", {})
            }
            
        return task
    
    def get_result_key_for_agent(self, agent_name: str) -> str:
        """
        Get the state key where an agent's results should be stored.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            str: The key in the state dict to store results
        """
        # Map agent names to their corresponding state keys
        key_mapping = {
            "market_data": "market_data",
            "technical_analyst": "technical_analysis",
            "fundamentals": "fundamentals",
            "sentiment": "sentiment",
            "risk_manager": "risk_assessment",
            "portfolio_manager": "portfolio_decisions"
        }
        
        return key_mapping.get(agent_name, agent_name)
    
    def get_currencies_from_instruments(self, instruments: List[str]) -> List[str]:
        """
        Extract unique currencies from instrument pairs.
        
        Args:
            instruments: List of currency pairs (e.g., ["EUR_USD", "GBP_JPY"])
            
        Returns:
            List[str]: List of unique currencies
        """
        currencies = set()
        for instrument in instruments:
            parts = instrument.split('_')
            if len(parts) == 2:
                currencies.add(parts[0])
                currencies.add(parts[1])
                
        return list(currencies)
    
    def define_edges(self, graph: StateGraph) -> None:
        """
        Define the connections between nodes in the graph.
        
        This method adds edges between agent nodes to define the workflow.
        
        Args:
            graph: The StateGraph to add edges to
        """
        # Start with market data agent
        graph.add_edge(START, "market_data")
        
        # Market data feeds into all analysis agents
        graph.add_conditional_edges(
            "market_data",
            self.route_after_market_data,
            {
                "technical_analyst": "technical_analyst",
                "fundamentals": "fundamentals",
                "sentiment": "sentiment"
            }
        )
        
        # After analysis agents, go to risk manager
        for analysis_agent in ["technical_analyst", "fundamentals", "sentiment"]:
            graph.add_edge(analysis_agent, "risk_manager")
        
        # Risk manager feeds into portfolio manager
        graph.add_edge("risk_manager", "portfolio_manager")
        
        # End after portfolio manager
        graph.add_edge("portfolio_manager", END)
    
    def route_after_market_data(self, state: Dict[str, Any]) -> str:
        """
        Conditional routing function after market data processing.
        
        In this implementation, we route to all analysis agents in parallel.
        
        Args:
            state: Current workflow state
            
        Returns:
            str: Next node to route to
        """
        # This function needs to return one of the keys from the conditional edges dict
        # In practice, we want to run all analysis agents in parallel
        # For LangGraph, we would need a true parallel execution strategy
        # As a workaround, we return a key to start the sequence
        return "technical_analyst"
    
    def start_workflow(self) -> bool:
        """
        Start the workflow execution.
        
        This method initializes the workflow and starts the execution cycle.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        self.logger.info("Starting workflow execution")
        
        try:
            # Check if agents are initialized
            if not self.agents:
                self.initialize_agents()
                
            # Start all agents
            if not self.start_all_agents():
                self.logger.error("Failed to start all agents")
                return False
                
            # Define workflow if not already defined
            if self.graph is None:
                self.define_workflow()
                
            # Initialize workflow state
            self.workflow_state = self.initialize_state()
            
            # Set workflow as running
            self.is_running = True
            self.is_paused = False
            self.metrics["start_time"] = datetime.now()
            
            # Start the execution thread
            self._start_execution_thread()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting workflow: {str(e)}", exc_info=True)
            return False
    
    def _start_execution_thread(self) -> None:
        """
        Start a background thread for workflow execution.
        
        This method runs the workflow in a separate thread to avoid blocking.
        """
        import threading
        
        execution_thread = threading.Thread(
            target=self._execution_loop,
            daemon=True
        )
        execution_thread.start()
        self.logger.info("Workflow execution thread started")
    
    def _execution_loop(self) -> None:
        """
        Main execution loop for the workflow.
        
        This method runs continuously while the workflow is active,
        processing market data and running the analysis cycle.
        """
        self.logger.info("Starting workflow execution loop")
        
        try:
            while self.is_running:
                if self.is_paused:
                    time.sleep(1)
                    continue
                
                # Run one complete workflow cycle
                self.complete_workflow_cycle()
                
                # Increment cycle count
                self.metrics["cycle_count"] += 1
                self.workflow_state["cycle_count"] += 1
                
                # Sleep between cycles based on configuration
                cycle_interval = self.config.get("orchestrator", {}).get("cycle_interval", 300)
                time.sleep(cycle_interval)
                
        except Exception as e:
            self.logger.error(f"Error in workflow execution loop: {str(e)}", exc_info=True)
            self.is_running = False
            
        finally:
            self.metrics["end_time"] = datetime.now()
            self.logger.info("Workflow execution loop ended")
    
    def complete_workflow_cycle(self) -> Dict[str, Any]:
        """
        Complete one cycle of the workflow.
        
        This method executes a full cycle through all agents in the workflow,
        from market data to portfolio decisions.
        
        Returns:
            Dict[str, Any]: Updated state after the cycle completion
        """
        self.logger.info("Starting workflow cycle")
        cycle_start_time = time.time()
        
        try:
            # Start with clean state for this cycle
            cycle_state = self.get_current_state()
            
            # Run the compiled graph
            config = {"configurable": {"thread_id": self.workflow_id}}
            result = self.compiled_graph.invoke(cycle_state, config=config)
            
            # Update the workflow state
            self.update_state(result)
            
            # Update metrics
            cycle_end_time = time.time()
            cycle_duration = cycle_end_time - cycle_start_time
            self.metrics.setdefault("cycle_durations", []).append(cycle_duration)
            self.metrics["successful_cycles"] += 1
            
            self.logger.info(f"Workflow cycle completed in {cycle_duration:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in workflow cycle: {str(e)}", exc_info=True)
            self.workflow_state["errors"].append({
                "type": "workflow_cycle",
                "error": str(e),
                "timestamp": datetime.now()
            })
            self.workflow_state["status"] = "error"
            return self.workflow_state
    
    def process_market_data(self) -> Dict[str, Any]:
        """
        Process incoming market data.
        
        This method handles the initial step in the workflow where
        market data is fetched and processed.
        
        Returns:
            Dict[str, Any]: Updated state with market data
        """
        self.logger.info("Processing market data")
        
        try:
            # Get market data agent
            market_data_agent = self.get_agent("market_data")
            if not market_data_agent:
                raise ValueError("Market data agent not found")
                
            # Create task for market data agent
            task = self.create_task_for_agent("market_data", self.workflow_state)
            
            # Run the task
            result = market_data_agent.run_task(task)
            
            # Update state
            updated_state = self.workflow_state.copy()
            updated_state["market_data"] = result
            updated_state["last_updated"] = datetime.now()
            self.update_state(updated_state)
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}", exc_info=True)
            self.handle_agent_failures("market_data", e)
            return self.workflow_state
    
    def run_analysis_agents(self) -> Dict[str, Any]:
        """
        Run the three analysis agents in parallel.
        
        This method executes the technical, fundamental, and sentiment
        analysis agents concurrently for optimal performance.
        
        Returns:
            Dict[str, Any]: Updated state with analysis results
        """
        self.logger.info("Running analysis agents in parallel")
        
        analysis_agents = ["technical_analyst", "fundamentals", "sentiment"]
        analysis_results = {}
        
        try:
            # Create a thread pool
            with ThreadPoolExecutor(max_workers=len(analysis_agents)) as executor:
                # Submit tasks for each analysis agent
                future_to_agent = {
                    executor.submit(self._run_analysis_agent, agent_name): agent_name
                    for agent_name in analysis_agents
                }
                
                # Collect results
                for future in future_to_agent:
                    agent_name = future_to_agent[future]
                    try:
                        result = future.result()
                        analysis_results[self.get_result_key_for_agent(agent_name)] = result
                    except Exception as e:
                        self.logger.error(f"Error in {agent_name}: {str(e)}", exc_info=True)
                        self.handle_agent_failures(agent_name, e)
            
            # Update state with all analysis results
            updated_state = self.workflow_state.copy()
            updated_state.update(analysis_results)
            updated_state["last_updated"] = datetime.now()
            self.update_state(updated_state)
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error running analysis agents: {str(e)}", exc_info=True)
            return self.workflow_state
    
    def _run_analysis_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Run a single analysis agent.
        
        Args:
            agent_name: Name of the analysis agent to run
            
        Returns:
            Dict[str, Any]: Results from the agent
        """
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")
            
        task = self.create_task_for_agent(agent_name, self.workflow_state)
        return agent.run_task(task)
    
    def process_risk_management(self) -> Dict[str, Any]:
        """
        Process risk management decisions.
        
        This method handles the risk assessment step after analysis.
        
        Returns:
            Dict[str, Any]: Updated state with risk assessment
        """
        self.logger.info("Processing risk management")
        
        try:
            # Get risk manager agent
            risk_manager = self.get_agent("risk_manager")
            if not risk_manager:
                raise ValueError("Risk manager agent not found")
                
            # Create task for risk manager
            task = self.create_task_for_agent("risk_manager", self.workflow_state)
            
            # Run the task
            result = risk_manager.run_task(task)
            
            # Update state
            updated_state = self.workflow_state.copy()
            updated_state["risk_assessment"] = result
            updated_state["last_updated"] = datetime.now()
            self.update_state(updated_state)
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error in risk management: {str(e)}", exc_info=True)
            self.handle_agent_failures("risk_manager", e)
            return self.workflow_state
    
    def execute_portfolio_decisions(self) -> Dict[str, Any]:
        """
        Execute portfolio management decisions.
        
        This method handles the final step in the workflow where
        trading decisions are made and executed.
        
        Returns:
            Dict[str, Any]: Updated state with execution results
        """
        self.logger.info("Executing portfolio decisions")
        
        try:
            # Get portfolio manager agent
            portfolio_manager = self.get_agent("portfolio_manager")
            if not portfolio_manager:
                raise ValueError("Portfolio manager agent not found")
                
            # Create task for portfolio manager
            task = self.create_task_for_agent("portfolio_manager", self.workflow_state)
            
            # Run the task
            result = portfolio_manager.run_task(task)
            
            # Update state
            updated_state = self.workflow_state.copy()
            updated_state["portfolio_decisions"] = result
            updated_state["execution_results"] = {
                "timestamp": datetime.now(),
                "status": "executed",
                "details": result.get("execution_details", {})
            }
            updated_state["last_updated"] = datetime.now()
            self.update_state(updated_state)
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error executing portfolio decisions: {str(e)}", exc_info=True)
            self.handle_agent_failures("portfolio_manager", e)
            return self.workflow_state
    
    def initialize_state(self) -> Dict[str, Any]:
        """
        Initialize the workflow state.
        
        This method creates a clean initial state for the workflow.
        
        Returns:
            Dict[str, Any]: Initial state dictionary
        """
        return self._initialize_state()
    
    def update_state(self, new_data: Dict[str, Any]) -> None:
        """
        Update the workflow state.
        
        Args:
            new_data: New data to update the state with
        """
        # Update state
        self.workflow_state.update(new_data)
        self.workflow_state["last_updated"] = datetime.now()
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current workflow state.
        
        Returns:
            Dict[str, Any]: Current state
        """
        return self.workflow_state
    
    def save_state(self, filepath: str) -> bool:
        """
        Save the current state to a file.
        
        Args:
            filepath: Path to save the state to
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Prepare state for serialization
            serializable_state = self._prepare_state_for_serialization(self.workflow_state)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serializable_state, f, indent=2)
                
            self.logger.info(f"Workflow state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}", exc_info=True)
            return False
    
    def _prepare_state_for_serialization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare state for JSON serialization by converting non-serializable objects.
        
        Args:
            state: State dictionary to prepare
            
        Returns:
            Dict[str, Any]: Serializable state dictionary
        """
        serializable_state = {}
        
        for key, value in state.items():
            # Handle datetime objects
            if isinstance(value, datetime):
                serializable_state[key] = value.isoformat()
            # Handle complex objects with their own serialization
            elif hasattr(value, 'to_dict'):
                serializable_state[key] = value.to_dict()
            # Handle dictionaries recursively
            elif isinstance(value, dict):
                serializable_state[key] = self._prepare_state_for_serialization(value)
            # Handle lists recursively
            elif isinstance(value, list):
                serializable_state[key] = [
                    self._prepare_state_for_serialization(item) if isinstance(item, dict) 
                    else (item.isoformat() if isinstance(item, datetime) else item)
                    for item in value
                ]
            # Handle basic serializable types
            else:
                serializable_state[key] = value
                
        return serializable_state
    
    def load_state(self, filepath: str) -> bool:
        """
        Load a state from a file.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                self.logger.error(f"State file not found: {filepath}")
                return False
                
            # Load from file
            with open(filepath, 'r') as f:
                loaded_state = json.load(f)
                
            # Convert back to proper types
            restored_state = self._restore_state_from_serialization(loaded_state)
            
            # Update state
            self.workflow_state = restored_state
            
            self.logger.info(f"Workflow state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}", exc_info=True)
            return False
    
    def _restore_state_from_serialization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore state from serialized format.
        
        Args:
            state: Serialized state dictionary
            
        Returns:
            Dict[str, Any]: Restored state dictionary
        """
        restored_state = {}
        
        for key, value in state.items():
            # Handle datetime strings
            if key in ["created_at", "last_updated"] and isinstance(value, str):
                try:
                    restored_state[key] = datetime.fromisoformat(value)
                except ValueError:
                    restored_state[key] = value
            # Handle dictionaries recursively
            elif isinstance(value, dict):
                restored_state[key] = self._restore_state_from_serialization(value)
            # Handle lists recursively
            elif isinstance(value, list):
                restored_state[key] = [
                    self._restore_state_from_serialization(item) if isinstance(item, dict) else item
                    for item in value
                ]
            # Handle other types
            else:
                restored_state[key] = value
                
        return restored_state
    
    def monitor_workflow(self) -> Dict[str, Any]:
        """
        Monitor the workflow execution.
        
        This method returns the current state of the workflow,
        including metrics and agent statuses.
        
        Returns:
            Dict[str, Any]: Workflow monitoring information
        """
        # Collect current data
        monitoring_data = {
            "workflow_id": self.workflow_id,
            "status": "running" if self.is_running else "stopped",
            "paused": self.is_paused,
            "cycle_count": self.metrics["cycle_count"],
            "start_time": self.metrics["start_time"],
            "current_time": datetime.now(),
            "agent_status": self.agent_status,
            "errors": self.workflow_state["errors"],
            "metrics": self.generate_workflow_metrics()
        }
        
        # Calculate running time if available
        if self.metrics["start_time"]:
            if self.metrics["end_time"]:
                running_time = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
            else:
                running_time = (datetime.now() - self.metrics["start_time"]).total_seconds()
            monitoring_data["running_time_seconds"] = running_time
        
        return monitoring_data
    
    def pause_workflow(self) -> bool:
        """
        Pause the workflow.
        
        Returns:
            bool: True if paused successfully, False otherwise
        """
        self.logger.info("Pausing workflow")
        
        if not self.is_running:
            self.logger.warning("Cannot pause: workflow is not running")
            return False
            
        self.is_paused = True
        return True
    
    def resume_workflow(self) -> bool:
        """
        Resume the workflow.
        
        Returns:
            bool: True if resumed successfully, False otherwise
        """
        self.logger.info("Resuming workflow")
        
        if not self.is_running:
            self.logger.warning("Cannot resume: workflow is not running")
            return False
            
        if not self.is_paused:
            self.logger.warning("Workflow is not paused")
            return False
            
        self.is_paused = False
        return True
    
    def handle_external_events(self, event: Dict[str, Any]) -> bool:
        """
        Handle external events.
        
        This method processes events from external sources that may
        affect the workflow execution, such as market disruptions,
        emergency stops, or configuration updates.
        
        Args:
            event: Event data
            
        Returns:
            bool: True if event handled successfully, False otherwise
        """
        self.logger.info(f"Handling external event: {event.get('type', 'unknown')}")
        
        try:
            event_type = event.get("type")
            
            if event_type == "emergency_stop":
                # Emergency stop workflow
                self.is_running = False
                self.stop_all_agents()
                self.workflow_state["status"] = "emergency_stopped"
                
            elif event_type == "market_disruption":
                # Pause workflow during market disruption
                self.pause_workflow()
                self.workflow_state["status"] = "paused_market_disruption"
                
            elif event_type == "config_update":
                # Update configuration
                new_config = event.get("config", {})
                self.config.update(new_config)
                self.workflow_state["status"] = "config_updated"
                
            elif event_type == "force_cycle":
                # Force a new workflow cycle
                self.complete_workflow_cycle()
                
            else:
                self.logger.warning(f"Unknown event type: {event_type}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling external event: {str(e)}", exc_info=True)
            return False
    
    def generate_workflow_metrics(self) -> Dict[str, Any]:
        """
        Generate metrics about the workflow execution.
        
        This method collects and calculates various metrics about
        the workflow execution, such as performance, error rates,
        and execution times.
        
        Returns:
            Dict[str, Any]: Workflow metrics
        """
        metrics = {
            "total_cycles": self.metrics["cycle_count"],
            "successful_cycles": self.metrics["successful_cycles"],
            "error_count": len(self.workflow_state["errors"]),
        }
        
        # Calculate success rate
        if self.metrics["cycle_count"] > 0:
            metrics["success_rate"] = self.metrics["successful_cycles"] / self.metrics["cycle_count"]
        else:
            metrics["success_rate"] = 0
            
        # Calculate average cycle duration
        cycle_durations = self.metrics.get("cycle_durations", [])
        if cycle_durations:
            metrics["avg_cycle_duration"] = sum(cycle_durations) / len(cycle_durations)
            metrics["min_cycle_duration"] = min(cycle_durations)
            metrics["max_cycle_duration"] = max(cycle_durations)
        
        # Calculate agent performance metrics
        agent_metrics = {}
        for agent_name, execution_times in self.metrics.get("agent_execution_times", {}).items():
            if execution_times:
                agent_metrics[agent_name] = {
                    "avg_execution_time": sum(execution_times) / len(execution_times),
                    "min_execution_time": min(execution_times),
                    "max_execution_time": max(execution_times),
                    "total_executions": len(execution_times)
                }
        metrics["agent_performance"] = agent_metrics
        
        # Calculate error distribution
        error_distribution = {}
        for error in self.workflow_state["errors"]:
            agent = error.get("agent", "unknown")
            error_distribution[agent] = error_distribution.get(agent, 0) + 1
        metrics["error_distribution"] = error_distribution
        
        return metrics
    
    def shutdown(self) -> bool:
        """
        Shutdown the orchestrator and all agents.
        
        This method performs a clean shutdown of the orchestrator
        and all agents, saving state and releasing resources.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down orchestrator")
        
        try:
            # Stop workflow
            self.is_running = False
            self.is_paused = False
            
            # Record end time
            self.metrics["end_time"] = datetime.now()
            
            # Update final state
            self.workflow_state["status"] = "shutdown"
            self.workflow_state["last_updated"] = datetime.now()
            
            # Stop all agents
            self.stop_all_agents()
            
            # Save final state
            self._save_final_state()
            
            self.logger.info("Orchestrator shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
            return False
    
    def _save_final_state(self) -> None:
        """
        Save the final state before shutdown.
        
        This method saves the current state to a file for later reference.
        """
        # Determine save path
        save_dir = self.config.get("system", {}).get("state_storage_path", "data/states")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/workflow_{self.workflow_id}_{timestamp}.json"
        
        # Save state
        self.save_state(filename)


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("orchestrator_example")
    
    # Create orchestrator
    orchestrator = Orchestrator(logger=logger)
    
    # Initialize agents
    orchestrator.initialize_agents()
    
    # Define workflow
    orchestrator.define_workflow()
    
    # Start workflow
    orchestrator.start_workflow()
    
    try:
        # Run for a while
        time.sleep(3600)  # Run for 1 hour
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        # Shutdown orchestrator
        orchestrator.shutdown() 