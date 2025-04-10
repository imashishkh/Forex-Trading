#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Agent Class for Forex Trading Platform

This module provides the foundation for all agent implementations
in the forex trading platform, defining standard interfaces and
functionality that all agents should implement.
"""

import logging
import os
import abc
from typing import Any, Dict, List, Optional, Union, Set, TypeVar, cast
from datetime import datetime
import uuid
import traceback

# LangChain and LangGraph imports
try:
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
    
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable import Runnable
import langgraph.graph as lg
from langgraph.checkpoint import MemorySaver

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Type definitions
T = TypeVar('T')
AgentState = Dict[str, Any]
AgentMessage = Dict[str, Any]

class BaseAgent(abc.ABC):
    """
    Base class for all agents in the forex trading platform.
    
    This abstract class defines the interface that all agent implementations
    should follow, including initialization, messaging, task execution,
    and lifecycle management. It serves as the foundation for building
    specialized agents with specific responsibilities in the system.
    """
    
    def __init__(
        self,
        agent_name: str,
        llm: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base agent with required components.
        
        Args:
            agent_name: A unique identifier for the agent
            llm: Language model to use (defaults to OpenAI if None)
            config: Configuration parameters for the agent
            logger: Logger instance for the agent
        """
        self.agent_name = agent_name
        self.agent_id = str(uuid.uuid4())
        self.config = config or {}
        
        # Initialize language model if not provided
        if llm is None:
            self.llm = ChatOpenAI(
                temperature=self.config.get('temperature', 0.1),
                model=self.config.get('model_name', 'gpt-3.5-turbo'),
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
        else:
            self.llm = llm
            
        # Initialize logger if not provided
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Initialize state
        self.state = self._initialize_state()
        
        # Track connected agents
        self.connected_agents = {}
        
        # Status tracking
        self.status = "initialized"
        self.last_active = datetime.now()
        
        self.log_action("init", f"Agent {self.agent_name} initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up a logger for the agent.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(f"agent.{self.agent_name}")
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        logger.setLevel(getattr(logging, log_level))
        
        # Check if handlers already exist to avoid duplicates
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level))
            
            # Use colored formatter if available
            try:
                from utils.logger import ColoredFormatter
                formatter = ColoredFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            except ImportError:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler (if configured)
            log_file = os.getenv('LOG_FILE_PATH')
            if log_file:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(getattr(logging, log_level))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_state(self) -> AgentState:
        """
        Initialize the agent's state.
        
        Returns:
            AgentState: Initial state dictionary
        """
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "messages": [],
            "tasks": [],
            "errors": [],
            "status": "initialized",
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "memory": {},  # For storing agent-specific memory/context
            "metrics": {}  # For tracking performance metrics
        }
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Set up the agent and its resources.
        
        This method should be implemented by subclasses to perform
        any necessary setup operations before the agent can be used.
        It should handle resource allocation, connection to external
        services, and any other initialization tasks.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.log_action("initialize", "Agent initialization started")
        # Subclasses should implement specific initialization logic
        # Example implementation:
        # self.status = "ready"
        # self.state["status"] = "ready"
        # return True
        pass
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages from other agents.
        
        This method handles communication between agents in the system.
        It validates incoming messages, updates the agent's state, and
        determines how to respond to the message.
        
        Args:
            message: The message to process, typically a dictionary with metadata

        Returns:
            Optional[AgentMessage]: Response message if applicable, None otherwise
        """
        self.log_action("process_message", f"Processing message: {message.get('type', 'Unknown')}")
        
        # Validate input
        if not self.validate_input(message):
            error_msg = f"Invalid message format: {message}"
            self.handle_error(ValueError(error_msg))
            return None
        
        # Update state with new message
        self.state["messages"].append(message)
        self.last_active = datetime.now()
        self.state["last_active"] = self.last_active
        
        # Basic implementation - subclasses should enhance this
        response = {
            "type": "response",
            "in_reply_to": message.get("message_id", "unknown"),
            "content": f"Received message of type: {message.get('type', 'Unknown')}",
            "timestamp": datetime.now(),
            "status": "acknowledged"
        }
        
        return response
    
    def send_message(self, recipient: str, message: AgentMessage) -> bool:
        """
        Send messages to other agents.
        
        This method handles outgoing communication to other agents.
        It packages the message with metadata, logs the communication,
        and delivers the message to the recipient.
        
        Args:
            recipient: Identifier of the agent to receive the message
            message: The message to send

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        self.log_action("send_message", f"Sending message to {recipient}")
        
        try:
            # Add metadata to message
            full_message = {
                **message,
                "sender": self.agent_name,
                "sender_id": self.agent_id,
                "timestamp": datetime.now(),
                "message_id": str(uuid.uuid4())
            }
            
            # In a real implementation, this would use some messaging system
            # For now, we'll just log it and assume the message will be delivered
            self.logger.debug(f"Message sent to {recipient}: {full_message}")
            
            # Update state
            self.state["messages"].append(full_message)
            self.last_active = datetime.now()
            self.state["last_active"] = self.last_active
            
            # If recipient is in connected_agents, we can deliver directly
            if recipient in self.connected_agents:
                recipient_agent = self.connected_agents[recipient]
                return recipient_agent.process_message(full_message) is not None
            
            return True
        except Exception as e:
            self.handle_error(e)
            return False
    
    @abc.abstractmethod
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary functionality.
        
        This method should be implemented by subclasses to perform
        the agent's main operations. It represents the core business
        logic of each specialized agent type.
        
        Args:
            task: Task description and parameters

        Returns:
            Dict[str, Any]: Task execution results
        """
        self.log_action("run_task", f"Running task: {task.get('type', 'Unknown')}")
        
        # Update state
        self.state["tasks"].append(task)
        self.last_active = datetime.now()
        self.state["last_active"] = self.last_active
        
        # Subclasses should implement specific task execution logic
        # Example implementation:
        # result = self._process_task_logic(task)
        # return {"status": "success", "result": result}
        pass
    
    def shutdown(self) -> bool:
        """
        Clean up resources when shutting down.
        
        This method handles the graceful termination of the agent,
        including resource cleanup, closing connections, and saving state.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.log_action("shutdown", "Agent shutting down")
        
        try:
            # Update status
            self.status = "shutdown"
            self.state["status"] = "shutdown"
            
            # Perform cleanup (subclasses may extend this)
            self.connected_agents = {}
            
            # Save final state if needed
            self._save_final_state()
            
            return True
        except Exception as e:
            self.handle_error(e)
            return False
    
    def _save_final_state(self) -> None:
        """
        Save the agent's final state before shutdown.
        
        This method can be overridden by subclasses to implement
        specific state persistence logic.
        """
        # Basic implementation - just log the final state
        self.logger.debug(f"Final agent state: {self.state}")
    
    def log_action(self, action: str, details: str) -> None:
        """
        Log agent actions with standardized formatting.
        
        This method provides consistent logging across all agents
        to facilitate monitoring and debugging.
        
        Args:
            action: The type of action being performed
            details: Description of the action
        """
        self.logger.info(f"[{self.agent_name}][{action}] {details}")
        
    def handle_error(self, error: Exception) -> None:
        """
        Standardized error handling for agents.
        
        This method provides a consistent approach to error handling,
        including logging, state updates, and optional error recovery.
        
        Args:
            error: The exception that occurred
        """
        error_details = {
            "timestamp": datetime.now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "agent_status": self.status
        }
        
        self.logger.error(f"Error in agent {self.agent_name}: {error}", exc_info=True)
        self.state["errors"].append(error_details)
        
        # Subclasses may implement specific error recovery strategies
    
    def get_status(self) -> Dict[str, Any]:
        """
        Return the agent's current status.
        
        This method provides a snapshot of the agent's current state,
        including operational status, message counts, and other metrics.
        
        Returns:
            Dict[str, Any]: Status information including state, message count, etc.
        """
        status_info = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "last_active": self.last_active,
            "message_count": len(self.state["messages"]),
            "task_count": len(self.state["tasks"]),
            "error_count": len(self.state["errors"]),
            "connected_agents": list(self.connected_agents.keys()),
            "up_time": (datetime.now() - self.state["created_at"]).total_seconds(),
            "metrics": self.state.get("metrics", {})
        }
        
        self.log_action("get_status", f"Status requested: {self.status}")
        return status_info
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate incoming data.
        
        This method checks that incoming data meets the agent's
        expected format and contains required fields.
        
        Args:
            input_data: The data to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Basic validation - subclasses should implement more specific validation
        if not isinstance(input_data, dict):
            return False
            
        # Check for required fields in messages
        required_fields = ["type"]
        for field in required_fields:
            if field not in input_data:
                self.log_action("validate_input", f"Missing required field: {field}")
                return False
                
        return True

    # LangGraph Integration Methods
    
    def setup_node(self) -> Any:
        """
        Configure this agent as a LangGraph node.
        
        This method integrates the agent with LangGraph workflows
        by defining its behavior as a graph node.
        
        Returns:
            Any: A LangGraph node representing this agent
        """
        # Define the node's processing function
        def process_in_graph(state: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Process the input through the agent
                result = self.run_task(input_data)
                
                # Update the graph state
                new_state = {**state}
                new_state[self.agent_name] = {
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.now()
                }
                
                return new_state
            except Exception as e:
                self.handle_error(e)
                # Return error state
                return {
                    **state,
                    self.agent_name: {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now()
                    }
                }
        
        # Create and return a LangGraph node using the node function
        return lg.node(process_in_graph, name=self.agent_name)
    
    def as_runnable(self) -> Runnable:
        """
        Convert the agent to a LangChain Runnable.
        
        This method allows the agent to be used in LangChain's
        composable Runnable interfaces.
        
        Returns:
            Runnable: A LangChain Runnable representing this agent
        """
        # Create a wrapper function that implements the Runnable interface
        def agent_runnable(input_data: Dict[str, Any]) -> Dict[str, Any]:
            return self.run_task(input_data)
            
        # Return as a Runnable
        from langchain.schema.runnable import RunnableLambda
        return RunnableLambda(agent_runnable)
    
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Define the input schema for this agent in LangGraph.
        
        This method specifies the expected format for inputs to the agent,
        which can be used for validation and documentation.
        
        Returns:
            Dict[str, Any]: JSON schema for agent inputs
        """
        # Basic schema - subclasses should enhance this
        return {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "description": "Type of task to execute"},
                "parameters": {"type": "object", "description": "Task-specific parameters"},
                "priority": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Task priority (1-10)"},
                "timestamp": {"type": "string", "format": "date-time", "description": "When the task was created"}
            },
            "required": ["task_type"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Define the output schema for this agent in LangGraph.
        
        This method specifies the format of the agent's outputs,
        which can be used for validation and documentation.
        
        Returns:
            Dict[str, Any]: JSON schema for agent outputs
        """
        # Basic schema - subclasses should enhance this
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failure", "pending", "error"],
                    "description": "Result status of the task execution"
                },
                "result": {"type": "object", "description": "Task execution results"},
                "error": {"type": "string", "description": "Error message if status is 'error'"},
                "metrics": {"type": "object", "description": "Performance metrics for the task"},
                "timestamp": {"type": "string", "format": "date-time", "description": "When the task completed"}
            },
            "required": ["status"]
        }
    
    def create_graph_with_dependencies(self, dependencies: List[str]) -> lg.Graph:
        """
        Create a LangGraph workflow with this agent and its dependencies.
        
        This method builds a workflow graph connecting this agent with others
        it depends on for its operation.
        
        Args:
            dependencies: List of agent names this agent depends on
        
        Returns:
            lg.Graph: A LangGraph workflow connecting the agents
        """
        # Initialize graph builder
        builder = lg.GraphBuilder()
        
        # Add this agent as a node
        builder.add_node(self.agent_name, self.setup_node())
        
        # Add connections to dependencies
        for dep in dependencies:
            if dep in self.connected_agents:
                dep_agent = self.connected_agents[dep]
                # Add dependency as a node
                builder.add_node(dep, dep_agent.setup_node())
                # Add edge from dependency to this agent
                builder.add_edge(dep, self.agent_name)
        
        # Build the graph
        workflow = builder.compile()
        
        # Add checkpointing
        memory = MemorySaver()
        workflow_with_memory = workflow.with_state(memory)
        
        return workflow_with_memory 