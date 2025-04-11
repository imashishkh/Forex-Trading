#!/bin/bash

echo "Stopping forex trading system..."

# Check if we have PIDs from the start script
if [ -f ".forex_pids" ]; then
    read MAIN_PID STREAMLIT_PID < .forex_pids
    
    # Stop the main trading platform process
    if ps -p $MAIN_PID > /dev/null; then
        echo "Stopping trading platform (PID: $MAIN_PID)..."
        kill $MAIN_PID
        sleep 2
        # Force kill if it's still running
        if ps -p $MAIN_PID > /dev/null; then
            echo "Force stopping trading platform..."
            kill -9 $MAIN_PID
        fi
    else
        echo "Trading platform is not running."
    fi
    
    # Stop the Streamlit dashboard process
    if ps -p $STREAMLIT_PID > /dev/null; then
        echo "Stopping monitoring dashboard (PID: $STREAMLIT_PID)..."
        kill $STREAMLIT_PID
        sleep 2
        # Force kill if it's still running
        if ps -p $STREAMLIT_PID > /dev/null; then
            echo "Force stopping monitoring dashboard..."
            kill -9 $STREAMLIT_PID
        fi
    else
        echo "Monitoring dashboard is not running."
    fi
    
    # Clean up the PID file
    rm -f .forex_pids
else
    # If no PID file, try to find and kill the processes
    echo "No PID file found. Attempting to find and stop processes..."
    
    # Find and kill Python processes related to our app
    pkill -f "python main.py"
    
    # Find and kill Streamlit processes
    pkill -f "streamlit run monitoring.py"
fi

echo "All services stopped." 