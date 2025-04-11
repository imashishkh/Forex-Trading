#!/bin/bash

# Set up environment
echo "Setting up environment..."

# --- ADDED: Load environment variables from .env file ---
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  set -a  # Automatically export all variables
  source .env
  set +a
else
  echo "Warning: .env file not found. Relying on potentially pre-set environment variables."
fi
# --- END ADDED ---

source venv/bin/activate 2>/dev/null || true  # Activate virtual environment if it exists

# Create necessary directories
mkdir -p logs
mkdir -p data

# Make sure the monitoring config exists
if [ ! -f "config/monitoring_config.json" ]; then
    echo "Creating monitoring configuration..."
    mkdir -p config
    cat > config/monitoring_config.json << 'EOF'
{
    "agents": ["market_data_agent", "wallet_manager", "technical_analysis", "portfolio_manager", "risk_manager"],
    "api_connections": {
        "oanda": {
            "url": "https://api-fxpractice.oanda.com",
            "timeout": 10
        },
        "news_api": {
            "url": "https://newsapi.org",
            "timeout": 10
        }
    },
    "alerts": {
        "email": {
            "enabled": false,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "recipients": []
        },
        "sms": {
            "enabled": false,
            "provider": "",
            "api_key": "",
            "phone_numbers": []
        }
    },
    "update_interval": 5,
    "anomaly_detection": {
        "cpu_threshold": 80,
        "memory_threshold": 80,
        "disk_threshold": 80,
        "response_time_threshold": 2
    },
    "instruments": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "EUR_GBP"]
}
EOF
fi

# Start the trading platform in the background with correct arguments
echo "Starting trading platform..."
python main.py --config config/settings.json start --mode live > logs/main.log 2>&1 &
MAIN_PID=$!
echo "Trading platform started with PID: $MAIN_PID"

# Wait for the platform to initialize
sleep 5

# Start the monitoring dashboard in a separate process
echo "Starting monitoring dashboard..."
streamlit run monitoring.py --server.port 8501 > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "Monitoring dashboard started with PID: $STREAMLIT_PID"

echo "System started successfully!"
echo "Trading platform is running in the background (PID: $MAIN_PID)"
echo "Monitoring dashboard should be available at http://localhost:8501"
echo "Check logs/main.log and logs/streamlit.log for application logs"

# Save the PIDs to a file for later cleanup
echo "$MAIN_PID $STREAMLIT_PID" > .forex_pids

echo ""
echo "To stop all services, run: ./stop_system.sh" 