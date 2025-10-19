#!/bin/bash

# AI Trading System - Production Deployment Script
echo "🚀 Deploying Platform Mind Reader to Production..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p logs
mkdir -p backups

# Set proper permissions
echo "🔒 Setting secure permissions..."
chmod 700 .env 2>/dev/null || true
chmod 600 *.db 2>/dev/null || true

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt --upgrade

# Run security check
echo "🛡️ Running security checks..."
python3 -c "
from production import PRODUCTION_CONFIG
issues = PRODUCTION_CONFIG.security_check()
if issues:
    print('⚠️ Security warnings:')
    for issue in issues:
        print(f'   - {issue}')
else:
    print('✅ All security checks passed')
"

# Start the application
echo "🤖 Starting AI Trading System..."
echo "📍 Health monitor: http://localhost:8080/health"
echo "📊 Logs: tail -f logs/trading_ai_*.log"

# Start with production settings
export PYTHONPATH=$(pwd)
python3 run.py
