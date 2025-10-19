"""
WSGI Entry Point for Web Deployment
Allows hosting our AI as a web application
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production import HEALTH_MONITOR
from http.server import HTTPServer

def create_health_server(port=8080):
    """Create a simple health check server"""
    from production import HEALTH_MONITOR
    
    class HealthHandler(HEALTH_MONITOR.create_health_endpoint()):
        pass
    
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    return server

def application(environ, start_response):
    """WSGI application interface"""
    # This allows our AI to run as a web app
    if environ['PATH_INFO'] == '/health':
        health_data = HEALTH_MONITOR.health_check()
        
        start_response('200 OK', [('Content-Type', 'application/json')])
        return [json.dumps(health_data).encode()]
    else:
        start_response('404 Not Found', [('Content-Type', 'text/plain')])
        return [b'AI Trading System - Use /health for status']

# For running directly
if __name__ == '__main__':
    print("🤖 AI Trading System - Health Monitor Starting...")
    server = create_health_server()
    print(f"📍 Health checks available at: http://localhost:8080/health")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Health monitor stopped")
