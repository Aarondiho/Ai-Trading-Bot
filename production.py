"""
Production Configuration & Security Settings
Makes our AI trading system production-ready
"""

import os
import logging
from datetime import datetime

class ProductionConfig:
    """Production-specific configuration"""
    
    def __init__(self):
        self.is_production = True
        self.debug_mode = False
        self.setup_environment()
    
    def setup_environment(self):
        """Setup production environment variables"""
        # Deriv API Configuration from Environment Variables
        os.environ.setdefault('DERIV_APP_ID', os.getenv('DERIV_APP_ID', '1089'))
        os.environ.setdefault('DERIV_ACCOUNT_TYPE', os.getenv('DERIV_ACCOUNT_TYPE', 'demo'))
        
        # Database configuration
        os.environ.setdefault('DB_PATH', os.getenv('DB_PATH', 'trading_data.db'))
        
        # Security settings
        os.environ.setdefault('ENCRYPTION_KEY', os.getenv('ENCRYPTION_KEY', 'change-in-production'))
    
    def setup_production_logging(self):
        """Setup production-appropriate logging"""
        log_directory = "logs"
        os.makedirs(log_directory, exist_ok=True)
        
        log_filename = f"{log_directory}/trading_ai_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Production logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5),  # 10MB files, keep 5
                logging.StreamHandler()
            ]
        )
        
        # Reduce verbose logging for third-party libraries
        logging.getLogger('websockets').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    def security_check(self):
        """Run production security checks"""
        security_issues = []
        
        # Check for hardcoded sensitive values
        if os.getenv('DERIV_APP_ID') == '1089':
            security_issues.append("Using default Deriv App ID - set DERIV_APP_ID environment variable")
        
        if os.getenv('ENCRYPTION_KEY') == 'change-in-production':
            security_issues.append("Using default encryption key - set ENCRYPTION_KEY environment variable")
        
        # Check file permissions
        if os.path.exists('trading_data.db'):
            import stat
            db_stat = os.stat('trading_data.db')
            if db_stat.st_mode & stat.S_IROTH:
                security_issues.append("Database file is world readable - fix permissions")
        
        return security_issues

class HealthMonitor:
    """Production health monitoring"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.health_checks = []
    
    def health_check(self):
        """Comprehensive system health check"""
        health_status = {
            'status': 'healthy',
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check critical components
        try:
            # Database health
            import sqlite3
            conn = sqlite3.connect(os.getenv('DB_PATH', 'trading_data.db'))
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            health_status['components']['database'] = 'healthy'
            conn.close()
        except Exception as e:
            health_status['components']['database'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            health_status['components']['disk_space'] = f'healthy ({free // (2**30)}GB free)'
        except Exception as e:
            health_status['components']['disk_space'] = f'check failed: {str(e)}'
        
        return health_status
    
    def create_health_endpoint(self):
        """Create a simple health check endpoint for web monitoring"""
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import json
        
        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    health_data = self.health_check()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(health_data).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        return HealthHandler

# Global production configuration
PRODUCTION_CONFIG = ProductionConfig()
HEALTH_MONITOR = HealthMonitor()
