"""
Enhanced Configuration Manager
Provides easy configuration management and validation
"""

import os
import json
import logging
from typing import Dict, Any

class ConfigManager:
    """Manages configuration with validation and environment support"""
    
    def __init__(self):
        self.config_path = "bot_config.json"
        self.default_config = self._get_default_config()
        self.current_config = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "deriv": {
                "app_id": os.getenv("DERIV_APP_ID", "1089"),
                "account_type": os.getenv("DERIV_ACCOUNT_TYPE", "demo"),
                "symbols": ["R_10", "R_25", "R_50", ""R_75","R_100"],
                "token" : os.getenv("DERIV_TOKEN")
            },
            "trading": {
                "initial_balance": 10000,
                "max_risk_per_trade": 0.02,
                "max_daily_drawdown": 0.05,
                "max_concurrent_trades": 3
            },
            "data": {
                "history_days": 180,
                "collection_interval": 10,
                "db_path": "trading_data.db"
            },
            "ml": {
                "lookback_window": 50,
                "min_confidence": 0.65,
                "training_interval": 1000
            },
            "phases": {
                "phase_1": True,
                "phase_2": False,
                "phase_3": False,
                "phase_4": False
            }
        }
    
    def load_config(self) -> bool:
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.current_config = json.load(f)
                logging.info("✅ Configuration loaded from file")
            else:
                self.current_config = self.default_config
                self.save_config()
                logging.info("✅ Default configuration created")
            
            return self.validate_config()
            
        except Exception as e:
            logging.error(f"❌ Configuration load failed: {e}")
            self.current_config = self.default_config
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.current_config, f, indent=2)
            logging.info("✅ Configuration saved to file")
            return True
        except Exception as e:
            logging.error(f"❌ Configuration save failed: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        try:
            required_sections = ["deriv", "trading", "data", "ml", "phases"]
            
            for section in required_sections:
                if section not in self.current_config:
                    logging.error(f"❌ Missing configuration section: {section}")
                    return False
            
            # Validate trading parameters
            trading = self.current_config["trading"]
            if trading["max_risk_per_trade"] > 0.1:
                logging.warning("⚠️ High risk per trade detected")
            
            if trading["max_daily_drawdown"] > 0.1:
                logging.warning("⚠️ High daily drawdown limit")
            
            logging.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logging.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def get(self, key: str, default=None):
        """Get configuration value by key (supports nested keys)"""
        try:
            keys = key.split('.')
            value = self.current_config
            
            for k in keys:
                value = value[k]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key"""
        try:
            keys = key.split('.')
            config = self.current_config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            logging.info(f"✅ Configuration updated: {key} = {value}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Configuration update failed: {e}")
            return False
    
    def show_config_summary(self):
        """Display configuration summary"""
        print("\n" + "="*50)
        print("PLATFORM MIND READER - CONFIGURATION SUMMARY")
        print("="*50)
        
        print(f"🔑 Deriv API: {self.get('deriv.account_type').upper()} Account")
        print(f"📈 Trading Symbols: {', '.join(self.get('deriv.symbols'))}")
        print(f"💰 Initial Balance: ${self.get('trading.initial_balance'):,}")
        print(f"🎯 Risk per Trade: {self.get('trading.max_risk_per_trade')*100}%")
        print(f"📊 Data Collection: {self.get('data.history_days')} days history")
        
        print("\n🔄 Active Phases:")
        phases = self.get('phases', {})
        for phase, active in phases.items():
            status = "ACTIVE" if active else "INACTIVE"
            print(f"  • {phase.replace('_', ' ').title()}: {status}")
        
        print("="*50)

# Global config manager instance
CONFIG_MANAGER = ConfigManager()
