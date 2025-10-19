"""
Platform Mind Reader - Master Configuration
Core configuration for the entire trading system
"""

import os
from datetime import time
from typing import Dict, List

class DeploymentPhases:
    """Control which phases are active"""
    def __init__(self):
        self.phases = {
            'phase_1_pattern_archaeology': {
                'active': True,
                'components': {
                    'data_collection': True,
                    'pattern_detection': True,
                    'regime_classification': True
                }
            },
            'phase_2_transition_prediction': {
                'active': False,
                'components': {
                    'state_transition_model': False,
                    'early_warning_system': False,
                    'temporal_analysis': False
                }
            },
            'phase_3_adaptive_execution': {
                'active': False,
                'components': {
                    'strategy_arsenal': False,
                    'live_trading': False,
                    'risk_management': False
                }
            },
            'phase_4_continuous_learning': {
                'active': False,
                'components': {
                    'meta_learning': False,
                    'knowledge_preservation': False,
                    'rapid_adaptation': False
                }
            }
        }
    
    def activate_phase(self, phase_name: str):
        """Activate a specific phase"""
        if phase_name in self.phases:
            self.phases[phase_name]['active'] = True
            for component in self.phases[phase_name]['components']:
                self.phases[phase_name]['components'][component] = True
            print(f"✅ Activated {phase_name}")
    
    def deactivate_phase(self, phase_name: str):
        """Deactivate a specific phase"""
        if phase_name in self.phases:
            self.phases[phase_name]['active'] = False
            for component in self.phases[phase_name]['components']:
                self.phases[phase_name]['components'][component] = False
            print(f"⏸️  Deactivated {phase_name}")

class DerivConfig:
    """Deriv API configuration"""
    def __init__(self):
        self.APP_ID = os.getenv('DERIV_APP_ID')
        self.ACCOUNT_TYPE = os.getenv('DERIV_ACCOUNT_TYPE')  # demo or real
        self.TOKEN = os.getenv('DERIV_TOKEN')
        
        # Trading symbols (Deriv Synthetic Indices)
        self.SYMBOLS = [
            'R_10', # Volatility 10 Index
            'R_25', # Volatility 25 Index
            'R_50',    # Volatility 50 Index 
            'R_75',    # Volatility 75 Index
            'R_100'  # Volatility 100 Index
        ]
        
        # API endpoints
        self.WEBSOCKET_URL = "wss://ws.binaryws.com/websockets/v3"
        self.API_URL = "https://api.deriv.com"

class DataConfig:
    """Data collection and storage configuration"""
    def __init__(self):
        # Database
        self.DB_PATH = "platform_mind_reader.db"
        
        # Data collection
        self.HISTORY_DAYS = 180  # 6 months of historical data
        self.TICK_BUFFER_SIZE = 10000
        self.COLLECTION_INTERVAL = 10  # seconds
        
        # Timeframes for analysis
        self.TIMEFRAMES = ['1m', '5m', '15m', '1h']

class MLConfig:
    """Machine learning configuration"""
    def __init__(self):
        self.LOOKBACK_WINDOW = 50
        self.TRAINING_INTERVAL = 1000  # Retrain every 1000 new samples
        self.MIN_CONFIDENCE = 0.65
        self.ENSEMBLE_WEIGHTS = {
            'lstm': 0.6,
            'random_forest': 0.4
        }

class RiskConfig:
    """Risk management configuration"""
    def __init__(self):
        self.INITIAL_BALANCE = 10000
        self.MAX_RISK_PER_TRADE = 0.02  # 2%
        self.MAX_DAILY_DRAWDOWN = 0.05  # 5%
        self.MAX_CONCURRENT_TRADES = 3
        self.EMERGENCY_STOP_LOSS = 0.10  # 10% total loss
        
        # Position sizing tiers based on confidence
        self.CONFIDENCE_TIERS = {
            'high': 0.8,      # 80%+ confidence - full position
            'medium': 0.6,    # 60-80% confidence - half position  
            'low': 0.0,       # Below 60% - no trade
        }

class PlatformPatterns:
    """Known platform algorithm patterns we're looking for"""
    def __init__(self):
        self.ALGORITHM_STATES = {
            'volatility_compression': {
                'description': 'Low volatility, price oscillating in tight range',
                'typical_duration': '10-30 minutes',
                'trading_opportunity': 'Mean reversion'
            },
            'trend_momentum': {
                'description': 'Strong directional movement with momentum',
                'typical_duration': '15-45 minutes', 
                'trading_opportunity': 'Trend following'
            },
            'random_walk': {
                'description': 'Unpredictable, noisy price action',
                'typical_duration': '5-20 minutes',
                'trading_opportunity': 'Avoid trading'
            },
            'scheduled_jump': {
                'description': 'Planned large movements at specific times',
                'typical_duration': 'Instant with aftermath',
                'trading_opportunity': 'Pre-positioning'
            },
            'volatility_expansion': {
                'description': 'High volatility, large price swings',
                'typical_duration': '20-60 minutes',
                'trading_opportunity': 'Breakout trading'
            }
        }

# Global configuration instances
PHASES = DeploymentPhases()
DERIV_CONFIG = DerivConfig()
DATA_CONFIG = DataConfig()
ML_CONFIG = MLConfig()
RISK_CONFIG = RiskConfig()
PLATFORM_PATTERNS = PlatformPatterns()
