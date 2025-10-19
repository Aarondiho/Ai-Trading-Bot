"""
Data Archaeologist - Data Collection Module
Responsible for collecting platform fingerprints and historical data
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import websockets
import pandas as pd

from config import DERIV_CONFIG, DATA_CONFIG
from core.deployment_orchestrator import ORCHESTRATOR

class DataArchaeologist:
    """Collects and stores platform data for pattern analysis"""
    
    def __init__(self):
        self.symbols = DERIV_CONFIG.SYMBOLS
        self.data_buffer = {symbol: [] for symbol in self.symbols}
        self.websocket = None
        self.is_collecting = False
        self.db_connection = None
        
        # Statistics
        self.collection_stats = {
            'total_ticks': 0,
            'symbol_ticks': {symbol: 0 for symbol in self.symbols},
            'last_collection': None,
            'errors': 0
        }
    
    def initialize_database(self):
        """Initialize SQLite database for data storage"""
        self.db_connection = sqlite3.connect(DATA_CONFIG.DB_PATH)
        cursor = self.db_connection.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tick_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                price REAL NOT NULL,
                quote_time INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlc_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open_time DATETIME NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL,
                records_count INTEGER,
                period_start DATETIME,
                period_end DATETIME,
                status TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.db_connection.commit()
        logging.info("✅ Database initialized successfully")
    
    async def connect_deriv_websocket(self):
        """Establish WebSocket connection to Deriv"""
        try:
            self.websocket = await websockets.connect(DERIV_CONFIG.WEBSOCKET_URL+'?app_id='+CONFIG.APP_ID)
            
            # Authorize connection
            auth_message = {
                "authorize": DERIV_CONFIG.TOKEN
            }
            await self.websocket.send(json.dumps(auth_message))
            response = await self.websocket.recv()
            logging.info("✅ WebSocket connected and authorized")
            return True
            
        except Exception as e:
            logging.error(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def collect_historical_data(self, symbol: str, days: int = 180):
        """Collect historical data for pattern analysis"""
        if not ORCHESTRATOR.should_run_component('phase_1_pattern_archaeology', 'data_collection'):
            logging.info(f"⏸️ Data collection paused for {symbol}")
            return
        
        try:
            # Request historical data
            history_request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": days * 24 * 60,  # Approximate minute data
                "end": "latest",
                "style": "ticks"
            }
            
            await self.websocket.send(json.dumps(history_request))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if 'history' in data and 'prices' in data['history']:
                prices = data['history']['prices']
                times = data['history']['times']
                
                # Store historical data
                self._store_historical_ticks(symbol, prices, times)
                logging.info(f"✅ Collected {len(prices)} historical ticks for {symbol}")
                
        except Exception as e:
            logging.error(f"❌ Historical data collection failed for {symbol}: {e}")
            self.collection_stats['errors'] += 1
    
    async def start_real_time_collection(self):
        """Start real-time tick data collection"""
        if not ORCHESTRATOR.should_run_component('phase_1_pattern_archaeology', 'data_collection'):
            return
        
        self.is_collecting = True
        logging.info("🎯 Starting real-time data collection")
        
        # Subscribe to ticks for all symbols
        for symbol in self.symbols:
            subscribe_message = {
                "ticks": symbol,
                "subscribe": 1
            }
            await self.websocket.send(json.dumps(subscribe_message))
        
        # Start listening for ticks
        await self._listen_for_ticks()
    
    async def _listen_for_ticks(self):
        """Listen for incoming tick data"""
        while self.is_collecting and self.websocket:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if 'tick' in data:
                    tick = data['tick']
                    symbol = tick['symbol']
                    
                    # Store tick data
                    self._store_tick_data(symbol, tick)
                    
                    # Update buffer
                    self.data_buffer[symbol].append(tick)
                    if len(self.data_buffer[symbol]) > DATA_CONFIG.TICK_BUFFER_SIZE:
                        self.data_buffer[symbol].pop(0)
                    
                    # Update statistics
                    self.collection_stats['total_ticks'] += 1
                    self.collection_stats['symbol_ticks'][symbol] += 1
                    self.collection_stats['last_collection'] = datetime.now()
                    
                    # Log periodically
                    if self.collection_stats['total_ticks'] % 1000 == 0:
                        logging.info(f"📊 Collected {self.collection_stats['total_ticks']} ticks total")
                        
            except Exception as e:
                logging.error(f"❌ Error in tick listener: {e}")
                self.collection_stats['errors'] += 1
                await asyncio.sleep(1)  # Brief pause before retry
    
    def _store_tick_data(self, symbol: str, tick: Dict):
        """Store tick data in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO tick_data (symbol, timestamp, bid, ask, price, quote_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.fromtimestamp(tick['quote_time']),
                tick.get('bid', tick['quote']),
                tick.get('ask', tick['quote']),
                tick['quote'],
                tick['quote_time']
            ))
            self.db_connection.commit()
        except Exception as e:
            logging.error(f"❌ Failed to store tick data: {e}")
    
    def _store_historical_ticks(self, symbol: str, prices: List[float], times: List[int]):
        """Store historical tick data"""
        try:
            cursor = self.db_connection.cursor()
            for price, timestamp in zip(prices, times):
                cursor.execute('''
                    INSERT OR IGNORE INTO tick_data (symbol, timestamp, bid, ask, price, quote_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    datetime.fromtimestamp(timestamp),
                    price, price, price, timestamp
                ))
            self.db_connection.commit()
            
            # Log collection
            cursor.execute('''
                INSERT INTO collection_log (symbol, data_type, records_count, period_start, period_end, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                'historical_ticks',
                len(prices),
                datetime.fromtimestamp(min(times)),
                datetime.fromtimestamp(max(times)),
                'success'
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logging.error(f"❌ Failed to store historical data: {e}")
    
    def get_recent_data(self, symbol: str, lookback: int = 1000) -> pd.DataFrame:
        """Get recent data for analysis"""
        try:
            query = '''
                SELECT timestamp, bid, ask, price, quote_time
                FROM tick_data 
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, self.db_connection, params=[symbol, lookback])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            logging.error(f"❌ Failed to get recent data: {e}")
            return pd.DataFrame()
    
    async def stop_collection(self):
        """Stop data collection"""
        self.is_collecting = False
        if self.websocket:
            await self.websocket.close()
        if self.db_connection:
            self.db_connection.close()
        logging.info("🛑 Data collection stopped")

# Global data collector instance
DATA_ARCHAEOLOGIST = DataArchaeologist()
