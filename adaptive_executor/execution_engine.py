"""
Execution Engine - Trade Execution System
Handles actual trade execution with Deriv API
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import websockets

from config import DERIV_CONFIG
from core.deployment_orchestrator import ORCHESTRATOR

class ExecutionEngine:
    """Handles trade execution with Deriv API"""
    
    def __init__(self):
        self.websocket = None
        self.is_connected = True
        self.pending_orders = {}
        self.open_positions = {}
        self.execution_history = []
        
        # Execution parameters
        self.execution_config = {
            'max_slippage': 0.0001,  # 0.01%
            'timeout_seconds': 30,
            'retry_attempts': 3,
            'confirmation_required': True
        }
    
    async def initialize_connection(self):
        """Initialize connection to Deriv API"""
        try:
            self.websocket = await websockets.connect(DERIV_CONFIG.WEBSOCKET_URL)
            
            
            self.is_connected = True
            logging.info("✅ Execution engine connected to Deriv API")
            return True
            
        except Exception as e:
            logging.error(f"❌ Execution engine connection failed: {e}")
            self.is_connected = False
            return False
    
    async def execute_trade(self, trade_signal: Dict[str, Any], 
                          risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on signal and risk assessment"""
        if not ORCHESTRATOR.should_run_component('phase_3_adaptive_execution', 'live_trading'):
            return self._create_rejected_trade("Trading not active in current phase")
        
        if not self.is_connected:
            return self._create_rejected_trade("Not connected to Deriv API")
        
        try:
            # Prepare trade parameters
            trade_params = self._prepare_trade_parameters(trade_signal, risk_assessment)
            
            # Validate trade parameters
            validation_result = self._validate_trade_parameters(trade_params)
            if not validation_result['valid']:
                return self._create_rejected_trade(validation_result['reason'])
            
            # Execute trade
            execution_result = await self._send_trade_request(trade_params)
            
            if execution_result['success']:
                # Record successful execution
                trade_record = self._create_trade_record(trade_params, execution_result)
                self.open_positions[execution_result['contract_id']] = trade_record
                self.execution_history.append(trade_record)
                
                logging.info(f"✅ Trade executed: {trade_signal['action']} {trade_signal['symbol']} "
                           f"@ {execution_result['price']} (Contract: {execution_result['contract_id']})")
                
                return {
                    'status': 'executed',
                    'contract_id': execution_result['contract_id'],
                    'execution_price': execution_result['price'],
                    'trade_details': trade_record,
                    'timestamp': datetime.now()
                }
            else:
                logging.error(f"❌ Trade execution failed: {execution_result['error']}")
                return {
                    'status': 'failed',
                    'error': execution_result['error'],
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            logging.error(f"❌ Trade execution error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _prepare_trade_parameters(self, trade_signal: Dict[str, Any], 
                                risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare trade parameters for Deriv API"""
        symbol = trade_signal['symbol']
        action = trade_signal['action']
        amount = risk_assessment['position_size']
        
        # Map action to Deriv trade direction
        if action == 'buy':
            trade_type = 'CALL'
        elif action == 'sell':
            trade_type = 'PUT'
        else:
            trade_type = None
        
        # Determine contract duration
        duration = trade_signal.get('duration', 5)  # Default 5 minutes
        duration_unit = 'm'  # minutes
        
        # Prepare base parameters
        params = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": trade_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol
        }
        
        # Add optional parameters
        if 'stop_loss' in trade_signal and trade_signal['stop_loss']:
            params["stop_loss"] = trade_signal['stop_loss']
        
        if 'take_profit' in trade_signal and trade_signal['take_profit']:
            params["take_profit"] = trade_signal['take_profit']
        
        return params
    
    def _validate_trade_parameters(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade parameters before execution"""
        required_fields = ['amount', 'contract_type', 'symbol', 'duration']
        
        for field in required_fields:
            if field not in trade_params or not trade_params[field]:
                return {'valid': False, 'reason': f'Missing required field: {field}'}
        
        # Validate amount
        if trade_params['amount'] <= 0:
            return {'valid': False, 'reason': 'Invalid trade amount'}
        
        # Validate contract type
        if trade_params['contract_type'] not in ['CALL', 'PUT']:
            return {'valid': False, 'reason': 'Invalid contract type'}
        
        # Validate symbol
        if trade_params['symbol'] not in DERIV_CONFIG.SYMBOLS:
            return {'valid': False, 'reason': f"Unknown symbol: {trade_params['symbol']}"}
        
        return {'valid': True, 'reason': 'Parameters valid'}
    
    async def _send_trade_request(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Send trade request to Deriv API"""
        try:
            # First, get proposal to validate parameters
            proposal_request = {
                "proposal": 1,
                "subscribe": 1,
                **{k: v for k, v in trade_params.items() if k != 'amount'}
            }
            
            await self.websocket.send(json.dumps(proposal_request))
            proposal_response = await asyncio.wait_for(
                self.websocket.recv(), 
                timeout=self.execution_config['timeout_seconds']
            )
            
            proposal_data = json.loads(proposal_response)
            if 'error' in proposal_data:
                return {'success': False, 'error': proposal_data['error']}
            
            # Now send the actual trade request
            trade_request = {
                "buy": trade_params['amount'],
                **trade_params
            }
            
            # Remove proposal field for actual trade
            if 'proposal' in trade_request:
                del trade_request['proposal']
            
            await self.websocket.send(json.dumps(trade_request))
            trade_response = await asyncio.wait_for(
                self.websocket.recv(), 
                timeout=self.execution_config['timeout_seconds']
            )
            
            trade_data = json.loads(trade_response)
            
            if 'error' in trade_data:
                return {'success': False, 'error': trade_data['error']}
            elif 'buy' in trade_data:
                contract_id = trade_data['buy']['contract_id']
                price = trade_data['buy']['price']
                
                return {
                    'success': True,
                    'contract_id': contract_id,
                    'price': price,
                    'raw_response': trade_data
                }
            else:
                return {'success': False, 'error': 'Unexpected response format'}
                
        except asyncio.TimeoutError:
            return {'success': False, 'error': 'Trade request timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_trade_record(self, trade_params: Dict[str, Any], 
                           execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive trade record"""
        return {
            'trade_id': f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'contract_id': execution_result['contract_id'],
            'symbol': trade_params['symbol'],
            'action': 'buy' if trade_params['contract_type'] == 'CALL' else 'sell',
            'contract_type': trade_params['contract_type'],
            'amount': trade_params['amount'],
            'execution_price': execution_result['price'],
            'stop_loss': trade_params.get('stop_loss'),
            'take_profit': trade_params.get('take_profit'),
            'duration': trade_params['duration'],
            'timestamp': datetime.now(),
            'status': 'open',
            'risk_assessment': trade_params.get('risk_assessment', {})
        }
    
    def _create_rejected_trade(self, reason: str) -> Dict[str, Any]:
        """Create a rejected trade response"""
        return {
            'status': 'rejected',
            'reason': reason,
            'timestamp': datetime.now()
        }
    
    async def monitor_open_positions(self):
        """Monitor and update open positions"""
        if not self.is_connected:
            return
        
        try:
            # Subscribe to updates for all open positions
            for contract_id, position in self.open_positions.items():
                if position['status'] == 'open':
                    await self._subscribe_to_contract(contract_id)
            
            # Listen for updates
            await self._listen_for_updates()
            
        except Exception as e:
            logging.error(f"❌ Position monitoring error: {e}")
    
    async def _subscribe_to_contract(self, contract_id: str):
        """Subscribe to contract updates"""
        try:
            subscribe_message = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            }
            await self.websocket.send(json.dumps(subscribe_message))
        except Exception as e:
            logging.error(f"❌ Contract subscription failed: {e}")
    
    async def _listen_for_updates(self):
        """Listen for contract updates"""
        try:
            while self.is_connected and self.open_positions:
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=10.0
                )
                
                data = json.loads(message)
                
                if 'proposal_open_contract' in data:
                    await self._handle_contract_update(data['proposal_open_contract'])
                    
        except asyncio.TimeoutError:
            pass  # Normal timeout, continue monitoring
        except Exception as e:
            logging.error(f"❌ Update listening error: {e}")
    
    async def _handle_contract_update(self, contract_data: Dict[str, Any]):
        """Handle contract update from Deriv"""
        contract_id = contract_data.get('contract_id')
        
        if contract_id in self.open_positions:
            position = self.open_positions[contract_id]
            
            # Update position details
            position['current_price'] = contract_data.get('bid_price')
            position['profit_loss'] = contract_data.get('profit', 0)
            position['is_expired'] = contract_data.get('is_expired', False)
            position['is_sold'] = contract_data.get('is_sold', False)
            position['is_valid_to_sell'] = contract_data.get('is_valid_to_sell', False)
            
            # Check if position is closed
            if contract_data.get('is_expired') or contract_data.get('is_sold'):
                position['status'] = 'closed'
                position['close_time'] = datetime.now()
                position['close_price'] = contract_data.get('sell_price')
                
                # Record final P&L
                final_pnl = contract_data.get('profit', 0)
                position['final_pnl'] = final_pnl
                
                logging.info(f"📊 Position closed: {position['symbol']} "
                           f"P&L: ${final_pnl:.2f}")
                
                # Remove from open positions
                del self.open_positions[contract_id]
    
    async def close_position(self, contract_id: str) -> Dict[str, Any]:
        """Manually close a position"""
        if contract_id not in self.open_positions:
            return {'success': False, 'error': 'Contract not found'}
        
        try:
            close_request = {
                "sell": contract_id,
                "price": 0  # Market sell
            }
            
            await self.websocket.send(json.dumps(close_request))
            response = await asyncio.wait_for(
                self.websocket.recv(), 
                timeout=self.execution_config['timeout_seconds']
            )
            
            close_data = json.loads(response)
            
            if 'error' in close_data:
                return {'success': False, 'error': close_data['error']}
            elif 'sell' in close_data:
                # Position will be updated via monitoring
                return {'success': True, 'message': 'Sell order placed'}
            else:
                return {'success': False, 'error': 'Unexpected response'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution engine statistics"""
        total_trades = len(self.execution_history)
        successful_trades = len([t for t in self.execution_history if t.get('status') != 'rejected'])
        open_positions = len(self.open_positions)
        
        return {
            'total_trades_executed': total_trades,
            'successful_executions': successful_trades,
            'current_open_positions': open_positions,
            'success_rate': successful_trades / total_trades if total_trades > 0 else 0,
            'connection_status': 'connected' if self.is_connected else 'disconnected',
            'last_execution': self.execution_history[-1]['timestamp'] if self.execution_history else None
        }
    
    async def close_connection(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logging.info("🔌 Execution engine connection closed")

# Global execution engine instance
EXECUTION_ENGINE = ExecutionEngine()
