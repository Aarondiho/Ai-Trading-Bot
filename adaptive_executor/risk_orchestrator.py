"""
Risk Orchestrator - Adaptive Risk Management
Manages risk and position sizing based on regime confidence
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from config import RISK_CONFIG
from core.deployment_orchestrator import ORCHESTRATOR

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    overall_risk: RiskLevel
    position_size: float
    max_drawdown: float
    daily_loss_limit: float
    recommended_action: str
    confidence: float
    risk_factors: List[str]

class AdaptiveRiskManager:
    """Manages risk adaptively based on market conditions and performance"""
    
    def __init__(self):
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.trade_history = []
        self.risk_assessments = []
        
        # Risk parameters
        self.risk_parameters = {
            'max_daily_loss': RISK_CONFIG.INITIAL_BALANCE * RISK_CONFIG.MAX_DAILY_DRAWDOWN,
            'max_position_size': RISK_CONFIG.INITIAL_BALANCE * 0.1,  # 10% max per trade
            'max_concurrent_trades': RISK_CONFIG.MAX_CONCURRENT_TRADES,
            'emergency_stop_loss': RISK_CONFIG.EMERGENCY_STOP_LOSS
        }
        
        # Confidence-based sizing tiers
        self.confidence_tiers = RISK_CONFIG.CONFIDENCE_TIERS
        
        # Performance tracking
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'current_streak': 0
        }
    
    def calculate_regime_aware_position(self, regime_confidence: float, 
                                      strategy_confidence: float,
                                      symbol: str) -> float:
        """Calculate position size based on regime and strategy confidence"""
        if not ORCHESTRATOR.should_run_component('phase_3_adaptive_execution', 'risk_management'):
            return 0.0
        
        # Combined confidence (weighted average)
        combined_confidence = (regime_confidence * 0.6 + strategy_confidence * 0.4)
        
        # Base position size from risk config
        base_size = RISK_CONFIG.INITIAL_BALANCE * RISK_CONFIG.MAX_RISK_PER_TRADE
        
        # Adjust based on confidence tier
        if combined_confidence >= self.confidence_tiers['high']:
            position_multiplier = 1.0
            risk_level = RiskLevel.LOW
        elif combined_confidence >= self.confidence_tiers['medium']:
            position_multiplier = 0.5
            risk_level = RiskLevel.MEDIUM
        else:
            position_multiplier = 0.0  # No trade
            risk_level = RiskLevel.HIGH
        
        # Calculate final position size
        position_size = base_size * position_multiplier
        
        # Apply maximum position size limit
        position_size = min(position_size, self.risk_parameters['max_position_size'])
        
        # Check if we have enough available balance
        available_balance = self._get_available_balance()
        position_size = min(position_size, available_balance * 0.5)  # Use max 50% of available
        
        logging.info(f"💰 Position size: ${position_size:.2f} (Confidence: {combined_confidence:.2f})")
        
        return position_size
    
    def assess_trade_risk(self, trade_signal: Dict[str, Any], 
                         market_conditions: Dict[str, Any]) -> RiskAssessment:
        """Comprehensive risk assessment for a potential trade"""
        risk_factors = []
        
        # 1. Confidence-based risk
        confidence = trade_signal.get('confidence', 0.0)
        if confidence < 0.6:
            risk_factors.append(f"Low confidence: {confidence:.2f}")
        
        # 2. Market volatility risk
        volatility = market_conditions.get('volatility', 0.0)
        if volatility > 0.01:
            risk_factors.append(f"High volatility: {volatility:.4f}")
        
        # 3. Concentration risk
        symbol = trade_signal.get('symbol', 'unknown')
        symbol_exposure = self._get_symbol_exposure(symbol)
        if symbol_exposure > 0:
            risk_factors.append(f"Existing exposure to {symbol}")
        
        # 4. Daily loss limit risk
        if self.daily_pnl < -self.risk_parameters['max_daily_loss'] * 0.5:
            risk_factors.append("Approaching daily loss limit")
        
        # 5. Concurrent trades risk
        current_trades = self._get_open_trades_count()
        if current_trades >= self.risk_parameters['max_concurrent_trades']:
            risk_factors.append(f"Max concurrent trades reached: {current_trades}")
        
        # Calculate overall risk level
        overall_risk = self._calculate_overall_risk(risk_factors, confidence)
        
        # Calculate position size with risk adjustment
        base_position = trade_signal.get('position_size', 0.0)
        risk_adjusted_position = self._adjust_position_for_risk(base_position, overall_risk)
        
        # Determine recommended action
        recommended_action = self._get_recommended_action(overall_risk, risk_factors)
        
        return RiskAssessment(
            overall_risk=overall_risk,
            position_size=risk_adjusted_position,
            max_drawdown=self.risk_parameters['max_daily_loss'],
            daily_loss_limit=self.risk_parameters['max_daily_loss'],
            recommended_action=recommended_action,
            confidence=confidence,
            risk_factors=risk_factors
        )
    
    def _calculate_overall_risk(self, risk_factors: List[str], confidence: float) -> RiskLevel:
        """Calculate overall risk level"""
        risk_score = len(risk_factors)
        
        # Adjust for confidence
        if confidence < 0.5:
            risk_score += 2
        elif confidence < 0.7:
            risk_score += 1
        
        # Classify risk level
        if risk_score >= 4:
            return RiskLevel.EXTREME
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _adjust_position_for_risk(self, base_position: float, risk_level: RiskLevel) -> float:
        """Adjust position size based on risk level"""
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.25,
            RiskLevel.EXTREME: 0.0
        }
        
        return base_position * risk_multipliers.get(risk_level, 0.0)
    
    def _get_recommended_action(self, risk_level: RiskLevel, risk_factors: List[str]) -> str:
        """Get recommended action based on risk level"""
        actions = {
            RiskLevel.LOW: "Proceed with trade",
            RiskLevel.MEDIUM: "Reduce position size",
            RiskLevel.HIGH: "Consider skipping trade",
            RiskLevel.EXTREME: "Do not trade"
        }
        
        base_action = actions.get(risk_level, "Do not trade")
        
        if risk_factors:
            return f"{base_action} - Factors: {', '.join(risk_factors)}"
        else:
            return base_action
    
    def can_execute_trade(self, trade_signal: Dict[str, Any], 
                         risk_assessment: RiskAssessment) -> bool:
        """Determine if trade can be executed based on risk assessment"""
        if risk_assessment.overall_risk == RiskLevel.EXTREME:
            logging.warning("🚨 Trade blocked: Extreme risk level")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.risk_parameters['max_daily_loss']:
            logging.warning("🚨 Trade blocked: Daily loss limit reached")
            return False
        
        # Check concurrent trades
        current_trades = self._get_open_trades_count()
        if current_trades >= self.risk_parameters['max_concurrent_trades']:
            logging.warning(f"🚨 Trade blocked: {current_trades} concurrent trades")
            return False
        
        # Check emergency stop loss
        total_balance = RISK_CONFIG.INITIAL_BALANCE + self.daily_pnl
        if total_balance <= RISK_CONFIG.INITIAL_BALANCE * (1 - RISK_CONFIG.EMERGENCY_STOP_LOSS):
            logging.warning("🚨 Trade blocked: Emergency stop loss triggered")
            return False
        
        # Check position size
        if risk_assessment.position_size <= 0:
            logging.warning("🚨 Trade blocked: Zero position size")
            return False
        
        return True
    
    def record_trade_execution(self, trade_data: Dict[str, Any]):
        """Record trade execution for risk tracking"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            **trade_data
        })
        
        # Update current exposure
        self.current_exposure += trade_data.get('position_size', 0)
        
        # Keep history manageable
        if len(self.trade_history) > 1000:
            self.trade_history.pop(0)
    
    def record_trade_outcome(self, trade_id: str, pnl: float):
        """Record trade outcome and update risk metrics"""
        # Find and update trade
        for trade in self.trade_history:
            if trade.get('trade_id') == trade_id:
                trade['pnl'] = pnl
                trade['closed_at'] = datetime.now()
                break
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Update current exposure
        self.current_exposure -= abs(pnl)  # Simplified exposure reduction
        
        # Update performance metrics
        self._update_performance_metrics()
        
        logging.info(f"📊 Trade outcome recorded: P&L ${pnl:.2f}, Daily P&L: ${self.daily_pnl:.2f}")
    
    def _update_performance_metrics(self):
        """Update performance metrics based on trade history"""
        if not self.trade_history:
            return
        
        # Calculate win rate
        completed_trades = [t for t in self.trade_history if 'pnl' in t]
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            self.performance_metrics['win_rate'] = len(winning_trades) / len(completed_trades)
            
            # Calculate profit factor
            total_profit = sum(t['pnl'] for t in winning_trades)
            total_loss = abs(sum(t['pnl'] for t in completed_trades if t['pnl'] < 0))
            self.performance_metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate current streak
            recent_trades = completed_trades[-10:]  # Last 10 trades
            if recent_trades:
                self.performance_metrics['current_streak'] = self._calculate_streak(recent_trades)
    
    def _calculate_streak(self, trades: List[Dict]) -> int:
        """Calculate current winning/losing streak"""
        if not trades:
            return 0
        
        streak = 0
        last_pnl_sign = np.sign(trades[-1]['pnl'])
        
        for trade in reversed(trades):
            if np.sign(trade['pnl']) == last_pnl_sign:
                streak += 1 if last_pnl_sign > 0 else -1
            else:
                break
        
        return streak
    
    def _get_available_balance(self) -> float:
        """Calculate available balance considering current exposure"""
        total_balance = RISK_CONFIG.INITIAL_BALANCE + self.daily_pnl
        return max(0, total_balance - self.current_exposure)
    
    def _get_symbol_exposure(self, symbol: str) -> float:
        """Get current exposure to a specific symbol"""
        open_trades = [t for t in self.trade_history if t.get('status') == 'open' and t.get('symbol') == symbol]
        return sum(t.get('position_size', 0) for t in open_trades)
    
    def _get_open_trades_count(self) -> int:
        """Get number of currently open trades"""
        return len([t for t in self.trade_history if t.get('status') == 'open'])
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard"""
        total_balance = RISK_CONFIG.INITIAL_BALANCE + self.daily_pnl
        available_balance = self._get_available_balance()
        
        return {
            'current_balance': total_balance,
            'available_balance': available_balance,
            'daily_pnl': self.daily_pnl,
            'current_exposure': self.current_exposure,
            'open_trades': self._get_open_trades_count(),
            'performance_metrics': self.performance_metrics,
            'risk_limits': {
                'max_daily_loss': self.risk_parameters['max_daily_loss'],
                'max_position_size': self.risk_parameters['max_position_size'],
                'max_concurrent_trades': self.risk_parameters['max_concurrent_trades'],
                'emergency_stop_loss': self.risk_parameters['emergency_stop_loss']
            },
            'current_risk_level': self._get_current_risk_level(),
            'recommendations': self._get_risk_recommendations()
        }
    
    def _get_current_risk_level(self) -> str:
        """Get current overall risk level"""
        risk_score = 0
        
        # Daily P&L contribution
        daily_loss_ratio = abs(self.daily_pnl) / self.risk_parameters['max_daily_loss']
        if daily_loss_ratio > 0.8:
            risk_score += 3
        elif daily_loss_ratio > 0.5:
            risk_score += 2
        elif daily_loss_ratio > 0.3:
            risk_score += 1
        
        # Exposure contribution
        exposure_ratio = self.current_exposure / RISK_CONFIG.INITIAL_BALANCE
        if exposure_ratio > 0.3:
            risk_score += 2
        elif exposure_ratio > 0.2:
            risk_score += 1
        
        # Concurrent trades contribution
        trades_ratio = self._get_open_trades_count() / self.risk_parameters['max_concurrent_trades']
        if trades_ratio >= 1.0:
            risk_score += 2
        elif trades_ratio > 0.7:
            risk_score += 1
        
        if risk_score >= 5:
            return "EXTREME"
        elif risk_score >= 3:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_risk_recommendations(self) -> List[str]:
        """Get risk management recommendations"""
        recommendations = []
        
        # Daily P&L recommendations
        daily_loss_ratio = abs(self.daily_pnl) / self.risk_parameters['max_daily_loss']
        if daily_loss_ratio > 0.7:
            recommendations.append("Consider stopping trading for today - approaching daily loss limit")
        elif daily_loss_ratio > 0.5:
            recommendations.append("Reduce position sizes - significant daily losses")
        
        # Exposure recommendations
        exposure_ratio = self.current_exposure / RISK_CONFIG.INITIAL_BALANCE
        if exposure_ratio > 0.25:
            recommendations.append("Reduce overall exposure - high concentration risk")
        
        # Performance recommendations
        if self.performance_metrics['current_streak'] < -3:
            recommendations.append("Recent losing streak - consider strategy review")
        
        return recommendations
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call this at start of new trading day)"""
        self.daily_pnl = 0.0
        logging.info("🔄 Daily risk metrics reset")

# Global risk manager instance
RISK_MANAGER = AdaptiveRiskManager()
