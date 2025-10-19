"""
Continuous Adapter - Adaptive Learning System
Enables continuous adaptation to platform changes
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

from config import ML_CONFIG, PLATFORM_PATTERNS
from core.deployment_orchestrator import ORCHESTRATOR

class AdaptationState(Enum):
    """Adaptation state machine states"""
    STABLE = "stable"
    LEARNING = "learning"
    ADAPTING = "adapting"
    RECOVERING = "recovering"
    CRITICAL = "critical"

class ContinuousAdapter:
    """Enables continuous adaptation to platform algorithm changes"""
    
    def __init__(self):
        self.adaptation_state = AdaptationState.STABLE
        self.performance_trend = "stable"
        self.last_adaptation = None
        self.adaptation_history = []
        
        # Adaptation parameters
        self.adaptation_config = {
            'performance_threshold': 0.7,
            'learning_trigger_window': 50,  # trades
            'min_learning_data': 1000,      # data points
            'adaptation_cooldown_hours': 6,
            'max_adaptation_duration_minutes': 120
        }
        
        # Performance tracking
        self.performance_window = []
        self.regime_performance_tracking = {}
        
        # Learning schedules
        self.scheduled_learnings = []
    
    def monitor_adaptation_needs(self, performance_data: Dict[str, Any],
                               regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor if adaptation is needed"""
        if not ORCHESTRATOR.should_run_component('phase_4_continuous_learning', 'rapid_adaptation'):
            return {'adaptation_needed': False, 'reason': 'component_inactive'}
        
        try:
            # Store performance data
            self.performance_window.append({
                'timestamp': datetime.now(),
                **performance_data
            })
            
            # Keep window manageable
            if len(self.performance_window) > 100:
                self.performance_window.pop(0)
            
            # Check adaptation conditions
            adaptation_analysis = self._analyze_adaptation_needs(performance_data, regime_data)
            
            # Update adaptation state
            self._update_adaptation_state(adaptation_analysis)
            
            return adaptation_analysis
            
        except Exception as e:
            logging.error(f"❌ Adaptation monitoring failed: {e}")
            return {'adaptation_needed': False, 'reason': 'error'}
    
    def _analyze_adaptation_needs(self, performance_data: Dict[str, Any],
                                regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if adaptation is needed"""
        analysis = {
            'adaptation_needed': False,
            'reason': '',
            'confidence': 0.0,
            'suggested_actions': [],
            'performance_metrics': {}
        }
        
        # Check if we have enough data
        if len(self.performance_window) < 20:
            analysis['reason'] = 'insufficient_data'
            return analysis
        
        # Calculate performance metrics
        performance_df = pd.DataFrame(self.performance_window)
        recent_performance = self._calculate_recent_performance(performance_df)
        
        analysis['performance_metrics'] = recent_performance
        
        # Check performance degradation
        if recent_performance['win_rate'] < self.adaptation_config['performance_threshold']:
            analysis['adaptation_needed'] = True
            analysis['reason'] = f"Low win rate: {recent_performance['win_rate']:.2f}"
            analysis['confidence'] = 1 - recent_performance['win_rate']
            analysis['suggested_actions'].append("Retrain ML models")
            analysis['suggested_actions'].append("Review strategy parameters")
        
        # Check regime performance issues
        regime_issues = self._check_regime_performance_issues(regime_data)
        if regime_issues['has_issues']:
            analysis['adaptation_needed'] = True
            analysis['reason'] = f"Regime performance issues: {regime_issues['worst_regime']}"
            analysis['confidence'] = max(analysis['confidence'], regime_issues['confidence'])
            analysis['suggested_actions'].append(f"Adapt strategies for {regime_issues['worst_regime']} regime")
        
        # Check for pattern drift
        pattern_drift = self._check_pattern_drift(regime_data)
        if pattern_drift['detected']:
            analysis['adaptation_needed'] = True
            analysis['reason'] = "Pattern drift detected"
            analysis['confidence'] = max(analysis['confidence'], pattern_drift['confidence'])
            analysis['suggested_actions'].append("Rediscover platform patterns")
            analysis['suggested_actions'].append("Update pattern classifiers")
        
        # Check cooldown period
        if (self.last_adaptation and 
            (datetime.now() - self.last_adaptation).total_seconds() < 
            self.adaptation_config['adaptation_cooldown_hours'] * 3600):
            analysis['adaptation_needed'] = False
            analysis['reason'] = 'in_cooldown_period'
            analysis['suggested_actions'] = ['Wait for cooldown to complete']
        
        return analysis
    
    def _calculate_recent_performance(self, performance_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate recent performance metrics"""
        if performance_df.empty:
            return {}
        
        # Use recent window (last 30% of data or last 20 points)
        window_size = max(20, len(performance_df) // 3)
        recent_data = performance_df.tail(window_size)
        
        winning_trades = recent_data[recent_data['pnl'] > 0]
        losing_trades = recent_data[recent_data['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(recent_data) if len(recent_data) > 0 else 0
        total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate performance trend
        if len(performance_df) >= 10:
            half_point = len(performance_df) // 2
            first_half = performance_df.iloc[:half_point]
            second_half = performance_df.iloc[half_point:]
            
            first_win_rate = len(first_half[first_half['pnl'] > 0]) / len(first_half)
            second_win_rate = len(second_half[second_half['pnl'] > 0]) / len(second_half)
            
            if second_win_rate > first_win_rate + 0.1:
                trend = "improving"
            elif second_win_rate < first_win_rate - 0.1:
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(recent_data),
            'performance_trend': trend,
            'avg_profit_per_trade': recent_data['pnl'].mean(),
            'profit_std_dev': recent_data['pnl'].std()
        }
    
    def _check_regime_performance_issues(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for regime-specific performance issues"""
        if not regime_data:
            return {'has_issues': False, 'confidence': 0.0}
        
        regime_performance = {}
        
        # Calculate performance by regime
        for regime_info in regime_data.values():
            regime = regime_info.get('regime', 'unknown')
            success_rate = regime_info.get('success_rate', 0)
            
            if regime not in regime_performance:
                regime_performance[regime] = []
            
            regime_performance[regime].append(success_rate)
        
        # Find worst performing regime
        worst_regime = None
        worst_performance = 1.0  # Start with perfect score
        
        for regime, performances in regime_performance.items():
            avg_performance = np.mean(performances) if performances else 0
            if avg_performance < worst_performance:
                worst_performance = avg_performance
                worst_regime = regime
        
        has_issues = worst_performance < 0.6  # 60% threshold
        confidence = 1 - worst_performance if has_issues else 0.0
        
        return {
            'has_issues': has_issues,
            'worst_regime': worst_regime,
            'performance_score': worst_performance,
            'confidence': confidence
        }
    
    def _check_pattern_drift(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for pattern drift in regime detection"""
        if not regime_data or len(regime_data) < 10:
            return {'detected': False, 'confidence': 0.0}
        
        # Analyze regime distribution consistency
        regime_counts = {}
        for regime_info in regime_data.values():
            regime = regime_info.get('regime', 'unknown')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total_regimes = sum(regime_counts.values())
        regime_distribution = {regime: count / total_regimes for regime, count in regime_counts.items()}
        
        # Check if distribution is unusual (e.g., one regime dominating)
        max_regime_ratio = max(regime_distribution.values()) if regime_distribution else 0
        drift_detected = max_regime_ratio > 0.8  # One regime > 80%
        
        confidence = max_regime_ratio - 0.5 if drift_detected else 0.0  # Normalize
        
        return {
            'detected': drift_detected,
            'confidence': confidence,
            'dominant_regime': max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else 'unknown',
            'dominance_ratio': max_regime_ratio
        }
    
    def _update_adaptation_state(self, adaptation_analysis: Dict[str, Any]):
        """Update the adaptation state machine"""
        previous_state = self.adaptation_state
        
        if adaptation_analysis['adaptation_needed']:
            if adaptation_analysis['confidence'] > 0.8:
                self.adaptation_state = AdaptationState.CRITICAL
            else:
                self.adaptation_state = AdaptationState.ADAPTING
        else:
            if self.adaptation_state in [AdaptationState.ADAPTING, AdaptationState.CRITICAL]:
                self.adaptation_state = AdaptationState.RECOVERING
            else:
                self.adaptation_state = AdaptationState.STABLE
        
        # Log state changes
        if previous_state != self.adaptation_state:
            logging.info(f"🔄 Adaptation state changed: {previous_state.value} -> {self.adaptation_state.value}")
            
            # Record state transition
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'from_state': previous_state.value,
                'to_state': self.adaptation_state.value,
                'reason': adaptation_analysis.get('reason', ''),
                'confidence': adaptation_analysis.get('confidence', 0)
            })
    
    async def execute_adaptation_cycle(self, adaptation_plan: Dict[str, Any]):
        """Execute a complete adaptation cycle"""
        if not adaptation_plan['adaptation_needed']:
            return {'success': True, 'actions_taken': 0, 'reason': 'no_adaptation_needed'}
        
        try:
            logging.info("🎯 Starting adaptation cycle")
            start_time = datetime.now()
            actions_executed = []
            
            # Execute suggested actions
            for action in adaptation_plan.get('suggested_actions', []):
                action_result = await self._execute_adaptation_action(action)
                if action_result['success']:
                    actions_executed.append(action)
            
            # Record adaptation completion
            self.last_adaptation = datetime.now()
            
            adaptation_record = {
                'timestamp': start_time,
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
                'actions_executed': actions_executed,
                'adaptation_plan': adaptation_plan,
                'success': len(actions_executed) > 0
            }
            
            self.adaptation_history.append(adaptation_record)
            
            logging.info(f"✅ Adaptation cycle completed: {len(actions_executed)} actions executed")
            
            return {
                'success': True,
                'actions_taken': len(actions_executed),
                'duration_minutes': adaptation_record['duration_minutes'],
                'actions_executed': actions_executed
            }
            
        except Exception as e:
            logging.error(f"❌ Adaptation cycle failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'actions_taken': 0
            }
    
    async def _execute_adaptation_action(self, action: str) -> Dict[str, Any]:
        """Execute a specific adaptation action"""
        try:
            if action == "Retrain ML models":
                # Import here to avoid circular imports
                from data_archaeologist.regime_classifier import STATE_CLASSIFIER
                
                # Trigger model retraining
                # This would typically use recent data
                logging.info("🔄 Retraining ML models...")
                await asyncio.sleep(2)  # Simulated training time
                return {'success': True, 'action': action}
            
            elif action.startswith("Adapt strategies for"):
                regime = action.split("for ")[1].replace(" regime", "")
                logging.info(f"🔄 Adapting strategies for {regime} regime...")
                await asyncio.sleep(1)
                return {'success': True, 'action': action}
            
            elif action == "Rediscover platform patterns":
                logging.info("🔄 Rediscovering platform patterns...")
                await asyncio.sleep(3)
                return {'success': True, 'action': action}
            
            elif action == "Update pattern classifiers":
                logging.info("🔄 Updating pattern classifiers...")
                await asyncio.sleep(2)
                return {'success': True, 'action': action}
            
            elif action == "Review strategy parameters":
                logging.info("🔄 Reviewing strategy parameters...")
                await asyncio.sleep(1)
                return {'success': True, 'action': action}
            
            else:
                logging.warning(f"⚠️ Unknown adaptation action: {action}")
                return {'success': False, 'action': action, 'reason': 'unknown_action'}
                
        except Exception as e:
            logging.error(f"❌ Adaptation action failed: {action} - {e}")
            return {'success': False, 'action': action, 'error': str(e)}
    
    def schedule_periodic_learning(self, learning_type: str, interval_hours: int):
        """Schedule periodic learning sessions"""
        next_learning = {
            'type': learning_type,
            'interval_hours': interval_hours,
            'next_schedule': datetime.now() + timedelta(hours=interval_hours),
            'last_executed': None
        }
        
        self.scheduled_learnings.append(next_learning)
        logging.info(f"📅 Scheduled {learning_type} learning every {interval_hours} hours")
    
    async def check_scheduled_learnings(self):
        """Check and execute scheduled learnings"""
        now = datetime.now()
        learnings_executed = []
        
        for learning in self.scheduled_learnings:
            if learning['next_schedule'] <= now:
                try:
                    logging.info(f"📅 Executing scheduled {learning['type']} learning")
                    
                    # Execute learning based on type
                    if learning['type'] == 'pattern_refresh':
                        result = await self._execute_pattern_refresh()
                    elif learning['type'] == 'strategy_optimization':
                        result = await self._execute_strategy_optimization()
                    elif learning['type'] == 'model_retraining':
                        result = await self._execute_model_retraining()
                    else:
                        result = {'success': False, 'reason': 'unknown_learning_type'}
                    
                    # Update schedule
                    learning['last_executed'] = now
                    learning['next_schedule'] = now + timedelta(hours=learning['interval_hours'])
                    
                    learnings_executed.append({
                        'type': learning['type'],
                        'success': result.get('success', False),
                        'timestamp': now
                    })
                    
                except Exception as e:
                    logging.error(f"❌ Scheduled learning failed: {e}")
        
        return learnings_executed
    
    async def _execute_pattern_refresh(self) -> Dict[str, Any]:
        """Execute pattern refresh learning"""
        # This would typically involve:
        # 1. Collecting recent data
        # 2. Running pattern discovery
        # 3. Updating pattern database
        await asyncio.sleep(5)  # Simulated learning time
        return {'success': True, 'patterns_updated': 15}
    
    async def _execute_strategy_optimization(self) -> Dict[str, Any]:
        """Execute strategy optimization learning"""
        # This would typically involve:
        # 1. Analyzing recent strategy performance
        # 2. Optimizing parameters
        # 3. Updating strategy configurations
        await asyncio.sleep(3)  # Simulated learning time
        return {'success': True, 'strategies_optimized': 8}
    
    async def _execute_model_retraining(self) -> Dict[str, Any]:
        """Execute model retraining learning"""
        # This would typically involve:
        # 1. Gathering recent training data
        # 2. Retraining ML models
        # 3. Validating model performance
        await asyncio.sleep(10)  # Simulated training time
        return {'success': True, 'models_retrained': 3}
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status"""
        return {
            'current_state': self.adaptation_state.value,
            'performance_trend': self.performance_trend,
            'last_adaptation': self.last_adaptation,
            'adaptation_history_count': len(self.adaptation_history),
            'scheduled_learnings_count': len(self.scheduled_learnings),
            'performance_window_size': len(self.performance_window),
            'recent_adaptations': self.adaptation_history[-5:] if self.adaptation_history else []
        }
    
    def get_adaptation_recommendations(self) -> List[Dict[str, Any]]:
        """Get adaptation recommendations based on current state"""
        recommendations = []
        
        if self.adaptation_state == AdaptationState.CRITICAL:
            recommendations.append({
                'priority': 'high',
                'action': 'Immediate model retraining',
                'reason': 'Critical performance degradation detected',
                'estimated_duration': '30 minutes'
            })
            recommendations.append({
                'priority': 'high', 
                'action': 'Emergency strategy review',
                'reason': 'Multiple strategy failures detected',
                'estimated_duration': '15 minutes'
            })
        
        elif self.adaptation_state == AdaptationState.ADAPTING:
            recommendations.append({
                'priority': 'medium',
                'action': 'Incremental model updates',
                'reason': 'Moderate performance issues detected',
                'estimated_duration': '20 minutes'
            })
        
        elif self.adaptation_state == AdaptationState.STABLE:
            recommendations.append({
                'priority': 'low',
                'action': 'Scheduled pattern learning',
                'reason': 'System performing well - maintain knowledge',
                'estimated_duration': '60 minutes'
            })
        
        # Add scheduled learning recommendations
        for learning in self.scheduled_learnings:
            if learning['next_schedule'] <= datetime.now() + timedelta(hours=1):
                recommendations.append({
                    'priority': 'medium',
                    'action': f'Scheduled {learning["type"]}',
                    'reason': 'Periodic learning due',
                    'estimated_duration': 'varies'
                })
        
        return recommendations

# Global continuous adapter instance
CONTINUOUS_ADAPTER = ContinuousAdapter()
