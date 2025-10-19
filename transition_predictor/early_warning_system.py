"""
Early Warning System - Change Point Detection
Detects when platform updates its algorithms
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
from scipy import stats
import ruptures as rpt

from config import PLATFORM_PATTERNS
from core.deployment_orchestrator import ORCHESTRATOR

class PlatformUpdateDetector:
    """Detects platform algorithm changes and provides early warnings"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=200)
        self.pattern_consistency = {}
        self.anomaly_buffer = deque(maxlen=50)
        self.last_detected_change = None
        
        # Detection thresholds
        self.detection_config = {
            'performance_threshold': 0.3,      # 30% performance drop
            'pattern_consistency_threshold': 0.6,
            'anomaly_threshold': 3.0,          # Z-score threshold
            'change_point_penalty': 2.0,
            'minimum_observations': 25
        }
        
        # Alert system
        self.alerts_issued = []
        self.alert_cooldown = timedelta(hours=1)
    
    def monitor_for_algorithm_changes(self, current_performance: Dict[str, Any], 
                                    recent_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor for platform algorithm changes"""
        if not ORCHESTRATOR.should_run_component('phase_2_transition_prediction', 'early_warning_system'):
            return {'status': 'monitoring_paused'}
        
        try:
            # Store current performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'accuracy': current_performance.get('accuracy', 0),
                'confidence': current_performance.get('confidence', 0),
                'win_rate': current_performance.get('win_rate', 0)
            })
            
            # Check if we have enough data
            if len(self.performance_history) < self.detection_config['minimum_observations']:
                return {'status': 'collecting_data', 'data_points': len(self.performance_history)}
            
            # Run detection algorithms
            detection_results = {
                'performance_degradation': self._detect_performance_degradation(),
                'pattern_inconsistency': self._detect_pattern_inconsistency(recent_predictions),
                'statistical_anomalies': self._detect_statistical_anomalies(),
                'change_points': self._detect_change_points()
            }
            
            # Calculate overall change probability
            change_probability = self._calculate_change_probability(detection_results)
            
            # Check if we should issue alert
            alert_info = self._evaluate_alert_conditions(detection_results, change_probability)
            
            return {
                'status': 'monitoring_active',
                'change_probability': change_probability,
                'detection_results': detection_results,
                'alert_issued': alert_info['alert_issued'],
                'alert_level': alert_info['alert_level'],
                'recommended_action': alert_info['recommended_action'],
                'data_points': len(self.performance_history)
            }
            
        except Exception as e:
            logging.error(f"❌ Change detection failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _detect_performance_degradation(self) -> Dict[str, Any]:
        """Detect significant performance degradation"""
        if len(self.performance_history) < 30:
            return {'detected': False, 'confidence': 0.0}
        
        # Extract performance metrics
        accuracies = [p['accuracy'] for p in self.performance_history]
        confidences = [p['confidence'] for p in self.performance_history]
        
        # Compare recent vs historical performance
        recent_window = 10
        if len(accuracies) >= recent_window * 2:
            recent_acc = np.mean(accuracies[-recent_window:])
            historical_acc = np.mean(accuracies[-recent_window*2:-recent_window])
            
            performance_drop = historical_acc - recent_acc
            drop_detected = performance_drop > self.detection_config['performance_threshold']
            
            return {
                'detected': drop_detected,
                'confidence': min(abs(performance_drop) / 0.5, 1.0),  # Normalize to 0-1
                'performance_drop': performance_drop,
                'recent_accuracy': recent_acc,
                'historical_accuracy': historical_acc
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_pattern_inconsistency(self, recent_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect inconsistencies in pattern predictions"""
        if not recent_predictions or len(recent_predictions) < 20:
            return {'detected': False, 'confidence': 0.0}
        
        # Analyze prediction consistency
        states = [p.get('predicted_state', 'unknown') for p in recent_predictions]
        confidences = [p.get('prediction_confidence', 0) for p in recent_predictions]
        
        # Calculate state consistency
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        if state_counts:
            dominant_state = max(state_counts.items(), key=lambda x: x[1])
            consistency_ratio = dominant_state[1] / len(states)
            
            inconsistency_detected = consistency_ratio < self.detection_config['pattern_consistency_threshold']
            
            return {
                'detected': inconsistency_detected,
                'confidence': 1 - consistency_ratio,  # Higher confidence when inconsistent
                'consistency_ratio': consistency_ratio,
                'dominant_state': dominant_state[0],
                'state_diversity': len(state_counts)
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_statistical_anomalies(self) -> Dict[str, Any]:
        """Detect statistical anomalies in performance data"""
        if len(self.performance_history) < 25:
            return {'detected': False, 'confidence': 0.0}
        
        accuracies = [p['accuracy'] for p in self.performance_history]
        confidences = [p['confidence'] for p in self.performance_history]
        
        # Calculate Z-scores for recent points
        recent_window = 5
        if len(accuracies) >= recent_window:
            recent_acc = accuracies[-recent_window:]
            mean_acc = np.mean(accuracies[:-recent_window])
            std_acc = np.std(accuracies[:-recent_window])
            
            if std_acc > 0:
                z_scores = [(x - mean_acc) / std_acc for x in recent_acc]
                anomalies = [abs(z) > self.detection_config['anomaly_threshold'] for z in z_scores]
                
                anomaly_count = sum(anomalies)
                anomaly_detected = anomaly_count >= 2  # At least 2 anomalies in recent window
                
                return {
                    'detected': anomaly_detected,
                    'confidence': min(anomaly_count / recent_window, 1.0),
                    'anomaly_count': anomaly_count,
                    'max_z_score': max(abs(z) for z in z_scores) if z_scores else 0
                }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_change_points(self) -> Dict[str, Any]:
        """Detect change points in performance time series"""
        if len(self.performance_history) < 50:
            return {'detected': False, 'confidence': 0.0}
        
        try:
            accuracies = [p['accuracy'] for p in self.performance_history]
            
            # Use ruptures for change point detection
            model = "rbf"  # Radial basis function kernel
            algo = rpt.Pelt(model=model, min_size=10, jump=5).fit(np.array(accuracies))
            change_points = algo.predict(pen=self.detection_config['change_point_penalty'])
            
            # Check if there's a recent change point
            recent_threshold = 15  # Consider change points in last 15 points as recent
            recent_changes = [cp for cp in change_points if cp >= len(accuracies) - recent_threshold]
            
            change_detected = len(recent_changes) > 0
            
            return {
                'detected': change_detected,
                'confidence': len(recent_changes) / recent_threshold,
                'change_points': change_points,
                'recent_change_points': recent_changes
            }
            
        except Exception as e:
            logging.warning(f"⚠️ Change point detection failed: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _calculate_change_probability(self, detection_results: Dict[str, Any]) -> float:
        """Calculate overall probability of platform algorithm change"""
        weights = {
            'performance_degradation': 0.4,
            'pattern_inconsistency': 0.3,
            'statistical_anomalies': 0.2,
            'change_points': 0.1
        }
        
        total_probability = 0.0
        total_weight = 0.0
        
        for detector, results in detection_results.items():
            if results.get('detected', False):
                weight = weights.get(detector, 0.1)
                confidence = results.get('confidence', 0.0)
                total_probability += weight * confidence
                total_weight += weight
        
        if total_weight > 0:
            return min(total_probability / total_weight, 1.0)
        else:
            return 0.0
    
    def _evaluate_alert_conditions(self, detection_results: Dict[str, Any], 
                                 change_probability: float) -> Dict[str, Any]:
        """Evaluate conditions for issuing alerts"""
        # Check cooldown period
        if self.alerts_issued:
            last_alert_time = self.alerts_issued[-1]['timestamp']
            if datetime.now() - last_alert_time < self.alert_cooldown:
                return {
                    'alert_issued': False,
                    'alert_level': 'none',
                    'recommended_action': 'continue_monitoring'
                }
        
        # Determine alert level
        if change_probability > 0.8:
            alert_level = 'high'
            recommended_action = 'pause_trading_immediately'
            issue_alert = True
        elif change_probability > 0.6:
            alert_level = 'medium'
            recommended_action = 'reduce_position_sizes'
            issue_alert = True
        elif change_probability > 0.4:
            alert_level = 'low'
            recommended_action = 'increase_monitoring'
            issue_alert = True
        else:
            alert_level = 'none'
            recommended_action = 'continue_normal_operations'
            issue_alert = False
        
        # Issue alert if needed
        if issue_alert:
            alert_info = {
                'timestamp': datetime.now(),
                'alert_level': alert_level,
                'change_probability': change_probability,
                'detection_results': detection_results,
                'recommended_action': recommended_action
            }
            self.alerts_issued.append(alert_info)
            self.last_detected_change = datetime.now()
            
            logging.warning(f"🚨 PLATFORM CHANGE ALERT - Level: {alert_level}, Probability: {change_probability:.2f}")
        
        return {
            'alert_issued': issue_alert,
            'alert_level': alert_level,
            'recommended_action': recommended_action
        }
    
    def get_early_warning_metrics(self) -> Dict[str, Any]:
        """Get current early warning system metrics"""
        return {
            'system_status': 'active',
            'total_alerts_issued': len(self.alerts_issued),
            'last_alert_time': self.alerts_issued[-1]['timestamp'] if self.alerts_issued else None,
            'last_detected_change': self.last_detected_change,
            'current_data_points': len(self.performance_history),
            'detection_sensitivity': self.detection_config,
            'recent_alerts': self.alerts_issued[-3:] if len(self.alerts_issued) >= 3 else self.alerts_issued
        }
    
    def adjust_detection_sensitivity(self, new_sensitivity: str):
        """Adjust detection sensitivity (conservative, normal, aggressive)"""
        sensitivity_profiles = {
            'conservative': {
                'performance_threshold': 0.4,
                'pattern_consistency_threshold': 0.5,
                'anomaly_threshold': 3.5,
                'change_point_penalty': 3.0
            },
            'normal': {
                'performance_threshold': 0.3,
                'pattern_consistency_threshold': 0.6,
                'anomaly_threshold': 3.0,
                'change_point_penalty': 2.0
            },
            'aggressive': {
                'performance_threshold': 0.2,
                'pattern_consistency_threshold': 0.7,
                'anomaly_threshold': 2.5,
                'change_point_penalty': 1.0
            }
        }
        
        if new_sensitivity in sensitivity_profiles:
            self.detection_config.update(sensitivity_profiles[new_sensitivity])
            logging.info(f"✅ Detection sensitivity set to: {new_sensitivity}")
        else:
            logging.warning(f"⚠️ Unknown sensitivity profile: {new_sensitivity}")

# Global early warning system instance
EARLY_WARNING_SYSTEM = PlatformUpdateDetector()
