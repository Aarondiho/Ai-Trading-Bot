"""
Knowledge Preserver - Pattern and Strategy Preservation
Preserves working patterns and strategies during platform updates
"""

import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

from config import PLATFORM_PATTERNS
from core.deployment_orchestrator import ORCHESTRATOR

@dataclass
class PreservedPattern:
    """Data structure for preserving pattern knowledge"""
    pattern_name: str
    pattern_signature: Dict[str, Any]
    performance_metrics: Dict[str, float]
    discovery_date: datetime
    last_verified: datetime
    verification_count: int
    success_rate: float
    stability_score: float
    is_active: bool

@dataclass
class PreservedStrategy:
    """Data structure for preserving strategy knowledge"""
    strategy_name: str
    regime: str
    parameters: Dict[str, Any]
    performance_history: List[Dict[str, Any]]
    total_trades: int
    win_rate: float
    profit_factor: float
    last_used: datetime
    is_active: bool

class KnowledgeBank:
    """Preserves and manages trading knowledge across platform updates"""
    
    def __init__(self):
        self.preserved_patterns: Dict[str, PreservedPattern] = {}
        self.preserved_strategies: Dict[str, PreservedStrategy] = {}
        self.platform_fingerprints: List[Dict[str, Any]] = []
        self.learning_cycles: List[Dict[str, Any]] = []
        
        # Knowledge preservation settings
        self.preservation_config = {
            'min_verifications': 5,
            'min_success_rate': 0.6,
            'max_pattern_age_days': 90,
            'backup_interval_hours': 24
        }
        
        self.last_backup = None
    
    def preserve_pattern_knowledge(self, pattern_data: Dict[str, Any], 
                                 performance_data: Dict[str, Any]) -> str:
        """Preserve a discovered pattern with its performance metrics"""
        if not ORCHESTRATOR.should_run_component('phase_4_continuous_learning', 'knowledge_preservation'):
            return "preservation_paused"
        
        try:
            pattern_name = pattern_data.get('pattern_name', 'unknown')
            pattern_signature = self._create_pattern_signature(pattern_data)
            
            # Calculate pattern stability and success metrics
            stability_score = self._calculate_pattern_stability(pattern_data, performance_data)
            success_rate = performance_data.get('success_rate', 0.0)
            
            # Check if pattern meets preservation criteria
            if not self._meets_preservation_criteria(success_rate, stability_score):
                logging.info(f"⏸️ Pattern {pattern_name} doesn't meet preservation criteria")
                return "below_threshold"
            
            # Create or update preserved pattern
            if pattern_name in self.preserved_patterns:
                self._update_existing_pattern(pattern_name, pattern_data, performance_data, stability_score)
            else:
                self._create_new_pattern(pattern_name, pattern_signature, pattern_data, 
                                       performance_data, stability_score)
            
            logging.info(f"💾 Preserved pattern: {pattern_name} (Success: {success_rate:.2f}, Stability: {stability_score:.2f})")
            
            # Create backup if needed
            self._check_and_backup()
            
            return "preserved"
            
        except Exception as e:
            logging.error(f"❌ Pattern preservation failed: {e}")
            return "error"
    
    def _create_pattern_signature(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a unique signature for the pattern"""
        signature_data = {
            'volatility_range': pattern_data.get('volatility_range', [0, 0]),
            'trend_strength_range': pattern_data.get('trend_strength_range', [0, 0]),
            'duration_range': pattern_data.get('duration_range', [0, 0]),
            'time_of_day_patterns': pattern_data.get('time_of_day_patterns', {}),
            'key_indicators': pattern_data.get('key_indicators', [])
        }
        
        # Create hash for quick comparison
        signature_json = json.dumps(signature_data, sort_keys=True)
        signature_hash = hashlib.md5(signature_json.encode()).hexdigest()
        
        return {
            'data': signature_data,
            'hash': signature_hash
        }
    
    def _calculate_pattern_stability(self, pattern_data: Dict[str, Any], 
                                   performance_data: Dict[str, Any]) -> float:
        """Calculate pattern stability score (0-1)"""
        stability_factors = []
        
        # Duration stability
        duration_std = pattern_data.get('duration_std', 0)
        duration_mean = pattern_data.get('duration_mean', 1)
        duration_stability = 1 - min(duration_std / duration_mean, 1.0)
        stability_factors.append(duration_stability * 0.3)
        
        # Performance consistency
        performance_std = performance_data.get('performance_std', 0)
        performance_mean = abs(performance_data.get('performance_mean', 1))
        performance_stability = 1 - min(performance_std / performance_mean, 1.0)
        stability_factors.append(performance_stability * 0.4)
        
        # Time consistency
        time_consistency = pattern_data.get('time_consistency', 0.5)
        stability_factors.append(time_consistency * 0.3)
        
        return sum(stability_factors)
    
    def _meets_preservation_criteria(self, success_rate: float, stability_score: float) -> bool:
        """Check if pattern meets preservation criteria"""
        return (success_rate >= self.preservation_config['min_success_rate'] and
                stability_score >= 0.6)
    
    def _update_existing_pattern(self, pattern_name: str, pattern_data: Dict[str, Any],
                               performance_data: Dict[str, Any], stability_score: float):
        """Update an existing preserved pattern"""
        pattern = self.preserved_patterns[pattern_name]
        
        # Update metrics
        pattern.last_verified = datetime.now()
        pattern.verification_count += 1
        
        # Update performance metrics (weighted average)
        old_weight = pattern.verification_count - 1
        new_weight = 1
        
        pattern.success_rate = (
            (pattern.success_rate * old_weight) + 
            (performance_data.get('success_rate', 0) * new_weight)
        ) / pattern.verification_count
        
        pattern.stability_score = (
            (pattern.stability_score * old_weight) + 
            (stability_score * new_weight)
        ) / pattern.verification_count
        
        # Update pattern signature if improved
        if stability_score > pattern.stability_score:
            pattern.pattern_signature = self._create_pattern_signature(pattern_data)
    
    def _create_new_pattern(self, pattern_name: str, pattern_signature: Dict[str, Any],
                          pattern_data: Dict[str, Any], performance_data: Dict[str, Any],
                          stability_score: float):
        """Create a new preserved pattern"""
        self.preserved_patterns[pattern_name] = PreservedPattern(
            pattern_name=pattern_name,
            pattern_signature=pattern_signature,
            performance_metrics=performance_data,
            discovery_date=datetime.now(),
            last_verified=datetime.now(),
            verification_count=1,
            success_rate=performance_data.get('success_rate', 0),
            stability_score=stability_score,
            is_active=True
        )
    
    def preserve_strategy_knowledge(self, strategy_name: str, regime: str,
                                  parameters: Dict[str, Any], 
                                  performance_data: Dict[str, Any]):
        """Preserve strategy knowledge and parameters"""
        if not ORCHESTRATOR.should_run_component('phase_4_continuous_learning', 'knowledge_preservation'):
            return
        
        try:
            strategy_key = f"{strategy_name}_{regime}"
            
            if strategy_key in self.preserved_strategies:
                self._update_existing_strategy(strategy_key, parameters, performance_data)
            else:
                self._create_new_strategy(strategy_key, strategy_name, regime, 
                                        parameters, performance_data)
            
            logging.info(f"💾 Preserved strategy: {strategy_name} for {regime} regime")
            
        except Exception as e:
            logging.error(f"❌ Strategy preservation failed: {e}")
    
    def _update_existing_strategy(self, strategy_key: str, parameters: Dict[str, Any],
                                performance_data: Dict[str, Any]):
        """Update an existing preserved strategy"""
        strategy = self.preserved_strategies[strategy_key]
        
        # Update parameters if performance improved
        current_performance = performance_data.get('win_rate', 0) * performance_data.get('profit_factor', 1)
        strategy_performance = strategy.win_rate * strategy.profit_factor
        
        if current_performance > strategy_performance:
            strategy.parameters = parameters
        
        # Update performance metrics
        strategy.win_rate = performance_data.get('win_rate', strategy.win_rate)
        strategy.profit_factor = performance_data.get('profit_factor', strategy.profit_factor)
        strategy.total_trades += performance_data.get('trades_count', 0)
        strategy.last_used = datetime.now()
        
        # Add to performance history
        strategy.performance_history.append({
            'timestamp': datetime.now(),
            **performance_data
        })
        
        # Keep history manageable
        if len(strategy.performance_history) > 100:
            strategy.performance_history.pop(0)
    
    def _create_new_strategy(self, strategy_key: str, strategy_name: str, regime: str,
                           parameters: Dict[str, Any], performance_data: Dict[str, Any]):
        """Create a new preserved strategy"""
        self.preserved_strategies[strategy_key] = PreservedStrategy(
            strategy_name=strategy_name,
            regime=regime,
            parameters=parameters,
            performance_history=[{
                'timestamp': datetime.now(),
                **performance_data
            }],
            total_trades=performance_data.get('trades_count', 0),
            win_rate=performance_data.get('win_rate', 0),
            profit_factor=performance_data.get('profit_factor', 1),
            last_used=datetime.now(),
            is_active=True
        )
    
    def get_preserved_patterns_for_regime(self, regime: str) -> List[PreservedPattern]:
        """Get preserved patterns for a specific regime"""
        return [
            pattern for pattern in self.preserved_patterns.values()
            if regime in pattern.pattern_name and pattern.is_active
        ]
    
    def get_best_strategy_for_regime(self, regime: str) -> Optional[PreservedStrategy]:
        """Get the best preserved strategy for a regime"""
        regime_strategies = [
            strategy for strategy in self.preserved_strategies.values()
            if strategy.regime == regime and strategy.is_active
        ]
        
        if not regime_strategies:
            return None
        
        # Score strategies by effectiveness
        scored_strategies = []
        for strategy in regime_strategies:
            effectiveness = strategy.win_rate * strategy.profit_factor
            recency_bonus = self._calculate_recency_bonus(strategy.last_used)
            score = effectiveness * recency_bonus
            scored_strategies.append((score, strategy))
        
        # Return highest scoring strategy
        return max(scored_strategies, key=lambda x: x[0])[1]
    
    def _calculate_recency_bonus(self, last_used: datetime) -> float:
        """Calculate recency bonus for strategies"""
        days_since_use = (datetime.now() - last_used).days
        return max(0.5, 1.0 - (days_since_use / 30))  # Linear decay over 30 days
    
    def detect_platform_fingerprint_change(self, current_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in platform fingerprint"""
        if not self.platform_fingerprints:
            return {'change_detected': False, 'confidence': 0.0}
        
        try:
            # Compare current patterns with historical fingerprints
            current_fingerprint = self._create_platform_fingerprint(current_patterns)
            historical_fingerprints = self.platform_fingerprints[-10:]  # Last 10 fingerprints
            
            similarity_scores = []
            for historical_fp in historical_fingerprints:
                similarity = self._calculate_fingerprint_similarity(current_fingerprint, historical_fp)
                similarity_scores.append(similarity)
            
            avg_similarity = np.mean(similarity_scores)
            change_detected = avg_similarity < 0.7  # 30% change threshold
            
            # Store current fingerprint
            self.platform_fingerprints.append(current_fingerprint)
            if len(self.platform_fingerprints) > 100:
                self.platform_fingerprints.pop(0)
            
            return {
                'change_detected': change_detected,
                'confidence': 1 - avg_similarity,
                'similarity_score': avg_similarity,
                'fingerprints_compared': len(similarity_scores)
            }
            
        except Exception as e:
            logging.error(f"❌ Fingerprint change detection failed: {e}")
            return {'change_detected': False, 'confidence': 0.0}
    
    def _create_platform_fingerprint(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Create platform fingerprint from current patterns"""
        fingerprint = {
            'timestamp': datetime.now(),
            'pattern_distribution': {},
            'average_volatility': 0,
            'average_trend_strength': 0,
            'regime_duration_stats': {}
        }
        
        if patterns:
            # Calculate pattern distribution
            pattern_counts = {}
            for pattern_name, pattern_data in patterns.items():
                regime = pattern_name.split('_')[0] if '_' in pattern_name else 'unknown'
                pattern_counts[regime] = pattern_counts.get(regime, 0) + 1
            
            total_patterns = sum(pattern_counts.values())
            fingerprint['pattern_distribution'] = {
                regime: count / total_patterns 
                for regime, count in pattern_counts.items()
            }
            
            # Calculate average metrics
            volatilities = [p.get('volatility', 0) for p in patterns.values()]
            trend_strengths = [p.get('trend_strength', 0) for p in patterns.values()]
            
            fingerprint['average_volatility'] = np.mean(volatilities) if volatilities else 0
            fingerprint['average_trend_strength'] = np.mean(trend_strengths) if trend_strengths else 0
        
        return fingerprint
    
    def _calculate_fingerprint_similarity(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> float:
        """Calculate similarity between two platform fingerprints"""
        similarity_scores = []
        
        # Compare pattern distributions
        dist1 = fp1.get('pattern_distribution', {})
        dist2 = fp2.get('pattern_distribution', {})
        
        all_regimes = set(dist1.keys()) | set(dist2.keys())
        for regime in all_regimes:
            p1 = dist1.get(regime, 0)
            p2 = dist2.get(regime, 0)
            similarity_scores.append(1 - abs(p1 - p2))
        
        # Compare average metrics
        vol_similarity = 1 - abs(fp1.get('average_volatility', 0) - fp2.get('average_volatility', 0))
        trend_similarity = 1 - abs(fp1.get('average_trend_strength', 0) - fp2.get('average_trend_strength', 0))
        
        similarity_scores.extend([vol_similarity, trend_similarity])
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def record_learning_cycle(self, cycle_data: Dict[str, Any]):
        """Record a learning cycle for meta-learning"""
        learning_cycle = {
            'timestamp': datetime.now(),
            'cycle_type': cycle_data.get('type', 'unknown'),
            'patterns_learned': cycle_data.get('patterns_learned', 0),
            'strategies_updated': cycle_data.get('strategies_updated', 0),
            'performance_improvement': cycle_data.get('performance_improvement', 0),
            'duration_minutes': cycle_data.get('duration_minutes', 0),
            'successful': cycle_data.get('successful', False)
        }
        
        self.learning_cycles.append(learning_cycle)
        
        # Keep cycles manageable
        if len(self.learning_cycles) > 50:
            self.learning_cycles.pop(0)
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of preserved knowledge"""
        active_patterns = [p for p in self.preserved_patterns.values() if p.is_active]
        active_strategies = [s for s in self.preserved_strategies.values() if s.is_active]
        
        return {
            'preserved_patterns_count': len(active_patterns),
            'preserved_strategies_count': len(active_strategies),
            'platform_fingerprints_count': len(self.platform_fingerprints),
            'learning_cycles_count': len(self.learning_cycles),
            'knowledge_health_score': self._calculate_knowledge_health_score(),
            'recent_learning_cycles': self.learning_cycles[-5:] if self.learning_cycles else [],
            'top_performing_patterns': self._get_top_performing_patterns(5),
            'knowledge_coverage': self._calculate_knowledge_coverage()
        }
    
    def _calculate_knowledge_health_score(self) -> float:
        """Calculate overall knowledge health score (0-100)"""
        if not self.preserved_patterns:
            return 0.0
        
        # Pattern quality score
        pattern_scores = []
        for pattern in self.preserved_patterns.values():
            if pattern.is_active:
                pattern_score = (pattern.success_rate * 0.6 + pattern.stability_score * 0.4) * 100
                pattern_scores.append(pattern_score)
        
        avg_pattern_score = np.mean(pattern_scores) if pattern_scores else 0
        
        # Strategy quality score
        strategy_scores = []
        for strategy in self.preserved_strategies.values():
            if strategy.is_active:
                strategy_score = (strategy.win_rate * strategy.profit_factor) * 100
                strategy_scores.append(strategy_score)
        
        avg_strategy_score = np.mean(strategy_scores) if strategy_scores else 0
        
        # Recency score (how recent is our knowledge)
        recent_patterns = [
            p for p in self.preserved_patterns.values() 
            if (datetime.now() - p.last_verified).days < 30
        ]
        recency_score = len(recent_patterns) / len(self.preserved_patterns) * 100 if self.preserved_patterns else 0
        
        return (avg_pattern_score * 0.4 + avg_strategy_score * 0.4 + recency_score * 0.2)
    
    def _get_top_performing_patterns(self, count: int) -> List[Dict[str, Any]]:
        """Get top performing patterns"""
        scored_patterns = []
        for pattern in self.preserved_patterns.values():
            if pattern.is_active:
                score = pattern.success_rate * pattern.stability_score
                scored_patterns.append((score, pattern))
        
        scored_patterns.sort(reverse=True, key=lambda x: x[0])
        
        return [
            {
                'name': pattern.pattern_name,
                'success_rate': pattern.success_rate,
                'stability_score': pattern.stability_score,
                'verifications': pattern.verification_count,
                'last_verified': pattern.last_verified
            }
            for score, pattern in scored_patterns[:count]
        ]
    
    def _calculate_knowledge_coverage(self) -> Dict[str, float]:
        """Calculate knowledge coverage across different regimes"""
        regime_coverage = {}
        all_regimes = list(PLATFORM_PATTERNS.ALGORITHM_STATES.keys())
        
        for regime in all_regimes:
            regime_patterns = self.get_preserved_patterns_for_regime(regime)
            regime_strategies = [
                s for s in self.preserved_strategies.values() 
                if s.regime == regime and s.is_active
            ]
            
            pattern_coverage = min(len(regime_patterns) / 3, 1.0)  # At least 3 patterns per regime
            strategy_coverage = min(len(regime_strategies) / 2, 1.0)  # At least 2 strategies per regime
            
            regime_coverage[regime] = (pattern_coverage * 0.6 + strategy_coverage * 0.4) * 100
        
        return regime_coverage
    
    def save_knowledge_backup(self, filepath: str = "knowledge_backup.pkl"):
        """Save knowledge backup to file"""
        try:
            backup_data = {
                'preserved_patterns': self.preserved_patterns,
                'preserved_strategies': self.preserved_strategies,
                'platform_fingerprints': self.platform_fingerprints,
                'learning_cycles': self.learning_cycles,
                'backup_timestamp': datetime.now()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(backup_data, f)
            
            self.last_backup = datetime.now()
            logging.info(f"💾 Knowledge backup saved: {filepath}")
            
        except Exception as e:
            logging.error(f"❌ Knowledge backup failed: {e}")
    
    def load_knowledge_backup(self, filepath: str = "knowledge_backup.pkl") -> bool:
        """Load knowledge backup from file"""
        try:
            with open(filepath, 'rb') as f:
                backup_data = pickle.load(f)
            
            self.preserved_patterns = backup_data.get('preserved_patterns', {})
            self.preserved_strategies = backup_data.get('preserved_strategies', {})
            self.platform_fingerprints = backup_data.get('platform_fingerprints', [])
            self.learning_cycles = backup_data.get('learning_cycles', [])
            
            logging.info(f"💾 Knowledge backup loaded: {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Knowledge backup load failed: {e}")
            return False
    
    def _check_and_backup(self):
        """Check if backup is needed and create one"""
        if (self.last_backup is None or 
            (datetime.now() - self.last_backup).total_seconds() > self.preservation_config['backup_interval_hours'] * 3600):
            self.save_knowledge_backup()

# Global knowledge bank instance
KNOWLEDGE_BANK = KnowledgeBank()
