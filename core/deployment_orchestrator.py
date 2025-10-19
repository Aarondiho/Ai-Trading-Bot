"""
Deployment Orchestrator
Controls which phases and components are active
"""

import logging
from datetime import datetime
from typing import Dict, Any

from config import PHASES

class DeploymentOrchestrator:
    """Orchestrates the activation of system phases"""
    
    def __init__(self):
        self.current_phase = 'phase_1_pattern_archaeology'
        self.phase_history = []
        self.performance_metrics = {}
        
    def should_run_component(self, phase: str, component: str) -> bool:
        """Check if a specific component should be running"""
        return (PHASES.phases.get(phase, {}).get('active', False) and
                PHASES.phases.get(phase, {}).get('components', {}).get(component, False))
    
    def get_active_components(self) -> Dict[str, list]:
        """Get all currently active components"""
        active = {}
        for phase_name, phase_config in PHASES.phases.items():
            if phase_config['active']:
                active[phase_name] = [
                    comp for comp, is_active in phase_config['components'].items() 
                    if is_active
                ]
        return active
    
    def progress_to_next_phase(self, current_phase_performance: Dict[str, Any]):
        """Evaluate current phase performance and decide if ready for next phase"""
        self.performance_metrics[self.current_phase] = current_phase_performance
        
        phase_requirements = {
            'phase_1_pattern_archaeology': {
                'min_regime_accuracy': 0.75,
                'min_data_quality': 0.90,
                'pattern_validation_passed': True
            },
            'phase_2_transition_prediction': {
                'min_transition_accuracy': 0.60,
                'early_warning_effectiveness': 0.70,
                'temporal_patterns_validated': True
            },
            'phase_3_adaptive_execution': {
                'backtest_profitability': True,
                'risk_controls_working': True,
                'live_demo_success': True
            }
        }
        
        current_req = phase_requirements.get(self.current_phase, {})
        meets_requirements = all(
            current_phase_performance.get(metric, 0) >= threshold
            for metric, threshold in current_req.items()
            if not isinstance(threshold, bool)
        ) and all(
            current_phase_performance.get(metric, False) == threshold
            for metric, threshold in current_req.items()
            if isinstance(threshold, bool)
        )
        
        if meets_requirements:
            self.activate_next_phase()
    
    def activate_next_phase(self):
        """Activate the next phase in sequence"""
        phase_sequence = [
            'phase_1_pattern_archaeology',
            'phase_2_transition_prediction', 
            'phase_3_adaptive_execution',
            'phase_4_continuous_learning'
        ]
        
        current_index = phase_sequence.index(self.current_phase)
        if current_index < len(phase_sequence) - 1:
            next_phase = phase_sequence[current_index + 1]
            PHASES.activate_phase(next_phase)
            self.current_phase = next_phase
            self.phase_history.append({
                'timestamp': datetime.now(),
                'phase': next_phase,
                'action': 'activated'
            })
            logging.info(f"🎯 Progressed to {next_phase}")
    
    def emergency_deactivate(self, reason: str):
        """Emergency shutdown of all trading components"""
        logging.error(f"🚨 EMERGENCY SHUTDOWN: {reason}")
        
        # Deactivate all trading-related components
        PHASES.deactivate_phase('phase_3_adaptive_execution')
        PHASES.deactivate_phase('phase_4_continuous_learning')
        
        # Keep only data collection and analysis
        PHASES.activate_phase('phase_1_pattern_archaeology')
        PHASES.activate_phase('phase_2_transition_prediction')
        
        self.phase_history.append({
            'timestamp': datetime.now(),
            'phase': 'all_trading',
            'action': 'emergency_deactivate',
            'reason': reason
        })

# Global orchestrator instance
ORCHESTRATOR = DeploymentOrchestrator()
