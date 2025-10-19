"""
State Transition Model - Markov Analysis
Models how platform algorithms transition between states
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from collections import defaultdict, deque
import markovify

from config import PLATFORM_PATTERNS
from core.deployment_orchestrator import ORCHESTRATOR

class TransitionPredictor:
    """Predicts transitions between platform algorithm states"""
    
    def __init__(self):
        self.transition_matrix = None
        self.state_durations = {}
        self.transition_history = deque(maxlen=1000)
        self.markov_chain = None
        
        # Transition analysis
        self.analysis_results = {
            'state_persistence': {},
            'transition_probabilities': {},
            'typical_sequences': [],
            'last_analysis': None
        }
    
    def build_state_transition_model(self, regime_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build Markov transition model from regime history"""
        if not ORCHESTRATOR.should_run_component('phase_2_transition_prediction', 'state_transition_model'):
            return {}
        
        if len(regime_history) < 10:
            logging.warning("⚠️ Insufficient regime history for transition modeling")
            return {}
        
        try:
            # Extract state sequences
            state_sequences = self._extract_state_sequences(regime_history)
            
            # Calculate transition probabilities
            self.transition_matrix = self._calculate_transition_matrix(state_sequences)
            
            # Analyze state durations
            self.state_durations = self._analyze_state_durations(regime_history)
            
            # Build Markov chain
            self.markov_chain = self._build_markov_chain(state_sequences)
            
            # Find typical sequences
            self.analysis_results['typical_sequences'] = self._find_typical_sequences(state_sequences)
            self.analysis_results['last_analysis'] = datetime.now()
            
            logging.info("✅ State transition model built successfully")
            
            return {
                'transition_matrix': self.transition_matrix,
                'state_durations': self.state_durations,
                'typical_sequences': self.analysis_results['typical_sequences'],
                'model_confidence': self._calculate_model_confidence(state_sequences)
            }
            
        except Exception as e:
            logging.error(f"❌ Transition model building failed: {e}")
            return {}
    
    def _extract_state_sequences(self, regime_history: List[Dict[str, Any]]) -> List[List[str]]:
        """Extract sequences of states from regime history"""
        sequences = []
        current_sequence = []
        
        for regime in sorted(regime_history, key=lambda x: x.get('timestamp', datetime.now())):
            state = regime.get('current_state', 'unknown')
            if current_sequence and state != current_sequence[-1]:
                sequences.append(current_sequence.copy())
                current_sequence = [state]
            else:
                current_sequence.append(state)
        
        if current_sequence:
            sequences.append(current_sequence)
            
        return sequences
    
    def _calculate_transition_matrix(self, state_sequences: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """Calculate Markov transition probabilities"""
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for sequence in state_sequences:
            for i in range(len(sequence) - 1):
                from_state = sequence[i]
                to_state = sequence[i + 1]
                transition_counts[from_state][to_state] += 1
        
        # Convert counts to probabilities
        transition_matrix = {}
        for from_state, to_states in transition_counts.items():
            total_transitions = sum(to_states.values())
            transition_matrix[from_state] = {
                to_state: count / total_transitions
                for to_state, count in to_states.items()
            }
        
        return transition_matrix
    
    def _analyze_state_durations(self, regime_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze typical durations for each state"""
        state_durations = defaultdict(list)
        
        for regime in regime_history:
            state = regime.get('current_state', 'unknown')
            duration = regime.get('duration_minutes', 0)
            if duration > 0:
                state_durations[state].append(duration)
        
        duration_stats = {}
        for state, durations in state_durations.items():
            if durations:
                duration_stats[state] = {
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'std': np.std(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'count': len(durations)
                }
            else:
                duration_stats[state] = {
                    'mean': 0, 'median': 0, 'std': 0, 
                    'min': 0, 'max': 0, 'count': 0
                }
        
        return duration_stats
    
    def _build_markov_chain(self, state_sequences: List[List[str]]) -> Any:
        """Build Markov chain model for sequence prediction"""
        # Convert sequences to text for markovify
        sequence_texts = [' '.join(sequence) for sequence in state_sequences]
        text = '\n'.join(sequence_texts)
        
        try:
            return markovify.Text(text, state_size=2)
        except:
            return None
    
    def _find_typical_sequences(self, state_sequences: List[List[str]]) -> List[Dict[str, Any]]:
        """Find frequently occurring state sequences"""
        sequence_counts = defaultdict(int)
        
        for sequence in state_sequences:
            if len(sequence) >= 2:  # Only consider sequences with at least 2 states
                seq_key = ' -> '.join(sequence)
                sequence_counts[seq_key] += 1
        
        # Return top 10 most frequent sequences
        top_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [
            {
                'sequence': seq,
                'frequency': count,
                'probability': count / len(state_sequences)
            }
            for seq, count in top_sequences
        ]
    
    def _calculate_model_confidence(self, state_sequences: List[List[str]]) -> float:
        """Calculate confidence in the transition model"""
        if not state_sequences:
            return 0.0
        
        # Confidence based on data quantity and variety
        total_transitions = sum(len(seq) - 1 for seq in state_sequences)
        unique_states = len(set(state for seq in state_sequences for state in seq))
        
        quantity_score = min(total_transitions / 100, 1.0)  # Normalize to 0-1
        variety_score = min(unique_states / 5, 1.0)  # Normalize to 0-1
        
        return (quantity_score * 0.6 + variety_score * 0.4)
    
    def predict_next_state(self, current_state: str, current_duration: float) -> Dict[str, Any]:
        """Predict the next likely state transition"""
        if not self.transition_matrix or current_state not in self.transition_matrix:
            return {'next_state': 'unknown', 'confidence': 0.0}
        
        try:
            # Get possible transitions
            possible_transitions = self.transition_matrix[current_state]
            
            # Adjust probabilities based on current state duration
            adjusted_probabilities = self._adjust_probabilities_by_duration(
                current_state, current_duration, possible_transitions
            )
            
            # Find most likely next state
            next_state = max(adjusted_probabilities.items(), key=lambda x: x[1])[0]
            confidence = adjusted_probabilities[next_state]
            
            # Check if transition is overdue
            is_overdue = self._is_transition_overdue(current_state, current_duration)
            
            return {
                'next_state': next_state,
                'confidence': confidence,
                'is_overdue': is_overdue,
                'all_probabilities': adjusted_probabilities,
                'expected_duration': self.state_durations.get(current_state, {}).get('mean', 0)
            }
            
        except Exception as e:
            logging.error(f"❌ State prediction failed: {e}")
            return {'next_state': 'unknown', 'confidence': 0.0}
    
    def _adjust_probabilities_by_duration(self, current_state: str, current_duration: float, 
                                        probabilities: Dict[str, float]) -> Dict[str, float]:
        """Adjust transition probabilities based on how long we've been in current state"""
        if current_state not in self.state_durations:
            return probabilities
        
        duration_stats = self.state_durations[current_state]
        mean_duration = duration_stats.get('mean', 0)
        std_duration = duration_stats.get('std', 1)
        
        if mean_duration == 0:
            return probabilities
        
        # Calculate how "overdue" we are for a transition
        duration_ratio = current_duration / mean_duration
        
        # If we're past the typical duration, increase transition probabilities
        if duration_ratio > 1.0:
            adjustment_factor = min(duration_ratio, 2.0)  # Cap at 2x
            adjusted_probs = {
                state: prob * adjustment_factor
                for state, prob in probabilities.items()
                if state != current_state
            }
            
            # Normalize probabilities
            total = sum(adjusted_probs.values())
            if total > 0:
                return {state: prob / total for state, prob in adjusted_probs.items()}
        
        return probabilities
    
    def _is_transition_overdue(self, current_state: str, current_duration: float) -> bool:
        """Check if we're overdue for a state transition"""
        if current_state not in self.state_durations:
            return False
        
        mean_duration = self.state_durations[current_state].get('mean', 0)
        std_duration = self.state_durations[current_state].get('std', 0)
        
        if mean_duration == 0:
            return False
        
        # Consider overdue if current duration > mean + 1 standard deviation
        return current_duration > (mean_duration + std_duration)
    
    def record_transition(self, from_state: str, to_state: str, duration: float):
        """Record a state transition for model improvement"""
        self.transition_history.append({
            'timestamp': datetime.now(),
            'from_state': from_state,
            'to_state': to_state,
            'duration': duration
        })
        
        # Update transition matrix with new data
        if len(self.transition_history) % 50 == 0:  # Update every 50 transitions
            self._update_model_from_history()
    
    def _update_model_from_history(self):
        """Update transition model with recent history"""
        try:
            recent_history = list(self.transition_history)
            if len(recent_history) >= 10:
                updated_regime_history = self._convert_to_regime_history(recent_history)
                self.build_state_transition_model(updated_regime_history)
                logging.info("🔄 Transition model updated with recent history")
        except Exception as e:
            logging.error(f"❌ Model update failed: {e}")
    
    def _convert_to_regime_history(self, transition_history: List[Dict]) -> List[Dict[str, Any]]:
        """Convert transition history to regime history format"""
        regime_history = []
        for transition in transition_history:
            regime_history.append({
                'timestamp': transition['timestamp'],
                'current_state': transition['from_state'],
                'duration_minutes': transition['duration']
            })
        return regime_history
    
    def get_transition_insights(self) -> Dict[str, Any]:
        """Get insights about state transitions"""
        if not self.transition_matrix:
            return {'status': 'model_not_built'}
        
        # Find most stable and most volatile states
        state_stability = {}
        for state, transitions in self.transition_matrix.items():
            self_transition_prob = transitions.get(state, 0)
            state_stability[state] = self_transition_prob
        
        most_stable = max(state_stability.items(), key=lambda x: x[1]) if state_stability else ('unknown', 0)
        most_volatile = min(state_stability.items(), key=lambda x: x[1]) if state_stability else ('unknown', 0)
        
        return {
            'most_stable_state': most_stable[0],
            'stability_score': most_stable[1],
            'most_volatile_state': most_volatile[0],
            'volatility_score': most_volatile[1],
            'total_transitions_recorded': len(self.transition_history),
            'model_confidence': self.analysis_results.get('model_confidence', 0)
        }

# Global transition predictor instance
TRANSITION_PREDICTOR = TransitionPredictor()
