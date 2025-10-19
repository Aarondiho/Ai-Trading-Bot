"""
Temporal Analyst - Time Pattern Analysis
Analyzes timing patterns in platform algorithms
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Dict, List, Any
from collections import defaultdict

from core.deployment_orchestrator import ORCHESTRATOR

class TemporalAnalyst:
    """Analyzes temporal patterns in platform algorithm behavior"""
    
    def __init__(self):
        self.time_patterns = {}
        self.seasonal_analysis = {}
        self.maintenance_periods = []
        
        # Analysis results
        self.analysis_results = {
            'hourly_patterns': {},
            'daily_patterns': {},
            'weekly_patterns': {},
            'duration_patterns': {},
            'last_analysis': None
        }
    
    def analyze_algorithm_timing(self, regime_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in algorithm behavior"""
        if not ORCHESTRATOR.should_run_component('phase_2_transition_prediction', 'temporal_analysis'):
            return {}
        
        if len(regime_history) < 50:
            logging.warning("⚠️ Insufficient data for temporal analysis")
            return {}
        
        try:
            # Convert to DataFrame for easier analysis
            df = self._prepare_temporal_data(regime_history)
            
            # Analyze different temporal dimensions
            self.analysis_results['hourly_patterns'] = self._analyze_hourly_patterns(df)
            self.analysis_results['daily_patterns'] = self._analyze_daily_patterns(df)
            self.analysis_results['weekly_patterns'] = self._analyze_weekly_patterns(df)
            self.analysis_results['duration_patterns'] = self._analyze_duration_patterns(df)
            
            # Detect maintenance periods
            self.maintenance_periods = self._detect_maintenance_periods(df)
            
            self.analysis_results['last_analysis'] = datetime.now()
            
            logging.info("✅ Temporal analysis completed")
            
            return self.analysis_results
            
        except Exception as e:
            logging.error(f"❌ Temporal analysis failed: {e}")
            return {}
    
    def _prepare_temporal_data(self, regime_history: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare regime history for temporal analysis"""
        data = []
        
        for regime in regime_history:
            if 'timestamp' not in regime:
                continue
                
            timestamp = regime['timestamp']
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            data.append({
                'timestamp': timestamp,
                'state': regime.get('current_state', 'unknown'),
                'duration': regime.get('duration_minutes', 0),
                'symbol': regime.get('symbol', 'unknown'),
                'volatility': regime.get('volatility', 0),
                'trend_strength': regime.get('trend_strength', 0)
            })
        
        return pd.DataFrame(data)
    
    def _analyze_hourly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns by hour of day"""
        if df.empty:
            return {}
        
        df['hour'] = df['timestamp'].dt.hour
        hourly_stats = {}
        
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            if len(hour_data) > 0:
                hourly_stats[hour] = {
                    'total_regimes': len(hour_data),
                    'state_distribution': hour_data['state'].value_counts().to_dict(),
                    'avg_duration': hour_data['duration'].mean(),
                    'avg_volatility': hour_data['volatility'].mean(),
                    'common_states': hour_data['state'].mode().tolist() if not hour_data['state'].mode().empty else []
                }
        
        # Find peak activity hours
        total_by_hour = {hour: stats['total_regimes'] for hour, stats in hourly_stats.items()}
        peak_hours = sorted(total_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'hourly_distribution': hourly_stats,
            'peak_activity_hours': [hour for hour, count in peak_hours],
            'quiet_hours': [hour for hour in range(24) if hourly_stats.get(hour, {}).get('total_regimes', 0) < 5]
        }
    
    def _analyze_daily_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns by day of week"""
        if df.empty:
            return {}
        
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        
        daily_stats = {}
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            day_data = df[df['day_of_week'] == day]
            if len(day_data) > 0:
                daily_stats[day] = {
                    'total_regimes': len(day_data),
                    'state_distribution': day_data['state'].value_counts().to_dict(),
                    'avg_volatility': day_data['volatility'].mean(),
                    'unique_states': day_data['state'].nunique()
                }
        
        # Compare weekdays vs weekends
        weekday_data = df[~df['is_weekend']]
        weekend_data = df[df['is_weekend']]
        
        return {
            'daily_distribution': daily_stats,
            'weekday_vs_weekend': {
                'weekday_volatility': weekday_data['volatility'].mean() if not weekday_data.empty else 0,
                'weekend_volatility': weekend_data['volatility'].mean() if not weekend_data.empty else 0,
                'weekday_regime_count': len(weekday_data),
                'weekend_regime_count': len(weekend_data)
            }
        }
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly seasonal patterns"""
        if df.empty:
            return {}
        
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        weekly_stats = {}
        
        for week in df['week_of_year'].unique():
            week_data = df[df['week_of_year'] == week]
            weekly_stats[week] = {
                'regime_count': len(week_data),
                'state_diversity': week_data['state'].nunique(),
                'avg_volatility': week_data['volatility'].mean(),
                'dominant_state': week_data['state'].mode().iloc[0] if not week_data['state'].mode().empty else 'unknown'
            }
        
        return weekly_stats
    
    def _analyze_duration_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duration patterns for each state"""
        if df.empty:
            return {}
        
        duration_stats = {}
        for state in df['state'].unique():
            state_data = df[df['state'] == state]
            durations = state_data['duration'].dropna()
            
            if len(durations) > 0:
                duration_stats[state] = {
                    'count': len(durations),
                    'mean': durations.mean(),
                    'median': durations.median(),
                    'std': durations.std(),
                    'min': durations.min(),
                    'max': durations.max(),
                    'consistency': 1 - (durations.std() / durations.mean()) if durations.mean() > 0 else 0
                }
        
        return duration_stats
    
    def _detect_maintenance_periods(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential platform maintenance periods"""
        if df.empty:
            return []
        
        # Look for periods with unusual gaps in data or specific state patterns
        df = df.sort_values('timestamp')
        df['time_gap'] = df['timestamp'].diff().dt.total_seconds() / 60  # Gap in minutes
        
        # Identify large gaps (potential maintenance)
        large_gaps = df[df['time_gap'] > 120]  # Gaps longer than 2 hours
        
        maintenance_periods = []
        for _, gap in large_gaps.iterrows():
            period_start = gap['timestamp'] - timedelta(minutes=gap['time_gap'])
            period_end = gap['timestamp']
            
            maintenance_periods.append({
                'start': period_start,
                'end': period_end,
                'duration_minutes': gap['time_gap'],
                'gap_type': 'data_interruption'
            })
        
        return maintenance_periods
    
    def predict_optimal_trading_times(self) -> Dict[str, Any]:
        """Predict optimal times for trading based on temporal patterns"""
        hourly_patterns = self.analysis_results.get('hourly_patterns', {})
        if not hourly_patterns.get('hourly_distribution'):
            return {'status': 'insufficient_data'}
        
        hourly_stats = hourly_patterns['hourly_distribution']
        
        # Score each hour for trading suitability
        hour_scores = {}
        for hour, stats in hourly_stats.items():
            score = self._calculate_trading_score(stats)
            hour_scores[hour] = score
        
        # Find best and worst hours
        best_hours = sorted(hour_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        worst_hours = sorted(hour_scores.items(), key=lambda x: x[1])[:3]
        
        return {
            'best_trading_hours': [{'hour': hour, 'score': score} for hour, score in best_hours],
            'worst_trading_hours': [{'hour': hour, 'score': score} for hour, score in worst_hours],
            'recommended_sessions': self._define_trading_sessions(hour_scores),
            'analysis_confidence': self._calculate_temporal_confidence()
        }
    
    def _calculate_trading_score(self, hour_stats: Dict[str, Any]) -> float:
        """Calculate trading suitability score for an hour"""
        score = 0.0
        
        # Prefer hours with moderate volatility (not too high, not too low)
        volatility = hour_stats.get('avg_volatility', 0)
        if 0.001 <= volatility <= 0.005:
            score += 0.3
        elif volatility > 0.005:
            score += 0.1
        else:
            score += 0.0
        
        # Prefer hours with good state diversity
        state_dist = hour_stats.get('state_distribution', {})
        state_diversity = len(state_dist)
        score += min(state_diversity / 5, 0.3)  # Max 0.3 for diversity
        
        # Prefer hours with reasonable duration (not too short, not too long)
        avg_duration = hour_stats.get('avg_duration', 0)
        if 10 <= avg_duration <= 45:
            score += 0.2
        elif avg_duration > 45:
            score += 0.1
        else:
            score += 0.0
        
        # Bonus for high data quality (many regimes)
        total_regimes = hour_stats.get('total_regimes', 0)
        score += min(total_regimes / 50, 0.2)  # Max 0.2 for data quantity
        
        return min(score, 1.0)
    
    def _define_trading_sessions(self, hour_scores: Dict[int, float]) -> List[Dict[str, Any]]:
        """Define optimal trading sessions based on hour scores"""
        sessions = []
        
        # Look for contiguous blocks of good hours
        good_hours = [hour for hour, score in hour_scores.items() if score > 0.6]
        good_hours.sort()
        
        if not good_hours:
            return sessions
        
        current_session = [good_hours[0]]
        for hour in good_hours[1:]:
            if hour == current_session[-1] + 1:
                current_session.append(hour)
            else:
                if len(current_session) >= 2:  # Only keep sessions with 2+ hours
                    sessions.append({
                        'start_hour': current_session[0],
                        'end_hour': current_session[-1],
                        'duration_hours': len(current_session),
                        'avg_score': np.mean([hour_scores[h] for h in current_session])
                    })
                current_session = [hour]
        
        # Add the last session
        if len(current_session) >= 2:
            sessions.append({
                'start_hour': current_session[0],
                'end_hour': current_session[-1],
                'duration_hours': len(current_session),
                'avg_score': np.mean([hour_scores[h] for h in current_session])
            })
        
        return sorted(sessions, key=lambda x: x['avg_score'], reverse=True)
    
    def _calculate_temporal_confidence(self) -> float:
        """Calculate confidence in temporal analysis"""
        hourly_data = self.analysis_results.get('hourly_patterns', {}).get('hourly_distribution', {})
        if not hourly_data:
            return 0.0
        
        # Confidence based on data coverage across hours
        hours_with_data = len(hourly_data)
        coverage_score = hours_with_data / 24
        
        # Confidence based on data quantity
        total_regimes = sum(stats.get('total_regimes', 0) for stats in hourly_data.values())
        quantity_score = min(total_regimes / 500, 1.0)  # Normalize
        
        return (coverage_score * 0.4 + quantity_score * 0.6)
    
    def get_temporal_insights(self) -> Dict[str, Any]:
        """Get key insights from temporal analysis"""
        hourly = self.analysis_results.get('hourly_patterns', {})
        daily = self.analysis_results.get('daily_patterns', {})
        
        return {
            'peak_activity_times': hourly.get('peak_activity_hours', []),
            'best_trading_windows': self.predict_optimal_trading_times().get('recommended_sessions', []),
            'weekly_patterns': {
                'most_active_day': max(daily.get('daily_distribution', {}).items(), 
                                     key=lambda x: x[1].get('total_regimes', 0))[0] if daily.get('daily_distribution') else 'unknown',
                'weekend_effect': daily.get('weekday_vs_weekend', {}).get('weekend_volatility', 0) > 
                                daily.get('weekday_vs_weekend', {}).get('weekday_volatility', 0)
            },
            'maintenance_periods_detected': len(self.maintenance_periods),
            'temporal_confidence': self._calculate_temporal_confidence()
        }

# Global temporal analyst instance
TEMPORAL_ANALYST = TemporalAnalyst()
