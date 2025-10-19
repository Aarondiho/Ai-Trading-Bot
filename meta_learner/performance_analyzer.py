"""
Performance Analyzer - Comprehensive Performance Tracking
Analyzes bot performance and identifies improvement areas
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from scipy import stats

from core.deployment_orchestrator import ORCHESTRATOR

class PerformanceAnalyzer:
    """Comprehensive performance analysis and tracking"""
    
    def __init__(self):
        self.performance_history = []
        self.regime_performance = {}
        self.strategy_performance = {}
        self.performance_metrics = {}
        
        # Analysis windows
        self.analysis_windows = {
            'short_term': 7,    # days
            'medium_term': 30,  # days
            'long_term': 90     # days
        }
    
    def record_trade_performance(self, trade_data: Dict[str, Any]):
        """Record trade performance for analysis"""
        if not ORCHESTRATOR.should_run_component('phase_4_continuous_learning', 'meta_learning'):
            return
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            **trade_data
        })
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Update regime-specific performance
        self._update_regime_performance(trade_data)
        
        # Update strategy-specific performance
        self._update_strategy_performance(trade_data)
        
        logging.info(f"📈 Performance recorded: {trade_data.get('symbol')} "
                   f"P&L: ${trade_data.get('pnl', 0):.2f}")
    
    def _update_performance_metrics(self):
        """Update overall performance metrics"""
        if not self.performance_history:
            return
        
        trades_df = pd.DataFrame(self.performance_history)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        # Win rate and profit factor
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(trades_df)
        max_drawdown = self._calculate_max_drawdown(trades_df)
        
        # Consistency metrics
        consistency_score = self._calculate_consistency(trades_df)
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'average_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
            'average_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
            'largest_win': winning_trades['pnl'].max() if not winning_trades.empty else 0,
            'largest_loss': losing_trades['pnl'].min() if not losing_trades.empty else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'consistency_score': consistency_score,
            'last_updated': datetime.now()
        }
    
    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from trade P&L"""
        if len(trades_df) < 2:
            return 0.0
        
        returns = trades_df['pnl'].pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if trades_df.empty:
            return 0.0
        
        cumulative = trades_df['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max.replace(0, 1)
        
        return abs(drawdown.min()) if not drawdown.empty else 0.0
    
    def _calculate_consistency(self, trades_df: pd.DataFrame) -> float:
        """Calculate trading consistency score"""
        if len(trades_df) < 5:
            return 0.0
        
        # Use coefficient of variation of returns (lower is more consistent)
        returns = trades_df['pnl']
        if returns.std() == 0:
            return 1.0
        
        cv = returns.std() / abs(returns.mean())
        consistency = 1 / (1 + cv)  # Convert to 0-1 scale
        
        return min(consistency, 1.0)
    
    def _update_regime_performance(self, trade_data: Dict[str, Any]):
        """Update performance metrics by regime"""
        regime = trade_data.get('regime', 'unknown')
        pnl = trade_data.get('pnl', 0)
        
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                'trades': [],
                'total_pnl': 0,
                'winning_trades': 0,
                'total_trades': 0
            }
        
        regime_data = self.regime_performance[regime]
        regime_data['trades'].append(trade_data)
        regime_data['total_pnl'] += pnl
        regime_data['total_trades'] += 1
        
        if pnl > 0:
            regime_data['winning_trades'] += 1
        
        # Calculate regime-specific metrics
        regime_data['win_rate'] = regime_data['winning_trades'] / regime_data['total_trades']
        regime_data['avg_pnl'] = regime_data['total_pnl'] / regime_data['total_trades']
    
    def _update_strategy_performance(self, trade_data: Dict[str, Any]):
        """Update performance metrics by strategy"""
        strategy = trade_data.get('strategy', 'unknown')
        pnl = trade_data.get('pnl', 0)
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'trades': [],
                'total_pnl': 0,
                'winning_trades': 0,
                'total_trades': 0
            }
        
        strategy_data = self.strategy_performance[strategy]
        strategy_data['trades'].append(trade_data)
        strategy_data['total_pnl'] += pnl
        strategy_data['total_trades'] += 1
        
        if pnl > 0:
            strategy_data['winning_trades'] += 1
        
        # Calculate strategy-specific metrics
        strategy_data['win_rate'] = strategy_data['winning_trades'] / strategy_data['total_trades']
        strategy_data['avg_pnl'] = strategy_data['total_pnl'] / strategy_data['total_trades']
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self.performance_history) < 10:
            return {'status': 'insufficient_data'}
        
        try:
            trades_df = pd.DataFrame(self.performance_history)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')
            
            # Daily performance
            daily_performance = self._analyze_daily_trends(trades_df)
            
            # Weekly performance
            weekly_performance = self._analyze_weekly_trends(trades_df)
            
            # Regime performance comparison
            regime_comparison = self._compare_regime_performance()
            
            # Strategy performance comparison
            strategy_comparison = self._compare_strategy_performance()
            
            # Performance degradation detection
            degradation_analysis = self._detect_performance_degradation(trades_df)
            
            return {
                'daily_trends': daily_performance,
                'weekly_trends': weekly_performance,
                'regime_comparison': regime_comparison,
                'strategy_comparison': strategy_comparison,
                'degradation_analysis': degradation_analysis,
                'overall_health_score': self._calculate_health_score()
            }
            
        except Exception as e:
            logging.error(f"❌ Performance trend analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_daily_trends(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze daily performance trends"""
        trades_df['date'] = trades_df['timestamp'].dt.date
        daily_stats = trades_df.groupby('date').agg({
            'pnl': ['sum', 'count', 'mean', 'std'],
            'symbol': 'nunique'
        }).round(4)
        
        # Calculate daily metrics
        daily_metrics = {
            'profitable_days': len(daily_stats[daily_stats[('pnl', 'sum')] > 0]),
            'total_days': len(daily_stats),
            'best_day': daily_stats[('pnl', 'sum')].max(),
            'worst_day': daily_stats[('pnl', 'sum')].min(),
            'avg_daily_trades': daily_stats[('pnl', 'count')].mean(),
            'daily_consistency': daily_stats[('pnl', 'sum')].std() / abs(daily_stats[('pnl', 'sum')].mean()) 
                                if daily_stats[('pnl', 'sum')].mean() != 0 else 0
        }
        
        daily_metrics['success_rate'] = daily_metrics['profitable_days'] / daily_metrics['total_days']
        
        return daily_metrics
    
    def _analyze_weekly_trends(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly performance trends"""
        trades_df['week'] = trades_df['timestamp'].dt.isocalendar().week
        trades_df['year'] = trades_df['timestamp'].dt.year
        trades_df['week_key'] = trades_df['year'].astype(str) + '_' + trades_df['week'].astype(str)
        
        weekly_stats = trades_df.groupby('week_key').agg({
            'pnl': ['sum', 'count', 'mean'],
            'symbol': 'nunique'
        }).round(4)
        
        if weekly_stats.empty:
            return {}
        
        return {
            'total_weeks': len(weekly_stats),
            'profitable_weeks': len(weekly_stats[weekly_stats[('pnl', 'sum')] > 0]),
            'best_week': weekly_stats[('pnl', 'sum')].max(),
            'worst_week': weekly_stats[('pnl', 'sum')].min(),
            'weekly_trend': self._calculate_trend(weekly_stats[('pnl', 'sum')].values)
        }
    
    def _compare_regime_performance(self) -> Dict[str, Any]:
        """Compare performance across different regimes"""
        if not self.regime_performance:
            return {}
        
        comparison = {}
        for regime, data in self.regime_performance.items():
            if data['total_trades'] >= 5:  # Minimum trades for meaningful comparison
                comparison[regime] = {
                    'win_rate': data['win_rate'],
                    'avg_pnl': data['avg_pnl'],
                    'total_trades': data['total_trades'],
                    'total_pnl': data['total_pnl'],
                    'efficiency_score': data['win_rate'] * abs(data['avg_pnl'])  # Simple efficiency metric
                }
        
        # Sort by efficiency score
        sorted_comparison = dict(sorted(
            comparison.items(), 
            key=lambda x: x[1]['efficiency_score'], 
            reverse=True
        ))
        
        return sorted_comparison
    
    def _compare_strategy_performance(self) -> Dict[str, Any]:
        """Compare performance across different strategies"""
        if not self.strategy_performance:
            return {}
        
        comparison = {}
        for strategy, data in self.strategy_performance.items():
            if data['total_trades'] >= 5:
                comparison[strategy] = {
                    'win_rate': data['win_rate'],
                    'avg_pnl': data['avg_pnl'],
                    'total_trades': data['total_trades'],
                    'total_pnl': data['total_pnl'],
                    'effectiveness_score': data['win_rate'] * data['avg_pnl']
                }
        
        # Sort by effectiveness score
        sorted_comparison = dict(sorted(
            comparison.items(), 
            key=lambda x: x[1]['effectiveness_score'], 
            reverse=True
        ))
        
        return sorted_comparison
    
    def _detect_performance_degradation(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect performance degradation over time"""
        if len(trades_df) < 20:
            return {'detected': False, 'confidence': 0.0}
        
        # Split data into halves for comparison
        split_point = len(trades_df) // 2
        first_half = trades_df.iloc[:split_point]
        second_half = trades_df.iloc[split_point:]
        
        # Compare key metrics
        first_win_rate = len(first_half[first_half['pnl'] > 0]) / len(first_half)
        second_win_rate = len(second_half[second_half['pnl'] > 0]) / len(second_half)
        
        first_avg_pnl = first_half['pnl'].mean()
        second_avg_pnl = second_half['pnl'].mean()
        
        # Calculate degradation scores
        win_rate_degradation = max(0, first_win_rate - second_win_rate)
        pnl_degradation = max(0, first_avg_pnl - second_avg_pnl)
        
        overall_degradation = (win_rate_degradation + pnl_degradation) / 2
        
        degradation_detected = overall_degradation > 0.1  # 10% threshold
        
        return {
            'detected': degradation_detected,
            'confidence': min(overall_degradation / 0.2, 1.0),  # Normalize to 0-1
            'win_rate_change': second_win_rate - first_win_rate,
            'pnl_change': second_avg_pnl - first_avg_pnl,
            'overall_degradation_score': overall_degradation
        }
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "insufficient_data"
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        if not self.performance_metrics:
            return 0.0
        
        metrics = self.performance_metrics
        
        # Score components
        win_rate_score = min(metrics['win_rate'] * 100, 100)  # Max 100
        profit_factor_score = min(metrics['profit_factor'] * 20, 100)  # Max 100
        consistency_score = metrics['consistency_score'] * 100
        sharpe_score = max(0, min(metrics['sharpe_ratio'] * 20, 100))  # Sharpe > 5 is excellent
        
        # Weighted average
        health_score = (
            win_rate_score * 0.3 +
            profit_factor_score * 0.3 + 
            consistency_score * 0.2 +
            sharpe_score * 0.2
        )
        
        return health_score
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance improvement recommendations"""
        recommendations = []
        
        if not self.performance_metrics:
            return ["Insufficient data for recommendations"]
        
        metrics = self.performance_metrics
        
        # Win rate recommendations
        if metrics['win_rate'] < 0.5:
            recommendations.append("Improve win rate - consider better entry timing or regime filtering")
        
        # Profit factor recommendations
        if metrics['profit_factor'] < 1.5:
            recommendations.append("Improve profit factor - work on risk-reward ratios")
        
        # Consistency recommendations
        if metrics['consistency_score'] < 0.7:
            recommendations.append("Improve consistency - reduce variance in trade outcomes")
        
        # Regime-specific recommendations
        regime_comparison = self._compare_regime_performance()
        if regime_comparison:
            worst_regime = min(regime_comparison.items(), key=lambda x: x[1]['efficiency_score'])
            if worst_regime[1]['efficiency_score'] < 0.1:
                recommendations.append(f"Review strategy for {worst_regime[0]} regime - poor performance")
        
        # Strategy-specific recommendations
        strategy_comparison = self._compare_strategy_performance()
        if strategy_comparison:
            worst_strategy = min(strategy_comparison.items(), key=lambda x: x[1]['effectiveness_score'])
            if worst_strategy[1]['effectiveness_score'] < 0:
                recommendations.append(f"Consider disabling {worst_strategy[0]} strategy - negative effectiveness")
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'summary_metrics': self.performance_metrics,
            'health_score': self._calculate_health_score(),
            'trend_analysis': self.analyze_performance_trends(),
            'regime_performance': self.regime_performance,
            'strategy_performance': self.strategy_performance,
            'recommendations': self.get_performance_recommendations(),
            'data_snapshot': {
                'total_trades_recorded': len(self.performance_history),
                'analysis_period_days': self._get_analysis_period_days(),
                'last_trade_date': self.performance_history[-1]['timestamp'] if self.performance_history else None
            }
        }
    
    def _get_analysis_period_days(self) -> int:
        """Get analysis period in days"""
        if not self.performance_history:
            return 0
        
        dates = [t['timestamp'] for t in self.performance_history]
        return (max(dates) - min(dates)).days + 1

# Global performance analyzer instance
PERFORMANCE_ANALYZER = PerformanceAnalyzer()
