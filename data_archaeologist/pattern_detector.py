"""
Pattern Archaeologist - Pattern Detection Module
Unsupervised learning to discover platform algorithm patterns
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import talib
from typing import Dict, List, Tuple, Any

from config import ML_CONFIG, PLATFORM_PATTERNS
from core.deployment_orchestrator import ORCHESTRATOR

class PatternArchaeologist:
    """Discovers platform algorithm patterns using unsupervised learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.pca_model = None
        self.tsne_model = None
        self.feature_names = []
        self.discovered_patterns = {}
        
        # Pattern discovery statistics
        self.discovery_stats = {
            'total_patterns_found': 0,
            'pattern_stability_score': 0.0,
            'last_analysis': None
        }
    
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical features for pattern detection"""
        if len(df) < 100:
            return df
            
        # Price-based features
        df['returns'] = df['price'].pct_change()
        df['price_change'] = df['price'].diff()
        df['high_low_ratio'] = (df['price'].rolling(50).max() / 
                               df['price'].rolling(50).min())
        
        # Volatility features
        df['volatility_5m'] = df['returns'].rolling(5).std()
        df['volatility_15m'] = df['returns'].rolling(15).std()
        df['volatility_30m'] = df['returns'].rolling(30).std()
        df['volatility_1h'] = df['returns'].rolling(60).std()
        
        # Trend features
        df['sma_10'] = df['price'].rolling(10).mean()
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50).mean()
        df['ema_12'] = df['price'].ewm(span=12).mean()
        df['ema_26'] = df['price'].ewm(span=26).mean()
        df['trend_strength'] = abs(df['sma_10'] - df['sma_50']) / df['volatility_5m'].replace(0, 0.001)
        
        # Momentum features
        df['rsi'] = talib.RSI(df['price'], timeperiod=14)
        df['momentum'] = df['price'] - df['price'].shift(10)
        
        # Mean reversion features
        df['price_vs_sma'] = (df['price'] - df['sma_20']) / df['sma_20']
        df['bollinger_position'] = self._calculate_bollinger_position(df)
        
        # Volume and liquidity features (if available)
        if 'volume' in df.columns:
            df['volume_profile'] = df['volume'].rolling(20).mean()
            df['volume_spike'] = df['volume'] / df['volume_profile'].replace(0, 1)
        else:
            df['volume_profile'] = 1.0
            df['volume_spike'] = 1.0
            
        # Rate of change features
        df['roc_5'] = df['price'].pct_change(5)
        df['roc_15'] = df['price'].pct_change(15)
        df['roc_30'] = df['price'].pct_change(30)
        
        # Price acceleration
        df['acceleration'] = df['returns'].diff()
        
        # Remove NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_bollinger_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Band position"""
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df['price'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2
        )
        return (df['price'] - bb_lower) / (bb_upper - bb_lower)
    
    def discover_platform_regimes(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Discover platform regimes using unsupervised learning"""
        if not ORCHESTRATOR.should_run_component('phase_1_pattern_archaeology', 'pattern_detection'):
            return {}
        
        if len(df) < 100:
            logging.warning(f"⚠️ Insufficient data for pattern discovery: {len(df)} points")
            return {}
        
        try:
            # Calculate features
            feature_df = self.calculate_technical_features(df)
            
            # Select feature columns
            self.feature_names = [col for col in feature_df.columns 
                                if col not in ['timestamp', 'symbol', 'price', 'bid', 'ask']]
            
            features = feature_df[self.feature_names].values
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Apply dimensionality reduction for visualization
            self.pca_model = PCA(n_components=3)
            pca_result = self.pca_model.fit_transform(scaled_features)
            
            # Cluster using multiple methods
            clusters_kmeans = self._cluster_with_kmeans(scaled_features)
            clusters_dbscan = self._cluster_with_dbscan(scaled_features)
            
            # Combine clustering results
            final_clusters = self._combine_clusters(clusters_kmeans, clusters_dbscan)
            
            # Analyze each cluster
            regime_analysis = self._analyze_clusters(feature_df, final_clusters, symbol)
            
            self.discovery_stats['total_patterns_found'] = len(regime_analysis)
            self.discovery_stats['last_analysis'] = pd.Timestamp.now()
            
            logging.info(f"🎯 Discovered {len(regime_analysis)} regimes for {symbol}")
            
            return regime_analysis
            
        except Exception as e:
            logging.error(f"❌ Pattern discovery failed: {e}")
            return {}
    
    def _cluster_with_kmeans(self, features: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """Cluster using KMeans"""
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return self.kmeans_model.fit_predict(features)
    
    def _cluster_with_dbscan(self, features: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN for anomaly detection"""
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        return dbscan.fit_predict(features)
    
    def _combine_clusters(self, clusters_kmeans: np.ndarray, clusters_dbscan: np.ndarray) -> np.ndarray:
        """Combine results from multiple clustering algorithms"""
        # Simple combination - you can enhance this
        final_clusters = clusters_kmeans.copy()
        
        # Mark DBSCAN outliers (-1) as a separate cluster
        outlier_mask = clusters_dbscan == -1
        if outlier_mask.any():
            final_clusters[outlier_mask] = np.max(final_clusters) + 1
            
        return final_clusters
    
    def _analyze_clusters(self, df: pd.DataFrame, clusters: np.ndarray, symbol: str) -> Dict[str, Any]:
        """Analyze each cluster to identify platform regimes"""
        regime_analysis = {}
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask]
            
            if len(cluster_data) < 10:  # Skip small clusters
                continue
                
            # Calculate cluster characteristics
            regime_profile = self._create_regime_profile(cluster_data, cluster_id, symbol)
            
            # Match with known platform patterns
            matched_pattern = self._match_with_known_patterns(regime_profile)
            regime_profile['matched_pattern'] = matched_pattern
            
            regime_analysis[f"regime_{cluster_id}"] = regime_profile
            
        return regime_analysis
    
    def _create_regime_profile(self, cluster_data: pd.DataFrame, cluster_id: int, symbol: str) -> Dict[str, Any]:
        """Create detailed profile for a regime"""
        returns = cluster_data['returns'].dropna()
        volatility = returns.std()
        trend_strength = cluster_data['trend_strength'].mean()
        mean_reversion = abs(cluster_data['price_vs_sma']).mean()
        
        profile = {
            'cluster_id': cluster_id,
            'symbol': symbol,
            'sample_size': len(cluster_data),
            'volatility': volatility,
            'trend_strength': trend_strength,
            'mean_reversion_tendency': mean_reversion,
            'avg_duration_minutes': self._estimate_regime_duration(cluster_data),
            'typical_time_of_day': self._analyze_temporal_patterns(cluster_data),
            'price_action_characteristics': {
                'avg_returns': returns.mean(),
                'returns_skewness': returns.skew(),
                'returns_kurtosis': returns.kurtosis(),
                'max_positive_move': returns.max(),
                'max_negative_move': returns.min()
            },
            'stability_score': self._calculate_regime_stability(cluster_data)
        }
        
        return profile
    
    def _estimate_regime_duration(self, cluster_data: pd.DataFrame) -> float:
        """Estimate typical duration of this regime"""
        if 'timestamp' not in cluster_data.columns or len(cluster_data) < 2:
            return 0.0
            
        time_diffs = cluster_data['timestamp'].diff().dt.total_seconds().dropna()
        if len(time_diffs) > 0:
            return time_diffs.median() / 60  # Convert to minutes
        return 0.0
    
    def _analyze_temporal_patterns(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze when this regime typically occurs"""
        if 'timestamp' not in cluster_data.columns:
            return {}
            
        cluster_data['hour'] = cluster_data['timestamp'].dt.hour
        hour_distribution = cluster_data['hour'].value_counts(normalize=True)
        
        return {
            'peak_hours': hour_distribution.head(3).to_dict(),
            'hourly_distribution': hour_distribution.to_dict()
        }
    
    def _calculate_regime_stability(self, cluster_data: pd.DataFrame) -> float:
        """Calculate how stable this regime pattern is"""
        if len(cluster_data) < 20:
            return 0.0
            
        # Calculate consistency of key metrics
        volatility_consistency = 1 - (cluster_data['volatility_15m'].std() / cluster_data['volatility_15m'].mean())
        trend_consistency = 1 - (cluster_data['trend_strength'].std() / cluster_data['trend_strength'].mean())
        
        return (volatility_consistency + trend_consistency) / 2
    
    def _match_with_known_patterns(self, regime_profile: Dict[str, Any]) -> str:
        """Match discovered regime with known platform patterns"""
        best_match = 'unknown'
        best_score = 0
        
        for pattern_name, pattern_info in PLATFORM_PATTERNS.ALGORITHM_STATES.items():
            match_score = self._calculate_pattern_match_score(regime_profile, pattern_name)
            if match_score > best_score:
                best_score = match_score
                best_match = pattern_name
        
        return best_match if best_score > 0.6 else 'unknown'
    
    def _calculate_pattern_match_score(self, regime_profile: Dict[str, Any], pattern_name: str) -> float:
        """Calculate how well regime matches a known pattern"""
        pattern_scores = {
            'volatility_compression': self._score_volatility_compression(regime_profile),
            'trend_momentum': self._score_trend_momentum(regime_profile),
            'random_walk': self._score_random_walk(regime_profile),
            'scheduled_jump': self._score_scheduled_jump(regime_profile),
            'volatility_expansion': self._score_volatility_expansion(regime_profile)
        }
        
        return pattern_scores.get(pattern_name, 0.0)
    
    def _score_volatility_compression(self, regime: Dict[str, Any]) -> float:
        """Score for volatility compression pattern"""
        if regime['volatility'] < 0.001 and regime['trend_strength'] < 0.5:
            return 0.8
        return 0.0
    
    def _score_trend_momentum(self, regime: Dict[str, Any]) -> float:
        """Score for trend momentum pattern"""
        if regime['trend_strength'] > 1.0 and regime['volatility'] > 0.002:
            return 0.8
        return 0.0
    
    def _score_random_walk(self, regime: Dict[str, Any]) -> float:
        """Score for random walk pattern"""
        if (regime['price_action_characteristics']['returns_skewness'] < 0.2 and
            regime['price_action_characteristics']['returns_kurtosis'] < 1.0):
            return 0.7
        return 0.0
    
    def _score_scheduled_jump(self, regime: Dict[str, Any]) -> float:
        """Score for scheduled jump pattern"""
        # Look for regimes with very short duration and high impact
        if (regime['avg_duration_minutes'] < 2 and 
            abs(regime['price_action_characteristics']['avg_returns']) > 0.005):
            return 0.8
        return 0.0
    
    def _score_volatility_expansion(self, regime: Dict[str, Any]) -> float:
        """Score for volatility expansion pattern"""
        if regime['volatility'] > 0.005 and regime['trend_strength'] < 0.8:
            return 0.8
        return 0.0

# Global pattern detector instance
PATTERN_ARCHAEOLOGIST = PatternArchaeologist()
