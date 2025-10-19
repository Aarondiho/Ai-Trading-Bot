"""
Strategy Arsenal - Regime-Specific Strategies
Specialized trading strategies for each platform algorithm state
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import PLATFORM_PATTERNS, RISK_CONFIG
from core.deployment_orchestrator import ORCHESTRATOR

@dataclass
class TradeSignal:
    """Standardized trade signal"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    position_size: float
    stop_loss: float
    take_profit: float
    duration: int  # in minutes
    rationale: str
    strategy: str

class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.performance_history = []
        self.is_active = True
    
    @abstractmethod
    def generate_signal(self, market_data: pd.DataFrame, regime_info: Dict[str, Any]) -> TradeSignal:
        """Generate trading signal based on market data and regime"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, confidence: float, account_balance: float) -> float:
        """Calculate position size based on confidence and account balance"""
        pass
    
    def record_performance(self, trade_result: Dict[str, Any]):
        """Record trade performance for strategy improvement"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            **trade_result
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        if not self.performance_history:
            return {'status': 'no_trades_yet'}
        
        wins = [t for t in self.performance_history if t.get('pnl', 0) > 0]
        losses = [t for t in self.performance_history if t.get('pnl', 0) <= 0]
        
        total_trades = len(self.performance_history)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t.get('pnl', 0) for t in wins]) if wins else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in losses]) if losses else 0
        profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if losses else float('inf')
        
        return {
            'strategy': self.strategy_name,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': sum(t.get('pnl', 0) for t in self.performance_history),
            'recent_performance': self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        }

class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy for volatility compression regimes"""
    
    def __init__(self):
        super().__init__("mean_reversion")
        self.oversold_threshold = 0.3
        self.overbought_threshold = 0.7
        self.min_confidence = 0.65
    
    def generate_signal(self, market_data: pd.DataFrame, regime_info: Dict[str, Any]) -> TradeSignal:
        if not ORCHESTRATOR.should_run_component('phase_3_adaptive_execution', 'strategy_arsenal'):
            return self._create_hold_signal()
        
        if len(market_data) < 20:
            return self._create_hold_signal()
        
        try:
            # Calculate Bollinger Bands position
            current_price = market_data['price'].iloc[-1]
            bb_position = self._calculate_bb_position(market_data)
            
            # Calculate RSI
            rsi = self._calculate_rsi(market_data)
            
            # Mean reversion logic
            if bb_position < self.oversold_threshold and rsi < 30:
                action = 'buy'
                confidence = min((self.oversold_threshold - bb_position) / self.oversold_threshold, 0.9)
                rationale = f"Oversold conditions: BB={bb_position:.3f}, RSI={rsi:.1f}"
                
            elif bb_position > self.overbought_threshold and rsi > 70:
                action = 'sell'
                confidence = min((bb_position - self.overbought_threshold) / (1 - self.overbought_threshold), 0.9)
                rationale = f"Overbought conditions: BB={bb_position:.3f}, RSI={rsi:.1f}"
                
            else:
                return self._create_hold_signal()
            
            # Check minimum confidence
            if confidence < self.min_confidence:
                return self._create_hold_signal()
            
            # Calculate position parameters
            position_size = self.calculate_position_size(confidence, RISK_CONFIG.INITIAL_BALANCE)
            stop_loss, take_profit = self._calculate_stop_take(current_price, action, market_data)
            
            return TradeSignal(
                symbol=regime_info.get('symbol', 'unknown'),
                action=action,
                confidence=confidence,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                duration=15,  # 15 minutes for mean reversion
                rationale=rationale,
                strategy=self.strategy_name
            )
            
        except Exception as e:
            logging.error(f"❌ Mean reversion signal generation failed: {e}")
            return self._create_hold_signal()
    
    def _calculate_bb_position(self, market_data: pd.DataFrame) -> float:
        """Calculate Bollinger Band position (0-1)"""
        prices = market_data['price'].tail(20)
        if len(prices) < 20:
            return 0.5
        
        sma = prices.mean()
        std = prices.std()
        
        if std == 0:
            return 0.5
        
        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std
        
        current_price = prices.iloc[-1]
        if bb_upper != bb_lower:
            return (current_price - bb_lower) / (bb_upper - bb_lower)
        else:
            return 0.5
    
    def _calculate_rsi(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        prices = market_data['price'].tail(period + 1)
        if len(prices) < period + 1:
            return 50
        
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean().iloc[-1]
        avg_loss = losses.rolling(window=period).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stop_take(self, current_price: float, action: str, market_data: pd.DataFrame) -> tuple:
        """Calculate stop loss and take profit levels"""
        atr = self._calculate_atr(market_data)
        
        if action == 'buy':
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 4 * atr  # 2:1 reward ratio
        else:  # sell
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 4 * atr
        
        return stop_loss, take_profit
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(market_data) < period + 1:
            return 0.001  # Default small value
        
        high = market_data['price'].tail(period + 1).rolling(window=2).max()
        low = market_data['price'].tail(period + 1).rolling(window=2).min()
        close_prev = market_data['price'].tail(period + 1).shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else 0.001
    
    def calculate_position_size(self, confidence: float, account_balance: float) -> float:
        """Calculate position size for mean reversion"""
        base_size = account_balance * RISK_CONFIG.MAX_RISK_PER_TRADE
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        
        return base_size * confidence_multiplier
    
    def _create_hold_signal(self) -> TradeSignal:
        """Create a hold signal"""
        return TradeSignal(
            symbol='unknown',
            action='hold',
            confidence=0.0,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            duration=0,
            rationale="No clear mean reversion opportunity",
            strategy=self.strategy_name
        )

class TrendFollowingStrategy(TradingStrategy):
    """Trend following strategy for trend momentum regimes"""
    
    def __init__(self):
        super().__init__("trend_following")
        self.trend_confirmation_period = 5
        self.min_trend_strength = 0.3
    
    def generate_signal(self, market_data: pd.DataFrame, regime_info: Dict[str, Any]) -> TradeSignal:
        if not ORCHESTRATOR.should_run_component('phase_3_adaptive_execution', 'strategy_arsenal'):
            return self._create_hold_signal()
        
        if len(market_data) < 30:
            return self._create_hold_signal()
        
        try:
            # Calculate trend direction and strength
            trend_direction, trend_strength = self._calculate_trend(market_data)
            
            # Check if trend is strong enough
            if trend_strength < self.min_trend_strength:
                return self._create_hold_signal()
            
            # Generate signal based on trend
            if trend_direction > 0:
                action = 'buy'
                confidence = min(trend_strength, 0.8)
                rationale = f"Uptrend detected: strength={trend_strength:.3f}"
            else:
                action = 'sell'
                confidence = min(trend_strength, 0.8)
                rationale = f"Downtrend detected: strength={trend_strength:.3f}"
            
            # Calculate position parameters
            position_size = self.calculate_position_size(confidence, RISK_CONFIG.INITIAL_BALANCE)
            stop_loss, take_profit = self._calculate_stop_take(market_data, action, trend_strength)
            
            return TradeSignal(
                symbol=regime_info.get('symbol', 'unknown'),
                action=action,
                confidence=confidence,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                duration=30,  # 30 minutes for trend following
                rationale=rationale,
                strategy=self.strategy_name
            )
            
        except Exception as e:
            logging.error(f"❌ Trend following signal generation failed: {e}")
            return self._create_hold_signal()
    
    def _calculate_trend(self, market_data: pd.DataFrame) -> tuple:
        """Calculate trend direction and strength"""
        prices = market_data['price'].tail(30)
        
        # Use multiple timeframes for trend confirmation
        short_ma = prices.tail(5).mean()
        medium_ma = prices.tail(15).mean()
        long_ma = prices.mean()
        
        # Trend direction (positive = uptrend, negative = downtrend)
        trend_direction = np.sign(short_ma - long_ma)
        
        # Trend strength (0-1)
        price_volatility = prices.std()
        if price_volatility == 0:
            trend_strength = 0
        else:
            trend_strength = min(abs(short_ma - long_ma) / price_volatility, 1.0)
        
        return trend_direction, trend_strength
    
    def _calculate_stop_take(self, market_data: pd.DataFrame, action: str, trend_strength: float) -> tuple:
        """Calculate dynamic stop loss and take profit for trends"""
        current_price = market_data['price'].iloc[-1]
        atr = self._calculate_atr(market_data)
        
        # Wider stops for stronger trends
        trend_multiplier = 1.0 + (trend_strength * 2.0)  # 1x to 3x
        
        if action == 'buy':
            stop_loss = current_price - (atr * trend_multiplier * 1.5)
            take_profit = current_price + (atr * trend_multiplier * 3.0)  # 2:1 ratio
        else:  # sell
            stop_loss = current_price + (atr * trend_multiplier * 1.5)
            take_profit = current_price - (atr * trend_multiplier * 3.0)
        
        return stop_loss, take_profit
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        return MeanReversionStrategy()._calculate_atr(market_data, period)
    
    def calculate_position_size(self, confidence: float, account_balance: float) -> float:
        """Calculate position size for trend following"""
        base_size = account_balance * RISK_CONFIG.MAX_RISK_PER_TRADE
        
        # Trend following can use larger positions with good confidence
        confidence_multiplier = 0.7 + (confidence * 0.3)  # 0.7 to 1.0
        
        return base_size * confidence_multiplier
    
    def _create_hold_signal(self) -> TradeSignal:
        """Create a hold signal"""
        return TradeSignal(
            symbol='unknown',
            action='hold',
            confidence=0.0,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            duration=0,
            rationale="No clear trend detected",
            strategy=self.strategy_name
        )

class BreakoutStrategy(TradingStrategy):
    """Breakout strategy for volatility expansion regimes"""
    
    def __init__(self):
        super().__init__("breakout")
        self.consolidation_period = 20
        self.breakout_threshold = 0.002  # 0.2% move
    
    def generate_signal(self, market_data: pd.DataFrame, regime_info: Dict[str, Any]) -> TradeSignal:
        if not ORCHESTRATOR.should_run_component('phase_3_adaptive_execution', 'strategy_arsenal'):
            return self._create_hold_signal()
        
        if len(market_data) < self.consolidation_period + 5:
            return self._create_hold_signal()
        
        try:
            # Check for consolidation pattern
            if not self._is_consolidating(market_data):
                return self._create_hold_signal()
            
            # Check for breakout
            breakout_direction, breakout_strength = self._detect_breakout(market_data)
            
            if breakout_direction == 0:
                return self._create_hold_signal()
            
            action = 'buy' if breakout_direction > 0 else 'sell'
            confidence = min(breakout_strength / self.breakout_threshold, 0.8)
            rationale = f"Breakout detected: {action.upper()}, strength={breakout_strength:.4f}"
            
            # Calculate position parameters
            position_size = self.calculate_position_size(confidence, RISK_CONFIG.INITIAL_BALANCE)
            stop_loss, take_profit = self._calculate_stop_take(market_data, action)
            
            return TradeSignal(
                symbol=regime_info.get('symbol', 'unknown'),
                action=action,
                confidence=confidence,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                duration=20,  # 20 minutes for breakout
                rationale=rationale,
                strategy=self.strategy_name
            )
            
        except Exception as e:
            logging.error(f"❌ Breakout signal generation failed: {e}")
            return self._create_hold_signal()
    
    def _is_consolidating(self, market_data: pd.DataFrame) -> bool:
        """Check if market is in consolidation"""
        recent_prices = market_data['price'].tail(self.consolidation_period)
        
        # Calculate price range and volatility
        price_range = recent_prices.max() - recent_prices.min()
        avg_price = recent_prices.mean()
        range_ratio = price_range / avg_price if avg_price > 0 else 0
        
        # Consolidation if range is small relative to recent volatility
        return range_ratio < 0.005  # 0.5% range
    
    def _detect_breakout(self, market_data: pd.DataFrame) -> tuple:
        """Detect breakout from consolidation"""
        consolidation_data = market_data['price'].tail(self.consolidation_period)
        recent_data = market_data['price'].tail(5)
        
        consolidation_high = consolidation_data.max()
        consolidation_low = consolidation_data.min()
        current_price = market_data['price'].iloc[-1]
        
        # Check for breakout above consolidation
        if current_price > consolidation_high:
            breakout_strength = (current_price - consolidation_high) / consolidation_high
            if breakout_strength >= self.breakout_threshold:
                return 1, breakout_strength  # Bullish breakout
        
        # Check for breakout below consolidation
        elif current_price < consolidation_low:
            breakout_strength = (consolidation_low - current_price) / consolidation_low
            if breakout_strength >= self.breakout_threshold:
                return -1, breakout_strength  # Bearish breakout
        
        return 0, 0  # No breakout
    
    def _calculate_stop_take(self, market_data: pd.DataFrame, action: str) -> tuple:
        """Calculate stop loss and take profit for breakouts"""
        current_price = market_data['price'].iloc[-1]
        consolidation_data = market_data['price'].tail(self.consolidation_period)
        
        consolidation_high = consolidation_data.max()
        consolidation_low = consolidation_data.min()
        consolidation_range = consolidation_high - consolidation_low
        
        if action == 'buy':
            stop_loss = consolidation_low - (consolidation_range * 0.1)
            take_profit = current_price + (consolidation_range * 1.5)
        else:  # sell
            stop_loss = consolidation_high + (consolidation_range * 0.1)
            take_profit = current_price - (consolidation_range * 1.5)
        
        return stop_loss, take_profit
    
    def calculate_position_size(self, confidence: float, account_balance: float) -> float:
        """Calculate position size for breakout trading"""
        base_size = account_balance * RISK_CONFIG.MAX_RISK_PER_TRADE
        
        # Breakout trading uses moderate position sizes
        confidence_multiplier = 0.6 + (confidence * 0.4)  # 0.6 to 1.0
        
        return base_size * confidence_multiplier
    
    def _create_hold_signal(self) -> TradeSignal:
        """Create a hold signal"""
        return TradeSignal(
            symbol='unknown',
            action='hold',
            confidence=0.0,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            duration=0,
            rationale="No breakout detected",
            strategy=self.strategy_name
        )

class StrategyArsenal:
    """Manages the collection of trading strategies"""
    
    def __init__(self):
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(),
            'trend_following': TrendFollowingStrategy(),
            'breakout': BreakoutStrategy()
        }
        
        self.regime_strategy_mapping = {
            'volatility_compression': 'mean_reversion',
            'trend_momentum': 'trend_following',
            'volatility_expansion': 'breakout',
            'random_walk': None,  # No trading
            'scheduled_jump': None  # No trading
        }
    
    def get_strategy_for_regime(self, regime: str) -> Optional[TradingStrategy]:
        """Get appropriate strategy for current regime"""
        strategy_name = self.regime_strategy_mapping.get(regime)
        return self.strategies.get(strategy_name) if strategy_name else None
    
    def generate_trade_signal(self, market_data: pd.DataFrame, 
                            regime_info: Dict[str, Any]) -> TradeSignal:
        """Generate trade signal using appropriate strategy"""
        regime = regime_info.get('current_state', 'unknown')
        strategy = self.get_strategy_for_regime(regime)
        
        if strategy and strategy.is_active:
            return strategy.generate_signal(market_data, regime_info)
        else:
            # Return hold signal for unknown or no-trading regimes
            return TradeSignal(
                symbol=regime_info.get('symbol', 'unknown'),
                action='hold',
                confidence=0.0,
                position_size=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                duration=0,
                rationale=f"No strategy for regime: {regime}",
                strategy="no_trading"
            )
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all strategies"""
        performance = {}
        for name, strategy in self.strategies.items():
            performance[name] = strategy.get_performance_metrics()
        return performance
    
    def disable_strategy(self, strategy_name: str):
        """Disable a specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_active = False
            logging.info(f"⏸️ Disabled strategy: {strategy_name}")
    
    def enable_strategy(self, strategy_name: str):
        """Enable a specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_active = True
            logging.info(f"✅ Enabled strategy: {strategy_name}")

# Global strategy arsenal instance
STRATEGY_ARSENAL = StrategyArsenal()
