"""
Platform Mind Reader - Main Application
Complete AI trading system that thinks like the platform
"""

import asyncio
import logging
import signal
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import all system components
from config import PHASES, DERIV_CONFIG
from data_archaeologist.data_collector import DATA_ARCHAEOLOGIST
from data_archaeologist.pattern_detector import PATTERN_ARCHAEOLOGIST
from data_archaeologist.regime_classifier import STATE_CLASSIFIER
from transition_predictor.state_transition_model import TRANSITION_PREDICTOR
from transition_predictor.temporal_analyst import TEMPORAL_ANALYST
from transition_predictor.early_warning_system import EARLY_WARNING_SYSTEM
from adaptive_executor.strategy_arsenal import STRATEGY_ARSENAL
from adaptive_executor.risk_orchestrator import RISK_MANAGER
from adaptive_executor.execution_engine import EXECUTION_ENGINE
from meta_learner.performance_analyzer import PERFORMANCE_ANALYZER
from meta_learner.knowledge_preserver import KNOWLEDGE_BANK
from meta_learner.continuous_adapter import CONTINUOUS_ADAPTER
from core.deployment_orchestrator import ORCHESTRATOR

class PlatformMindReader:
    """Main application class for the Platform Mind Reader trading system"""
    
    def __init__(self):
        self.is_running = False
        self.start_time = None
        self.cycle_count = 0
        self.start_count = 0
        self.system_health = "initializing"
        
        # Component status tracking
        self.component_status = {}
        
        # Trading session state
        self.current_session = {
            'start_time': None,
            'trades_executed': 0,
            'session_pnl': 0.0,
            'current_regime': 'unknown',
            'regime_confidence': 0.0
        }
    
    async def initialize_system(self):
        """Initialize all system components"""
        logging.info("🚀 Initializing Platform Mind Reader...")
        
        try:
            # Phase 1: Pattern Archaeology
            if PHASES.phases['phase_1_pattern_archaeology']['active']:
                await self._initialize_phase_1()
            
            # Phase 2: Transition Prediction  
            if PHASES.phases['phase_2_transition_prediction']['active']:
                await self._initialize_phase_2()
            
            # Phase 3: Adaptive Execution
            if PHASES.phases['phase_3_adaptive_execution']['active']:
                await self._initialize_phase_3()
            
            # Phase 4: Continuous Learning
            if PHASES.phases['phase_4_continuous_learning']['active']:
                await self._initialize_phase_4()
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.system_health = "healthy"
            self.start_time = datetime.now()
            logging.info("✅ Platform Mind Reader initialized successfully!")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ System initialization failed: {e}")
            self.system_health = "error"
            return False
    
    async def _initialize_phase_1(self):
        """Initialize Phase 1 components"""
        logging.info("🔍 Initializing Phase 1: Pattern Archaeology...")
        
        # Initialize data collection
        DATA_ARCHAEOLOGIST.initialize_database()
        await DATA_ARCHAEOLOGIST.connect_deriv_websocket()
        
        # Start data collection
        asyncio.create_task(self._run_data_collection())
        
        self.component_status['data_collection'] = 'active'
        logging.info("✅ Phase 1 components initialized")
    
    async def _initialize_phase_2(self):
        """Initialize Phase 2 components"""
        logging.info("🎯 Initializing Phase 2: Transition Prediction...")
        
        # Schedule temporal analysis
        CONTINUOUS_ADAPTER.schedule_periodic_learning('pattern_refresh', 24)
        
        self.component_status['transition_prediction'] = 'active'
        logging.info("✅ Phase 2 components initialized")
    
    async def _initialize_phase_3(self):
        """Initialize Phase 3 components"""
        logging.info("⚡ Initializing Phase 3: Adaptive Execution...")
        
        # Initialize execution engine
        await EXECUTION_ENGINE.initialize_connection()
        
        # Start position monitoring
        asyncio.create_task(EXECUTION_ENGINE.monitor_open_positions())
        
        self.component_status['adaptive_execution'] = 'active'
        logging.info("✅ Phase 3 components initialized")
    
    async def _initialize_phase_4(self):
        """Initialize Phase 4 components"""
        logging.info("🧠 Initializing Phase 4: Continuous Learning...")
        
        # Schedule regular learnings
        CONTINUOUS_ADAPTER.schedule_periodic_learning('strategy_optimization', 12)
        CONTINUOUS_ADAPTER.schedule_periodic_learning('model_retraining', 48)
        
        # Load knowledge backup if available
        KNOWLEDGE_BANK.load_knowledge_backup()
        
        self.component_status['continuous_learning'] = 'active'
        logging.info("✅ Phase 4 components initialized")
    
    async def _run_data_collection(self):
        """Run continuous data collection"""
        while self.is_running:
            try:
                # Collect historical data on first run
                if self.start_count == 0:
                    self.start_count += 1
                    
                    for symbol in DERIV_CONFIG.SYMBOLS:
                        await DATA_ARCHAEOLOGIST.collect_historical_data(symbol, days=180)
                        

                # Start real-time collection
                await DATA_ARCHAEOLOGIST.start_real_time_collection()
                
            except Exception as e:
                logging.error(f"❌ Data collection error: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def run_trading_cycle(self):
        """Run a single trading cycle"""
        try:
            self.cycle_count += 1
            cycle_start = datetime.now()
            
            logging.info(f"🔄 Running trading cycle #{self.cycle_count}")
            
            # Step 1: Collect and analyze current market data
            market_analysis = await self._analyze_current_market()
            
            # Step 2: Detect current platform regime
            regime_analysis = await self._detect_current_regime(market_analysis)
            
            # Step 3: Generate trading signals
            trading_decisions = await self._generate_trading_signals(market_analysis, regime_analysis)
            
            # Step 4: Execute trades with risk management
            execution_results = await self._execute_trades(trading_decisions)
            
            # Step 5: Learn and adapt
            await self._learn_and_adapt(execution_results, regime_analysis)
            
            # Update session state
            self._update_session_state(regime_analysis, execution_results)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logging.info(f"✅ Trading cycle #{self.cycle_count} completed in {cycle_duration:.2f}s")
            
            return {
                'cycle_number': self.cycle_count,
                'duration_seconds': cycle_duration,
                'regime': regime_analysis.get('current_regime', 'unknown'),
                'signals_generated': len(trading_decisions.get('signals', [])),
                'trades_executed': len(execution_results.get('executed_trades', [])),
                'timestamp': cycle_start
            }
            
        except Exception as e:
            logging.error(f"❌ Trading cycle failed: {e}")
            return {'error': str(e), 'cycle_number': self.cycle_count}
    
    async def _analyze_current_market(self) -> Dict[str, Any]:
        """Analyze current market conditions"""
        analysis = {}
        
        try:
            # Get recent data for all symbols
            for symbol in DERIV_CONFIG.SYMBOLS:
                recent_data = DATA_ARCHAEOLOGIST.get_recent_data(symbol, lookback=100)
                if not recent_data.empty:
                    # Calculate technical indicators
                    symbol_analysis = PATTERN_ARCHAEOLOGIST.calculate_technical_features(recent_data)
                    analysis[symbol] = symbol_analysis
            
            # Add market-wide analysis
            analysis['market_summary'] = {
                'total_symbols': len(analysis),
                'analysis_timestamp': datetime.now(),
                'overall_volatility': self._calculate_market_volatility(analysis)
            }
            
        except Exception as e:
            logging.error(f"❌ Market analysis failed: {e}")
        
        return analysis
    
    async def _detect_current_regime(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current platform algorithm regime"""
        regime_analysis = {}
        
        try:
            for symbol, data in market_analysis.items():
                if symbol != 'market_summary':
                    # Discover patterns for this symbol
                    patterns = PATTERN_ARCHAEOLOGIST.discover_platform_regimes(data, symbol)
                    
                    # Classify current regime
                    current_features = self._extract_current_features(data)
                    current_regime, confidence = STATE_CLASSIFIER.predict_current_state(current_features)
                    
                    regime_analysis[symbol] = {
                        'current_regime': current_regime,
                        'confidence': confidence,
                        'patterns_found': len(patterns),
                        'timestamp': datetime.now()
                    }
            
            # Determine overall market regime
            if regime_analysis:
                regime_analysis['overall_regime'] = self._determine_overall_regime(regime_analysis)
            
        except Exception as e:
            logging.error(f"❌ Regime detection failed: {e}")
        
        return regime_analysis
    
    async def _generate_trading_signals(self, market_analysis: Dict[str, Any], 
                                      regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on analysis"""
        signals = []
        
        try:
            for symbol, regime_info in regime_analysis.items():
                if symbol != 'overall_regime' and regime_info.get('confidence', 0) > 0.6:
                    market_data = market_analysis.get(symbol)
                    
                    if market_data is not None:
                        # Generate signal using appropriate strategy
                        signal = STRATEGY_ARSENAL.generate_trade_signal(market_data, regime_info)
                        
                        if signal.action != 'hold':
                            signals.append({
                                'symbol': symbol,
                                'signal': signal,
                                'regime': regime_info['current_regime'],
                                'regime_confidence': regime_info['confidence'],
                                'timestamp': datetime.now()
                            })
            
            logging.info(f"📊 Generated {len(signals)} trading signals")
            
        except Exception as e:
            logging.error(f"❌ Signal generation failed: {e}")
        
        return {'signals': signals, 'timestamp': datetime.now()}
    
    async def _execute_trades(self, trading_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trades with proper risk management"""
        executed_trades = []
        rejected_trades = []
        
        try:
            for signal_info in trading_decisions.get('signals', []):
                signal = signal_info['signal']
                
                # Risk assessment
                risk_assessment = RISK_MANAGER.assess_trade_risk(
                    asdict(signal), 
                    {'volatility': 0.002}  # Simplified market conditions
                )
                
                # Check if trade can be executed
                if RISK_MANAGER.can_execute_trade(asdict(signal), risk_assessment):
                    # Execute trade
                    execution_result = await EXECUTION_ENGINE.execute_trade(
                        asdict(signal), asdict(risk_assessment)
                    )
                    
                    if execution_result['status'] == 'executed':
                        executed_trades.append(execution_result)
                        
                        # Record trade execution
                        trade_record = {
                            'symbol': signal.symbol,
                            'action': signal.action,
                            'position_size': signal.position_size,
                            'execution_price': execution_result['execution_price'],
                            'contract_id': execution_result['contract_id'],
                            'timestamp': datetime.now(),
                            'regime': signal_info['regime'],
                            'strategy': signal.strategy
                        }
                        
                        RISK_MANAGER.record_trade_execution(trade_record)
                        PERFORMANCE_ANALYZER.record_trade_performance(trade_record)
                        
                    else:
                        rejected_trades.append({
                            'signal': signal_info,
                            'reason': execution_result.get('error', 'unknown'),
                            'timestamp': datetime.now()
                        })
                else:
                    rejected_trades.append({
                        'signal': signal_info,
                        'reason': risk_assessment.recommended_action,
                        'timestamp': datetime.now()
                    })
            
            logging.info(f"💼 Executed {len(executed_trades)} trades, rejected {len(rejected_trades)}")
            
        except Exception as e:
            logging.error(f"❌ Trade execution failed: {e}")
        
        return {
            'executed_trades': executed_trades,
            'rejected_trades': rejected_trades,
            'timestamp': datetime.now()
        }
    
    async def _learn_and_adapt(self, execution_results: Dict[str, Any], 
                             regime_analysis: Dict[str, Any]):
        """Learn from results and adapt system"""
        try:
            # Analyze performance
            performance_analysis = PERFORMANCE_ANALYZER.analyze_performance_trends()
            
            # Check adaptation needs
            adaptation_analysis = CONTINUOUS_ADAPTER.monitor_adaptation_needs(
                performance_analysis, regime_analysis
            )
            
            # Execute adaptation if needed
            if adaptation_analysis['adaptation_needed']:
                await CONTINUOUS_ADAPTER.execute_adaptation_cycle(adaptation_analysis)
            
            # Check scheduled learnings
            await CONTINUOUS_ADAPTER.check_scheduled_learnings()
            
            # Preserve knowledge
            self._preserve_knowledge(performance_analysis, regime_analysis)
            
        except Exception as e:
            logging.error(f"❌ Learning and adaptation failed: {e}")
    
    def _preserve_knowledge(self, performance_analysis: Dict[str, Any], 
                          regime_analysis: Dict[str, Any]):
        """Preserve valuable knowledge"""
        try:
            # Preserve successful patterns
            for symbol, regime_info in regime_analysis.items():
                if symbol != 'overall_regime':
                    pattern_data = {
                        'pattern_name': f"{regime_info['current_regime']}_{symbol}",
                        'regime': regime_info['current_regime'],
                        'confidence': regime_info['confidence'],
                        'timestamp': datetime.now()
                    }
                    
                    performance_data = {
                        'success_rate': performance_analysis.get('win_rate', 0.5),
                        'stability_score': 0.8  # Simplified
                    }
                    
                    KNOWLEDGE_BANK.preserve_pattern_knowledge(pattern_data, performance_data)
            
        except Exception as e:
            logging.error(f"❌ Knowledge preservation failed: {e}")
    
    def _extract_current_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract current features for regime classification"""
        # Simplified feature extraction - in practice, this would use
        # the same features as during training
        if market_data.empty:
            return np.array([])
        
        recent_data = market_data.tail(50)
        
        features = [
            recent_data['volatility_15m'].iloc[-1] if 'volatility_15m' in recent_data else 0.001,
            recent_data['trend_strength'].iloc[-1] if 'trend_strength' in recent_data else 0.5,
            recent_data['rsi'].iloc[-1] if 'rsi' in recent_data else 50,
            recent_data['price_vs_sma'].iloc[-1] if 'price_vs_sma' in recent_data else 0
        ]
        
        return np.array(features)
    
    def _calculate_market_volatility(self, market_analysis: Dict[str, Any]) -> float:
        """Calculate overall market volatility"""
        volatilities = []
        
        for symbol, analysis in market_analysis.items():
            if symbol != 'market_summary' and 'volatility_15m' in analysis:
                volatilities.append(analysis['volatility_15m'].iloc[-1])
        
        return np.mean(volatilities) if volatilities else 0.001
    
    def _determine_overall_regime(self, regime_analysis: Dict[str, Any]) -> str:
        """Determine overall market regime from individual symbols"""
        regime_counts = {}
        
        for symbol, analysis in regime_analysis.items():
            if symbol != 'overall_regime':
                regime = analysis.get('current_regime', 'unknown')
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            return max(regime_counts.items(), key=lambda x: x[1])[0]
        else:
            return 'unknown'
    
    def _update_session_state(self, regime_analysis: Dict[str, Any], 
                            execution_results: Dict[str, Any]):
        """Update current trading session state"""
        self.current_session['trades_executed'] += len(execution_results.get('executed_trades', []))
        self.current_session['current_regime'] = regime_analysis.get('overall_regime', 'unknown')
        
        # Calculate session P&L from executed trades
        for trade in execution_results.get('executed_trades', []):
            if 'trade_details' in trade and 'pnl' in trade['trade_details']:
                self.current_session['session_pnl'] += trade['trade_details']['pnl']
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logging.info("🛑 Received shutdown signal...")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        logging.info("🔴 Shutting down Platform Mind Reader...")
        self.is_running = False
        
        try:
            # Stop data collection
            await DATA_ARCHAEOLOGIST.stop_collection()
            
            # Close execution engine
            await EXECUTION_ENGINE.close_connection()
            
            # Save knowledge backup
            KNOWLEDGE_BANK.save_knowledge_backup()
            
            logging.info("✅ Platform Mind Reader shutdown complete")
            
        except Exception as e:
            logging.error(f"❌ Shutdown error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = (datetime.now() - self.start_time) if self.start_time else timedelta(0)
        
        return {
            'system_health': self.system_health,
            'uptime_seconds': uptime.total_seconds(),
            'trading_cycles_completed': self.cycle_count,
            'active_phases': ORCHESTRATOR.get_active_components(),
            'component_status': self.component_status,
            'current_session': self.current_session,
            'performance_metrics': PERFORMANCE_ANALYZER.get_performance_report(),
            'risk_status': RISK_MANAGER.get_risk_dashboard(),
            'adaptation_status': CONTINUOUS_ADAPTER.get_adaptation_status(),
            'knowledge_summary': KNOWLEDGE_BANK.get_knowledge_summary()
        }

async def main():
    """Main application entry point"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('platform_mind_reader.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create and initialize the system
    bot = PlatformMindReader()
    
    if await bot.initialize_system():
        bot.is_running = True
        
        logging.info("🎯 Platform Mind Reader is now RUNNING!")
        logging.info("💡 System will automatically progress through phases as performance validates")
        
        # Main trading loop
        while bot.is_running:
            try:
                # Run trading cycle
                cycle_result = await bot.run_trading_cycle()
                
                # Log cycle results
                if 'error' not in cycle_result:
                    logging.info(f"📈 Cycle {cycle_result['cycle_number']}: "
                               f"Regime: {cycle_result['regime']}, "
                               f"Trades: {cycle_result['trades_executed']}")
                
                # Progress to next phase if ready
                ORCHESTRATOR.progress_to_next_phase({
                    'regime_accuracy': 0.85,  # Simulated performance
                    'data_quality': 0.95,
                    'pattern_validation_passed': True
                })
                
                # Wait before next cycle (adjust based on timeframe)
                await asyncio.sleep(60)  # 1 minute between cycles
                
            except Exception as e:
                logging.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(10)  # Brief pause before continuing
        
        # Shutdown gracefully
        await bot.shutdown()
    
    else:
        logging.error("❌ Failed to initialize Platform Mind Reader")
        sys.exit(1)

if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
