"""
Algorithm State Classifier
Classifies current market conditions into platform algorithm states
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Tuple, Any

from config import ML_CONFIG, PLATFORM_PATTERNS
from core.deployment_orchestrator import ORCHESTRATOR

class AlgorithmStateClassifier:
    """Classifies current market conditions into platform algorithm states"""
    
    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.is_trained = False
        self.feature_names = []
        self.class_labels = list(PLATFORM_PATTERNS.ALGORITHM_STATES.keys()) + ['unknown']
        
        # Performance tracking
        self.performance_history = {
            'accuracy': [],
            'training_time': [],
            'last_trained': None
        }
    
    def prepare_training_data(self, regime_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from discovered regimes"""
        if not regime_data:
            return np.array([]), np.array([])
        
        features_list = []
        labels_list = []
        
        for regime_id, regime_info in regime_data.items():
            # Extract features for training
            features = self._extract_regime_features(regime_info)
            if features is not None:
                features_list.append(features)
                labels_list.append(self.class_labels.index(regime_info.get('matched_pattern', 'unknown')))
        
        return np.array(features_list), np.array(labels_list)
    
    def _extract_regime_features(self, regime_info: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from regime information"""
        try:
            features = [
                regime_info['volatility'],
                regime_info['trend_strength'],
                regime_info['mean_reversion_tendency'],
                regime_info['price_action_characteristics']['avg_returns'],
                regime_info['price_action_characteristics']['returns_skewness'],
                regime_info['price_action_characteristics']['returns_kurtosis'],
                regime_info['stability_score']
            ]
            return np.array(features)
        except KeyError as e:
            logging.warning(f"⚠️ Missing feature in regime data: {e}")
            return None
    
    def build_lstm_model(self, input_shape: tuple) -> Sequential:
        """Build LSTM model for sequence classification"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape, dropout=0.2),
            BatchNormalization(),
            LSTM(32, dropout=0.2),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(len(self.class_labels), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train both LSTM and Random Forest models"""
        if not ORCHESTRATOR.should_run_component('phase_1_pattern_archaeology', 'regime_classification'):
            return
        
        if len(X_train) < 50:
            logging.warning("⚠️ Insufficient training data")
            return
        
        try:
            # Prepare data for LSTM (reshape for sequences)
            if len(X_train.shape) == 2:
                X_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            else:
                X_lstm = X_train
            
            # Build and train LSTM
            self.lstm_model = self.build_lstm_model((X_lstm.shape[1], 1))
            
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                validation_data = (X_val_lstm, y_val)
            
            lstm_history = self.lstm_model.fit(
                X_lstm, y_train,
                epochs=100,
                batch_size=32,
                validation_data=validation_data,
                verbose=0,
                shuffle=True,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            
            # Train Random Forest
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Calculate accuracy
            lstm_accuracy = max(lstm_history.history['val_accuracy']) if validation_data else max(lstm_history.history['accuracy'])
            rf_accuracy = accuracy_score(y_train, self.rf_model.predict(X_train))
            
            self.performance_history['accuracy'].append((lstm_accuracy + rf_accuracy) / 2)
            self.performance_history['last_trained'] = pd.Timestamp.now()
            
            logging.info(f"✅ Models trained - LSTM: {lstm_accuracy:.3f}, RF: {rf_accuracy:.3f}")
            
        except Exception as e:
            logging.error(f"❌ Model training failed: {e}")
    
    def predict_current_state(self, current_features: np.ndarray) -> Tuple[str, float]:
        """Predict current algorithm state using ensemble"""
        if not self.is_trained or self.lstm_model is None or self.rf_model is None:
            return 'unknown', 0.0
        
        try:
            # Prepare features for prediction
            if len(current_features.shape) == 1:
                current_features = current_features.reshape(1, -1)
            
            # LSTM prediction
            current_lstm = current_features.reshape(current_features.shape[0], current_features.shape[1], 1)
            lstm_pred = self.lstm_model.predict(current_lstm, verbose=0)[0]
            lstm_class_idx = np.argmax(lstm_pred)
            lstm_confidence = np.max(lstm_pred)
            
            # Random Forest prediction
            rf_pred = self.rf_model.predict_proba(current_features)[0]
            rf_class_idx = np.argmax(rf_pred)
            rf_confidence = np.max(rf_pred)
            
            # Ensemble decision
            if lstm_confidence > ML_CONFIG.MIN_CONFIDENCE and rf_confidence > ML_CONFIG.MIN_CONFIDENCE:
                if lstm_class_idx == rf_class_idx:
                    # Models agree
                    final_class_idx = lstm_class_idx
                    final_confidence = (lstm_confidence * ML_CONFIG.ENSEMBLE_WEIGHTS['lstm'] + 
                                      rf_confidence * ML_CONFIG.ENSEMBLE_WEIGHTS['random_forest'])
                else:
                    # Models disagree - choose higher confidence
                    if lstm_confidence > rf_confidence:
                        final_class_idx = lstm_class_idx
                        final_confidence = lstm_confidence
                    else:
                        final_class_idx = rf_class_idx
                        final_confidence = rf_confidence
            else:
                # Low confidence - return unknown
                final_class_idx = self.class_labels.index('unknown')
                final_confidence = 0.0
            
            final_class = self.class_labels[final_class_idx]
            
            return final_class, final_confidence
            
        except Exception as e:
            logging.error(f"❌ Prediction failed: {e}")
            return 'unknown', 0.0
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        if not self.performance_history['accuracy']:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'average_accuracy': np.mean(self.performance_history['accuracy']),
            'last_trained': self.performance_history['last_trained'],
            'training_samples': len(self.performance_history['accuracy']),
            'class_labels': self.class_labels
        }
    
    def should_retrain(self, recent_accuracy: float, threshold: float = 0.7) -> bool:
        """Determine if models need retraining"""
        if not self.performance_history['accuracy']:
            return True
        
        avg_accuracy = np.mean(self.performance_history['accuracy'][-5:])  # Last 5 trainings
        return recent_accuracy < threshold or avg_accuracy < threshold

# Global classifier instance
STATE_CLASSIFIER = AlgorithmStateClassifier()
