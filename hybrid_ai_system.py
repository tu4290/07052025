"""
HYBRID AI ARCHITECTURE - Foundation + Learning
==============================================

A hybrid system that combines your trading expertise (foundation rules) 
with adaptive learning to discover new patterns and opportunities.

Perfect for limited data scenarios where domain expertise provides the base.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Any

class HybridMOE:
    """
    Hybrid MOE with Foundation Rules + Adaptive Learning Layer
    
    Architecture:
    1. Foundation Layer: Your trading expertise as rules
    2. Learning Layer: Discovers new patterns
    3. Confidence System: Distinguishes known vs discovered patterns
    4. Evolution Tracker: Learns from outcomes
    """
    
    def __init__(self):
        # Foundation rules (your trading expertise)
        self.foundation_rules = self._initialize_foundation_rules()
        
        # Learning layer
        self.discovered_patterns = {}
        self.pattern_performance = {}
        self.learning_confidence_threshold = 0.6
        
        # Evolution tracking
        self.outcome_history = []
        self.pattern_evolution = {}
        
        # Confidence weights
        self.foundation_weight = 0.7  # High confidence in your expertise
        self.learning_weight = 0.3    # Lower weight for discovered patterns
    
    def _initialize_foundation_rules(self):
        """
        FOUNDATION RULES - Your Trading Expertise
        ========================================
        
        These are the core scenarios you know work, encoded as rules.
        High confidence, battle-tested patterns.
        """
        return {
            'VOLATILE_DOWN_IGNITION': {
                'conditions': {
                    'VAPI-FA': {'range': [0.5, 1.2], 'weight': 0.3},
                    'DWFD': {'range': [1.5, 2.5], 'weight': 0.4},
                    'VRI 2.0': {'range': [-3.0, -2.0], 'weight': 0.3}
                },
                'confluence_required': 2,  # At least 2 conditions must match
                'output': {
                    'signal': 'Ignition Point: Volatile Down',
                    'action': 'Prepare for volatility expansion',
                    'color': 'RED',
                    'confidence': 0.9,
                    'timeframe': 'Immediate to 1 hour'
                }
            },
            
            'MOMENTUM_EXPLOSION': {
                'conditions': {
                    'MSPI': {'range': [1.8, 3.0], 'weight': 0.4},
                    'A-DAG': {'range': [1.5, 3.0], 'weight': 0.3},
                    'AOFM': {'range': [1.2, 2.5], 'weight': 0.3}
                },
                'confluence_required': 2,
                'output': {
                    'signal': 'Momentum Explosion Imminent',
                    'action': 'Aggressive trend following',
                    'color': 'BRIGHT_GREEN',
                    'confidence': 0.85,
                    'timeframe': '15 minutes to 2 hours'
                }
            },
            
            'EOD_HEDGING_SETUP': {
                'conditions': {
                    'GIB': {'range': [2.5, 4.0], 'weight': 0.5},
                    'TDPI': {'range': [1.5, 3.0], 'weight': 0.3},
                    'time_of_day': {'range': [14, 16], 'weight': 0.2}  # 2-4 PM
                },
                'confluence_required': 2,
                'output': {
                    'signal': 'EOD Hedging Opportunity',
                    'action': 'Position for end-of-day flows',
                    'color': 'PURPLE',
                    'confidence': 0.8,
                    'timeframe': '30 minutes to close'
                }
            },
            
            'PREMIUM_CRUSH': {
                'conditions': {
                    'VRI 2.0': {'range': [-1.0, 0.5], 'weight': 0.4},
                    'VAPI-FA': {'range': [-0.5, 0.5], 'weight': 0.3},
                    'GIB': {'range': [-1.0, 1.0], 'weight': 0.3}
                },
                'confluence_required': 3,  # All conditions for premium selling
                'output': {
                    'signal': 'Premium Selling Environment',
                    'action': 'Sell options premium',
                    'color': 'BLUE',
                    'confidence': 0.75,
                    'timeframe': 'Several hours to days'
                }
            },
            
            'GAMMA_STORM_WARNING': {
                'conditions': {
                    'GIB': {'range': [3.0, 5.0], 'weight': 0.4},
                    'A-DAG': {'range': [2.5, 4.0], 'weight': 0.3},
                    'VRI 2.0': {'range': [2.0, 4.0], 'weight': 0.3}
                },
                'confluence_required': 2,
                'output': {
                    'signal': 'Gamma Storm Detected',
                    'action': 'Extreme caution - reduce size',
                    'color': 'ELECTRIC_BLUE',
                    'confidence': 0.95,
                    'timeframe': 'Immediate'
                }
            }
        }
    
    def check_foundation_rules(self, metrics: Dict[str, float]) -> Tuple[str, Dict, float]:
        """
        Check metrics against foundation rules (your expertise).
        Returns the best matching rule with confidence.
        """
        best_match = None
        best_score = 0
        best_rule_name = None
        
        current_hour = datetime.now().hour
        
        for rule_name, rule in self.foundation_rules.items():
            score = 0
            conditions_met = 0
            total_weight = 0
            
            for metric, condition in rule['conditions'].items():
                if metric == 'time_of_day':
                    # Special handling for time
                    if condition['range'][0] <= current_hour <= condition['range'][1]:
                        score += condition['weight']
                        conditions_met += 1
                    total_weight += condition['weight']
                elif metric in metrics:
                    value = metrics[metric]
                    min_val, max_val = condition['range']
                    
                    if min_val <= value <= max_val:
                        # Perfect match
                        score += condition['weight']
                        conditions_met += 1
                    elif min_val - 0.5 <= value <= max_val + 0.5:
                        # Close match (partial credit)
                        score += condition['weight'] * 0.5
                        conditions_met += 0.5
                    
                    total_weight += condition['weight']
            
            # Normalize score and check confluence requirement
            if total_weight > 0:
                normalized_score = score / total_weight
                if conditions_met >= rule['confluence_required']:
                    if normalized_score > best_score:
                        best_score = normalized_score
                        best_match = rule
                        best_rule_name = rule_name
        
        if best_match:
            # Adjust confidence based on how well conditions were met
            adjusted_confidence = best_match['output']['confidence'] * best_score
            return best_rule_name, best_match['output'], adjusted_confidence
        
        return None, {}, 0.0
    
    def discover_new_patterns(self, metrics: Dict[str, float], outcome: str = None) -> Tuple[str, Dict, float]:
        """
        LEARNING LAYER - Discover New Patterns
        =====================================
        
        Uses clustering and correlation analysis to find new metric combinations
        that predict specific outcomes.
        """
        # Convert metrics to pattern signature
        pattern_signature = self._create_pattern_signature(metrics)
        
        # Check if we've seen this pattern before
        if pattern_signature in self.discovered_patterns:
            pattern = self.discovered_patterns[pattern_signature]
            confidence = self._calculate_learning_confidence(pattern_signature)
            return f"DISCOVERED_{pattern_signature}", pattern, confidence
        
        # Try to discover new pattern
        new_pattern = self._analyze_for_new_pattern(metrics)
        
        if new_pattern and new_pattern['confidence'] > self.learning_confidence_threshold:
            self.discovered_patterns[pattern_signature] = new_pattern
            return f"NEW_DISCOVERY_{pattern_signature}", new_pattern, new_pattern['confidence']
        
        return None, {}, 0.0
    
    def _create_pattern_signature(self, metrics: Dict[str, float]) -> str:
        """Create a signature for the current metric pattern."""
        # Discretize metrics into ranges for pattern matching
        signature_parts = []
        
        for metric, value in sorted(metrics.items()):
            if abs(value) > 2.5:
                level = "EXTREME"
            elif abs(value) > 1.5:
                level = "HIGH"
            elif abs(value) > 0.5:
                level = "MODERATE"
            else:
                level = "LOW"
            
            direction = "POS" if value > 0 else "NEG"
            signature_parts.append(f"{metric}_{level}_{direction}")
        
        return "_".join(signature_parts[:4])  # Limit signature length
    
    def _analyze_for_new_pattern(self, metrics: Dict[str, float]) -> Dict:
        """
        Analyze current metrics for potential new patterns.
        Uses statistical analysis to identify interesting combinations.
        """
        # Look for unusual metric combinations
        metric_values = list(metrics.values())
        
        # Statistical measures
        mean_abs = np.mean([abs(v) for v in metric_values])
        std_dev = np.std(metric_values)
        max_val = max([abs(v) for v in metric_values])
        
        # Pattern detection heuristics
        if mean_abs > 2.0 and std_dev > 1.0:
            # High volatility pattern
            return {
                'signal': f'High Volatility Cluster (μ={mean_abs:.1f})',
                'action': 'Monitor for breakout',
                'color': 'YELLOW',
                'confidence': min(mean_abs / 3.0, 0.8),
                'pattern_type': 'DISCOVERED_VOLATILITY'
            }
        
        elif mean_abs < 0.8 and std_dev < 0.5:
            # Low activity pattern
            return {
                'signal': f'Low Activity Environment (μ={mean_abs:.1f})',
                'action': 'Consider premium strategies',
                'color': 'LIGHT_BLUE',
                'confidence': min((1.0 - mean_abs), 0.7),
                'pattern_type': 'DISCOVERED_CALM'
            }
        
        # Check for divergence patterns
        positive_metrics = [k for k, v in metrics.items() if v > 1.0]
        negative_metrics = [k for k, v in metrics.items() if v < -1.0]
        
        if len(positive_metrics) >= 2 and len(negative_metrics) >= 2:
            return {
                'signal': 'Divergence Pattern Detected',
                'action': 'Conflicting signals - reduce conviction',
                'color': 'ORANGE',
                'confidence': 0.6,
                'pattern_type': 'DISCOVERED_DIVERGENCE'
            }
        
        return None
    
    def _calculate_learning_confidence(self, pattern_signature: str) -> float:
        """Calculate confidence in a discovered pattern based on historical performance."""
        if pattern_signature not in self.pattern_performance:
            return 0.5  # Default confidence for new patterns
        
        performance = self.pattern_performance[pattern_signature]
        success_rate = performance.get('success_rate', 0.5)
        sample_size = performance.get('sample_size', 1)
        
        # Confidence increases with success rate and sample size
        confidence = success_rate * min(sample_size / 10.0, 1.0)
        return min(confidence, 0.8)  # Cap learning confidence below foundation
    
    def hybrid_analysis(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        MASTER HYBRID ANALYSIS
        =====================
        
        Combines foundation rules with learning layer.
        Returns the best signal with confidence breakdown.
        """
        # Check foundation rules first (your expertise)
        foundation_rule, foundation_output, foundation_confidence = self.check_foundation_rules(metrics)
        
        # Check learning layer
        learning_pattern, learning_output, learning_confidence = self.discover_new_patterns(metrics)
        
        # Combine results
        if foundation_confidence > 0:
            # Foundation rule triggered
            if learning_confidence > 0.5:
                # Both systems agree - boost confidence
                combined_confidence = min(foundation_confidence * 1.2, 1.0)
                signal_source = "FOUNDATION + LEARNING"
            else:
                combined_confidence = foundation_confidence
                signal_source = "FOUNDATION"
            
            return {
                'signal': foundation_output.get('signal', 'Unknown'),
                'action': foundation_output.get('action', 'Monitor'),
                'color': foundation_output.get('color', 'GRAY'),
                'confidence': combined_confidence,
                'source': signal_source,
                'foundation_rule': foundation_rule,
                'learning_pattern': learning_pattern,
                'timeframe': foundation_output.get('timeframe', 'Unknown'),
                'type': 'HYBRID_FOUNDATION'
            }
        
        elif learning_confidence > self.learning_confidence_threshold:
            # Only learning layer triggered
            return {
                'signal': learning_output.get('signal', 'Unknown Pattern'),
                'action': learning_output.get('action', 'Monitor'),
                'color': learning_output.get('color', 'GRAY'),
                'confidence': learning_confidence,
                'source': "LEARNING_ONLY",
                'foundation_rule': None,
                'learning_pattern': learning_pattern,
                'timeframe': 'Variable',
                'type': 'HYBRID_LEARNING'
            }
        
        else:
            # No clear signal
            return {
                'signal': 'No Clear Pattern',
                'action': 'Continue monitoring',
                'color': 'GRAY',
                'confidence': 0.3,
                'source': "NONE",
                'foundation_rule': None,
                'learning_pattern': None,
                'timeframe': 'N/A',
                'type': 'HYBRID_NEUTRAL'
            }
    
    def record_outcome(self, metrics: Dict[str, float], prediction: Dict, actual_outcome: str):
        """
        Record actual outcomes to improve learning layer.
        This is how the system evolves and gets smarter.
        """
        outcome_record = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'success': self._evaluate_prediction_success(prediction, actual_outcome)
        }
        
        self.outcome_history.append(outcome_record)
        
        # Update pattern performance
        if prediction.get('learning_pattern'):
            pattern_sig = prediction['learning_pattern'].split('_', 1)[-1]
            if pattern_sig not in self.pattern_performance:
                self.pattern_performance[pattern_sig] = {
                    'success_rate': 0.5,
                    'sample_size': 0,
                    'total_successes': 0
                }
            
            perf = self.pattern_performance[pattern_sig]
            perf['sample_size'] += 1
            if outcome_record['success']:
                perf['total_successes'] += 1
            perf['success_rate'] = perf['total_successes'] / perf['sample_size']
    
    def _evaluate_prediction_success(self, prediction: Dict, actual_outcome: str) -> bool:
        """Evaluate if the prediction was successful."""
        # Simple success criteria - can be made more sophisticated
        predicted_action = prediction.get('action', '').lower()
        actual_outcome = actual_outcome.lower()
        
        success_keywords = {
            'buy': ['profit', 'gain', 'up', 'bullish'],
            'sell': ['profit', 'gain', 'down', 'bearish'],
            'reduce': ['avoided', 'safe', 'protected'],
            'monitor': ['stable', 'unchanged']
        }
        
        for action_key, outcome_keys in success_keywords.items():
            if action_key in predicted_action:
                return any(keyword in actual_outcome for keyword in outcome_keys)
        
        return False  # Default to unsuccessful if unclear

def demonstrate_hybrid_system():
    """Demonstrate the hybrid system with examples."""
    
    # Test scenarios
    scenarios = {
        "Foundation Rule Match": {
            'VAPI-FA': 0.88,
            'DWFD': 1.96,
            'VRI 2.0': -2.53,
            'GIB': 1.2,
            'MSPI': 0.5,
            'A-DAG': 0.3
        },
        "Learning Discovery": {
            'VAPI-FA': 2.8,
            'DWFD': 2.9,
            'VRI 2.0': 2.7,
            'GIB': 2.6,
            'MSPI': 2.5,
            'A-DAG': 2.4
        },
        "No Clear Pattern": {
            'VAPI-FA': 0.1,
            'DWFD': -0.2,
            'VRI 2.0': 0.3,
            'GIB': -0.1,
            'MSPI': 0.2,
            'A-DAG': -0.1
        }
    }
    
    hybrid_moe = HybridMOE()
    
    print("🤖 HYBRID AI SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    for scenario_name, metrics in scenarios.items():
        print(f"\n📊 SCENARIO: {scenario_name}")
        print("-" * 30)
        
        result = hybrid_moe.hybrid_analysis(metrics)
        
        print(f"🎯 SIGNAL: {result['signal']}")
        print(f"🎬 ACTION: {result['action']}")
        print(f"🎨 COLOR: {result['color']}")
        print(f"🎪 CONFIDENCE: {result['confidence']:.1%}")
        print(f"🔍 SOURCE: {result['source']}")
        print(f"⏰ TIMEFRAME: {result['timeframe']}")
        print(f"🏗️ TYPE: {result['type']}")
        
        if result['foundation_rule']:
            print(f"📚 FOUNDATION RULE: {result['foundation_rule']}")
        if result['learning_pattern']:
            print(f"🧠 LEARNING PATTERN: {result['learning_pattern']}")

if __name__ == "__main__":
    demonstrate_hybrid_system()

