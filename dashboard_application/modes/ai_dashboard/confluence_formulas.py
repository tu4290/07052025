"""
CONFLUENCE COMBINATION FORMULAS - Next Level Intelligence
=========================================================

Composite signals where multiple metrics create specific trading setups.
These combinations reveal hidden market patterns and opportunities.
"""

import numpy as np
import math

class ConfluenceMOE:
    """Advanced MOE that detects confluence patterns between metrics."""
    
    def __init__(self):
        # Confluence thresholds
        self.CONFLUENCE_THRESHOLD = 0.7  # How aligned metrics need to be
        self.STRONG_SIGNAL_THRESHOLD = 1.5
        self.EXTREME_SIGNAL_THRESHOLD = 2.0
        
        # Confluence pattern definitions
        self.confluence_patterns = {
            'MOMENTUM_EXPLOSION': {
                'metrics': ['MSPI', 'A-DAG'],
                'description': 'Momentum + Gamma Acceleration',
                'color': 'BRIGHT_GREEN',
                'action': 'Aggressive trend following'
            },
            'EOD_HEDGING': {
                'metrics': ['GIB', 'TDPI'],
                'description': 'Gamma Imbalance + Time Decay Pressure',
                'color': 'PURPLE',
                'action': 'End of day hedging opportunity'
            },
            'FLOW_CONVERGENCE': {
                'metrics': ['MSPI', 'TDPI', 'VAPI-FA'],
                'description': 'Multi-timeframe flow alignment',
                'color': 'CYAN',
                'action': 'High conviction directional play'
            },
            'VOLATILITY_SQUEEZE': {
                'metrics': ['VRI 2.0', 'GIB', 'AOFM'],
                'description': 'Low vol + High gamma + Aggressive flow',
                'color': 'YELLOW',
                'action': 'Volatility breakout imminent'
            },
            'SMART_MONEY_DIVERGENCE': {
                'metrics': ['DWFD', 'MSPI'],
                'description': 'Smart money vs retail divergence',
                'color': 'ORANGE_RED',
                'action': 'Follow the smart money'
            },
            'GAMMA_STORM': {
                'metrics': ['GIB', 'A-DAG', 'VRI 2.0'],
                'description': 'Triple gamma confluence',
                'color': 'ELECTRIC_BLUE',
                'action': 'Extreme volatility event'
            }
        }
    
    def calculate_momentum_explosion(self, metrics):
        """
        MOMENTUM EXPLOSION FORMULA
        =========================
        
        MSPI + A-DAG Confluence
        
        Momentum_Score = |MSPI| * direction_alignment * gamma_amplifier
        Gamma_Amplifier = 1 + (|A-DAG| / 3.0)
        Direction_Alignment = 1 if same sign, 0.5 if opposite
        
        If Momentum_Score > 2.0: BRIGHT_GREEN (Aggressive trend following)
        """
        mspi = metrics.get('MSPI', 0)
        a_dag = metrics.get('A-DAG', 0)
        
        # Direction alignment
        if np.sign(mspi) == np.sign(a_dag):
            direction_alignment = 1.0
        else:
            direction_alignment = 0.5
        
        # Gamma amplifier
        gamma_amplifier = 1 + (abs(a_dag) / 3.0)
        
        # Momentum score
        momentum_score = abs(mspi) * direction_alignment * gamma_amplifier
        
        confidence = min(momentum_score / 3.0, 1.0)
        
        return {
            'score': momentum_score,
            'confidence': confidence,
            'trigger': momentum_score > 2.0,
            'description': f"Momentum: {momentum_score:.2f}, Alignment: {direction_alignment:.1f}"
        }
    
    def calculate_eod_hedging(self, metrics):
        """
        END OF DAY HEDGING FORMULA
        =========================
        
        GIB + TDPI Confluence
        
        Hedging_Pressure = |GIB| * time_decay_factor * imbalance_multiplier
        Time_Decay_Factor = |TDPI| / 2.0
        Imbalance_Multiplier = 1 + (|GIB| > 2.0 ? 0.5 : 0)
        
        If Hedging_Pressure > 1.5: PURPLE (EOD hedging opportunity)
        """
        gib = metrics.get('GIB', 0)
        tdpi = metrics.get('TDPI', 0)
        
        # Time decay factor
        time_decay_factor = abs(tdpi) / 2.0
        
        # Imbalance multiplier
        imbalance_multiplier = 1.0 + (0.5 if abs(gib) > 2.0 else 0)
        
        # Hedging pressure
        hedging_pressure = abs(gib) * time_decay_factor * imbalance_multiplier
        
        confidence = min(hedging_pressure / 2.5, 1.0)
        
        return {
            'score': hedging_pressure,
            'confidence': confidence,
            'trigger': hedging_pressure > 1.5,
            'description': f"Hedging: {hedging_pressure:.2f}, Time Factor: {time_decay_factor:.2f}"
        }
    
    def calculate_flow_convergence(self, metrics):
        """
        FLOW CONVERGENCE FORMULA
        =======================
        
        MSPI + TDPI + VAPI-FA Confluence
        
        Convergence_Score = geometric_mean(|metrics|) * direction_consistency^2
        Direction_Consistency = |sum(signs)| / count(metrics)
        
        If Convergence_Score > 1.8 AND consistency > 0.8: CYAN (High conviction)
        """
        mspi = metrics.get('MSPI', 0)
        tdpi = metrics.get('TDPI', 0)
        vapi_fa = metrics.get('VAPI-FA', 0)
        
        values = [mspi, tdpi, vapi_fa]
        abs_values = [abs(v) for v in values if v != 0]
        
        if len(abs_values) < 2:
            return {'score': 0, 'confidence': 0, 'trigger': False, 'description': 'Insufficient data'}
        
        # Geometric mean of absolute values
        geometric_mean = np.power(np.prod(abs_values), 1.0/len(abs_values))
        
        # Direction consistency
        signs = [np.sign(v) for v in values if v != 0]
        direction_consistency = abs(sum(signs)) / len(signs)
        
        # Convergence score
        convergence_score = geometric_mean * (direction_consistency ** 2)
        
        confidence = min(convergence_score / 2.5, 1.0)
        
        return {
            'score': convergence_score,
            'confidence': confidence,
            'trigger': convergence_score > 1.8 and direction_consistency > 0.8,
            'description': f"Convergence: {convergence_score:.2f}, Consistency: {direction_consistency:.2f}"
        }
    
    def calculate_volatility_squeeze(self, metrics):
        """
        VOLATILITY SQUEEZE FORMULA
        =========================
        
        VRI 2.0 + GIB + AOFM Confluence
        
        Squeeze_Potential = (low_vol_factor * gamma_pressure * flow_intensity)^0.5
        Low_Vol_Factor = max(0, (1.0 - |VRI|/2.0))
        Gamma_Pressure = |GIB| / 2.0
        Flow_Intensity = |AOFM| / 2.0
        
        If Squeeze_Potential > 0.8: YELLOW (Breakout imminent)
        """
        vri = metrics.get('VRI 2.0', 0)
        gib = metrics.get('GIB', 0)
        aofm = metrics.get('AOFM', 0)
        
        # Low volatility factor (higher when VRI is low)
        low_vol_factor = max(0, (1.0 - abs(vri)/2.0))
        
        # Gamma pressure
        gamma_pressure = abs(gib) / 2.0
        
        # Flow intensity
        flow_intensity = abs(aofm) / 2.0
        
        # Squeeze potential (geometric mean)
        if low_vol_factor > 0 and gamma_pressure > 0 and flow_intensity > 0:
            squeeze_potential = (low_vol_factor * gamma_pressure * flow_intensity) ** (1/3)
        else:
            squeeze_potential = 0
        
        confidence = min(squeeze_potential / 1.0, 1.0)
        
        return {
            'score': squeeze_potential,
            'confidence': confidence,
            'trigger': squeeze_potential > 0.8,
            'description': f"Squeeze: {squeeze_potential:.2f}, Low Vol: {low_vol_factor:.2f}"
        }
    
    def calculate_smart_money_divergence(self, metrics):
        """
        SMART MONEY DIVERGENCE FORMULA
        =============================
        
        DWFD + MSPI Confluence
        
        Divergence_Strength = |DWFD - MSPI| * max(|DWFD|, |MSPI|)
        Smart_Money_Edge = |DWFD| - |MSPI|
        
        If Divergence_Strength > 2.0: ORANGE_RED (Follow smart money)
        """
        dwfd = metrics.get('DWFD', 0)
        mspi = metrics.get('MSPI', 0)
        
        # Divergence strength
        divergence = abs(dwfd - mspi)
        magnitude = max(abs(dwfd), abs(mspi))
        divergence_strength = divergence * magnitude
        
        # Smart money edge (positive = smart money stronger)
        smart_money_edge = abs(dwfd) - abs(mspi)
        
        confidence = min(divergence_strength / 3.0, 1.0)
        
        return {
            'score': divergence_strength,
            'confidence': confidence,
            'trigger': divergence_strength > 2.0,
            'smart_money_stronger': smart_money_edge > 0,
            'description': f"Divergence: {divergence_strength:.2f}, Edge: {smart_money_edge:.2f}"
        }
    
    def calculate_gamma_storm(self, metrics):
        """
        GAMMA STORM FORMULA
        ==================
        
        GIB + A-DAG + VRI 2.0 Confluence
        
        Storm_Intensity = (|GIB| * |A-DAG| * |VRI|)^(1/3) * alignment_factor
        Alignment_Factor = 1 + (all_extreme_bonus)
        All_Extreme_Bonus = 0.5 if all metrics > 2.0, else 0
        
        If Storm_Intensity > 2.5: ELECTRIC_BLUE (Extreme volatility event)
        """
        gib = metrics.get('GIB', 0)
        a_dag = metrics.get('A-DAG', 0)
        vri = metrics.get('VRI 2.0', 0)
        
        abs_values = [abs(gib), abs(a_dag), abs(vri)]
        
        # Geometric mean
        if all(v > 0 for v in abs_values):
            geometric_mean = np.power(np.prod(abs_values), 1/3)
        else:
            geometric_mean = 0
        
        # All extreme bonus
        all_extreme_bonus = 0.5 if all(v > 2.0 for v in abs_values) else 0
        alignment_factor = 1.0 + all_extreme_bonus
        
        # Storm intensity
        storm_intensity = geometric_mean * alignment_factor
        
        confidence = min(storm_intensity / 3.5, 1.0)
        
        return {
            'score': storm_intensity,
            'confidence': confidence,
            'trigger': storm_intensity > 2.5,
            'all_extreme': all_extreme_bonus > 0,
            'description': f"Storm: {storm_intensity:.2f}, All Extreme: {all_extreme_bonus > 0}"
        }
    
    def detect_confluence_patterns(self, metrics):
        """
        MASTER CONFLUENCE DETECTOR
        =========================
        
        Analyzes all confluence patterns and returns the strongest signal.
        """
        patterns = {}
        
        # Calculate all confluence patterns
        patterns['MOMENTUM_EXPLOSION'] = self.calculate_momentum_explosion(metrics)
        patterns['EOD_HEDGING'] = self.calculate_eod_hedging(metrics)
        patterns['FLOW_CONVERGENCE'] = self.calculate_flow_convergence(metrics)
        patterns['VOLATILITY_SQUEEZE'] = self.calculate_volatility_squeeze(metrics)
        patterns['SMART_MONEY_DIVERGENCE'] = self.calculate_smart_money_divergence(metrics)
        patterns['GAMMA_STORM'] = self.calculate_gamma_storm(metrics)
        
        # Find triggered patterns
        triggered_patterns = {name: data for name, data in patterns.items() if data['trigger']}
        
        if triggered_patterns:
            # Return strongest triggered pattern
            strongest = max(triggered_patterns.items(), key=lambda x: x[1]['confidence'])
            return strongest[0], strongest[1], self.confluence_patterns[strongest[0]]
        else:
            # Return strongest pattern even if not triggered
            strongest = max(patterns.items(), key=lambda x: x[1]['confidence'])
            return strongest[0], strongest[1], self.confluence_patterns[strongest[0]]

def demonstrate_confluence():
    """Demonstrate confluence patterns with examples."""
    
    scenarios = {
        "Momentum Explosion": {
            'MSPI': 2.1,
            'A-DAG': 1.8,
            'VAPI-FA': 1.5,
            'GIB': 1.2,
            'VRI 2.0': 1.0,
            'AOFM': 1.6,
            'DWFD': 1.3,
            'TDPI': 0.8
        },
        "EOD Hedging Setup": {
            'GIB': 2.8,
            'TDPI': 1.9,
            'MSPI': 0.5,
            'A-DAG': 0.3,
            'VAPI-FA': 0.7,
            'VRI 2.0': 1.1,
            'AOFM': 0.6,
            'DWFD': 0.4
        },
        "Flow Convergence": {
            'MSPI': 1.9,
            'TDPI': 1.7,
            'VAPI-FA': 2.1,
            'GIB': 1.0,
            'A-DAG': 0.8,
            'VRI 2.0': 1.2,
            'AOFM': 1.5,
            'DWFD': 1.6
        },
        "Gamma Storm": {
            'GIB': 3.1,
            'A-DAG': 2.7,
            'VRI 2.0': 2.9,
            'MSPI': 1.2,
            'TDPI': 1.0,
            'VAPI-FA': 1.5,
            'AOFM': 1.8,
            'DWFD': 1.1
        }
    }
    
    moe = ConfluenceMOE()
    
    print("🌊 CONFLUENCE PATTERN DETECTION")
    print("=" * 50)
    
    for scenario_name, metrics in scenarios.items():
        print(f"\n📊 SCENARIO: {scenario_name}")
        print("-" * 30)
        
        pattern_name, pattern_data, pattern_info = moe.detect_confluence_patterns(metrics)
        
        print(f"🎯 DETECTED PATTERN: {pattern_name}")
        print(f"📈 SCORE: {pattern_data['score']:.2f}")
        print(f"🎪 CONFIDENCE: {pattern_data['confidence']:.1%}")
        print(f"🚨 TRIGGERED: {'YES' if pattern_data['trigger'] else 'NO'}")
        print(f"🎨 COLOR: {pattern_info['color']}")
        print(f"💡 ACTION: {pattern_info['action']}")
        print(f"📝 DETAILS: {pattern_data['description']}")
        
        if pattern_name == 'SMART_MONEY_DIVERGENCE':
            stronger = "Smart Money" if pattern_data.get('smart_money_stronger') else "Retail"
            print(f"💰 STRONGER: {stronger}")

if __name__ == "__main__":
    demonstrate_confluence()

