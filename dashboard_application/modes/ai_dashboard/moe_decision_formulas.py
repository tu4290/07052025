"""
MOE DECISION FORMULAS - The Secret Sauce
=======================================

The mathematical algorithms that convert raw metrics into visual intelligence.
These formulas determine COLOR (action) and SHAPE (focus) decisions.
"""

import numpy as np
import math

class MOEDecisionEngine:
    """The brain that converts metrics into visual decisions."""
    
    def __init__(self):
        # Thresholds for decision making
        self.VOLATILITY_EXPLOSION_THRESHOLD = 2.0
        self.TREND_STRENGTH_THRESHOLD = 1.5
        self.GAMMA_DANGER_THRESHOLD = 2.5
        self.FLOW_INTENSITY_THRESHOLD = 1.2
        
        # Weights for composite scores
        self.VOLATILITY_WEIGHTS = {
            'VRI 2.0': 0.4,
            'GIB': 0.3,
            'VAPI-FA': 0.2,
            'AOFM': 0.1
        }
        
        self.TREND_WEIGHTS = {
            'VAPI-FA': 0.35,
            'DWFD': 0.25,
            'LWPAI': 0.25,
            'AOFM': 0.15
        }
    
    def calculate_volatility_explosion_score(self, metrics):
        """
        VOLATILITY EXPLOSION FORMULA
        ===========================
        
        Score = Σ(metric_value * weight) for volatility-related metrics
        
        If Score > THRESHOLD and GIB > 2.0 and VRI > 1.5:
            COLOR = RED (Buy Options)
        """
        score = 0
        for metric, weight in self.VOLATILITY_WEIGHTS.items():
            if metric in metrics:
                # Use absolute value for volatility (direction doesn't matter)
                score += abs(metrics[metric]) * weight
        
        # Bonus multiplier if gamma and volatility are both extreme
        gib = abs(metrics.get('GIB', 0))
        vri = abs(metrics.get('VRI 2.0', 0))
        
        if gib > 2.0 and vri > 1.5:
            score *= 1.5  # Amplify score for extreme conditions
        
        return score
    
    def calculate_trend_strength_score(self, metrics):
        """
        TREND STRENGTH FORMULA
        =====================
        
        Score = Σ(metric_value * weight * direction_consistency)
        
        Direction consistency = 1 if all major flows align, 0.5 if mixed
        
        If Score > THRESHOLD and direction_consistent:
            COLOR = GREEN (Ride Trend)
        """
        # Calculate weighted trend score
        score = 0
        direction_sum = 0
        
        for metric, weight in self.TREND_WEIGHTS.items():
            if metric in metrics:
                value = metrics[metric]
                score += abs(value) * weight
                direction_sum += np.sign(value) * weight
        
        # Direction consistency factor
        direction_consistency = abs(direction_sum) / sum(self.TREND_WEIGHTS.values())
        
        # Apply consistency multiplier
        final_score = score * direction_consistency
        
        return final_score, direction_consistency
    
    def calculate_premium_selling_score(self, metrics):
        """
        PREMIUM SELLING FORMULA
        ======================
        
        Conditions for BLUE (Sell Premium):
        1. Low volatility: VRI < 0.8
        2. High time decay potential: |VAPI-FA| < 0.5
        3. Stable gamma: |GIB| < 1.0
        4. Sideways action: |DWFD| < 0.5
        
        Score = (1 - normalized_volatility) * stability_factor
        """
        vri = abs(metrics.get('VRI 2.0', 0))
        vapi_fa = abs(metrics.get('VAPI-FA', 0))
        gib = abs(metrics.get('GIB', 0))
        dwfd = abs(metrics.get('DWFD', 0))
        
        # Normalize to 0-1 scale (assuming max values around 3.0)
        normalized_vri = min(vri / 3.0, 1.0)
        normalized_vapi = min(vapi_fa / 3.0, 1.0)
        normalized_gib = min(gib / 3.0, 1.0)
        normalized_dwfd = min(dwfd / 3.0, 1.0)
        
        # Low volatility score (higher is better for premium selling)
        low_vol_score = 1.0 - normalized_vri
        
        # Stability score (lower values = more stable)
        stability_score = 1.0 - (normalized_vapi + normalized_gib + normalized_dwfd) / 3.0
        
        # Combined score
        premium_score = low_vol_score * stability_score
        
        return premium_score
    
    def calculate_danger_score(self, metrics):
        """
        DANGER ZONE FORMULA
        ==================
        
        Conditions for ORANGE (Reduce Size):
        1. Extreme gamma: |GIB| > 2.5
        2. Conflicting signals: trend_consistency < 0.3
        3. High volatility + weak flow: VRI > 2.0 and |VAPI-FA| < 1.0
        4. Whipsaw pattern: rapid direction changes
        
        Score = max(gamma_danger, signal_conflict, whipsaw_risk)
        """
        gib = abs(metrics.get('GIB', 0))
        vri = abs(metrics.get('VRI 2.0', 0))
        vapi_fa = abs(metrics.get('VAPI-FA', 0))
        
        # Gamma danger (extreme gamma can cause violent moves)
        gamma_danger = max(0, (gib - 2.5) / 2.0)  # Scales from 0 at GIB=2.5 to 1.0 at GIB=4.5
        
        # Signal conflict (calculate trend consistency)
        _, trend_consistency = self.calculate_trend_strength_score(metrics)
        signal_conflict = max(0, (0.5 - trend_consistency) / 0.5)  # Higher when consistency < 0.5
        
        # Whipsaw risk (high vol + weak flow = dangerous)
        if vri > 2.0 and vapi_fa < 1.0:
            whipsaw_risk = min(1.0, vri / 3.0)
        else:
            whipsaw_risk = 0
        
        danger_score = max(gamma_danger, signal_conflict, whipsaw_risk)
        
        return danger_score
    
    def determine_color_action(self, metrics):
        """
        MASTER COLOR DECISION ALGORITHM
        ==============================
        
        Priority order:
        1. ORANGE (Danger) - Safety first
        2. RED (Volatility Explosion) - High profit potential
        3. BLUE (Premium Selling) - Steady income
        4. GREEN (Trend) - Default momentum play
        """
        # Calculate all scores
        vol_explosion_score = self.calculate_volatility_explosion_score(metrics)
        trend_score, trend_consistency = self.calculate_trend_strength_score(metrics)
        premium_score = self.calculate_premium_selling_score(metrics)
        danger_score = self.calculate_danger_score(metrics)
        
        # Decision logic (priority order)
        if danger_score > 0.6:
            return "ORANGE", danger_score, "High risk detected - reduce exposure"
        
        elif vol_explosion_score > self.VOLATILITY_EXPLOSION_THRESHOLD:
            return "RED", vol_explosion_score, "Volatility explosion imminent - buy options"
        
        elif premium_score > 0.7:
            return "BLUE", premium_score, "Low volatility environment - sell premium"
        
        elif trend_score > self.TREND_STRENGTH_THRESHOLD and trend_consistency > 0.6:
            return "GREEN", trend_score, "Strong directional trend - ride momentum"
        
        else:
            return "ORANGE", 0.5, "Mixed signals - reduce size and wait"
    
    def calculate_shape_focus(self, metrics):
        """
        SHAPE FOCUS FORMULA
        ==================
        
        Focus = metric with highest weighted significance score
        
        Significance = |metric_value| * importance_weight * recency_factor
        
        The shape bulges towards the metric with highest significance.
        """
        # Importance weights (how much each metric matters for trading decisions)
        importance_weights = {
            'VAPI-FA': 1.0,    # Institutional flow - very important
            'GIB': 0.9,        # Gamma risk - critical for options
            'VRI 2.0': 0.8,    # Volatility - key for timing
            'AOFM': 0.7,       # Aggressive flow - momentum indicator
            'LWPAI': 0.6,      # Custom momentum - proprietary edge
            'DWFD': 0.5        # Smart money - confirmation signal
        }
        
        # Calculate significance scores
        significance_scores = {}
        for metric, value in metrics.items():
            if metric in importance_weights:
                # Significance = |value| * importance * (1 + extreme_bonus)
                abs_value = abs(value)
                importance = importance_weights[metric]
                
                # Bonus for extreme readings (>2.0 gets extra weight)
                extreme_bonus = max(0, (abs_value - 2.0) / 2.0)
                
                significance = abs_value * importance * (1 + extreme_bonus)
                significance_scores[metric] = significance
        
        # Find metric with highest significance
        if significance_scores:
            focus_metric = max(significance_scores.items(), key=lambda x: x[1])
            return focus_metric[0], focus_metric[1]
        else:
            return "VAPI-FA", 0.0  # Default fallback
    
    def calculate_shape_deformation(self, metrics, focus_metric):
        """
        SHAPE DEFORMATION FORMULA
        ========================
        
        Deformation intensity = focus_significance / max_possible_significance
        
        Bulge factor = 1.0 + (deformation_intensity * max_bulge_multiplier)
        Compress factor = 1.0 - (deformation_intensity * max_compress_multiplier)
        
        Focus metric gets bulge_factor, others get compress_factor
        """
        focus_value = abs(metrics.get(focus_metric, 0))
        
        # Calculate deformation intensity (0 to 1)
        max_expected_value = 3.0  # Assuming metrics typically range -3 to +3
        deformation_intensity = min(focus_value / max_expected_value, 1.0)
        
        # Deformation parameters
        max_bulge_multiplier = 0.8    # Focus metric can be up to 1.8x normal size
        max_compress_multiplier = 0.4  # Other metrics can be compressed to 0.6x normal size
        
        # Calculate factors
        bulge_factor = 1.0 + (deformation_intensity * max_bulge_multiplier)
        compress_factor = 1.0 - (deformation_intensity * max_compress_multiplier)
        
        # Apply deformation to all metrics
        deformation_factors = {}
        for metric in metrics.keys():
            if metric == focus_metric:
                deformation_factors[metric] = bulge_factor
            else:
                deformation_factors[metric] = compress_factor
        
        return deformation_factors, deformation_intensity

def demonstrate_formulas():
    """Demonstrate the formulas with example data."""
    
    # Example market scenarios
    scenarios = {
        "Volatility Explosion": {
            'VAPI-FA': 1.8,
            'DWFD': -0.5,
            'VRI 2.0': 2.2,
            'GIB': 2.8,
            'LWPAI': 1.1,
            'AOFM': 1.6
        },
        "Premium Selling": {
            'VAPI-FA': 0.3,
            'DWFD': -0.2,
            'VRI 2.0': 0.6,
            'GIB': 0.8,
            'LWPAI': 0.4,
            'AOFM': 0.5
        },
        "Strong Trend": {
            'VAPI-FA': 2.1,
            'DWFD': 1.8,
            'VRI 2.0': 1.2,
            'GIB': 1.5,
            'LWPAI': 1.9,
            'AOFM': 1.7
        },
        "Danger Zone": {
            'VAPI-FA': 0.8,
            'DWFD': -1.2,
            'VRI 2.0': 2.5,
            'GIB': 3.2,
            'LWPAI': -0.6,
            'AOFM': 1.1
        }
    }
    
    engine = MOEDecisionEngine()
    
    print("🧮 MOE DECISION FORMULAS DEMONSTRATION")
    print("=" * 50)
    
    for scenario_name, metrics in scenarios.items():
        print(f"\n📊 SCENARIO: {scenario_name}")
        print("-" * 30)
        
        # Color decision
        color, score, reason = engine.determine_color_action(metrics)
        print(f"COLOR: {color} (Score: {score:.2f})")
        print(f"REASON: {reason}")
        
        # Shape focus
        focus_metric, significance = engine.calculate_shape_focus(metrics)
        print(f"FOCUS: {focus_metric} (Significance: {significance:.2f})")
        
        # Shape deformation
        deformation_factors, intensity = engine.calculate_shape_deformation(metrics, focus_metric)
        print(f"DEFORMATION INTENSITY: {intensity:.2f}")
        print(f"BULGE FACTOR: {deformation_factors[focus_metric]:.2f}")
        
        print(f"\n💡 TRANSLATION: {color} compass bulging towards {focus_metric}")
        print(f"   Meaning: {reason}")

if __name__ == "__main__":
    demonstrate_formulas()

