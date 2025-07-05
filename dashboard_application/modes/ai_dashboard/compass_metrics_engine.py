"""
LEGENDARY MARKET COMPASS METRICS ENGINE
======================================

Calculates all 12 dimensions for the compass from EOTS data.
Integrates with existing MOE formulas and confluence patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, confloat

from data_models import (
    ProcessedOptionsFlowV2_5,
    MarketMicrostructureV2_5,
    AdvancedOptionsMetricsV2_5,
    ProcessedDataBundleV2_5
)

class CompassDimension(BaseModel):
    """Model for a single compass dimension."""
    name: str
    label: str
    value: confloat(ge=0.0, le=1.0)  # Normalized 0-1
    raw_value: float
    category: str  # 'flow', 'volatility', 'momentum', 'liquidity'
    description: str
    color: str
    alert_threshold: float = 0.9
    
class CompassMetricsEngine:
    """
    Calculates all 12 dimensions of the Legendary Market Compass.
    Integrates with existing EOTS metrics and adds advanced calculations.
    """
    
    def __init__(self):
        # Dimension configurations
        self.dimensions_config = {
            # Core Dimensions (Inner Ring)
            'VAPI_FA': {
                'label': 'VAPI-FA Intensity',
                'category': 'flow',
                'color': '#00ff00',
                'description': 'Volatility-Adjusted Premium Intensity with Flow Acceleration'
            },
            'DWFD': {
                'label': 'DWFD Smart Money',
                'category': 'flow', 
                'color': '#00ffff',
                'description': 'Delta-Weighted Flow Divergence (institutional detection)'
            },
            'TW_LAF': {
                'label': 'TW-LAF Conviction',
                'category': 'flow',
                'color': '#0080ff',
                'description': 'Time-Weighted Liquidity-Adjusted Flow'
            },
            'VRI_2': {
                'label': 'VRI 2.0 Risk',
                'category': 'volatility',
                'color': '#ff0000',
                'description': 'Volatility Risk Index 2.0'
            },
            'A_DAG': {
                'label': 'A-DAG Pressure',
                'category': 'volatility',
                'color': '#ff8000',
                'description': 'Adaptive Delta-Adjusted Gamma'
            },
            'GIB': {
                'label': 'GIB Imbalance',
                'category': 'volatility',
                'color': '#ffff00',
                'description': 'Gamma Imbalance Barometer'
            },
            
            # Advanced Dimensions (Outer Ring)
            'LWPAI': {
                'label': 'LWPAI Signal',
                'category': 'momentum',
                'color': '#80ff00',
                'description': 'Liquidity-Weighted Price Action Indicator'
            },
            'VABAI': {
                'label': 'VABAI Bias',
                'category': 'liquidity',
                'color': '#00ff80',
                'description': 'Volatility-Adjusted Bid/Ask Imbalance'
            },
            'AOFM': {
                'label': 'AOFM Momentum',
                'category': 'momentum',
                'color': '#ff00ff',
                'description': 'Aggressive Order Flow Momentum'
            },
            'LIDB': {
                'label': 'LIDB Direction',
                'category': 'liquidity',
                'color': '#8000ff',
                'description': 'Liquidity-Implied Directional Bias'
            },
            'SVR': {
                'label': 'SVR Efficiency',
                'category': 'liquidity',
                'color': '#ff0080',
                'description': 'Spread-to-Volatility Ratio'
            },
            'TPDLF': {
                'label': 'TPDLF Quality',
                'category': 'momentum',
                'color': '#00ffff',
                'description': 'Theoretical Price Deviation with Liquidity Filter'
            }
        }
        
        # Normalization parameters (calibrated from historical data)
        self.normalization_params = {
            'VAPI_FA': {'min': -3.0, 'max': 3.0},
            'DWFD': {'min': -2.0, 'max': 2.0},
            'TW_LAF': {'min': -2.5, 'max': 2.5},
            'VRI_2': {'min': 0.0, 'max': 3.0},
            'A_DAG': {'min': -3.0, 'max': 3.0},
            'GIB': {'min': -2.5, 'max': 2.5},
            'LWPAI': {'min': -2.0, 'max': 2.0},
            'VABAI': {'min': -1.5, 'max': 1.5},
            'AOFM': {'min': -3.0, 'max': 3.0},
            'LIDB': {'min': -2.0, 'max': 2.0},
            'SVR': {'min': 0.0, 'max': 2.0},
            'TPDLF': {'min': -1.5, 'max': 1.5}
        }
    
    def calculate_all_dimensions(
        self, 
        data_bundle: ProcessedDataBundleV2_5,
        advanced_metrics: Optional[AdvancedOptionsMetricsV2_5] = None
    ) -> List[CompassDimension]:
        """
        Calculate all 12 compass dimensions from EOTS data.
        Returns normalized values (0-1) for compass visualization.
        """
        dimensions = []
        
        # Extract base data
        flow_data = data_bundle.flow_analysis
        microstructure = data_bundle.market_microstructure
        
        # Core Dimensions
        dimensions.append(self._calculate_vapi_fa(flow_data, microstructure))
        dimensions.append(self._calculate_dwfd(flow_data, microstructure))
        dimensions.append(self._calculate_tw_laf(flow_data, microstructure))
        dimensions.append(self._calculate_vri_2(microstructure, advanced_metrics))
        dimensions.append(self._calculate_a_dag(flow_data, microstructure))
        dimensions.append(self._calculate_gib(microstructure, advanced_metrics))
        
        # Advanced Dimensions
        dimensions.append(self._calculate_lwpai(flow_data, microstructure))
        dimensions.append(self._calculate_vabai(microstructure))
        dimensions.append(self._calculate_aofm(flow_data))
        dimensions.append(self._calculate_lidb(microstructure))
        dimensions.append(self._calculate_svr(microstructure))
        dimensions.append(self._calculate_tpdlf(microstructure, flow_data))
        
        return dimensions
    
    def _normalize_value(self, value: float, metric_name: str) -> float:
        """Normalize raw value to 0-1 range for compass display."""
        params = self.normalization_params[metric_name]
        normalized = (value - params['min']) / (params['max'] - params['min'])
        return max(0.0, min(1.0, normalized))
    
    def _calculate_vapi_fa(
        self, 
        flow: ProcessedOptionsFlowV2_5, 
        micro: MarketMicrostructureV2_5
    ) -> CompassDimension:
        """Calculate VAPI-FA: Volatility-Adjusted Premium Intensity with Flow Acceleration"""
        # Use existing VAPI calculation if available
        if hasattr(flow, 'vapi_score'):
            raw_value = flow.vapi_score
        else:
            # Calculate from components
            premium_flow = flow.net_premium_flow
            volatility = micro.implied_volatility_stats.current_iv
            
            # Add flow acceleration component
            if hasattr(flow, 'flow_momentum'):
                acceleration = abs(flow.flow_momentum.rate_of_change)
            else:
                acceleration = 1.0
            
            raw_value = (premium_flow / volatility) * acceleration if volatility > 0 else 0
        
        return CompassDimension(
            name='VAPI_FA',
            label=self.dimensions_config['VAPI_FA']['label'],
            value=self._normalize_value(raw_value, 'VAPI_FA'),
            raw_value=raw_value,
            category=self.dimensions_config['VAPI_FA']['category'],
            description=self.dimensions_config['VAPI_FA']['description'],
            color=self.dimensions_config['VAPI_FA']['color']
        )
    
    def _calculate_dwfd(
        self, 
        flow: ProcessedOptionsFlowV2_5, 
        micro: MarketMicrostructureV2_5
    ) -> CompassDimension:
        """Calculate DWFD: Delta-Weighted Flow Divergence"""
        # Smart money detection through delta-weighted analysis
        call_delta_flow = sum(t.delta * t.size for t in flow.significant_trades 
                             if t.contract_type == 'CALL' and hasattr(t, 'delta'))
        put_delta_flow = sum(abs(t.delta) * t.size for t in flow.significant_trades 
                            if t.contract_type == 'PUT' and hasattr(t, 'delta'))
        
        total_delta_flow = call_delta_flow - put_delta_flow
        
        # Compare to regular flow for divergence
        regular_flow = flow.net_call_flow - flow.net_put_flow
        divergence = (total_delta_flow - regular_flow) / (abs(regular_flow) + 1)
        
        return CompassDimension(
            name='DWFD',
            label=self.dimensions_config['DWFD']['label'],
            value=self._normalize_value(divergence, 'DWFD'),
            raw_value=divergence,
            category=self.dimensions_config['DWFD']['category'],
            description=self.dimensions_config['DWFD']['description'],
            color=self.dimensions_config['DWFD']['color']
        )
    
    def _calculate_tw_laf(
        self, 
        flow: ProcessedOptionsFlowV2_5, 
        micro: MarketMicrostructureV2_5
    ) -> CompassDimension:
        """Calculate TW-LAF: Time-Weighted Liquidity-Adjusted Flow"""
        # Weight recent flow more heavily
        time_weights = np.exp(-np.linspace(0, 1, len(flow.flow_sequence)))
        weighted_flow = sum(f.net_flow * w for f, w in zip(flow.flow_sequence, time_weights))
        
        # Adjust for liquidity
        liquidity_factor = micro.liquidity_score if hasattr(micro, 'liquidity_score') else 1.0
        tw_laf = weighted_flow * liquidity_factor
        
        return CompassDimension(
            name='TW_LAF',
            label=self.dimensions_config['TW_LAF']['label'],
            value=self._normalize_value(tw_laf, 'TW_LAF'),
            raw_value=tw_laf,
            category=self.dimensions_config['TW_LAF']['category'],
            description=self.dimensions_config['TW_LAF']['description'],
            color=self.dimensions_config['TW_LAF']['color']
        )
    
    def _calculate_vri_2(
        self, 
        micro: MarketMicrostructureV2_5,
        advanced: Optional[AdvancedOptionsMetricsV2_5]
    ) -> CompassDimension:
        """Calculate VRI 2.0: Enhanced Volatility Risk Index"""
        # Start with IV
        current_iv = micro.implied_volatility_stats.current_iv
        
        # Add term structure component
        if hasattr(micro, 'iv_term_structure'):
            term_spread = micro.iv_term_structure.front_back_spread
        else:
            term_spread = 0
        
        # Add skew component
        if hasattr(micro, 'iv_surface'):
            skew = abs(micro.iv_surface.atm_skew)
        else:
            skew = 0
        
        # Combine components
        vri_2 = current_iv * (1 + abs(term_spread)) * (1 + skew/100)
        
        return CompassDimension(
            name='VRI_2',
            label=self.dimensions_config['VRI_2']['label'],
            value=self._normalize_value(vri_2, 'VRI_2'),
            raw_value=vri_2,
            category=self.dimensions_config['VRI_2']['category'],
            description=self.dimensions_config['VRI_2']['description'],
            color=self.dimensions_config['VRI_2']['color']
        )
    
    def _calculate_a_dag(
        self, 
        flow: ProcessedOptionsFlowV2_5, 
        micro: MarketMicrostructureV2_5
    ) -> CompassDimension:
        """Calculate A-DAG: Adaptive Delta-Adjusted Gamma"""
        # Get gamma exposure
        if hasattr(micro, 'greek_exposure'):
            gamma_exposure = micro.greek_exposure.net_gamma
        else:
            gamma_exposure = 0
        
        # Adjust for market conditions
        volatility_regime = micro.implied_volatility_stats.regime
        if volatility_regime == 'HIGH_VOL':
            adjustment_factor = 1.5
        elif volatility_regime == 'LOW_VOL':
            adjustment_factor = 0.7
        else:
            adjustment_factor = 1.0
        
        a_dag = gamma_exposure * adjustment_factor
        
        return CompassDimension(
            name='A_DAG',
            label=self.dimensions_config['A_DAG']['label'],
            value=self._normalize_value(a_dag, 'A_DAG'),
            raw_value=a_dag,
            category=self.dimensions_config['A_DAG']['category'],
            description=self.dimensions_config['A_DAG']['description'],
            color=self.dimensions_config['A_DAG']['color']
        )
    
    def _calculate_gib(
        self, 
        micro: MarketMicrostructureV2_5,
        advanced: Optional[AdvancedOptionsMetricsV2_5]
    ) -> CompassDimension:
        """Calculate GIB: Gamma Imbalance Barometer"""
        # Get call and put gamma
        if hasattr(micro, 'greek_exposure'):
            call_gamma = micro.greek_exposure.call_gamma
            put_gamma = abs(micro.greek_exposure.put_gamma)
            
            # Calculate imbalance
            total_gamma = call_gamma + put_gamma
            if total_gamma > 0:
                gib = (call_gamma - put_gamma) / total_gamma
            else:
                gib = 0
        else:
            gib = 0
        
        return CompassDimension(
            name='GIB',
            label=self.dimensions_config['GIB']['label'],
            value=self._normalize_value(gib, 'GIB'),
            raw_value=gib,
            category=self.dimensions_config['GIB']['category'],
            description=self.dimensions_config['GIB']['description'],
            color=self.dimensions_config['GIB']['color']
        )
    
    def _calculate_lwpai(
        self, 
        flow: ProcessedOptionsFlowV2_5, 
        micro: MarketMicrostructureV2_5
    ) -> CompassDimension:
        """Calculate LWPAI: Liquidity-Weighted Price Action Indicator"""
        # Price momentum weighted by liquidity
        if hasattr(micro, 'price_action'):
            price_momentum = micro.price_action.momentum
            liquidity = micro.liquidity_score if hasattr(micro, 'liquidity_score') else 1.0
            lwpai = price_momentum * np.sqrt(liquidity)
        else:
            lwpai = 0
        
        return CompassDimension(
            name='LWPAI',
            label=self.dimensions_config['LWPAI']['label'],
            value=self._normalize_value(lwpai, 'LWPAI'),
            raw_value=lwpai,
            category=self.dimensions_config['LWPAI']['category'],
            description=self.dimensions_config['LWPAI']['description'],
            color=self.dimensions_config['LWPAI']['color']
        )
    
    def _calculate_vabai(self, micro: MarketMicrostructureV2_5) -> CompassDimension:
        """Calculate VABAI: Volatility-Adjusted Bid/Ask Imbalance"""
        # Bid/ask imbalance adjusted for volatility
        if hasattr(micro, 'microstructure_stats'):
            bid_volume = micro.microstructure_stats.bid_volume
            ask_volume = micro.microstructure_stats.ask_volume
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
                volatility = micro.implied_volatility_stats.current_iv
                vabai = imbalance / (volatility + 0.1)  # Avoid division by zero
            else:
                vabai = 0
        else:
            vabai = 0
        
        return CompassDimension(
            name='VABAI',
            label=self.dimensions_config['VABAI']['label'],
            value=self._normalize_value(vabai, 'VABAI'),
            raw_value=vabai,
            category=self.dimensions_config['VABAI']['category'],
            description=self.dimensions_config['VABAI']['description'],
            color=self.dimensions_config['VABAI']['color']
        )
    
    def _calculate_aofm(self, flow: ProcessedOptionsFlowV2_5) -> CompassDimension:
        """Calculate AOFM: Aggressive Order Flow Momentum"""
        # Focus on large, aggressive orders
        aggressive_trades = [t for t in flow.significant_trades 
                           if t.size > flow.block_trade_threshold * 0.5]
        
        aggressive_momentum = sum(t.size * (1 if t.side == 'BUY' else -1) 
                                for t in aggressive_trades)
        
        # Normalize by total flow
        total_flow = flow.total_volume if flow.total_volume > 0 else 1
        aofm = aggressive_momentum / total_flow
        
        return CompassDimension(
            name='AOFM',
            label=self.dimensions_config['AOFM']['label'],
            value=self._normalize_value(aofm, 'AOFM'),
            raw_value=aofm,
            category=self.dimensions_config['AOFM']['category'],
            description=self.dimensions_config['AOFM']['description'],
            color=self.dimensions_config['AOFM']['color']
        )
    
    def _calculate_lidb(self, micro: MarketMicrostructureV2_5) -> CompassDimension:
        """Calculate LIDB: Liquidity-Implied Directional Bias"""
        # Analyze where liquidity is concentrated
        if hasattr(micro, 'market_depth'):
            bid_liquidity = sum(micro.market_depth.bids.values())
            ask_liquidity = sum(micro.market_depth.asks.values())
            total_liquidity = bid_liquidity + ask_liquidity
            
            if total_liquidity > 0:
                lidb = (bid_liquidity - ask_liquidity) / total_liquidity
            else:
                lidb = 0
        else:
            lidb = 0
        
        return CompassDimension(
            name='LIDB',
            label=self.dimensions_config['LIDB']['label'],
            value=self._normalize_value(lidb, 'LIDB'),
            raw_value=lidb,
            category=self.dimensions_config['LIDB']['category'],
            description=self.dimensions_config['LIDB']['description'],
            color=self.dimensions_config['LIDB']['color']
        )
    
    def _calculate_svr(self, micro: MarketMicrostructureV2_5) -> CompassDimension:
        """Calculate SVR: Spread-to-Volatility Ratio"""
        # Efficiency metric: tighter spreads relative to volatility = better
        if hasattr(micro, 'spread_analysis'):
            avg_spread = micro.spread_analysis.average_spread_percent
            volatility = micro.implied_volatility_stats.current_iv
            
            if volatility > 0:
                svr = avg_spread / volatility
            else:
                svr = avg_spread
        else:
            svr = 1.0  # Neutral value
        
        return CompassDimension(
            name='SVR',
            label=self.dimensions_config['SVR']['label'],
            value=self._normalize_value(svr, 'SVR'),
            raw_value=svr,
            category=self.dimensions_config['SVR']['category'],
            description=self.dimensions_config['SVR']['description'],
            color=self.dimensions_config['SVR']['color']
        )
    
    def _calculate_tpdlf(
        self, 
        micro: MarketMicrostructureV2_5, 
        flow: ProcessedOptionsFlowV2_5
    ) -> CompassDimension:
        """Calculate TPDLF: Theoretical Price Deviation with Liquidity Filter"""
        # How far current price deviates from theoretical value
        if hasattr(micro, 'pricing_analysis'):
            theoretical_price = micro.pricing_analysis.theoretical_value
            current_price = micro.pricing_analysis.current_price
            
            if theoretical_price > 0:
                deviation = (current_price - theoretical_price) / theoretical_price
                
                # Filter by liquidity quality
                liquidity_quality = micro.liquidity_score if hasattr(micro, 'liquidity_score') else 1.0
                tpdlf = deviation * liquidity_quality
            else:
                tpdlf = 0
        else:
            tpdlf = 0
        
        return CompassDimension(
            name='TPDLF',
            label=self.dimensions_config['TPDLF']['label'],
            value=self._normalize_value(tpdlf, 'TPDLF'),
            raw_value=tpdlf,
            category=self.dimensions_config['TPDLF']['category'],
            description=self.dimensions_config['TPDLF']['description'],
            color=self.dimensions_config['TPDLF']['color']
        )
    
    def detect_confluence_patterns(
        self, 
        dimensions: List[CompassDimension]
    ) -> List[Dict[str, any]]:
        """
        Detect confluence patterns across dimensions.
        Integrates with confluence_formulas.py patterns.
        """
        patterns = []
        
        # Convert to dict for easier access
        dim_dict = {d.name: d for d in dimensions}
        
        # Momentum Explosion: VAPI-FA + A-DAG both > 0.8
        if dim_dict['VAPI_FA'].value > 0.8 and dim_dict['A_DAG'].value > 0.8:
            patterns.append({
                'name': 'MOMENTUM_EXPLOSION',
                'strength': (dim_dict['VAPI_FA'].value + dim_dict['A_DAG'].value) / 2,
                'description': 'Strong momentum with gamma acceleration',
                'action': 'Aggressive trend following opportunity'
            })
        
        # Volatility Squeeze: Low VRI + High GIB + AOFM
        if (dim_dict['VRI_2'].value < 0.3 and 
            abs(dim_dict['GIB'].value) > 0.7 and 
            dim_dict['AOFM'].value > 0.6):
            patterns.append({
                'name': 'VOLATILITY_SQUEEZE',
                'strength': dim_dict['GIB'].value,
                'description': 'Low volatility with gamma build-up',
                'action': 'Breakout imminent - position for volatility expansion'
            })
        
        # Smart Money Divergence: DWFD opposite to AOFM
        if (dim_dict['DWFD'].value > 0.7 and dim_dict['AOFM'].value < 0.3) or \
           (dim_dict['DWFD'].value < 0.3 and dim_dict['AOFM'].value > 0.7):
            patterns.append({
                'name': 'SMART_MONEY_DIVERGENCE',
                'strength': abs(dim_dict['DWFD'].value - dim_dict['AOFM'].value),
                'description': 'Institutional flow diverging from retail',
                'action': 'Follow the smart money (DWFD direction)'
            })
        
        return patterns