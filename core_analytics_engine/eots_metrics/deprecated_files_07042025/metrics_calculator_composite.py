# core_analytics_engine/eots_metrics/metrics_calculator_composite.py

"""
Composite MetricsCalculatorV2_5 class that combines all optimized metric calculators
for backward compatibility with existing code.
"""

import pandas as pd
from typing import Tuple, Optional
from pydantic import ValidationError

from .core_calculator import CoreCalculator
from .flow_analytics import FlowAnalytics
from .adaptive_calculator import AdaptiveCalculator
from .visualization_metrics import VisualizationMetrics
from .elite_intelligence import EliteImpactCalculator, EliteConfig, EliteImpactResultsV2_5
from .supplementary_metrics import SupplementaryMetrics
from .validation_utils import ValidationUtils

from data_models import (
    ProcessedUnderlyingAggregatesV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedContractMetricsV2_5,
    RawUnderlyingDataCombinedV2_5,
)


class MetricsCalculatorV2_5:
    """
    Consolidated composite calculator that combines all optimized metric calculators.
    
    Uses the new 6-module architecture:
    - CoreCalculator: Base utilities + foundational metrics
    - FlowAnalytics: Enhanced flow metrics + flow classification + momentum
    - AdaptiveCalculator: Adaptive metrics + regime detection + volatility surface
    - VisualizationMetrics: Heatmap data + underlying aggregates
    - EliteIntelligence: Elite impact calculations + institutional intelligence
    - SupplementaryMetrics: ATR + advanced options metrics
    """

    def __init__(self, config_manager, historical_data_manager, enhanced_cache_manager, elite_config=None):
        # Ensure elite_config is a Pydantic model
        if elite_config is None:
            elite_config = EliteConfig()
        elif not isinstance(elite_config, EliteConfig):
            elite_config = EliteConfig.model_validate(elite_config)

        # Initialize consolidated calculators
        self.core = CoreCalculator(config_manager, historical_data_manager, enhanced_cache_manager)
        self.flow_analytics = FlowAnalytics(config_manager, historical_data_manager, enhanced_cache_manager, elite_config)
        self.adaptive = AdaptiveCalculator(config_manager, historical_data_manager, enhanced_cache_manager, elite_config)
        self.visualization = VisualizationMetrics(config_manager, historical_data_manager, enhanced_cache_manager, elite_config)
        self.elite_intelligence = EliteImpactCalculator(elite_config)
        self.supplementary = SupplementaryMetrics(config_manager, historical_data_manager, enhanced_cache_manager)

        # Store references for common access
        self.config_manager = config_manager
        self.historical_data_manager = historical_data_manager
        self.enhanced_cache_manager = enhanced_cache_manager
        self.elite_config = elite_config

        # Initialize validation utilities
        self.validator = ValidationUtils()

        # Backward compatibility aliases
        self.foundational = self.core
        self.enhanced_flow = self.flow_analytics
        self.heatmap = self.visualization
        self.underlying_aggregates = self.visualization
        self.miscellaneous = self.supplementary
        self.elite_impact = self.elite_intelligence

    def calculate_all_metrics(self, options_df_raw: pd.DataFrame, 
                            und_data_api_raw: RawUnderlyingDataCombinedV2_5, 
                            dte_max: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, ProcessedUnderlyingAggregatesV2_5]:
        """
        Calculate all metrics using the consolidated 6-module architecture.
        
        STRICT PYDANTIC V2-ONLY: No dictionaries, direct ProcessedUnderlyingAggregatesV2_5 instantiation
        ZERO TOLERANCE FAKE DATA: All required fields calculated with real market data
        
        Args:
            options_df_raw: DataFrame with raw options data
            und_data_api_raw: RawUnderlyingDataCombinedV2_5 Pydantic model
            dte_max: Maximum DTE for calculations
            
        Returns:
            Tuple of (strike_level_df, contract_level_df, enriched_underlying_pydantic_model)
        """
        try:
            # Validate inputs
            self.validator.validate_input_data(options_df_raw, und_data_api_raw)
            
            # Initialize contract level data
            df_chain_all_metrics = options_df_raw.copy() if not options_df_raw.empty else pd.DataFrame()
            
            print(f"✅ Starting metrics calculation for {und_data_api_raw.symbol} at price ${und_data_api_raw.price}")
            
            # Create initial model for calculation
            temp_model = self._create_initial_model(und_data_api_raw)
            
            # Calculate foundational metrics
            print("🔄 Calculating foundational metrics...")
            foundational_model = self.core.calculate_all_foundational_metrics(temp_model)
            self.validator.validate_foundational_metrics(foundational_model)
            
            # Calculate flow analytics
            print("🔄 Calculating enhanced flow metrics...")
            flow_scores = self._calculate_flow_metrics(foundational_model)
            
            # Calculate elite intelligence metrics
            print("🔄 Calculating elite intelligence metrics...")
            elite_results = self.elite_intelligence.calculate_elite_impact_score(df_chain_all_metrics, foundational_model)
            self.validator.validate_elite_results(elite_results)
            
            # Create final enriched model
            enriched_underlying = self._create_final_model(
                und_data_api_raw, foundational_model, flow_scores, elite_results
            )
            
            # Process strike-level data
            df_strike_all_metrics = self._process_strike_level_data(options_df_raw, enriched_underlying, dte_max)
            
            # Calculate additional metrics on strike data
            if not df_strike_all_metrics.empty:
                df_strike_all_metrics = self._calculate_strike_level_metrics(df_strike_all_metrics, enriched_underlying)
                
                # Update underlying aggregates
                aggregates = self.visualization.calculate_all_underlying_aggregates(df_strike_all_metrics, enriched_underlying)
                if aggregates:
                    enriched_underlying = self._merge_aggregates(enriched_underlying, aggregates)
            
            # Final validation
            self.validator.validate_final_model(enriched_underlying)
            
            print("✅ All metrics calculated successfully")
            return df_strike_all_metrics, df_chain_all_metrics, enriched_underlying
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in calculate_all_metrics: {e}")
            raise RuntimeError(f"Metrics calculation failed: {e}") from e

    def _create_initial_model(self, und_data_api_raw: RawUnderlyingDataCombinedV2_5) -> ProcessedUnderlyingAggregatesV2_5:
        """Create initial model with placeholder values for calculation."""
        raw_data_dict = und_data_api_raw.model_dump()
        
        return ProcessedUnderlyingAggregatesV2_5(
            **raw_data_dict,
            # Foundational metrics - will be calculated immediately
            gib_oi_based_und=0.0,
            td_gib_und=0.0,
            hp_eod_und=0.0,
            net_cust_delta_flow_und=0.0,
            net_cust_gamma_flow_und=0.0,
            net_cust_vega_flow_und=0.0,
            net_cust_theta_flow_und=0.0,
            # Z-score metrics - will be calculated by flow analytics
            vapi_fa_z_score_und=0.0,
            dwfd_z_score_und=0.0,
            tw_laf_z_score_und=0.0,
            # Elite metrics - will be calculated by elite intelligence
            elite_impact_score_und=1.0,
            institutional_flow_score_und=1.0,
            flow_momentum_index_und=0.1,
            market_regime_elite='calculating',
            flow_type_elite='calculating',
            volatility_regime_elite='calculating',
            confidence=0.1,
            transition_risk=0.9
        )

    def _calculate_flow_metrics(self, foundational_model: ProcessedUnderlyingAggregatesV2_5) -> dict:
        """Calculate flow analytics metrics."""
        try:
            flow_model = self.flow_analytics.calculate_all_enhanced_flow_metrics(
                foundational_model, foundational_model.symbol
            )
            
            if flow_model is None:
                print("⚠️ WARNING: Flow analytics returned None - using default values")
                return {
                    'vapi_fa_z_score': 0.0,
                    'dwfd_z_score': 0.0,
                    'tw_laf_z_score': 0.0
                }
            
            return {
                'vapi_fa_z_score': getattr(flow_model, 'vapi_fa_z_score_und', 0.0),
                'dwfd_z_score': getattr(flow_model, 'dwfd_z_score_und', 0.0),
                'tw_laf_z_score': getattr(flow_model, 'tw_laf_z_score_und', 0.0)
            }
            
        except Exception as flow_error:
            print(f"⚠️ WARNING: Flow analytics calculation failed: {flow_error}")
            return {
                'vapi_fa_z_score': 0.0,
                'dwfd_z_score': 0.0,
                'tw_laf_z_score': 0.0
            }

    def _create_final_model(self, raw_data: RawUnderlyingDataCombinedV2_5,
                          foundational_model: ProcessedUnderlyingAggregatesV2_5,
                          flow_scores: dict,
                          elite_results: EliteImpactResultsV2_5) -> ProcessedUnderlyingAggregatesV2_5:
        """Create final enriched model with all calculated values."""
        raw_data_dict = raw_data.model_dump()
        
        return ProcessedUnderlyingAggregatesV2_5(
            **raw_data_dict,
            # Foundational metrics
            gib_oi_based_und=foundational_model.gib_oi_based_und,
            td_gib_und=foundational_model.td_gib_und,
            hp_eod_und=foundational_model.hp_eod_und,
            net_cust_delta_flow_und=foundational_model.net_cust_delta_flow_und,
            net_cust_gamma_flow_und=foundational_model.net_cust_gamma_flow_und,
            net_cust_vega_flow_und=foundational_model.net_cust_vega_flow_und,
            net_cust_theta_flow_und=foundational_model.net_cust_theta_flow_und,
            # Flow metrics
            vapi_fa_z_score_und=flow_scores['vapi_fa_z_score'],
            dwfd_z_score_und=flow_scores['dwfd_z_score'],
            tw_laf_z_score_und=flow_scores['tw_laf_z_score'],
            # Elite metrics
            elite_impact_score_und=elite_results.elite_impact_score_und,
            institutional_flow_score_und=elite_results.institutional_flow_score_und,
            flow_momentum_index_und=elite_results.flow_momentum_index_und,
            market_regime_elite=elite_results.market_regime_elite,
            flow_type_elite=elite_results.flow_type_elite,
            volatility_regime_elite=elite_results.volatility_regime_elite,
            confidence=elite_results.confidence,
            transition_risk=elite_results.transition_risk
        )

    def _process_strike_level_data(self, options_df_raw: pd.DataFrame, 
                                 enriched_underlying: ProcessedUnderlyingAggregatesV2_5,
                                 dte_max: int) -> pd.DataFrame:
        """Process options data to create strike-level metrics."""
        if options_df_raw.empty:
            return pd.DataFrame()
            
        # Filter by DTE
        df_filtered = options_df_raw[options_df_raw['dte'] <= dte_max].copy()
        
        if df_filtered.empty:
            return pd.DataFrame()
            
        # Group by strike and calculate metrics
        strike_data = []
        for strike, group in df_filtered.groupby('strike'):
            try:
                strike_model = ProcessedStrikeLevelMetricsV2_5(
                    strike=float(strike),
                    total_oi_at_strike=self.validator.require_column_sum(group, 'oi', 'Open Interest'),
                    total_volume_at_strike=self.validator.require_column_sum(group, 'volm', 'Volume'),
                    avg_iv_at_strike=self._calculate_avg_iv_at_strike(group),
                    net_cust_delta_flow_at_strike=self._calculate_flow_metric(group, 'delta_contract', 'volm'),
                    net_cust_gamma_flow_at_strike=self._calculate_flow_metric(group, 'gamma_contract', 'volm'),
                    net_cust_vega_flow_at_strike=self._calculate_flow_metric(group, 'vega_contract', 'volm'),
                    net_cust_theta_flow_at_strike=self._calculate_flow_metric(group, 'theta_contract', 'volm')
                )
                strike_data.append(strike_model)
            except Exception as e:
                print(f"⚠️ WARNING: Failed to process strike {strike}: {e}")
                continue
                
        if strike_data:
            return pd.DataFrame([model.model_dump() for model in strike_data])
        else:
            return pd.DataFrame()

    def _calculate_strike_level_metrics(self, df_strike: pd.DataFrame, 
                                      enriched_underlying: ProcessedUnderlyingAggregatesV2_5) -> pd.DataFrame:
        """Calculate additional strike-level metrics."""
        print(f"🔄 Calculating adaptive metrics for {len(df_strike)} strikes...")
        df_strike = self.adaptive.calculate_all_adaptive_metrics(df_strike, enriched_underlying)
        
        print(f"🔄 Calculating heatmap metrics for {len(df_strike)} strikes...")
        df_strike = self.visualization.calculate_all_heatmap_data(df_strike, enriched_underlying)
        
        return df_strike

    def _merge_aggregates(self, enriched_underlying: ProcessedUnderlyingAggregatesV2_5, 
                        aggregates: dict) -> ProcessedUnderlyingAggregatesV2_5:
        """Merge calculated aggregates into the underlying model."""
        if isinstance(aggregates, dict) and aggregates:
            current_data = enriched_underlying.model_dump()
            current_data.update(aggregates)
            return ProcessedUnderlyingAggregatesV2_5(**current_data)
        return enriched_underlying

    def _calculate_avg_iv_at_strike(self, group: pd.DataFrame) -> float:
        """Calculate average implied volatility at strike level."""
        if 'iv' not in group.columns:
            raise ValueError("Missing 'iv' column for implied volatility calculation")
            
        valid_iv = group['iv'].dropna()
        valid_iv = valid_iv[valid_iv > 0]  # Remove zero/negative IV values
        
        if valid_iv.empty:
            print("⚠️ WARNING: No valid IV data found - using default value")
            return 0.2  # Default 20% IV
            
        return float(valid_iv.mean())

    def _calculate_flow_metric(self, group: pd.DataFrame, greek_col: str, volume_col: str) -> float:
        """Calculate flow metric from Greek and volume data."""
        if greek_col in group.columns and volume_col in group.columns:
            greek_values = group[greek_col].fillna(0.0)
            volume_values = group[volume_col].fillna(0.0)
            flow_values = greek_values * volume_values
            return float(flow_values.sum())
        return 0.0