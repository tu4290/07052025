# core_analytics_engine/market_regime_engine_v2_5.py
# EOTS v2.5 - S-GRADE PRODUCTION HARDENED & OPTIMIZED ARTIFACT

import logging
import re
from typing import Any, Dict, Optional
from datetime import datetime

from pydantic import ValidationError

import numpy as np
import pandas as pd

from data_models import ProcessedUnderlyingAggregatesV2_5, ProcessedDataBundleV2_5
from data_models.configuration_models import MarketRegimeEngineSettings
from data_models.elite_intelligence import MarketRegime, FlowType, EliteImpactColumns, EliteConfig
from utils.config_manager_v2_5 import ConfigManagerV2_5

logger = logging.getLogger(__name__)

class MarketRegimeEngineV2_5:
    """Determines market regime based on configuration-driven rules."""
    
    def __init__(self, config_manager: ConfigManagerV2_5, elite_config: EliteConfig, tradier_fetcher=None, convex_fetcher=None, enhanced_cache=None):
        """
        Initialize the MarketRegimeEngineV2_5.

        Args:
            config_manager: The system's configuration manager
            elite_config: The EliteConfig instance for elite metric settings
            tradier_fetcher: Optional Tradier data fetcher for VIX data
            convex_fetcher: Optional ConvexValue data fetcher for VIX data
            enhanced_cache: Optional enhanced cache manager for accessing intraday VIX data
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.elite_config = elite_config
        self.tradier_fetcher = tradier_fetcher
        self.convex_fetcher = convex_fetcher
        self.enhanced_cache = enhanced_cache
        
        # Get settings from config manager
        raw_settings = self.config_manager.get_setting("market_regime_engine_settings")
        print(f"Raw market_regime_engine_settings from config: {raw_settings}")
        print(f"Type of raw_settings: {type(raw_settings)}")
        self.logger.debug(f"Raw market_regime_engine_settings from config: {raw_settings}")
        self.logger.debug(f"Type of raw_settings: {type(raw_settings)}")

        # raw_settings is already an instance of MarketRegimeEngineSettings, so assign directly without model_validate
        self.settings = raw_settings
            
        # Access regime rules from the Pydantic model
        self.regime_rules = self.settings.regime_rules
        self.evaluation_order = self.settings.regime_evaluation_order
        self.default_regime = self.settings.default_regime
        self.start_time = datetime.now() # For processing time calculation
        
        self.logger.info("MarketRegimeEngineV2_5 initialized.")
        
    async def determine_regime(self, und_data: ProcessedUnderlyingAggregatesV2_5, df_strike: pd.DataFrame, df_chain: pd.DataFrame) -> str:
        """
        Determines the current market regime by evaluating rules in the specified order.
        """
        if not all(isinstance(arg, (ProcessedUnderlyingAggregatesV2_5, pd.DataFrame)) for arg in [und_data, df_strike, df_chain]):
            self.logger.error("Invalid input types to determine_market_regime. Falling back to default.")
            return self.default_regime

        self.logger.debug(f"Determining market regime for {und_data.symbol}...")
        
        # Dynamic thresholds are now expected to be an attribute of the und_data object
        dynamic_thresholds = getattr(und_data, 'dynamic_thresholds', {})

        for regime_name in self.evaluation_order:
            rule_block = getattr(self.regime_rules, regime_name, None)
            if not rule_block:
                self.logger.warning(f"Skipping missing rule block for regime '{regime_name}'.")
                continue

            try:
                if await self._evaluate_condition_block(rule_block, und_data, df_strike, df_chain, dynamic_thresholds):
                    self.logger.info(f"Market regime matched: {regime_name}")
                    return regime_name
            except Exception as e:
                self.logger.error(f"Unhandled exception during evaluation of regime '{regime_name}': {e}", exc_info=True)
                continue

        self.logger.info(f"No regime rules matched. Falling back to default: {self.default_regime}")
        return self.default_regime

    async def determine_market_regime(self, data_bundle: ProcessedDataBundleV2_5) -> MarketRegime:
        """
        Determines the current market regime based on the processed data bundle.
        This is the primary entry point for the Market Regime Engine.
        """
        und_data = data_bundle.underlying_data_enriched
        df_strike = pd.DataFrame([s.model_dump() for s in data_bundle.strike_level_data_with_metrics])
        df_chain = pd.DataFrame([c.model_dump() for c in data_bundle.options_data_with_metrics])

        self.logger.debug(f"Determining market regime for {und_data.symbol} using elite metrics...")

        logger.info(f"[RegimeEngine] elite_config type: {type(self.elite_config)}, value: {repr(self.elite_config)}")
        logger.info(f"[RegimeEngine] und_data type: {type(und_data)}, value: {repr(und_data)[:500]}")
        logger.info(f"[RegimeEngine] About to access self.elite_config.enable_elite_regime_detection")

        # Prioritize elite-driven regime if enabled and available
        if self.elite_config.enable_elite_regime_detection and und_data.market_regime_elite:
            # CRITICAL FIX: market_regime_elite is already a string (enum value), not an enum object
            self.logger.info(f"Elite market regime detected: {und_data.market_regime_elite}")
            return und_data.market_regime_elite

        # Fallback to rule-based determination if elite is not enabled or available
        self.logger.debug("Falling back to rule-based market regime determination.")
        
        # Dynamic thresholds are now expected to be an attribute of the und_data object
        dynamic_thresholds = getattr(und_data, 'dynamic_thresholds', {})

        for regime_name in self.evaluation_order:
            rule_block = getattr(self.regime_rules, regime_name, None)
            if not rule_block:
                self.logger.warning(f"Skipping missing rule block for regime '{regime_name}'.")
                continue

            try:
                # Convert Pydantic model to dictionary for evaluation
                rule_block_dict = rule_block.model_dump() if hasattr(rule_block, 'model_dump') else rule_block
                if await self._evaluate_condition_block(rule_block_dict, und_data, df_strike, df_chain, dynamic_thresholds):
                    self.logger.info(f"Market regime matched: {regime_name}")
                    return MarketRegime(regime_name) # Convert string to Enum
            except Exception as e:
                self.logger.error(f"Unhandled exception during evaluation of regime '{regime_name}': {e}", exc_info=True)
                continue

        self.logger.info(f"No regime rules matched. Falling back to default: {self.default_regime}")
        # Find the enum member by value
        for regime in MarketRegime:
            if regime.value == self.default_regime:
                return regime
        # If not found, return the unclear regime as fallback
        return MarketRegime.REGIME_UNCLEAR_OR_TRANSITIONING

    async def _evaluate_condition_block(self, block: Dict, und_data: ProcessedUnderlyingAggregatesV2_5, df_strike: pd.DataFrame, df_chain: pd.DataFrame, dynamic_thresholds: Dict) -> bool:
        """A condition block is TRUE only if ALL its conditions are TRUE (AND logic)."""
        thresholds = dynamic_thresholds or {}
        for key, target_value in block.items():
            if key == "_any_of":
                if not any([await self._evaluate_condition_block(sub_block, und_data, df_strike, df_chain, thresholds) for sub_block in target_value]):
                    return False
                continue

            # Handle special boolean rules that don't need operators
            boolean_rules = {
                'A_MSPI_flips_negative_at_key_support',
                'A_SSI_very_low',
                'VRI_2_0_trend_down',
                'is_SPX_0DTE_Friday_eq',
                'is_FOMC_eve_eq'
            }
            
            # Handle special threshold rules that default to 'gt' operator
            threshold_rules = {
                'vix_threshold',
                'flow_alignment_threshold',
                'confidence_threshold'
            }
            
            if key in boolean_rules:
                # For boolean rules, treat as equality check
                metric_name = key
                operator = 'eq'
            elif key in threshold_rules:
                # For threshold rules, default to greater than
                metric_name = key
                operator = 'gt'
            else:
                # Extract operator from the end of the key (e.g., metric_name_gt)
                match = re.match(r"^(.*)_(gt|lt|lte|gte|eq|neq|abs_gt|abs_lt|in_list|contains)$", key)
                if not match:
                    self.logger.warning(f"Rule key '{key}' has no valid operator. Condition fails.")
                    return False
                
                metric_name = match.group(1)
                operator = match.group(2)

            context = getattr(und_data, 'ticker_context_dict_v2_5', {}) or {}
            if metric_name in context:
                if not self._check_special_condition(metric_name, target_value, context, operator):
                    return False
                continue

            resolved_target = thresholds.get(target_value.split(':')[1]) if isinstance(target_value, str) and target_value.startswith("dynamic_threshold:") else target_value
            
            # Special handling for VRI_2_0_trend_down
            if key == 'VRI_2_0_trend_down':
                vri_value = await self._resolve_metric_value('VRI_2_0_trend_down', None, und_data, df_strike, df_chain)
                if vri_value is None:
                    self.logger.debug(f"Metric '{key}' resolved to None. Condition fails.")
                    return False
                # Check if VRI indicates downward trend (negative value)
                actual_value = vri_value < 0
            else:
                actual_value = await self._resolve_metric_value(metric_name, None, und_data, df_strike, df_chain) # Selector is handled within _resolve_metric_value
            
            if actual_value is None:
                self.logger.debug(f"Metric '{key}' resolved to None. Condition fails.")
                return False 
            
            if not self._perform_comparison(actual_value, operator, resolved_target):
                return False
        return True

    async def _resolve_metric_value(self, metric: str, selector: Optional[str], und_data: ProcessedUnderlyingAggregatesV2_5, df_strike: pd.DataFrame, df_chain: pd.DataFrame) -> Any:
        """Resolves a metric's value, handling complex selectors and aggregations."""
        try:
            # Create metric name mapping for regime rules to actual attributes
            metric_mapping = {
                # 0DTE Aggregates
                'vci_0dte_agg_gt': 'vci_0dte_agg',
                'vri_0dte_agg_roc_gt': 'vri_0dte_und_sum',  # Using available metric
                'vvr_0dte_agg_gt': 'vvr_0dte_und_avg',
                'vfi_0dte_lt': 'vfi_0dte_und_sum',
                'vri_0dte_und_sum_gt': 'vri_0dte_und_sum',
                'vfi_0dte_und_sum_gt': 'vfi_0dte_und_sum',
                
                # Flow and Pressure Metrics
                'HP_EOD_gt': 'hp_eod_und',
                'HP_EOD_lt': 'hp_eod_und',
                'DWFD_Und_gt': 'dwfd_z_score_und',
                'VAPI_FA_Und_gt': 'vapi_fa_z_score_und',
                'TW_LAF_Und_gt': 'tw_laf_z_score_und',
                
                # Adaptive Metrics
                'A_MSPI_flips_negative_at_key_support': 'a_mspi_und_summary_score',
                'A_SSI_very_low': 'a_ssi_und_avg',
                
                # Strike-level metrics (will be handled in df_strike section)
                'NVP_at_key_strike_gt': 'nvp_at_strike',
                
                # Trend and volatility thresholds (calculated values)
                'trend_threshold_abs_lt': 'trend_strength',
                'volatility_threshold_lt': 'hist_vol_20d',
                'volatility_threshold_gt': 'hist_vol_20d',
                
                # VIX threshold (special handling - will be resolved separately)
                'vix_threshold': None,  # Special case - requires VIX data lookup
                
                # VRI 2.0 trend (boolean check)
                'VRI_2_0_trend_down': 'vri_2_0_und_aggregate',
                
                # Special context flags
                'is_SPX_0DTE_Friday_eq': None,  # Context flag
                'is_FOMC_eve_eq': None,  # Context flag
            }
            
            # Special handling for VIX threshold - get VIX level directly from market data
            if metric == 'vix_threshold':
                try:
                    # Try to get VIX level from market data directly (synchronous version)
                    vix_data = self._get_market_data_sync('VIX')
                    if not vix_data.empty:
                        vix_level = vix_data['close'].iloc[-1]
                        self.logger.debug(f"Retrieved VIX level from market data: {vix_level}")
                        return vix_level
                except Exception as e:
                    self.logger.warning(f"Error fetching VIX data: {str(e)}")
                # Fallback: return a default VIX level if not available
                self.logger.warning("VIX level not found in market data, using default value of 20.0")
                return 20.0
            
            # Map the metric name if it exists in our mapping
            actual_metric = metric_mapping.get(metric, metric)
            
            # Handle special context flags that should come from ticker_context_dict_v2_5
            if actual_metric is None and hasattr(und_data, 'ticker_context_dict_v2_5') and und_data.ticker_context_dict_v2_5:
                context_value = getattr(und_data.ticker_context_dict_v2_5, metric, None)
                if context_value is not None:
                    return context_value
                # If not found in context, return None (will fail condition as expected)
                return None
            
            # First, try to get from und_data (which contains all aggregated metrics)
            value = getattr(und_data, actual_metric, None) if actual_metric else None
            if value is not None: # If found in und_data, return it
                return value

            # If not in und_data, check df_strike for strike-level metrics
            strike_metric = actual_metric if actual_metric else metric
            if strike_metric in df_strike.columns:
                if selector and selector.startswith('[AGG='):
                    agg_type = selector[5:-1]
                    if agg_type == 'sum':
                        return df_strike[strike_metric].sum()
                    elif agg_type == 'mean':
                        return df_strike[strike_metric].mean()
                    elif agg_type == 'max':
                        return df_strike[strike_metric].max()
                    elif agg_type == 'min':
                        return df_strike[strike_metric].min()
                    # Add more aggregation types as needed
                    return None
                elif selector and selector.startswith('[PERCENTILE='):
                    percentile = float(selector[12:-1])
                    return df_strike[strike_metric].quantile(percentile / 100.0)
                elif selector == '@ATM':
                    if und_data.price is None: return None
                    strike_diffs = pd.Series(df_strike['strike'] - und_data.price).abs()
                    target_strike_row = df_strike.loc[strike_diffs.idxmin()]
                    return target_strike_row[strike_metric]
                else:
                    # If no selector or unknown selector, return the series (or first value if appropriate)
                    return df_strike[strike_metric]
            
            # Fallback if not found
            return None
        except (AttributeError, KeyError, IndexError, ValueError, TypeError) as e:
            self.logger.warning(f"Could not resolve metric key '{metric}{selector or ''}': {e}", exc_info=True)
            return None
            
    def _check_special_condition(self, context_key: str, target_value: Any, context: Dict, operator: str) -> bool:
        """Data-driven evaluation of flags in the ticker context dictionary."""
        actual_value = context.get(context_key)
        if actual_value is None: return False
        return self._perform_comparison(actual_value, operator, target_value)

    def _perform_comparison(self, actual: Any, op: str, target: Any) -> bool:
        """Performs the actual comparison between a metric's value and its target."""
        try:
            if op == "gt": return actual > target
            if op == "lt": return actual < target
            if op == "lte": return actual <= target
            if op == "gte": return actual >= target
            if op == "eq": return actual == target
            if op == "neq": return actual != target
            if op == "abs_gt": return abs(actual) > target
            if op == "abs_lt": return abs(actual) < target
            if op == "in_list": return actual in target
            if op == "contains": return isinstance(actual, str) and target in actual
        except (TypeError, ValueError):
            # This can happen if comparing incompatible types, e.g., number and None
            return False
        return False

    def calculate_volatility_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate volatility regime score using VRI 2.0 and enhanced metrics"""
        try:
            # Extract VRI 2.0 base score
            vri_2_0 = market_data.vri_2_0_und or 0.0
            
            # Extract volatility metrics
            hist_vol = market_data.hist_vol_20d or 0.0
            impl_vol = market_data.impl_vol_atm or 0.0
            
            # Calculate regime score
            vol_ratio = impl_vol / max(hist_vol, 0.0001)
            vol_regime = np.clip(vri_2_0 * vol_ratio, -1.0, 1.0)
            
            return float(vol_regime)
            
        except Exception as e:
            self.logger.error(f"Volatility regime calculation failed: {e}")
            return 0.0
    
    def calculate_flow_intensity(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate flow intensity score using VFI and enhanced metrics"""
        try:
            # Extract flow metrics
            vfi_score = market_data.vfi_0dte_und_avg or 0.0
            vapi_fa = market_data.vapi_fa_z_score_und or 0.0
            dwfd = market_data.dwfd_z_score_und or 0.0
            
            # Calculate intensity score
            intensity_components = [
                vfi_score * 0.4,
                vapi_fa * 0.3,
                dwfd * 0.3
            ]
            
            flow_intensity = np.clip(sum(intensity_components), -1.0, 1.0)
            
            return float(flow_intensity)
            
        except Exception as e:
            self.logger.error(f"Flow intensity calculation failed: {e}")
            return 0.0
    
    def calculate_regime_stability(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate regime stability score"""
        try:
            # Extract stability metrics
            mspi = market_data.a_mspi_und_summary_score or 0.0
            trend_strength = market_data.trend_strength or 0.0
            
            # Calculate stability score
            stability_components = [
                mspi * 0.6,
                trend_strength * 0.4
            ]
            
            stability = np.clip(sum(stability_components), 0.0, 1.0)
            
            return float(stability)
            
        except Exception as e:
            self.logger.error(f"Regime stability calculation failed: {e}")
            return 0.0
    
    def calculate_transition_momentum(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate transition momentum score"""
        try:
            # Extract momentum metrics
            dag_total = market_data.a_dag_total_und or 0.0
            vapi_fa = market_data.vapi_fa_z_score_und or 0.0
            
            # Calculate momentum score
            momentum_components = [
                dag_total * 0.5,
                vapi_fa * 0.5
            ]
            
            momentum = np.clip(sum(momentum_components), -1.0, 1.0)
            
            return float(momentum)
            
        except Exception as e:
            self.logger.error(f"Transition momentum calculation failed: {e}")
            return 0.0
    
    def calculate_vri3_composite(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate VRI 3.0 composite score"""
        try:
            # Calculate components
            vol_regime = self.calculate_volatility_regime(market_data)
            flow_intensity = self.calculate_flow_intensity(market_data)
            stability = self.calculate_regime_stability(market_data)
            momentum = self.calculate_transition_momentum(market_data)
            
            # Calculate composite score
            component_weights = [0.3, 0.3, 0.2, 0.2]
            components = [vol_regime, flow_intensity, stability, momentum]
            
            composite = sum(w * c for w, c in zip(component_weights, components))
            
            return float(np.clip(composite, -1.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"VRI 3.0 composite calculation failed: {e}")
            return 0.0
    
    def calculate_confidence_level(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> float:
        """Calculate confidence level for the analysis"""
        try:
            # Extract quality metrics
            data_quality = 1.0  # Placeholder for data quality calculation
            signal_strength = abs(self.calculate_vri3_composite(market_data))
            
            # Calculate confidence
            confidence = min(data_quality * signal_strength, 1.0)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Confidence level calculation failed: {e}")
            return 0.0
    
    def calculate_regime_transition_probabilities(
        self,
        current_regime: str,
        vri_components: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate transition probabilities to other regimes"""
        try:
            # Extract components
            stability = vri_components.get('regime_stability_score', 0.0)
            momentum = vri_components.get('transition_momentum_score', 0.0)
            
            # Base transition probability
            base_prob = 1.0 - stability
            
            # Calculate directional probabilities
            if momentum > 0:
                up_prob = base_prob * (1.0 + momentum)
                down_prob = base_prob * (1.0 - momentum)
            else:
                up_prob = base_prob * (1.0 + momentum)
                down_prob = base_prob * (1.0 - momentum)
            
            # Return probabilities
            return {
                'remain': stability,
                'transition_up': up_prob,
                'transition_down': down_prob
            }
            
        except Exception as e:
            self.logger.error(f"Transition probability calculation failed: {e}")
            return {'remain': 1.0, 'transition_up': 0.0, 'transition_down': 0.0}
    
    def calculate_transition_timeframe(self, vri_components: Dict[str, float]) -> int:
        """Calculate expected transition timeframe in days"""
        try:
            # Extract components
            stability = vri_components.get('regime_stability_score', 0.0)
            momentum = abs(vri_components.get('transition_momentum_score', 0.0))
            
            # Calculate base timeframe
            if momentum > 0.8:
                base_days = 1
            elif momentum > 0.5:
                base_days = 3
            elif momentum > 0.3:
                base_days = 5
            else:
                base_days = 10
            
            # Adjust for stability
            adjusted_days = int(base_days * (1.0 + stability))
            
            return max(adjusted_days, 1)
            
        except Exception as e:
            self.logger.error(f"Transition timeframe calculation failed: {e}")
            return 5
    
    def get_processing_time(self) -> float:
        """Calculate processing time in milliseconds"""
        try:
            end_time = datetime.now()
            processing_time = (end_time - self.start_time).total_seconds() * 1000
            return float(processing_time)
            
        except Exception as e:
            self.logger.error(f"Processing time calculation failed: {e}")
            return 0.0
    
    def analyze_equity_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze equity market regime"""
        try:
            # Extract equity metrics
            vri_composite = self.calculate_vri3_composite(market_data)
            trend = market_data.trend_direction or "neutral"
            
            # Classify regime
            if vri_composite > 0.5:
                if trend == "up":
                    return "bullish_trending"
                else:
                    return "bullish_consolidation"
            elif vri_composite < -0.5:
                if trend == "down":
                    return "bearish_trending"
                else:
                    return "bearish_consolidation"
            else:
                return "neutral"
            
        except Exception as e:
            self.logger.error(f"Equity regime analysis failed: {e}")
            return "undefined"
    
    def analyze_bond_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze bond market regime"""
        try:
            # Extract bond metrics
            vri_composite = self.calculate_vri3_composite(market_data)
            
            # Simple classification
            if vri_composite > 0.3:
                return "yield_rising"
            elif vri_composite < -0.3:
                return "yield_falling"
            else:
                return "yield_stable"
            
        except Exception as e:
            self.logger.error(f"Bond regime analysis failed: {e}")
            return "undefined"
    
    def analyze_commodity_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze commodity market regime"""
        try:
            # Extract commodity metrics
            vri_composite = self.calculate_vri3_composite(market_data)
            volatility = self.calculate_volatility_regime(market_data)
            
            # Classify regime
            if vri_composite > 0.5:
                if volatility > 0.5:
                    return "strong_uptrend"
                else:
                    return "steady_uptrend"
            elif vri_composite < -0.5:
                if volatility > 0.5:
                    return "strong_downtrend"
                else:
                    return "steady_downtrend"
            else:
                return "consolidation"
            
        except Exception as e:
            self.logger.error(f"Commodity regime analysis failed: {e}")
            return "undefined"
    
    def analyze_currency_regime(self, market_data: ProcessedUnderlyingAggregatesV2_5) -> str:
        """Analyze currency market regime"""
        try:
            # Extract currency metrics
            vri_composite = self.calculate_vri3_composite(market_data)
            flow = self.calculate_flow_intensity(market_data)
            
            # Classify regime
            if vri_composite > 0.5:
                if flow > 0.3:
                    return "strong_appreciation"
                else:
                    return "mild_appreciation"
            elif vri_composite < -0.5:
                if flow < -0.3:
                    return "strong_depreciation"
                else:
                    return "mild_depreciation"
            else:
                return "range_bound"
            
        except Exception as e:
            self.logger.error(f"Currency regime analysis failed: {e}")
            return "undefined"
    
    def generate_regime_description(self, regime_name: str, vri_components: Dict[str, float]) -> str:
        """Generate detailed regime description"""
        try:
            # Extract components
            vol_regime = vri_components.get('volatility_regime_score', 0.0)
            flow = vri_components.get('flow_intensity_score', 0.0)
            stability = vri_components.get('regime_stability_score', 0.0)
            
            # Generate description
            desc_parts = []
            
            # Add volatility description
            if abs(vol_regime) > 0.7:
                desc_parts.append("extremely volatile")
            elif abs(vol_regime) > 0.4:
                desc_parts.append("moderately volatile")
            else:
                desc_parts.append("stable volatility")
            
            # Add flow description
            if abs(flow) > 0.7:
                flow_desc = "strong " + ("buying" if flow > 0 else "selling")
                desc_parts.append(flow_desc)
            elif abs(flow) > 0.3:
                flow_desc = "moderate " + ("buying" if flow > 0 else "selling")
                desc_parts.append(flow_desc)
            
            # Add stability description
            if stability > 0.7:
                desc_parts.append("highly stable")
            elif stability > 0.4:
                desc_parts.append("moderately stable")
            else:
                desc_parts.append("transitioning")
            
            return f"Market showing {', '.join(desc_parts)} characteristics"
            
        except Exception as e:
            self.logger.error(f"Regime description generation failed: {e}")
            return "Regime description unavailable"
    
    def classify_regime(self, vri_components: Dict[str, float]) -> str:
        """Classify the current market regime"""
        try:
            # Extract components
            vol_regime = vri_components.get('volatility_regime_score', 0.0)
            flow = vri_components.get('flow_intensity_score', 0.0)
            stability = vri_components.get('regime_stability_score', 0.0)
            momentum = vri_components.get('transition_momentum_score', 0.0)
            
            # Classify based on components
            if stability < 0.3:
                if momentum > 0:
                    return "transition_bear_to_bull"
                else:
                    return "transition_bull_to_bear"
            elif abs(vol_regime) > 0.7:
                if flow > 0.5:
                    return "bull_trending_high_vol"
                elif flow < -0.5:
                    return "bear_trending_high_vol"
                else:
                    return "volatile_consolidation"
            elif abs(flow) > 0.7:
                if vol_regime > 0:
                    return "momentum_acceleration"
                else:
                    return "mean_reversion"
            else:
                return "sideways_low_vol"
            
        except Exception as e:
            self.logger.error(f"Regime classification failed: {e}")
    
    def _get_market_data_sync(self, symbol: str) -> pd.DataFrame:
        """Synchronous version of _get_market_data for use in non-async contexts."""
        try:
            # First, try to get data from enhanced cache (intraday collector)
            if self.enhanced_cache is not None:
                try:
                    # Try to get the underlying data from cache
                    cached_data = self.enhanced_cache.get(symbol, "underlying_data_enriched")
                    if cached_data is not None:
                        # Extract price data from cached underlying data
                        if hasattr(cached_data, 'current_price') and cached_data.current_price > 0:
                            current_data = {
                                'close': [float(cached_data.current_price)],
                                'high': [float(getattr(cached_data, 'high', cached_data.current_price))],
                                'low': [float(getattr(cached_data, 'low', cached_data.current_price))],
                                'open': [float(getattr(cached_data, 'open', cached_data.current_price))],
                                'volume': [int(getattr(cached_data, 'volume', 0))]
                            }
                            data = pd.DataFrame(current_data, index=[pd.Timestamp.now()])
                            self.logger.debug(f"Retrieved cached market data for {symbol}: price={cached_data.current_price}")
                            return data
                except Exception as cache_e:
                    self.logger.debug(f"Cache lookup failed for {symbol}: {cache_e}")
            
            # For synchronous calls, we can't use async API calls, so return empty DataFrame
            # This will trigger the fallback to default values
            self.logger.debug(f"No cached data available for {symbol} in synchronous call")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for a given symbol, first trying enhanced cache, then API."""
        try:
            # First, try to get data from enhanced cache (intraday collector)
            if self.enhanced_cache is not None:
                try:
                    # Try to get the underlying data from cache
                    cached_data = self.enhanced_cache.get(symbol, "underlying_data_enriched")
                    if cached_data is not None:
                        # Extract price data from cached underlying data
                        if hasattr(cached_data, 'current_price') and cached_data.current_price > 0:
                            current_data = {
                                'close': [float(cached_data.current_price)],
                                'high': [float(getattr(cached_data, 'high', cached_data.current_price))],
                                'low': [float(getattr(cached_data, 'low', cached_data.current_price))],
                                'open': [float(getattr(cached_data, 'open', cached_data.current_price))],
                                'volume': [int(getattr(cached_data, 'volume', 0))]
                            }
                            data = pd.DataFrame(current_data, index=[pd.Timestamp.now()])
                            self.logger.debug(f"Retrieved cached market data for {symbol}: price={cached_data.current_price}")
                            return data
                except Exception as cache_e:
                    self.logger.debug(f"Cache lookup failed for {symbol}: {cache_e}")
            
            # Fallback to API if cache is not available or doesn't have the data
            if self.tradier_fetcher is None:
                self.logger.warning(f"Tradier fetcher not available, cannot fetch data for {symbol}")
                return pd.DataFrame()
            
            # Get raw quote data from Tradier (for VIX and other symbols)
            quote_data = await self.tradier_fetcher.fetch_raw_quote_data(symbol)
            
            if not quote_data or 'quotes' not in quote_data:
                self.logger.warning(f"No quote data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Extract the quote information
            quotes = quote_data['quotes']
            if isinstance(quotes, dict) and 'quote' in quotes:
                quote = quotes['quote']
            elif isinstance(quotes, list) and len(quotes) > 0:
                quote = quotes[0]
            elif isinstance(quotes, dict):
                quote = quotes
            else:
                self.logger.warning(f"Unexpected quote data format for {symbol}")
                return pd.DataFrame()
            
            # Create a DataFrame with current price data
            current_data = {
                'close': [float(quote.get('last', 0.0))],
                'high': [float(quote.get('high', 0.0))],
                'low': [float(quote.get('low', 0.0))],
                'open': [float(quote.get('open', 0.0))],
                'volume': [int(quote.get('volume', 0))]
            }
            
            data = pd.DataFrame(current_data, index=[pd.Timestamp.now()])
            
            self.logger.debug(f"Retrieved API market data for {symbol}: last={quote.get('last', 0.0)}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return pd.DataFrame()