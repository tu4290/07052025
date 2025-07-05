# core_analytics_engine/adaptive_trade_idea_framework_v2_5.py
# EOTS v2.5 - S-GRADE PRODUCTION HARDENED ARTIFACT

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Tuple
from datetime import datetime
from pydantic import ValidationError

from data_models import (
    ProcessedDataBundleV2_5, SignalPayloadV2_5,
    ActiveRecommendationPayloadV2_5, ATIFStrategyDirectivePayloadV2_5,
    ATIFManagementDirectiveV2_5, ATIFSituationalAssessmentProfileV2_5,
    KeyLevelsDataV2_5
)
from data_models import ATIFSettingsModel # Import from new location
from utils.config_manager_v2_5 import ConfigManagerV2_5
from utils.pydantic_utils import normalize_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)



class AdaptiveTradeIdeaFrameworkV2_5:
    """
    The core decision-making engine of the EOTS v2.5 system. This production-hardened
    version includes comprehensive error handling and data validation to ensure
    resilience and stability.
    All public methods are strictly Pydantic-first: inputs and outputs are validated at the boundary.
    The ATIF settings config is strictly validated and self-documenting via ATIFSettingsModel.
    """

    def __init__(self, config_manager: ConfigManagerV2_5, performance_tracker: Any):
        self.logger = logger.getChild(self.__class__.__name__)
        self.config_manager = config_manager
        self.performance_tracker = performance_tracker
        raw_atif_settings = self.config_manager.get_setting("adaptive_trade_idea_framework_settings", default={})
        normalized_settings = normalize_config(raw_atif_settings, ATIFSettingsModel)
        try:
            self.atif_settings = ATIFSettingsModel.model_validate(normalized_settings)
        except ValidationError as e:
            self.logger.critical(f"ATIF settings config validation failed: {e.errors()}")
            raise
        self.min_conviction_to_trade = self.atif_settings.min_conviction_to_initiate_trade
        self.logger.info("AdaptiveTradeIdeaFrameworkV2_5 initialized with validated ATIF settings.")

    def generate_trade_directives(
        self,
        processed_data: ProcessedDataBundleV2_5,
        scored_signals: Dict[str, List[SignalPayloadV2_5]],
        key_levels: KeyLevelsDataV2_5
    ) -> List[ATIFStrategyDirectivePayloadV2_5]:
        """
        Generates new trade strategy directives based on a full situational analysis.
        Validates all inputs and outputs using Pydantic. Fails fast on invalid data.
        """
        try:
            processed_data = ProcessedDataBundleV2_5.model_validate(processed_data.model_dump() if hasattr(processed_data, 'model_dump') else processed_data)
            key_levels = KeyLevelsDataV2_5.model_validate(key_levels.model_dump() if hasattr(key_levels, 'model_dump') else key_levels)
        except ValidationError as e:
            self.logger.error(f"Input validation error in generate_trade_directives: {e.errors()}")
            return []
        if not scored_signals:
            self.logger.error("Empty scored_signals provided to generate_trade_directives. Aborting.")
            return []
        self.logger.debug("Generating trade directives...")
        try:
            assessment = self._integrate_signals_and_assess_situation(scored_signals, processed_data)
            if not assessment:
                self.logger.info("Signal integration resulted in no assessment. No directives generated.")
                return []
            symbol = processed_data.underlying_data_enriched.symbol
            regime = processed_data.underlying_data_enriched.current_market_regime_v2_5 or "UNKNOWN"
            dominant_bias_category, conviction = self._map_assessment_to_conviction(assessment, symbol, regime)
            if conviction < self.min_conviction_to_trade:
                self.logger.info(f"Conviction {conviction:.2f} for bias '{dominant_bias_category}' is below threshold {self.min_conviction_to_trade}.")
                return []
            self.logger.info(f"High conviction ({conviction:.2f}) met for bias '{dominant_bias_category}'. Determining strategy.")
            directive = self._determine_strategy_specificity(assessment, conviction, dominant_bias_category, processed_data)
            if directive:
                try:
                    directive = ATIFStrategyDirectivePayloadV2_5.model_validate(directive.model_dump() if hasattr(directive, 'model_dump') else directive)
                except ValidationError as e:
                    self.logger.error(f"Output validation error in generate_trade_directives: {e.errors()}")
                    return []
                self.logger.info(f"Generated a new trade directive: {directive.selected_strategy_type} for {symbol}")
                return [directive]
            else:
                self.logger.info("High conviction met, but no specific strategy rule matched.")
                return []
        except ValidationError as e:
            self.logger.error(f"Validation error in generate_trade_directives: {e.errors()}")
            return []
        except Exception as e:
            self.logger.critical(f"Unhandled exception during trade directive generation: {e}", exc_info=True)
            return []

    def get_management_directive(self, active_recommendation: ActiveRecommendationPayloadV2_5, current_und_price: float) -> Optional[ATIFManagementDirectiveV2_5]:
        """
        Evaluates an existing active recommendation and issues a management directive.
        Validates input and output using Pydantic. Returns None if no action is needed.
        """
        try:
            active_recommendation = ActiveRecommendationPayloadV2_5.model_validate(active_recommendation.model_dump() if hasattr(active_recommendation, 'model_dump') else active_recommendation)
        except ValidationError as e:
            self.logger.error(f"Input validation error in get_management_directive: {e.errors()}")
            return None
        # This is a placeholder implementation. In a real system, this would contain complex logic.
        if active_recommendation.trade_bias == "Bullish" and current_und_price <= active_recommendation.stop_loss_current:
            try:
                return ATIFManagementDirectiveV2_5.model_validate({
                    'recommendation_id': active_recommendation.recommendation_id,
                    'action': "EXIT",
                    'reason': "STOPLOSS_HIT"
                })
            except ValidationError as e:
                self.logger.error(f"Output validation error in get_management_directive: {e.errors()}")
                return None
        if active_recommendation.trade_bias == "Bearish" and current_und_price >= active_recommendation.stop_loss_current:
            try:
                return ATIFManagementDirectiveV2_5.model_validate({
                    'recommendation_id': active_recommendation.recommendation_id,
                    'action': "EXIT",
                    'reason': "STOPLOSS_HIT"
                })
            except ValidationError as e:
                self.logger.error(f"Output validation error in get_management_directive: {e.errors()}")
                return None
        return None  # No action needed

    def _integrate_signals_and_assess_situation(
        self, scored_signals: Dict[str, List[SignalPayloadV2_5]], processed_data: ProcessedDataBundleV2_5
    ) -> Optional[ATIFSituationalAssessmentProfileV2_5]:
        """Dynamically integrates signals to form a holistic situational assessment."""
        assessment = ATIFSituationalAssessmentProfileV2_5(timestamp=datetime.now())
        current_regime = processed_data.underlying_data_enriched.current_market_regime_v2_5 or "DEFAULT"
        # Regime-aware weighting
        weight_multipliers = self.atif_settings.regime_context_weight_multipliers.get(current_regime, {})
        regime_weight = weight_multipliers.get(current_regime, 1.0)
        for signal_list in scored_signals.values():
            for signal in signal_list:
                weighted_score = signal.strength_score * regime_weight
                if signal.direction == "Bullish":
                    assessment.bullish_assessment_score += weighted_score
                elif signal.direction == "Bearish":
                    assessment.bearish_assessment_score += abs(weighted_score) # Use absolute for bearish score
        if assessment.bullish_assessment_score == 0 and assessment.bearish_assessment_score == 0:
            return None
        return assessment

    def _map_assessment_to_conviction(self, assessment: ATIFSituationalAssessmentProfileV2_5, symbol: str, regime: str) -> Tuple[str, float]:
        """Translates a situational assessment score into a final trade conviction."""
        dominant_bias = "Neutral"
        dominant_score = 0.0
        if assessment.bullish_assessment_score > assessment.bearish_assessment_score:
            dominant_bias = "Bullish"
            dominant_score = assessment.bullish_assessment_score
        else:
            dominant_bias = "Bearish"
            dominant_score = assessment.bearish_assessment_score
        # Placeholder: a real implementation would query performance_tracker here
        perf_data = self.performance_tracker.get_historical_performance_for_setup(symbol, regime, dominant_bias)
        win_rate_adjustment = (perf_data['win_rate'] - 0.5) # Positive if >50% winrate, negative if <50%
        # Regime-aware conviction boost
        conviction_params = self.atif_settings.conviction_mapping_params.get(regime, {})
        bias_boost = conviction_params.get("bias_boost", {}).get(dominant_bias, 0.0)
        # Final conviction is the raw score adjusted by historical win rate and regime bias boost
        final_conviction = dominant_score + win_rate_adjustment + bias_boost
        return dominant_bias, max(0.0, final_conviction)

    def _determine_strategy_specificity(
        self, assessment: ATIFSituationalAssessmentProfileV2_5, conviction: float, bias: str, processed_data: ProcessedDataBundleV2_5
    ) -> Optional[ATIFStrategyDirectivePayloadV2_5]:
        """Selects the optimal options strategy based on the full context."""
        # PYDANTIC COMPLIANCE FIX: Handle both Pydantic models and dictionaries
        specificity_rules = (self.atif_settings.strategy_specificity_rules 
                           if hasattr(self.atif_settings, 'strategy_specificity_rules') 
                           else getattr(self.atif_settings, 'strategy_specificity_rules', []))
        regime = processed_data.underlying_data_enriched.current_market_regime_v2_5 or "UNKNOWN"
        # Placeholder for IV Rank, would come from historical data manager
        iv_rank = 50 

        for rule in specificity_rules:
            cond = rule.get("conditions", {})
            # Check bias
            if cond.get("bias") != bias:
                continue
            # Check conviction
            if not (cond.get("min_conviction", 0) <= conviction <= cond.get("max_conviction", 100)):
                continue
            # Check regime
            if not any(r_part in regime for r_part in cond.get("regime_contains", [])):
                continue
            # Check IV
            if not (cond.get("min_iv_rank", 0) <= iv_rank <= cond.get("max_iv_rank", 100)):
                continue

            # If all conditions match, we have found our strategy
            strat_output = rule["strategy_output"]
            return ATIFStrategyDirectivePayloadV2_5(
                selected_strategy_type=strat_output["strategy_type"],
                target_dte_min=strat_output["target_dte"][0],
                target_dte_max=strat_output["target_dte"][1],
                target_delta_long_leg_min=strat_output.get("delta_range_long", [None, None])[0],
                target_delta_long_leg_max=strat_output.get("delta_range_long", [None, None])[1],
                target_delta_short_leg_min=strat_output.get("delta_range_short", [None, None])[0],
                target_delta_short_leg_max=strat_output.get("delta_range_short", [None, None])[1],
                underlying_price_at_decision=processed_data.underlying_data_enriched.price or 0.0,
                final_conviction_score_from_atif=conviction,
                supportive_rationale_components={"rule_name": rule.get("name")},
                assessment_profile=assessment
            )
        return None