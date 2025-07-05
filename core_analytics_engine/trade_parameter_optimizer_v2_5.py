# core_analytics_engine/trade_parameter_optimizer_v2_5.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE & FINAL

import logging
import uuid
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import pandas as pd

from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_models import ATIFStrategyDirectivePayloadV2_5
from data_models import ActiveRecommendationPayloadV2_5
from data_models import KeyLevelsDataV2_5
from data_models import ProcessedDataBundleV2_5
from data_models import TradeParameterOptimizerSettings

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class TradeParameterOptimizerV2_5:
    """
    Translates ATIF directives into precise, executable trade parameters.
    """
    def __init__(self, config_manager: ConfigManagerV2_5):
        self.logger = logger.getChild(self.__class__.__name__)
        self.config_manager = config_manager
        
        # Use default settings if not found in config
        self.settings = self.config_manager.get_setting("trade_parameter_optimizer_settings", default={})
        if not self.settings:
            self.logger.warning("trade_parameter_optimizer_settings not found in configuration. Using default behaviors.")
            # Fallback to a default Pydantic model instance if settings are not found
            self.settings = TradeParameterOptimizerSettings()
        
        self.logger.info("TradeParameterOptimizerV2_5 Initialized with settings:")
        self.logger.debug(f"Settings: {self.settings}")

    def optimize_parameters_for_directive(
        self, directive: ATIFStrategyDirectivePayloadV2_5,
        processed_data: ProcessedDataBundleV2_5, key_levels: KeyLevelsDataV2_5
    ) -> Optional[ActiveRecommendationPayloadV2_5]:
        """Main method to generate a fully parameterized trade recommendation."""
        if not all(isinstance(arg, (ATIFStrategyDirectivePayloadV2_5, ProcessedDataBundleV2_5, KeyLevelsDataV2_5)) for arg in [directive, processed_data, key_levels]):
            self.logger.error("Invalid input types to TPO. Aborting.")
            return None
        try:
            options_chain_df = pd.DataFrame(processed_data.options_data_with_metrics)
            if options_chain_df.empty:
                self.logger.warning("Options chain is empty. Cannot select contract.")
                return None

            selected_contracts = self._select_optimal_contracts(directive, options_chain_df)
            if not selected_contracts:
                self.logger.warning(f"Could not find a suitable contract for directive: {directive.selected_strategy_type}")
                return None

            trade_bias = "Bullish" if directive.final_conviction_score_from_atif > 0 else "Bearish"
            atr = processed_data.underlying_data_enriched.atr_und or (processed_data.underlying_data_enriched.price * 0.01)

            und_sl, und_t1, und_t2 = self._calculate_sl_and_targets(
                trade_bias, processed_data.underlying_data_enriched.price, key_levels, atr
            )

            return self._construct_recommendation_payload(
                directive, processed_data, selected_contracts, und_sl, und_t1, und_t2, trade_bias
            )
        except Exception as e:
            self.logger.critical(f"Unhandled exception during parameter optimization: {e}", exc_info=True)
            return None

    def _select_optimal_contracts(self, directive: ATIFStrategyDirectivePayloadV2_5, chain_df: pd.DataFrame) -> Optional[List[Dict]]:
        """Selects the best option contracts from the chain based on ATIF directives."""
        if directive.selected_strategy_type in ["LongCall", "LongPut"]:
            is_call = directive.selected_strategy_type == "LongCall"
            
            # Filter by DTE and option type
            candidates = chain_df[
                (chain_df['opt_kind'] == ('call' if is_call else 'put')) &
                (chain_df['dte_calc'] >= directive.target_dte_min) &
                (chain_df['dte_calc'] <= directive.target_dte_max)
            ].copy()

            # Filter by delta
            if directive.target_delta_long_leg_min is not None:
                candidates = candidates[candidates['delta'].abs().between(
                    directive.target_delta_long_leg_min, directive.target_delta_long_leg_max
                )]

            if candidates.empty: return None

            # Score and select best contract
            candidates['liquidity_score'] = candidates['oi'].fillna(0) + candidates['volm'].fillna(0) * 5
            best_contract = candidates.loc[candidates['liquidity_score'].idxmax()]
            return [best_contract.to_dict()]
        
        self.logger.warning(f"Contract selection for '{directive.selected_strategy_type}' is not yet implemented.")
        return None

    def _calculate_sl_and_targets(self, trade_bias: str, entry_price_und: float, key_levels: KeyLevelsDataV2_5, atr: float) -> Tuple[float, float, Optional[float]]:
        """Calculates SL and multiple TP levels for the underlying."""
        # PYDANTIC COMPLIANCE FIX: Use config manager for proper settings access
        try:
            sl_mult = float(self.config_manager.get_setting("strategy_settings.targets.target_atr_stop_loss_multiplier", default=1.5))
            t1_mult = float(self.config_manager.get_setting("strategy_settings.targets.t1_mult_no_sr", default=1.0))
            t2_mult = float(self.config_manager.get_setting("strategy_settings.targets.t2_mult_no_sr", default=2.0))
        except Exception:
            # Fallback to defaults if config access fails
            sl_mult = 1.5
            t1_mult = 1.0
            t2_mult = 2.0

        if trade_bias == "Bullish":
            stop_loss = entry_price_und - (atr * sl_mult)
            supports = sorted([lvl.level_price for lvl in key_levels.supports if lvl.level_price < entry_price_und], reverse=True)
            if supports: stop_loss = min(stop_loss, supports[0] - (atr * 0.1))

            target_1 = entry_price_und + (atr * t1_mult)
            resistances = sorted([lvl.level_price for lvl in key_levels.resistances if lvl.level_price > entry_price_und])
            if resistances: target_1 = min(target_1, resistances[0])
            target_2 = target_1 + (atr * (t2_mult - t1_mult)) if resistances else None
        else: # Bearish
            stop_loss = entry_price_und + (atr * sl_mult)
            resistances = sorted([lvl.level_price for lvl in key_levels.resistances if lvl.level_price > entry_price_und])
            if resistances: stop_loss = max(stop_loss, resistances[0] + (atr * 0.1))

            target_1 = entry_price_und - (atr * t1_mult)
            supports = sorted([lvl.level_price for lvl in key_levels.supports if lvl.level_price < entry_price_und], reverse=True)
            if supports: target_1 = max(target_1, supports[0])
            target_2 = target_1 - (atr * (t2_mult - t1_mult)) if supports else None
        
        return stop_loss, target_1, target_2

    def _calculate_atr_based_targets_no_sr(self, entry_price: float, direction: str, atr: float) -> Dict[str, float]:
        """Calculate ATR-based targets when no support/resistance levels are available."""
        
        # PYDANTIC COMPLIANCE FIX: Use config manager for proper settings access
        try:
            sl_mult = float(self.config_manager.get_setting("strategy_settings.targets.target_atr_stop_loss_multiplier", default=1.5))
            t1_mult = float(self.config_manager.get_setting("strategy_settings.targets.t1_mult_no_sr", default=1.0))
            t2_mult = float(self.config_manager.get_setting("strategy_settings.targets.t2_mult_no_sr", default=2.0))
        except Exception:
            # Fallback to defaults if config access fails
            sl_mult = 1.5
            t1_mult = 1.0
            t2_mult = 2.0
        
        if direction.lower() == "bullish":
            stop_loss = entry_price - (atr * sl_mult)
            target_1 = entry_price + (atr * t1_mult)
            target_2 = entry_price + (atr * t2_mult)
        else:  # bearish
            stop_loss = entry_price + (atr * sl_mult)
            target_1 = entry_price - (atr * t1_mult)
            target_2 = entry_price - (atr * t2_mult)
        
        return {
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2
        }

    def _calculate_atr_based_targets_with_sr(self, entry_price: float, direction: str, atr: float, 
                                           support_level: Optional[float], resistance_level: Optional[float]) -> Dict[str, float]:
        """Calculate targets using both ATR and support/resistance levels."""
        
        # PYDANTIC COMPLIANCE FIX: Use config manager for proper settings access
        try:
            sl_mult = float(self.config_manager.get_setting("strategy_settings.targets.target_atr_stop_loss_multiplier", default=1.5))
            t1_mult = float(self.config_manager.get_setting("strategy_settings.targets.t1_mult_no_sr", default=1.0))
            t2_mult = float(self.config_manager.get_setting("strategy_settings.targets.t2_mult_no_sr", default=2.0))
        except Exception:
            # Fallback to defaults if config access fails
            sl_mult = 1.5
            t1_mult = 1.0
            t2_mult = 2.0
        
        # Calculate ATR-based targets first
        atr_targets = self._calculate_atr_based_targets_no_sr(entry_price, direction, atr)
        
        if direction.lower() == "bullish":
            # For bullish trades, use support for stop loss, resistance for targets
            stop_loss = support_level if support_level else atr_targets["stop_loss"]
            target_1 = resistance_level if resistance_level else atr_targets["target_1"]
            target_2 = atr_targets["target_2"]  # Keep ATR-based for second target
        else:  # bearish
            # For bearish trades, use resistance for stop loss, support for targets
            stop_loss = resistance_level if resistance_level else atr_targets["stop_loss"]
            target_1 = support_level if support_level else atr_targets["target_1"]
            target_2 = atr_targets["target_2"]  # Keep ATR-based for second target
        
        return {
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2
        }

    def _select_best_contract(self, options_df: pd.DataFrame, target_delta: float, target_dte: int, 
                            option_type: str) -> Optional[Dict[str, Any]]:
        """Select the best contract based on delta and DTE targets."""
        try:
            # Filter by option type and DTE
            filtered_df = options_df[
                (options_df['opt_kind'] == option_type) & 
                (options_df['dte'] <= target_dte + 2) &  # Allow some flexibility
                (options_df['dte'] >= max(1, target_dte - 2))
            ].copy()
            
            if filtered_df.empty:
                return None
            
            # Find contract closest to target delta
            filtered_df['delta_diff'] = abs(filtered_df['delta'] - abs(target_delta))
            best_contract = filtered_df.loc[filtered_df['delta_diff'].idxmin()]
            
            # PYDANTIC COMPLIANCE FIX: Handle contract data properly
            contract_dict = best_contract.to_dict()
            opt_entry = contract_dict.get('price') or ((contract_dict.get('bid_price', 0) + contract_dict.get('ask_price', 0)) / 2)
            opt_delta = contract_dict.get('delta', 0.5)
            
            return {
                'strike': best_contract['strike'],
                'expiration': best_contract['expiration'],
                'option_type': option_type,
                'delta': opt_delta,
                'entry_price': opt_entry,
                'dte': best_contract['dte']
            }
            
        except Exception as e:
            self.logger.error(f"Error selecting best contract: {str(e)}")
            return None

    def _construct_recommendation_payload(self, directive, processed_data, contracts, und_sl, und_t1, und_t2, bias) -> ActiveRecommendationPayloadV2_5:
        """Constructs the final recommendation payload."""
        contract = contracts[0]
        opt_entry = contract.get('price') or ((contract.get('bid_price', 0) + contract.get('ask_price', 0)) / 2)
        opt_delta = contract.get('delta', 0.5)

        opt_sl = opt_entry - abs(und_sl - processed_data.underlying_data_enriched.price) * abs(opt_delta)
        opt_t1 = opt_entry + abs(und_t1 - processed_data.underlying_data_enriched.price) * abs(opt_delta)
        opt_t2 = (opt_entry + abs(und_t2 - processed_data.underlying_data_enriched.price) * abs(opt_delta)) if und_t2 else None

        return ActiveRecommendationPayloadV2_5(
            recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
            symbol=processed_data.underlying_data_enriched.symbol,
            timestamp_issued=datetime.now(),
            strategy_type=directive.selected_strategy_type,
            selected_option_details=contracts,
            trade_bias=bias,
            entry_price_initial=opt_entry,
            stop_loss_initial=opt_sl,
            target_1_initial=opt_t1,
            target_2_initial=opt_t2,
            stop_loss_current=opt_sl,
            target_1_current=opt_t1,
            target_2_current=opt_t2,
            target_rationale=f"ATR/Level Based. UND SL: {und_sl:.2f}, T1: {und_t1:.2f}",
            status="ACTIVE_NEW",
            atif_conviction_score_at_issuance=directive.final_conviction_score_from_atif,
            triggering_signals_summary=str(directive.supportive_rationale_components),
            regime_at_issuance=processed_data.underlying_data_enriched.current_market_regime_v2_5
        )