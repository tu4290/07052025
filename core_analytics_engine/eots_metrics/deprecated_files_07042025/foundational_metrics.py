import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, time

# Canonical models are now imported from data_models/processed_data.py and related modules.

logger = logging.getLogger(__name__)
EPSILON = 1e-9

class FoundationalMetricsCalculator:
    """
    Calculates Tier 1 Foundational Metrics for EOTS v2.5.
    These metrics provide the bedrock of market understanding.
    """

    def __init__(self, config_manager: Any, historical_data_manager: Any):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.historical_data_manager = historical_data_manager

    def calculate_all_foundational_metrics(self, und_data: Dict) -> Dict:
        """
        Orchestrates the calculation of all foundational metrics.
        """
        self.logger.debug("Calculating foundational metrics...")
        
        # Calculate Net Customer Greek Flows
        und_data = self._calculate_net_customer_greek_flows(und_data)
        
        # Calculate GIB, HP_EOD, TD_GIB
        und_data = self._calculate_gib_based_metrics(und_data)

        # Standard Rolling Net Signed Flows are assumed to be part of the raw data or handled by initial processor
        # If not, they would be calculated here based on raw options data.
        
        self.logger.debug("Foundational metrics calculation complete.")
        return und_data

    def _calculate_net_customer_greek_flows(self, und_data: Dict) -> Dict:
        """
        Calculates Net Customer Greek Flows (Delta, Gamma, Vega, Theta) for the underlying.
        Uses granular get_chain data for precision.
        """
        self.logger.debug("Calculating Net Customer Greek Flows...")
        
        # Ensure fields exist, default to 0 if not
        deltas_buy = und_data.get('deltas_buy', 0) or 0
        deltas_sell = und_data.get('deltas_sell', 0) or 0
        und_data['net_cust_delta_flow_und'] = deltas_buy - deltas_sell
        
        gammas_call_buy = und_data.get('gammas_call_buy', 0) or 0
        gammas_put_buy = und_data.get('gammas_put_buy', 0) or 0
        gammas_call_sell = und_data.get('gammas_call_sell', 0) or 0
        gammas_put_sell = und_data.get('gammas_put_sell', 0) or 0
        und_data['net_cust_gamma_flow_und'] = (gammas_call_buy + gammas_put_buy) - (gammas_call_sell + gammas_put_sell)
        
        vegas_buy = und_data.get('vegas_buy', 0) or 0
        vegas_sell = und_data.get('vegas_sell', 0) or 0
        und_data['net_cust_vega_flow_und'] = vegas_buy - vegas_sell
        
        thetas_buy = und_data.get('thetas_buy', 0) or 0
        thetas_sell = und_data.get('thetas_sell', 0) or 0
        und_data['net_cust_theta_flow_und'] = thetas_buy - thetas_sell

        self.logger.debug("Net Customer Greek Flows calculated.")
        return und_data

    def _calculate_gib_based_metrics(self, und_data: Dict) -> Dict:
        """
        Calculate GIB (Gamma Imbalance from Open Interest), HP_EOD (End-of-Day Hedging Pressure),
        and TD_GIB (Traded Dealer Gamma Imbalance) metrics.
        """
        self.logger.debug("Calculating GIB-based metrics...")

        # GIB (Gamma Imbalance from Open Interest)
        call_gxoi_safe = und_data.get('call_gxoi', 0.0) or 0.0
        put_gxoi_safe = und_data.get('put_gxoi', 0.0) or 0.0
        gib_raw_gamma_units = put_gxoi_safe - call_gxoi_safe

        underlying_price = und_data.get('price', 100.0)
        contract_multiplier = 100  # Standard options contract multiplier
        gib_dollar_value_full = gib_raw_gamma_units * underlying_price * contract_multiplier

        # Scale down for dashboard display (Pydantic-first approach)
        gib_display_value = gib_dollar_value_full / 10000.0

        und_data['gib_oi_based_und'] = gib_display_value
        und_data['gib_raw_gamma_units_und'] = gib_raw_gamma_units
        und_data['gib_dollar_value_full_und'] = gib_dollar_value_full

        self.logger.debug(f"GIB calculated: raw={gib_raw_gamma_units:.2f}, display={gib_display_value:.2f}")

        # HP_EOD (End-of-Day Hedging Pressure) calculation
        hp_eod_value = self._calculate_hp_eod_und_v2_5(und_data)
        und_data['hp_eod_und'] = hp_eod_value
        self.logger.debug(f"HP_EOD calculated: {hp_eod_value}")

        # TD_GIB (Traded Dealer Gamma Imbalance)
        net_cust_gamma_flow = und_data.get('net_cust_gamma_flow_und', 0.0) or 0.0
        td_gib_value = -net_cust_gamma_flow
        und_data['td_gib_und'] = td_gib_value
        self.logger.debug(f"TD_GIB calculated: {td_gib_value}")

        self.logger.debug("GIB-based metrics calculation complete.")
        return und_data

    def _calculate_hp_eod_und_v2_5(self, und_data: Dict) -> float:
        """
        Calculate HP_EOD (End-of-Day Hedging Pressure).
        """
        try:
            gib_full = und_data.get('gib_dollar_value_full_und', 0.0)
            if gib_full == 0.0:
                gib_display = und_data.get('gib_oi_based_und', 0.0)
                gib_full = gib_display * 10000.0

            current_price = und_data.get('price', 0.0)
            reference_price = (
                und_data.get('day_open_price_und') or
                und_data.get('tradier_open') or
                und_data.get('prev_day_close_price_und') or
                current_price * 0.995
            )

            current_time = datetime.now().time()
            market_open = time(9, 30)
            market_close = time(16, 0)

            if market_open <= current_time <= market_close:
                total_market_minutes = (market_close.hour - market_open.hour) * 60 + (market_close.minute - market_open.minute)
                current_minutes = (current_time.hour - market_open.hour) * 60 + (current_time.minute - market_open.minute)
                time_progression = current_minutes / total_market_minutes
                time_multiplier = 0.5 + (time_progression * 0.5)

                price_change = current_price - reference_price
                hp_eod_full = gib_full * price_change * time_multiplier
                hp_eod_display = hp_eod_full / 10000.0

                self.logger.debug(f"HP_EOD calculation: gib_full={gib_full}, price_change={price_change}, "
                                f"time_multiplier={time_multiplier:.3f}, hp_eod_full={hp_eod_full}, "
                                f"hp_eod_display={hp_eod_display}")
                return float(hp_eod_display)
            else:
                self.logger.debug(f"Current time {current_time} is outside trading hours, HP_EOD = 0")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating HP_EOD: {e}")
            return 0.0
