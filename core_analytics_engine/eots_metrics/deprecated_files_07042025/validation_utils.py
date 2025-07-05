# core_analytics_engine/eots_metrics/validation_utils.py

"""
Validation utilities for EOTS metrics calculations.
Consolidates all validation logic to eliminate redundancy.
"""

import pandas as pd
from typing import Any, Union
from data_models import (
    RawUnderlyingDataCombinedV2_5,
    ProcessedUnderlyingAggregatesV2_5,
)
from .elite_intelligence import EliteImpactResultsV2_5


class ValidationUtils:
    """Centralized validation utilities for metrics calculations."""
    
    def validate_input_data(self, options_df_raw: pd.DataFrame, 
                          und_data_api_raw: RawUnderlyingDataCombinedV2_5) -> None:
        """Validate input data for metrics calculation."""
        # Validate underlying data model
        if not hasattr(und_data_api_raw, 'model_dump'):
            raise TypeError(f"und_data_api_raw must be a Pydantic model, got {type(und_data_api_raw)}")
            
        if not isinstance(und_data_api_raw, RawUnderlyingDataCombinedV2_5):
            raise TypeError(f"und_data_api_raw must be RawUnderlyingDataCombinedV2_5, got {type(und_data_api_raw)}")
            
        # Validate critical price data
        if not hasattr(und_data_api_raw, 'price') or und_data_api_raw.price <= 0.0:
            raise ValueError(f"Invalid price data {getattr(und_data_api_raw, 'price', 'MISSING')}")

    def validate_foundational_metrics(self, foundational_model: ProcessedUnderlyingAggregatesV2_5) -> None:
        """Validate foundational metrics were calculated properly."""
        if foundational_model.gib_oi_based_und == 0.0:
            print("⚠️ WARNING: gib_oi_based_und is 0.0 - verify this is real market data!")
        if foundational_model.td_gib_und == 0.0:
            print("⚠️ WARNING: td_gib_und is 0.0 - verify this is real market data!")
        if foundational_model.hp_eod_und == 0.0:
            print("⚠️ WARNING: hp_eod_und is 0.0 - verify this is real market data!")
            
        print(f"✅ Foundational metrics: GIB={foundational_model.gib_oi_based_und:.2f}, "
              f"TD_GIB={foundational_model.td_gib_und:.2f}, HP_EOD={foundational_model.hp_eod_und:.2f}")

    def validate_elite_results(self, elite_results: EliteImpactResultsV2_5) -> None:
        """Validate elite intelligence results."""
        if not isinstance(elite_results, EliteImpactResultsV2_5):
            raise TypeError(f"Elite intelligence must return EliteImpactResultsV2_5, got {type(elite_results)}")
            
        if elite_results.elite_impact_score_und <= 0.0:
            raise ValueError(f"elite_impact_score_und={elite_results.elite_impact_score_und} is invalid")
        if elite_results.institutional_flow_score_und <= 0.0:
            raise ValueError(f"institutional_flow_score_und={elite_results.institutional_flow_score_und} is invalid")
            
        print(f"✅ Elite metrics: Impact={elite_results.elite_impact_score_und:.2f}, "
              f"Institutional={elite_results.institutional_flow_score_und:.2f}")

    def validate_final_model(self, enriched_underlying: ProcessedUnderlyingAggregatesV2_5) -> None:
        """Validate final enriched model has all required fields."""
        required_fields = [
            'gib_oi_based_und', 'td_gib_und', 'hp_eod_und',
            'net_cust_delta_flow_und', 'net_cust_gamma_flow_und', 
            'net_cust_vega_flow_und', 'net_cust_theta_flow_und',
            'vapi_fa_z_score_und', 'dwfd_z_score_und', 'tw_laf_z_score_und',
            'elite_impact_score_und', 'institutional_flow_score_und', 
            'flow_momentum_index_und', 'market_regime_elite', 
            'flow_type_elite', 'volatility_regime_elite',
            'confidence', 'transition_risk'
        ]
        
        for field in required_fields:
            value = getattr(enriched_underlying, field, None)
            if value is None:
                raise ValueError(f"Required field {field} is None")
            if isinstance(value, (int, float)) and value == 0.0 and field in ['elite_impact_score_und', 'institutional_flow_score_und']:
                raise ValueError(f"Required field {field}={value} is zero - calculation failed!")
                
        print("✅ All required fields validated with real calculated values")

    def require_pydantic_field(self, pydantic_model: Any, field_name: str, field_description: str) -> Any:
        """Require field from Pydantic model with validation."""
        if not hasattr(pydantic_model, field_name):
            raise ValueError(f"Required field '{field_name}' ({field_description}) missing from Pydantic model!")
            
        value = getattr(pydantic_model, field_name)
        if value is None:
            raise ValueError(f"Field '{field_name}' ({field_description}) is None!")
            
        # Additional validation for suspicious values
        if isinstance(value, (int, float)):
            if value == 0 and field_name in ['day_volume', 'u_volatility']:
                import warnings
                warnings.warn(f"WARNING: {field_description} is exactly 0 - verify this is real market data!")
                
        return value

    def require_column_sum(self, dataframe: pd.DataFrame, column_name: str, column_description: str) -> float:
        """Require column to exist and sum it with validation."""
        if dataframe.empty:
            raise ValueError(f"DataFrame is empty - cannot sum {column_description}")
            
        if column_name not in dataframe.columns:
            raise ValueError(f"Required column '{column_name}' ({column_description}) is missing!")
            
        column_sum = dataframe[column_name].sum()
        
        if pd.isna(column_sum):
            raise ValueError(f"Sum of {column_description} is NaN!")
            
        return float(column_sum)

    def get_pydantic_field_optional(self, pydantic_model: Any, field_name: str) -> Any:
        """Get optional field from Pydantic model."""
        return getattr(pydantic_model, field_name, None)