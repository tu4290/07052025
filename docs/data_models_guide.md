# EOTS v2.5 Data Models Guide

## Overview

The EOTS v2.5 data models have been completely refactored and consolidated into a clean, maintainable structure. This guide provides comprehensive documentation for all data models, their usage, validation rules, and best practices.

## Architecture

### Consolidated Structure

The data models are organized into 6 main modules:

1. **`core_models.py`** - Base types, system state, raw/processed data, bundles, advanced metrics
2. **`configuration_models.py`** - All configuration schemas and settings  
3. **`ai_ml_models.py`** - AI/ML, MOE, learning, and performance models
4. **`trading_market_models.py`** - Trading, market context, signals, recommendations
5. **`dashboard_ui_models.py`** - Dashboard and UI component models
6. **`validation_utils.py`** - Validation utilities and helper functions

### Key Benefits

- **Reduced Complexity**: From 20+ scattered files to 6 organized modules
- **Better Maintainability**: Clear separation of concerns
- **Enhanced Validation**: Comprehensive field validation and business rules
- **Type Safety**: Full Pydantic v2 compliance with proper type hints
- **Backward Compatibility**: All existing imports continue to work

## Core Models (`core_models.py`)

### Raw Data Models

#### `RawOptionsContractV2_5`
Represents raw options contract data from data providers.

```python
from data_models import RawOptionsContractV2_5

contract = RawOptionsContractV2_5(
    contract_symbol="SPY240315C00500000",
    strike=500.0,
    opt_kind="call",
    dte_calc=30.0,
    iv=0.25,
    delta_contract=0.5
)
```

**Validation Rules:**
- `strike` must be positive
- `opt_kind` must be "call" or "put"
- `iv` (implied volatility) must be between 0.0 and 10.0
- `delta_contract` must be between -1.0 and 1.0
- `gamma_contract` and `vega_contract` must be non-negative

#### `RawUnderlyingDataV2_5`
Raw underlying asset data.

```python
underlying = RawUnderlyingDataV2_5(
    symbol="SPY",
    timestamp=datetime.now(timezone.utc),
    price=450.25
)
```

### Processed Data Models

#### `ProcessedContractMetricsV2_5`
Extends raw contract data with calculated metrics.

#### `ProcessedStrikeLevelMetricsV2_5`
Strike-level aggregated metrics with validation.

**Validation Rules:**
- `prediction_confidence` and `signal_strength` must be between 0.0 and 1.0
- `strike` must be positive

#### `ProcessedDataBundleV2_5`
Complete processed data bundle for analysis.

### Bundle Models

#### `FinalAnalysisBundleV2_5`
Top-level data structure for complete analysis results.

```python
bundle = FinalAnalysisBundleV2_5(
    processed_data_bundle=processed_bundle,
    scored_signals_v2_5={},
    bundle_timestamp=datetime.now(timezone.utc),
    target_symbol="SPY"
)
```

## Trading & Market Models (`trading_market_models.py`)

### Market Regime

#### `MarketRegimeState`
Comprehensive market regime enumeration.

```python
from data_models import MarketRegimeState

regime = MarketRegimeState.BULLISH_TREND
# Available values: BULLISH_TREND, BEARISH_TREND, SIDEWAYS, 
# VOLATILITY_EXPANSION, VOLATILITY_CONTRACTION, etc.
```

### Signal Models

#### `SignalPayloadV2_5`
Trading signal representation with comprehensive validation.

```python
signal = SignalPayloadV2_5(
    signal_id=str(uuid.uuid4()),
    signal_name="VAPI_FA_Bullish_Surge",
    symbol="SPY",
    timestamp=datetime.now(timezone.utc),
    signal_type="Directional",
    direction="Bullish",
    strength_score=2.5
)
```

**Validation Rules:**
- `direction` must be "Bullish", "Bearish", or "Neutral"
- `signal_type` must be one of: "Directional", "Volatility", "Structural", "Flow_Divergence", "Warning", "Momentum", "Mean_Reversion"
- `strength_score` must be between -5.0 and 5.0
- `strike_impacted` must be positive if provided

#### `KeyLevelV2_5`
Key price level identification.

```python
level = KeyLevelV2_5(
    level_price=500.0,
    level_type="Support",
    conviction_score=0.85,
    contributing_metrics=["A-MSPI", "NVP_Peak"]
)
```

**Validation Rules:**
- `level_price` must be positive
- `level_type` must be one of: "Support", "Resistance", "PinZone", "VolTrigger", "MajorWall", "GammaWall", "DeltaWall"
- `conviction_score` must be between 0.0 and 5.0

#### `DynamicThresholdsV2_5`
Dynamic threshold configuration with built-in validation.

```python
thresholds = DynamicThresholdsV2_5(
    vapi_fa_bullish_thresh=1.5,
    vapi_fa_bearish_thresh=-1.5,
    vri_bullish_thresh=0.6,
    vri_bearish_thresh=-0.6
)
```

## AI/ML Models (`ai_ml_models.py`)

### HuiHui Expert System

#### `HuiHuiExpertType`
Expert type enumeration.

```python
from data_models import HuiHuiExpertType

expert = HuiHuiExpertType.MARKET_REGIME
# Available: MARKET_REGIME, OPTIONS_FLOW, SENTIMENT, ORCHESTRATOR, 
# VOLATILITY, LIQUIDITY, RISK, EXECUTION
```

#### `HuiHuiModelConfigV2_5`
Configuration for HuiHui AI models.

### MOE (Mixture of Experts) System

#### `MOEExpertRegistryV2_5`
Expert registration and management.

#### `MOEGatingNetworkV2_5`
Routing decisions and expert selection.

### Performance Monitoring

#### `PerformanceMetricV2_5`
System performance tracking.

#### `SystemResourceSnapshotV2_5`
Resource utilization monitoring.

## Configuration Models (`configuration_models.py`)

Comprehensive configuration management through modular sub-files:

- **`core_system_config.py`** - Core system, dashboard, and data management
- **`expert_ai_config.py`** - Expert systems, MOE, and AI configurations  
- **`trading_analytics_config.py`** - Trading parameters, performance, and analytics
- **`learning_intelligence_config.py`** - Learning systems, HuiHui, and intelligence

## Dashboard & UI Models (`dashboard_ui_models.py`)

### Dashboard Configuration

#### `DashboardModeType`
Dashboard mode enumeration.

#### `DashboardConfigV2_5`
Complete dashboard configuration.

### Compliance Tracking

#### `ComponentComplianceV2_5`
Component compliance monitoring.

## Validation & Best Practices

### Field Validation

All models include comprehensive field validation:

```python
# Automatic validation on creation
try:
    contract = RawOptionsContractV2_5(
        contract_symbol="SPY240315C00500000",
        strike=-500.0  # Invalid: negative strike
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Custom Validators

Models include custom validators for business logic:

```python
@field_validator('direction')
@classmethod
def validate_direction(cls, v):
    if v is not None and v not in ['Bullish', 'Bearish', 'Neutral']:
        raise ValueError("direction must be 'Bullish', 'Bearish', or 'Neutral'")
    return v
```

### Serialization

All models support JSON serialization:

```python
# Serialize to dict
data = signal.model_dump()

# Serialize to JSON string
json_str = signal.model_dump_json()

# Deserialize from dict
signal = SignalPayloadV2_5.model_validate(data)
```

## Migration Guide

### From Legacy Schemas

The consolidation maintains backward compatibility:

```python
# Old import (still works)
from data_models.deprecated_files.eots_schemas_v2_5 import SignalPayloadV2_5

# New import (recommended)
from data_models import SignalPayloadV2_5
```

### Breaking Changes

- Deprecated `datetime.utcnow()` replaced with `datetime.now(timezone.utc)`
- Some placeholder classes removed and replaced with proper implementations
- Enhanced validation may reject previously accepted invalid data

## Testing

Comprehensive test suite available in `tests/test_data_models.py`:

```bash
# Run all data model tests
pytest tests/test_data_models.py

# Run specific test class
pytest tests/test_data_models.py::TestSignalPayloadV2_5
```

## Performance Considerations

- Models use Pydantic v2 for optimal performance
- Field validation is performed at creation time
- Serialization is optimized for JSON output
- Memory usage reduced through efficient field definitions

## Future Enhancements

- Schema versioning support
- Advanced validation rules
- Performance monitoring integration
- Enhanced documentation generation
