# EOTS v2.5 Data Models Migration Guide

## Overview

This guide helps you migrate from the legacy scattered data model files to the new consolidated structure. The migration maintains backward compatibility while providing enhanced features and validation.

## What Changed

### Before (Legacy Structure)
```
data_models/
├── deprecated_files/
│   ├── eots_schemas_v2_5.py (4000+ lines)
│   ├── ai_adaptations.py
│   ├── ai_predictions.py
│   ├── moe_schemas_v2_5.py
│   ├── learning_schemas.py
│   ├── hui_hui_schemas.py
│   ├── performance_schemas.py
│   ├── context_schemas.py
│   ├── signal_schemas.py
│   └── ... (20+ files)
```

### After (Consolidated Structure)
```
data_models/
├── core_models.py              # Base types, raw/processed data, bundles
├── configuration_models.py     # All configuration schemas
├── ai_ml_models.py            # AI/ML, MOE, HuiHui, performance
├── trading_market_models.py   # Trading, signals, market context
├── dashboard_ui_models.py     # Dashboard and UI components
├── validation_utils.py        # Validation utilities
└── __init__.py               # Backward compatibility exports
```

## Migration Steps

### Step 1: Update Imports (Recommended)

#### Old Imports
```python
# Legacy imports (still work but deprecated)
from data_models.deprecated_files.eots_schemas_v2_5 import SignalPayloadV2_5
from data_models.deprecated_files.hui_hui_schemas import HuiHuiExpertType
from data_models.deprecated_files.moe_schemas_v2_5 import MOEExpertRegistryV2_5
```

#### New Imports
```python
# New consolidated imports (recommended)
from data_models import SignalPayloadV2_5, HuiHuiExpertType, MOEExpertRegistryV2_5

# Or specific module imports
from data_models.trading_market_models import SignalPayloadV2_5
from data_models.ai_ml_models import HuiHuiExpertType, MOEExpertRegistryV2_5
```

### Step 2: Update Model Usage

Most model usage remains the same, but some improvements are available:

#### Enhanced Validation
```python
# Old: Limited validation
signal = SignalPayloadV2_5(
    signal_id="test",
    signal_name="Test Signal",
    symbol="SPY",
    timestamp=datetime.utcnow(),  # Deprecated
    signal_type="Invalid",        # Would be accepted
    strength_score=100.0          # Would be accepted
)

# New: Comprehensive validation
signal = SignalPayloadV2_5(
    signal_id="test",
    signal_name="Test Signal", 
    symbol="SPY",
    timestamp=datetime.now(timezone.utc),  # Recommended
    signal_type="Directional",             # Validated
    strength_score=2.5                     # Validated range
)
```

#### Improved Field Constraints
```python
# Old: No validation
contract = RawOptionsContractV2_5(
    contract_symbol="SPY240315C00500000",
    strike=-500.0,      # Negative strike accepted
    opt_kind="invalid", # Invalid type accepted
    dte_calc=30.0
)

# New: Automatic validation
try:
    contract = RawOptionsContractV2_5(
        contract_symbol="SPY240315C00500000",
        strike=-500.0,      # ValidationError: must be positive
        opt_kind="invalid", # ValidationError: must be 'call' or 'put'
        dte_calc=30.0
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Step 3: Handle Breaking Changes

#### DateTime Usage
```python
# Old (deprecated)
from datetime import datetime
timestamp = datetime.utcnow()

# New (recommended)
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc)
```

#### Enum Values
```python
# Old: String values
regime = "bullish_trend"

# New: Enum values (recommended)
from data_models import MarketRegimeState
regime = MarketRegimeState.BULLISH_TREND
```

#### Configuration Models
```python
# Old: Dict-based configuration
config = {
    "vapi_fa_bullish_thresh": 1.5,
    "vapi_fa_bearish_thresh": -1.5
}

# New: Validated configuration models
from data_models import DynamicThresholdsV2_5
config = DynamicThresholdsV2_5(
    vapi_fa_bullish_thresh=1.5,
    vapi_fa_bearish_thresh=-1.5
)
```

## Compatibility Matrix

| Component | Legacy Support | New Features | Migration Required |
|-----------|---------------|--------------|-------------------|
| Core Models | ✅ Full | Enhanced validation | Optional |
| Signal Models | ✅ Full | Field validation | Optional |
| AI/ML Models | ✅ Full | Complete implementations | Optional |
| Configuration | ✅ Full | Structured validation | Recommended |
| Dashboard Models | ✅ Full | Enhanced compliance | Optional |

## Common Migration Patterns

### Pattern 1: Simple Import Update
```python
# Before
from data_models.deprecated_files.eots_schemas_v2_5 import (
    SignalPayloadV2_5,
    KeyLevelV2_5,
    ProcessedDataBundleV2_5
)

# After
from data_models import (
    SignalPayloadV2_5,
    KeyLevelV2_5, 
    ProcessedDataBundleV2_5
)
```

### Pattern 2: Configuration Migration
```python
# Before: Dict-based config
def create_signal_config():
    return {
        "signal_types": ["Directional", "Volatility"],
        "strength_threshold": 1.0,
        "confidence_threshold": 0.8
    }

# After: Validated config model
from data_models.configuration_models import SignalGeneratorConfig

def create_signal_config():
    return SignalGeneratorConfig(
        signal_types=["Directional", "Volatility"],
        strength_threshold=1.0,
        confidence_threshold=0.8
    )
```

### Pattern 3: Enhanced Validation
```python
# Before: Manual validation
def create_key_level(price, level_type, conviction):
    if price <= 0:
        raise ValueError("Price must be positive")
    if level_type not in ["Support", "Resistance"]:
        raise ValueError("Invalid level type")
    if not 0 <= conviction <= 1:
        raise ValueError("Conviction must be 0-1")
    
    return KeyLevelV2_5(
        level_price=price,
        level_type=level_type,
        conviction_score=conviction
    )

# After: Automatic validation
def create_key_level(price, level_type, conviction):
    # Validation happens automatically
    return KeyLevelV2_5(
        level_price=price,      # Auto-validated: must be > 0
        level_type=level_type,  # Auto-validated: must be valid type
        conviction_score=conviction  # Auto-validated: must be 0-5
    )
```

## Testing Your Migration

### 1. Import Tests
```python
# Test that all your imports still work
try:
    from data_models import (
        SignalPayloadV2_5,
        KeyLevelV2_5,
        ProcessedDataBundleV2_5,
        # ... all your models
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

### 2. Model Creation Tests
```python
# Test that your existing model creation still works
try:
    signal = SignalPayloadV2_5(
        signal_id="test",
        signal_name="Test",
        symbol="SPY",
        timestamp=datetime.now(timezone.utc),
        signal_type="Directional",
        strength_score=1.0
    )
    print("✅ Model creation successful")
except Exception as e:
    print(f"❌ Model creation error: {e}")
```

### 3. Serialization Tests
```python
# Test JSON serialization/deserialization
signal_dict = signal.model_dump()
reconstructed = SignalPayloadV2_5.model_validate(signal_dict)
assert reconstructed.signal_name == signal.signal_name
print("✅ Serialization working")
```

## Performance Impact

### Improvements
- **Faster imports**: Reduced from 20+ files to 6 modules
- **Better validation**: Pydantic v2 performance optimizations
- **Memory efficiency**: Optimized field definitions
- **Type safety**: Enhanced IDE support and error detection

### Potential Issues
- **Stricter validation**: Some previously accepted invalid data may now raise errors
- **Import changes**: Deprecated imports may show warnings

## Rollback Plan

If you encounter issues, you can temporarily use legacy imports:

```python
# Temporary fallback to legacy imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from data_models.deprecated_files.eots_schemas_v2_5 import SignalPayloadV2_5
```

## Getting Help

### Common Issues

1. **ValidationError on model creation**
   - Check field constraints in the documentation
   - Ensure all required fields are provided
   - Verify field types match expectations

2. **Import errors**
   - Update import statements to use new structure
   - Check for typos in model names
   - Ensure you're importing from the correct module

3. **Serialization issues**
   - Use `model_dump()` instead of `dict()`
   - Use `model_validate()` instead of direct construction

### Support Resources

- **Documentation**: `docs/data_models_guide.md`
- **Test Examples**: `tests/test_data_models.py`
- **API Reference**: Generated from docstrings
- **Migration Support**: Contact the development team

## Timeline

- **Phase 1** (Current): Legacy imports supported with deprecation warnings
- **Phase 2** (Next release): Legacy files moved to archive
- **Phase 3** (Future): Legacy files removed entirely

We recommend migrating to the new structure as soon as possible to take advantage of enhanced validation and improved maintainability.
