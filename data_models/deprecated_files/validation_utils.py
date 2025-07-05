"""
Validation & Utility Functions for EOTS v2.5

Consolidated from: validators.py
"""


# FROM validators.py
"""
Centralized validation functions for Pydantic models.
"""
from typing import Optional

def validate_probability(v: Optional[float]) -> Optional[float]:
    """Validate that a value is a probability (between 0.0 and 1.0)."""
    if v is not None and not (0.0 <= v <= 1.0):
        raise ValueError("Value must be between 0.0 and 1.0")
    return v

def validate_neg_one_to_one(v: Optional[float]) -> Optional[float]:
    """Validate that a value is within the range [-1.0, 1.0]."""
    if v is not None and not (-1.0 <= v <= 1.0):
        raise ValueError("Value must be between -1.0 and 1.0")
    return v

def validate_non_negative(v: Optional[float]) -> Optional[float]:
    """Validate that a value is non-negative."""
    if v is not None and v < 0:
        raise ValueError("Value must be non-negative")
    return v