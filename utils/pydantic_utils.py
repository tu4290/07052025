from pydantic import BaseModel
from typing import Any, Dict

def normalize_config(raw_config: Dict[str, Any], model: BaseModel) -> Dict[str, Any]:
    """
    Normalizes a raw configuration dictionary against a Pydantic model.
    This function attempts to convert the raw_config into a format that
    can be validated by the given Pydantic model, handling cases where
    the raw config might contain extra fields or fields with incorrect types
    that Pydantic's .model_validate() would otherwise reject.

    Args:
        raw_config (Dict[str, Any]): The raw configuration dictionary.
        model (BaseModel): The Pydantic model to normalize against.

    Returns:
        Dict[str, Any]: A normalized dictionary that should be compatible
                        with the Pydantic model.
    """
    normalized = {}
    for field_name, field_info in model.model_fields.items():
        if field_name in raw_config:
            # Attempt to convert value to the expected type if necessary
            # This is a basic attempt; more complex type conversions might be needed
            # based on the specific Pydantic model field types.
            if field_info.annotation == bool and isinstance(raw_config[field_name], str):
                normalized[field_name] = raw_config[field_name].lower() == 'true'
            else:
                normalized[field_name] = raw_config[field_name]
        elif field_info.default is not None:
            normalized[field_name] = field_info.default
        # If a field is required and not in raw_config and has no default, it will be missing.
        # Pydantic's validation will catch this later.

    return normalized
