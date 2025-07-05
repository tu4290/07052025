from typing import Any, TypeVar, Generic
from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
import pandas as pd
from pydantic import GetCoreSchemaHandler


class DataFrameSchema(Generic[TypeVar('T')]):
    """
    Pydantic v2 custom type for pandas DataFrames with schema validation.
    
    Usage:
        class MyModel(BaseModel):
            df: DataFrameSchema[MyRowModel]  # Validates each row against MyRowModel
    
    Attributes:
        schema: The Pydantic model to validate each row against
        strict: If True, raises ValidationError if any rows fail validation
    """
    def __class_getitem__(cls, item):
        return type(f'DataFrameSchema[{item.__name__}]', (cls,), {'__schema__': item})
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        schema = cls.__dict__.get('__schema__')
        if schema is None:
            raise ValueError("DataFrameSchema must be parameterized with a row type")
            
        def validate(v: Any) -> pd.DataFrame:
            if not isinstance(v, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(v).__name__}")
                
            # Convert each row to the schema and back to catch validation errors
            try:
                rows = [schema.model_validate(row.to_dict()) for _, row in v.iterrows()]
                return pd.DataFrame([row.model_dump() for row in rows])
            except Exception as e:
                raise ValueError(f"DataFrame validation failed: {str(e)}")
        
        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda df: df.to_dict('records'),
                when_used='json',
            ),
        )
    
    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = cls.__dict__.get('__schema__')
        return {
            'type': 'array',
            'items': handler(schema.__pydantic_core_schema__) if hasattr(schema, '__pydantic_core_schema__') else {},
            'title': f'DataFrame[{schema.__name__ if hasattr(schema, "__name__") else str(schema)}]',
        }


# Backward compatible type alias that can be used when full schema validation isn't needed
PandasDataFrame = pd.DataFrame

# Note: This file can be expanded with other common simple type aliases
# or very basic shared Pydantic models if they arise in the future.
# For now, it primarily serves to define PandasDataFrame for type hinting.

# Example of another base type if needed in the future:
# class Identifier(BaseModel):
#     id: str = Field(..., description="A unique identifier.")
#     system: Optional[str] = Field(None, description="The system or namespace of the identifier.")

"""
Base type definitions used across various EOTS v2.5 data models.
This module centralizes common type aliases or very simple base Pydantic models
to promote consistency and ease of maintenance.
"""