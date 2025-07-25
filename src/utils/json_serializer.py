#!/usr/bin/env python3
"""
Comprehensive JSON serialization utility to handle all numpy/pandas types.

This module provides a single, robust solution for JSON serialization that handles:
- numpy integers, floats, arrays, and other types
- pandas data types
- nested dictionaries with numpy keys/values
- complex objects with __dict__ attributes

Usage:
    from src.utils.json_serializer import json_serialize, json_dumps

    # Convert any object to JSON-serializable format
    clean_data = json_serialize(your_data)

    # Direct JSON string conversion
    json_string = json_dumps(your_data)
"""

import json

from typing import Any

import numpy as np
import pandas as pd


def json_serialize(obj: Any) -> Any:
    """
    Convert any object to a JSON-serializable format.

    This function recursively processes objects and converts:
    - numpy types to native Python types
    - pandas types to native Python types
    - dictionaries with numpy keys to string keys
    - complex objects to their dictionary representation

    Args:
        obj: Any object to serialize

    Returns:
        JSON-serializable version of the object
    """

    # Handle None
    if obj is None:
        return None

    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.str_):
        return str(obj)

    # Handle pandas types
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
        return json_serialize(obj.item()) if hasattr(obj, 'item') else str(obj)
    if hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
        return int(obj)
    if hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
        return float(obj)

    # Handle native Python types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle collections
    if isinstance(obj, dict):
        # Clean both keys and values
        cleaned_dict = {}
        for k, v in obj.items():
            # Convert numpy/pandas keys to appropriate types
            if isinstance(k, (np.integer, np.int64)):
                clean_key = int(k)
            elif isinstance(k, (np.floating, np.float64)):
                clean_key = float(k)
            elif isinstance(k, (str, int, float, bool)) or k is None:
                clean_key = k
            else:
                clean_key = str(k)  # Convert other types to string

            cleaned_dict[clean_key] = json_serialize(v)
        return cleaned_dict

    if isinstance(obj, (list, tuple, set)):
        return [json_serialize(item) for item in obj]

    # Handle objects with __dict__ (dataclasses, custom objects)
    if hasattr(obj, '__dict__'):
        return json_serialize(obj.__dict__)

    # Handle objects with _asdict (namedtuples)
    if hasattr(obj, '_asdict'):
        return json_serialize(obj._asdict())

    # Handle objects that can be converted to dict
    if hasattr(obj, 'to_dict'):
        return json_serialize(obj.to_dict())

    # Handle iterables
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            return [json_serialize(item) for item in obj]
        except TypeError:
            pass

    # Fallback: convert to string
    return str(obj)


def json_dumps(obj: Any, **kwargs) -> str:
    """
    Convert object to JSON string with comprehensive type handling.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string representation
    """
    cleaned_obj = json_serialize(obj)
    return json.dumps(cleaned_obj, **kwargs)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely convert object to JSON string with error handling.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string representation, or error message if serialization fails
    """
    try:
        return json_dumps(obj, **kwargs)
    except Exception as e:
        # Return a safe error representation
        return json.dumps({
            "error": f"JSON serialization failed: {e!s}",
            "type": str(type(obj)),
            "repr": repr(obj)[:200] + "..." if len(repr(obj)) > 200 else repr(obj)
        })
