"""
Common constants for entity matching models.
"""

# OpenAI and other model costs per 1M tokens: (input_cost, output_cost)
MODEL_COSTS = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.2, 0.8),
    "o3": (0.4, 1.6),
    "o3-mini": (0.1, 0.4),
    "o4": (10.0, 30.0),
    "o4-mini": (0.15, 0.60),
    "claude-4-sonnet": (3.0, 15.0),
}

__all__ = ["MODEL_COSTS"] 