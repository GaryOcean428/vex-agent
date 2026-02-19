"""Coordizer — Euclidean → Fisher-Rao coordinate transformation.

The coordizer provides geometric purity enforcement at data ingestion points,
transforming Euclidean embeddings to Fisher-Rao coordinates on the probability
simplex.

Key Functions:
    coordize: Transform embedding to Fisher-Rao coordinates
    validate_simplex: Validate simplex properties
    
Key Classes:
    CoordinatorPipeline: End-to-end coordinate transformation pipeline
    
Usage:
    >>> from kernel.coordizer import coordize
    >>> import numpy as np
    >>> 
    >>> # Euclidean embedding from LLM
    >>> embedding = np.array([0.5, -0.3, 0.8, -0.1])
    >>> 
    >>> # Transform to Fisher-Rao coordinates
    >>> coordinates = coordize(embedding)
    >>> 
    >>> # Verify simplex properties
    >>> assert np.all(coordinates >= 0), "Non-negative"
    >>> assert np.isclose(coordinates.sum(), 1.0), "Sum to 1"
"""

from .transform import coordize, coordize_batch
from .validate import validate_simplex, ensure_simplex
from .pipeline import CoordinatorPipeline
from .types import CoordinateTransform, ValidationResult, TransformMethod

__all__ = [
    "coordize",
    "coordize_batch",
    "validate_simplex",
    "ensure_simplex",
    "CoordinatorPipeline",
    "CoordinateTransform",
    "ValidationResult",
    "TransformMethod",
]

__version__ = "0.1.0"
