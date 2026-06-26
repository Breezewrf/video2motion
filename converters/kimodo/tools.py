# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities: batching decorator, tensor conversion."""

import inspect
from collections.abc import Mapping
from functools import wraps
from math import prod
from typing import Any, Callable, ParamSpec, TypeVar

import numpy as np
import torch

T = TypeVar("R")
P = ParamSpec("P")
R = TypeVar("R")

Tensor = Any


def ensure_batched(**spec: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to flatten complex batch dimensions.

    Fixes included:
    1. Handles 1D tensors (tail_ndim=0) correctly without slicing errors.
    2. Skips .reshape() if the input is already purely flat (Optimization).
    """
    if not spec:
        raise ValueError("At least one argument spec must be provided.")

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            def _sequence_shape(name: str, value: Any) -> tuple[int, ...]:
                if not isinstance(value, (list, tuple)):
                    return ()
                if len(value) == 0:
                    return (0,)
                first_shape = _sequence_shape(name, value[0])
                for item in value[1:]:
                    item_shape = _sequence_shape(name, item)
                    if item_shape != first_shape:
                        raise ValueError(f"'{name}' must be a rectangular nested sequence, got ragged shape.")
                return (len(value), *first_shape)

            def _shape_and_ndim(name: str, value: Any) -> tuple[tuple[int, ...], int]:
                if hasattr(value, "shape") and hasattr(value, "ndim"):
                    shape = tuple(value.shape)
                    return shape, int(value.ndim)
                if isinstance(value, (list, tuple)):
                    shape = _sequence_shape(name, value)
                    return shape, len(shape)
                raise TypeError(f"'{name}' must be tensor-like or a nested list/tuple, got {type(value)}.")

            def _reshape_like(value: Any, shape: tuple[int, ...], name: str) -> Any:
                if hasattr(value, "reshape"):
                    return value.reshape(*shape)

                if not isinstance(value, (list, tuple)):
                    raise TypeError(f"Cannot reshape '{name}' of type {type(value)}.")

                flat: list[Any] = []

                def _flatten(x: Any) -> None:
                    if isinstance(x, (list, tuple)):
                        for item in x:
                            _flatten(item)
                    else:
                        flat.append(x)

                _flatten(value)
                expected_size = prod(shape) if shape else 1
                if len(flat) != expected_size:
                    raise ValueError(f"Cannot reshape '{name}' with {len(flat)} elements into shape {shape}.")

                def _build(index: int, dims: tuple[int, ...]) -> tuple[Any, int]:
                    if not dims:
                        return flat[index], index + 1
                    items = []
                    for _ in range(dims[0]):
                        item, index = _build(index, dims[1:])
                        items.append(item)
                    return items, index

                rebuilt, used = _build(0, shape)
                if used != len(flat):
                    raise ValueError(f"Internal reshape error for '{name}': used {used}/{len(flat)} elements.")
                if isinstance(value, tuple) and isinstance(rebuilt, list):
                    return tuple(rebuilt)
                return rebuilt

            # --- 1. CANONICAL ARGUMENT ---
            spec_items = list(spec.items())
            canonical_name = None
            canonical_ndim = None
            x0 = None
            for name, ndim in spec_items:
                candidate = bound.arguments.get(name, None)
                if candidate is not None:
                    canonical_name = name
                    canonical_ndim = ndim
                    x0 = candidate
                    break
            if canonical_name is None:
                raise ValueError(
                    "All canonical candidates are None: " + ", ".join(f"'{name}'" for name, _ in spec_items)
                )

            expected_tail_dims = canonical_ndim - 1
            x0_shape, x0_ndim = _shape_and_ndim(canonical_name, x0)

            if x0_ndim < expected_tail_dims:
                raise ValueError(f"'{canonical_name}' ndim={x0_ndim} < expected {expected_tail_dims} tail dims.")

            if expected_tail_dims == 0:
                orig_batch_shape = x0_shape
                tail_shape = ()
            else:
                orig_batch_shape = x0_shape[:-expected_tail_dims]
                tail_shape = x0_shape[-expected_tail_dims:]

            B_flat = prod(orig_batch_shape) if orig_batch_shape else 1

            is_unbatched_input = len(orig_batch_shape) == 0

            is_already_flat = len(orig_batch_shape) == 1

            if is_unbatched_input:
                x0_batched = _reshape_like(x0, (1, *tail_shape), canonical_name)
            elif is_already_flat:
                x0_batched = x0
            else:
                x0_batched = _reshape_like(x0, (B_flat, *tail_shape), canonical_name)

            bound.arguments[canonical_name] = x0_batched

            # --- 2. OTHER ARGUMENTS ---
            for name, target_ndim in spec_items:
                if name == canonical_name:
                    continue
                val = bound.arguments.get(name, None)
                if val is None:
                    continue

                arg_tail_dims = target_ndim - 1
                val_shape, val_ndim = _shape_and_ndim(name, val)

                if val_ndim < arg_tail_dims:
                    raise ValueError(f"'{name}' ndim={val_ndim} too small.")

                if arg_tail_dims == 0:
                    val_batch_shape = val_shape
                    val_tail_shape = ()
                else:
                    val_batch_shape = val_shape[:-arg_tail_dims]
                    val_tail_shape = val_shape[-arg_tail_dims:]

                if len(val_batch_shape) == 0:
                    if not is_unbatched_input:
                        raise ValueError(f"'{name}' is unbatched but canonical is batched.")
                    val_batched = _reshape_like(val, (1, *val_tail_shape), name)
                else:
                    if val_batch_shape != orig_batch_shape:
                        raise ValueError(
                            f"Batch dimensions mismatch! '{canonical_name}' has {orig_batch_shape}, "
                            f"but '{name}' has {val_batch_shape}."
                        )

                    if is_already_flat:
                        val_batched = val
                    else:
                        val_batched = _reshape_like(val, (B_flat, *val_tail_shape), name)

                bound.arguments[name] = val_batched

            # --- 3. EXECUTION ---
            out = fn(**bound.arguments)

            # --- 4. RESTORE ---
            def restore(obj):
                if isinstance(obj, Mapping):
                    return {k: restore(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return type(obj)(restore(x) for x in obj)

                if hasattr(obj, "shape"):
                    if obj.ndim == 0:
                        return obj

                    if obj.shape[0] != B_flat:
                        return obj

                    if is_already_flat:
                        return obj

                    rest = obj.shape[1:]

                    if is_unbatched_input:
                        assert obj.shape[0] == 1, "The batch size should be 1 for unbatched."
                        return obj[0]

                    return obj.reshape(*orig_batch_shape, *rest)
                return obj

            return restore(out)

        return wrapper

    return decorator


def to_numpy(obj):
    """Recursively convert tensors in dicts/lists/tuples to numpy arrays; leave other types
    unchanged."""
    if isinstance(obj, Mapping):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(x) for x in obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    return obj


def to_torch(obj, device=None, dtype=None):
    """Recursively convert numpy arrays in dicts/lists/tuples to torch tensors; optionally move to
    device/dtype."""
    if isinstance(obj, Mapping):
        return {k: to_torch(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_torch(x, device, dtype) for x in obj)
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if isinstance(obj, torch.Tensor):
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        if device is None:
            return obj
        return obj.to(device)
    return obj
