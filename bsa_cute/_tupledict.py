# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copied verbatim from python/cudnn/api_base.py (cudnn-frontend v1.28.0) so the
# vendored block-sparse attention package does not import the cudnn package.

class TupleDict(dict):
    """A dictionary that supports tuple unpacking.

    This class extends dict to allow unpacking like a tuple while still
    providing dictionary-style key access. The unpacking order is determined
    by the _keys attribute which preserves insertion order.

    Example:
        >>> result = TupleDict(a=1, b=2, c=3)
        >>> x, y, z = result  # Unpacks as (1, 2, 3)
        >>> result['a']  # Returns 1
        >>> result[0]  # Returns 1 (integer indexing)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store keys in order for tuple unpacking
        self._keys = list(self.keys())

    def __iter__(self):
        """Iterate over values in insertion order for tuple unpacking."""
        return (self[k] for k in self._keys)

    def __getitem__(self, key):
        """Support both string keys and integer indices."""
        if isinstance(key, int):
            if key < 0 or key >= len(self._keys):
                raise IndexError(f"index {key} out of range for TupleDict with {len(self._keys)} items")
            return super().__getitem__(self._keys[key])
        return super().__getitem__(key)
