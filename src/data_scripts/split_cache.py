"""Disk-backed cache for data splits.

Cache entries are keyed by a dict of all split-determining parameters (mode,
seed, fractions, iso-path mtime/size, neg_ratio).  Each entry is stored as a
single pickle file so a cache hit avoids reading the source CSV, running
train_test_split, and sampling gene–isoform negatives entirely.

Usage:
    cache = SplitCache('.cache/data_splits')
    key   = {'mode': 'inductive', 'seed': 42, ...}
    data  = cache.get(key)          # None on miss
    if data is None:
        data = build_splits(...)
        cache.put(key, data)
"""

import hashlib
import json
import os
import pickle


class SplitCache:
    def __init__(self, cache_dir='.cache/data_splits'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key_dict):
        serialised = json.dumps(key_dict, sort_keys=True, default=str)
        h = hashlib.sha256(serialised.encode()).hexdigest()[:24]
        return os.path.join(self.cache_dir, f'split_{h}.pkl')

    def get(self, key_dict):
        """Return the cached data dict, or None on a miss or corrupt file."""
        path = self._path(key_dict)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def put(self, key_dict, data_dict):
        """Persist data_dict to disk; return the file path written."""
        path = self._path(key_dict)
        with open(path, 'wb') as f:
            pickle.dump(data_dict, f)
        return path

    def describe(self, key_dict):
        """Return the cache file path for this key (for diagnostic messages)."""
        return self._path(key_dict)
