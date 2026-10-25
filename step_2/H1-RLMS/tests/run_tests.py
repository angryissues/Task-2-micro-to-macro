"""Minimal fallback runner (used when pytest is not installed):  python tests/run_tests.py"""
import importlib
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
failed = passed = 0
for fname in sorted(os.listdir(os.path.join(ROOT, "tests"))):
    if not (fname.startswith("test_") and fname.endswith(".py")):
        continue
    mod = importlib.import_module(fname[:-3])
    for name in sorted(dir(mod)):
        if name.startswith("test_") and callable(getattr(mod, name)):
            t = time.time()
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {fname}::{name} ({time.time() - t:.1f}s)")
            except Exception:
                failed += 1
                print(f"FAIL {fname}::{name}")
                traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
