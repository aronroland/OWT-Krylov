#!/usr/bin/env python3
"""Run matrix-analysis regressions and retain their logs and source hashes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unittest


def main():
    source = Path(__file__).resolve().parent
    root = source.parents[1]
    output = root / "build/matrix-analysis-tests"
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results.json"
    history = json.loads(results.read_text()) if results.exists() else []
    log = output / f"test-{len(history)+1:03d}.log"
    suite = unittest.defaultTestLoader.discover(str(source), pattern="test_*.py")
    with log.open("x") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    evidence = {"finished": datetime.now(timezone.utc).isoformat(), "passed": result.wasSuccessful(),
        "tests": result.testsRun, "log": str(log), "command": sys.argv,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for pattern in ("*.py", "*.cpp") for p in source.glob(pattern)},
        "fixture_audit": json.loads((root / "build/ilu-audit/results.json").read_text())[-1]}
    history.append(evidence)
    results.write_text(json.dumps(history, indent=2)+"\n")
    print(log.read_text(), end="")
    print("Evidence:", results)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
