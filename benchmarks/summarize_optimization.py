"""Summarize paired benchmark CSV without discarding slower observations."""
import csv
import statistics
import sys
from collections import defaultdict

pairs = defaultdict(dict)
with open(sys.argv[1], newline="", encoding="ascii") as source:
    for row in csv.DictReader(source):
        if row["phase"] not in {"products", "preconditioner", "solve"}:
            continue
        key = (row["precision"], int(row["side"]), int(row["block"]), row["phase"], int(row["sample"]))
        if row["variant"] in pairs[key]:
            raise ValueError(f"duplicate observation: {key}")
        pairs[key][row["variant"]] = row

groups = defaultdict(list)
for key, pair in pairs.items():
    if set(pair) != {"native", "reference"}:
        raise ValueError(f"unpaired observation: {key}")
    for field in ("iterations", "operator_apps", "preconditioner_apps", "reductions"):
        if pair["native"][field] != pair["reference"][field]:
            raise ValueError(f"solver work differs for {key}: {field}")
    groups[key[:-1]].append((float(pair["reference"]["seconds"]), float(pair["native"]["seconds"])))

print("precision,side,block,phase,pairs,reference_median_seconds,native_median_seconds,median_paired_speedup,min_paired_speedup,max_paired_speedup")
for key, samples in sorted(groups.items()):
    reference, native = zip(*samples)
    ratios = [a / b for a, b in samples]
    print(",".join(map(str, (*key, len(samples), statistics.median(reference),
                            statistics.median(native), statistics.median(ratios), min(ratios), max(ratios)))))
