#!/usr/bin/env python
import pandas as pd

# Check iTunes-Amazon dataset
tableA = pd.read_csv("data/raw/itunes_amazon/tableA.csv")
tableB = pd.read_csv("data/raw/itunes_amazon/tableB.csv")
test_pairs = pd.read_csv("data/raw/itunes_amazon/test.csv")

print(f"TableA: {len(tableA)} records, ID range: {tableA.id.min()}-{tableA.id.max()}")
print(f"TableB: {len(tableB)} records, ID range: {tableB.id.min()}-{tableB.id.max()}")
print(f"Test pairs: {len(test_pairs)} pairs")
print(f"Test ltable_id range: {test_pairs.ltable_id.min()}-{test_pairs.ltable_id.max()}")
print(f"Test rtable_id range: {test_pairs.rtable_id.min()}-{test_pairs.rtable_id.max()}")

# Check if all test IDs exist in tables
ltable_ids_missing = set(test_pairs.ltable_id) - set(tableA.id)
rtable_ids_missing = set(test_pairs.rtable_id) - set(tableB.id)

print(f"Missing ltable_ids: {len(ltable_ids_missing)}")
print(f"Missing rtable_ids: {len(rtable_ids_missing)}")
if len(ltable_ids_missing) > 0:
    print(f"  First few missing: {sorted(ltable_ids_missing)[:10]}")
if len(rtable_ids_missing) > 0:
    print(f"  First few missing: {sorted(rtable_ids_missing)[:10]}")
