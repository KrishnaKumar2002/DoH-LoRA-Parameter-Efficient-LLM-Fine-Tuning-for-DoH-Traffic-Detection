#!/usr/bin/env python3
"""Quick script to check CSV data."""

import pandas as pd

print("=" * 60)
print("CHECKING CSV FILES")
print("=" * 60)

try:
    df1 = pd.read_csv('data/merge_first_layer.csv')
    print("\n✅ Stage 1 (First Layer):")
    print(f"   Shape: {df1.shape}")
    print(f"   Columns: {df1.columns.tolist()[:5]}...")
    print(f"   DoH distribution:\n{df1['DoH'].value_counts()}")
except Exception as e:
    print(f"\n❌ Stage 1 Error: {e}")

try:
    df2 = pd.read_csv('data/merge_second_layer.csv')
    print("\n✅ Stage 2 (Second Layer):")
    print(f"   Shape: {df2.shape}")
    print(f"   Columns: {df2.columns.tolist()[:5]}...")
    print(f"   Label distribution:\n{df2['Label'].value_counts()}")
except Exception as e:
    print(f"\n❌ Stage 2 Error: {e}")

print("\n" + "=" * 60)
