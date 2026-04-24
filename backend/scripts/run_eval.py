"""Standalone evaluation and comparison table generator."""
import sys
sys.path.insert(0, ".")

from evaluation.comparisons import generate_comparison_table
import pandas as pd

if __name__ == "__main__":
    print("Generating comparison table...")
    df = generate_comparison_table("./logs")
    print("\n=== AFSPL Experiment Comparison ===")
    print(df.to_string(index=False))
    print(f"\nSaved to ./logs/comparison_table.csv")
