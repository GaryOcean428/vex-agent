# Eigenvalue Analysis Pipeline Results

## What this verifies

The synthetic pipeline test verifies the entire compression pipeline using synthetic data:

1. Construct a `HarvestResult` object with `token_fingerprints` populated by Dirichlet samples and `vocab_size` set.
2. Pass this HarvestResult to `compress()` to obtain a CompressionResult.
3. Read the eigenvalues directly from `CompressionResult.eigenvalues`.

## Purpose

This test validates that the compression pipeline is operational end-to-end with synthetic data. It ensures HarvestResult → compress() → CompressionResult → eigenvalue analysis works as expected.

## Results

The successful run produced the following eigenvalue spectrum:

```
Synthetic pipeline test eigenvalues: [2.431 2.287 2.207 2.091 2.04  1.988 1.956 1.884 1.851 1.796 1.729 1.678 1.587 1.541 1.535 1.52  1.495 1.484 1.451 1.421 1.39  1.363 1.33  1.307 1.291 1.258 1.241 1.202 1.176 1.17  1.133 1.112 1.085 1.083 1.048 1.034 0.998 0.981 0.95  0.932 0.924 0.893 0.882 0.861 0.856 0.83  0.817 0.782 0.779 0.757 0.742 0.717 0.71  0.707 0.681 0.657 0.627 0.613 0.59  0.569 0.564 0.527 0.484 0.452 0.417]
```

## Usage

Run `PYTHONPATH=./vex python3 vex/kernel/coordizer_v2/eigenvalue_analysis.py` from the project root.
