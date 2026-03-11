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
Synthetic pipeline test eigenvalues: [ 5.41873805e-03  5.38299793e-03  5.30065220e-03  5.22036788e-03
  4.96650354e-03  4.94198383e-03  4.91917346e-03  4.84505662e-03
  4.76604757e-03  4.69153772e-03  4.63035120e-03  4.49873483e-03
  4.44092947e-03  4.37798861e-03  4.36196399e-03  4.35379995e-03
  4.27438800e-03  4.21829465e-03  4.19331819e-03  4.07573412e-03
  4.01644403e-03  3.96184297e-03  3.93625058e-03  3.90206129e-03
  3.86505401e-03  3.78295039e-03  3.70533902e-03  3.64126202e-03
  3.62396020e-03  3.55061702e-03  3.53811700e-03  3.53237728e-03
  3.44295938e-03  3.41587031e-03  3.36160036e-03  3.31783272e-03
  3.31636696e-03  3.27187427e-03  3.19404678e-03  3.16236643e-03
  3.12635479e-03  3.10682456e-03  2.99846811e-03  2.97755062e-03
  2.94944232e-03  2.90486986e-03  2.84498480e-03  2.83148612e-03
  2.79261077e-03  2.73798033e-03  2.70602986e-03  2.63609523e-03
  2.60545564e-03  2.59165685e-03  2.52577954e-03  2.50283649e-03
  2.42675064e-03  2.36479532e-03  2.31584020e-03  2.25629367e-03
  2.17724776e-03  2.14719920e-03  2.13298368e-03 -2.93193974e-18]
```

## Usage

Run `PYTHONPATH=./vex python3 vex/kernel/coordizer_v2/eigenvalue_analysis.py` from the project root.
