# Parallel Longest Common Subsequence (LCS) with OpenMP

This repository contains a high-performance C implementation of the Longest Common Subsequence (LCS) algorithm. The project leverages OpenMP and Blocked Dynamic Programming (BDP) to optimize parallel execution across multi-core architectures. 

## Technical Approach
*   **Blocked Dynamic Programming (BDP):** The scoring matrix is divided into independent blocks (tiles). Processing is parallelized at the block level using `#pragma omp task depend`, ensuring that data dependencies (top, left, and top-left diagonal blocks) are strictly maintained while drastically reducing task management overhead.
*   **Cache Optimization:** Tiles are sized at 128x128 elements (2 bytes each), occupying exactly 32 KB. This perfectly aligns with the L1 cache capacity per core on the target architecture, preventing cache misses and false sharing.

## Benchmarking & Environment Setup
To ensure statistically significant and reliable performance metrics, the benchmarking environment was strictly controlled to prevent dynamic hardware scaling:
*   **CPU Governor:** Forced continuous maximum frequency (`sudo cpufreq-set -g performance`).
*   **Turbo Boost:** Disabled to eliminate automatic clock variations.
*   **Process Isolation:** Managed CPU allocation via `cgroups` and deprioritized background processes using `renice`.
*   **OpenMP Tuning:** Bound threads physically close to cores (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`) using dynamic scheduling (`OMP_SCHEDULE=dynamic`).

## Compilation
Aggressive optimization and auto-vectorization were enabled using GCC 13.3.0:
```bash
gcc -O3 -ftree-vectorize -fopenmp lcs.c -o lcs
