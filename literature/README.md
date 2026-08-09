# OWT-Krylov literature

This directory indexes the local research material used to design and validate
OWT-Krylov. PDF files live under `literature/papers/` and are intentionally
ignored by Git. The tracked index records provenance and prevents accidental
duplication or inclusion of unrelated downloads.

Source collection:

`/home/aron/git/SpecWave/TRITON-C/literature/le-solvers/`

Reviews:

- [Literature review](LITERATURE_REVIEW.md)
- [GitHub repository review](REPOSITORY_REVIEW.md)
- [SpecWave implementation port matrix](../docs/SPECWAVE_PORT_MATRIX.md)

## Foundations

| Local file | Work | Relevance |
|---|---|---|
| `01_0907058.pdf` | Saad and Schultz, *GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems* | Restarted GMRES foundation. |
| `02_0913035.pdf` | van der Vorst, *Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG* | BiCGSTAB foundation. |

## Parallelism, communication, and numerical reliability

| Local file | Work | Relevance |
|---|---|---|
| `03_Carson_Demmel_2014_Residual_Replacement.pdf` | Carson and Demmel, *A Residual Replacement Strategy for Improving the Maximum Attainable Accuracy of s-step Krylov Subspace Methods* | True-residual checks and residual replacement. |
| `04_Cools_2017_Communication_Avoiding_BiCGSTAB.pdf` | Cools and Vanroose, *The Communication-Hiding Pipelined BiCGStab Method for the Parallel Solution of Large Unsymmetric Linear Systems* | Hiding global reductions behind operator work. |
| `05_Havdiak_2024_Pipelined_BiCGSTAB.pdf` | Havdiak, Aliaga, and Iakymchuk, *Robustness and Accuracy in Pipelined Bi-Conjugate Gradient Stabilized Method* | Accuracy and robustness tradeoffs in pipelined BiCGSTAB. |
| `16_Numerical_Reliability_Krylov_2023.pdf` | Iakymchuk and Aliaga, *General Framework for Re-assuring Numerical Reliability in Parallel Krylov Solvers* | Reproducibility and floating-point reliability. |

## Preconditioning and domain decomposition

| Local file | Work | Relevance |
|---|---|---|
| `06_HDG_GPU_Preconditioning_2024.pdf` | Welter and Nguyen (2025 preprint), *Preconditioning Techniques for Hybridizable Discontinuous Galerkin Discretizations on GPU Architectures* | Sparse unstructured-discretization preconditioners. The filename year is incorrect. |
| `10_Pearson_2020_Preconditioners_Overview.pdf` | Pearson and Pestana, *Preconditioners for Krylov Subspace Methods: An Overview* | General preconditioner taxonomy and solver compatibility. |
| `18_Two_Level_GPU_ILU_2025.pdf` | Xu, Li, and Osei-Kuffuor (2023 preprint; 2025 journal article), *A Two-level GPU-Accelerated Incomplete LU Preconditioner for General Sparse Linear Systems* | Distributed domain decomposition, local ILU, and coarse correction. |

## Surveys and learned acceleration

| Local file | Work | Relevance |
|---|---|---|
| `13_Sparse_Solvers_Survey_EDA_2025.pdf` | Rai, *A Technical Survey of Sparse Linear Solvers in Electronic Design Automation* | Broad sparse-solver comparison. |
| `08_NOWS_Neural_Warm_Starts_2025.pdf` | Eshaghi et al., *NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers* | Learned initial guesses. |
| `09_Online_Learned_Preconditioners_2025.pdf` | Khodak et al., *One-shot Acceleration of Transient PDE Solvers via Online-learned Preconditioners* | Updating preconditioners across related systems. |
| `19_Learning_SPAI_Preconditioners_GPU.pdf` | Yang et al., *Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs* | Learned sparse approximate inverses. |
| `21_Graph_Neural_Preconditioners_2024.pdf` | Chen (ICLR 2025), *Graph Neural Preconditioners for Iterative Solutions of Sparse Linear Systems* | Graph-aware learned preconditioning. |
| `22_GNN_Preconditioner_CG_2024.pdf` | Trifonov et al. (2025 revision), *Learning from Linear Algebra: A Graph Neural Network Approach to Preconditioner Design for Conjugate Gradient Solvers* | Graph-based preconditioner design. |

## Excluded source files

| Source file | Reason |
|---|---|
| `07_NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers.pdf` | Alternate rendering of the NOWS work; `08` is retained as the canonical copy. |
| `11_Pipelined_BiCGSTAB_arXiv_2024.pdf` | Invalid/incomplete one-page download with no extractable paper text. |
| `12_Sonneveld_vanGijzen_IDRs.pdf` | Mislabelled file containing an unrelated Einstein--Yang--Mills paper. |
| `14_Pipelined_BiCGSTAB_Robustness_2024.pdf` | Byte-for-byte duplicate of `05`. |

## SpecWave implementation references

The literature should be read alongside these existing implementations and
design notes in the SpecWave working copy:

- `TRITON-C/libwwx/GMRES.hpp`
- `TRITON-C/libwwx/BiCGSTAB.hpp`
- `TRITON-C/libwwx/IDRs.hpp`
- `TRITON-C/libwwx/ILU0.hpp`
- `TRITON-C/libwwx/Multigrid.hpp`
- `TRITON-C/libwwx/Exchange.hpp`
- `TRITON-C/libwwx/NodeOrdering.hpp`
- `TRITON-C/libwcore/libwcore/Array3D.hpp`
- `TRITON-C/manual/technical_manual.tex`, especially the Krylov and domain
  decomposition sections

The concrete review baseline is SpecWave commit
`acb6865c05124ffef2a35be420d15cf3cd0ad7ae`, not the historical solver branch
names alone.
