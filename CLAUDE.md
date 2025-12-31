# Repository Orientation for Claude

## Purpose
Large-scale scientific codebase for molecular simulation and analysis.
Primary goals: correctness, reproducibility, performance portability.

## Languages
- Python (orchestration, analysis)
- C++ (kernels)
- CUDA (optional)
- Bash / Slurm

## Critical Constraints
- No API-breaking changes without explicit request
- Maintain backward compatibility
- Tests must pass on CPU-only systems
- Avoid adding heavyweight dependencies

## Key Directories
- src/core/        → performance-critical
- src/python/     → user-facing APIs
- workflows/      → Slurm / HPC jobs
- tests/          → pytest + CTest

## Style
- Prefer minimal diffs
- Explain reasoning before refactors
- Ask before touching src/core/*
