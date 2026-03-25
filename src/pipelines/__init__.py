"""
Pipeline modules for FourD project.

This package provides end-to-end analysis pipelines:
- run_qrng_pipeline.py: QRNG bitstream → features → drives → coordinator simulation
- run_surrogate_comparison.py: Compare real data against PRNG/shuffled/phase-randomized surrogates
"""

from __future__ import annotations

# QRNG Analysis Pipeline
from .run_qrng_pipeline import (
    load_bitstream,
    extract_features_from_stream,
    run_qrng_analysis,
    export_results,
    compute_summary_statistics,
)

# Surrogate Comparison Pipeline  
from .run_surrogate_comparison import (
    generate_prng_bits,
    create_shuffled_surrogate,
    create_phase_randomized_surrogate,
    run_surrogate_comparison,
    export_comparison_results,
)


__all__ = [
    # QRNG Pipeline
    "load_bitstream",
    "extract_features_from_stream", 
    "run_qrng_analysis",
    "export_results",
    "compute_summary_statistics",
    
    # Surrogate Comparison
    "generate_prng_bits",
    "create_shuffled_surrogate",
    "create_phase_randomized_surrogate",
    "run_surrogate_comparison",
    "export_comparison_results",
]