#!/usr/bin/env python3
"""
Enhanced Batch Simulation Runner for FourD Consciousness Simulator

This script runs comprehensive parameter sweeps across all subsystem configurations,
including tuned parameters based on previous findings and optional QRNG data integration.

Usage:
    python run_batch_simulations.py [--qrng-data PATH] [--parallel N]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============================================================================
# Configuration Definitions
# ============================================================================

# Subsystem configurations (all combinations) - based on actual simulator parameters
SUBSYSTEM_CONFIGS = [
    {"name": "minimal", "n_subsystems": 4, "n_dims": 2, "n_timesteps": 100},
    {"name": "standard", "n_subsystems": 8, "n_dims": 4, "n_timesteps": 300},
    {"name": "detailed", "n_subsystems": 12, "n_dims": 6, "n_timesteps": 500},
]

# Basin dynamics configurations (all combinations)
BASIN_CONFIGS = [
    {"name": "low_ambiguity", "basin_ambiguity_threshold": 0.02, "basin_pull_strength": 0.01},
    {"name": "standard_basin", "basin_ambiguity_threshold": 0.05, "basin_pull_strength": 0.02},
    {"name": "high_ambiguity", "basin_ambiguity_threshold": 0.1, "basin_pull_strength": 0.03},
]

# Tuned configurations based on previous findings and analysis
TUNED_CONFIGS = [
    # High coherence regime (from analysis) - high emotion/planning gains
    {
        "name": "high_coherence",
        "n_subsystems": 8,
        "n_dims": 4,
        "n_timesteps": 300,
        "emotion_gain": 0.35,
        "planning_gain": 0.15,
        "perception_gain": 0.12,
        "basin_ambiguity_threshold": 0.04,
        "description": "Optimized for high coherence states"
    },
    # Critical regime (near phase transition) - balanced gains
    {
        "name": "critical",
        "n_subsystems": 8,
        "n_dims": 4,
        "n_timesteps": 300,
        "emotion_gain": 0.25,
        "planning_gain": 0.12,
        "perception_gain": 0.08,
        "language_gain": 0.05,
        "basin_ambiguity_threshold": 0.06,
        "description": "Near critical point for maximum complexity"
    },
    # Balanced regime (good all-around) - default-like but optimized
    {
        "name": "balanced",
        "n_subsystems": 8,
        "n_dims": 4,
        "n_timesteps": 300,
        "emotion_gain": 0.22,
        "planning_gain": 0.12,
        "perception_gain": 0.08,
        "memory_gain": 0.08,
        "basin_pull_strength": 0.025,
        "description": "Balanced complexity and coherence"
    },
]

# Comprehensive parameter sweeps for detailed investigation
PARAMETER_SWEEPS = {
    # Fine-grained subsystem sweep (recommended)
    "n_subsystems_fine": [2, 3, 4, 5, 6, 7, 8],
    # Dimensionality sweep
    "n_dims": [2, 3, 4, 5, 6],
    # Time steps
    "n_timesteps": [100, 200, 300, 400, 500],
    # Basin dynamics
    "basin_ambiguity_threshold": [0.02, 0.04, 0.05, 0.06, 0.08, 0.1],
    "basin_pull_strength": [0.01, 0.015, 0.02, 0.025, 0.03],
}

# Multi-dimensional configuration matrix for comprehensive testing
CONFIG_MATRIX = {
    # (n_subsystems, n_dims) combinations to test together
    "combinations": [
        {"name": "minimal_2d", "n_subsystems": 4, "n_dims": 2},
        {"name": "standard_4d", "n_subsystems": 8, "n_dims": 4},
        {"name": "high_dim_6d", "n_subsystems": 12, "n_dims": 6},
        {"name": "low_sub_high_dim", "n_subsystems": 4, "n_dims": 6},
        {"name": "high_sub_low_dim", "n_subsystems": 12, "n_dims": 2},
    ]
}

# Lesion study configurations - which subsystems to disable
LESION_SUBSYSTEMS = [
    "Perception", "Language", "Planning", "Emotion",
    "Memory", "Motor Control", "Attention", "Executive Control"
]


def run_lesion_studies(runner: "BatchSimulationRunner", subsystems: List[str]) -> Dict[str, Any]:
    """Run lesion studies on all specified subsystems."""
    print("\n" + "="*60)
    print("COMPREHENSIVE LESION STUDIES")
    print("="*60)
    
    start_time = time.time()
    all_results = []
    
    # Get seed from QRNG if available
    qrng_seed = get_qrng_seed(runner.qrng_data) if runner.qrng_data.get("samples") else None
    
    for subsystem in subsystems:
        print(f"\n--- Lesion Study: {subsystem} ---")
        
        # Create config with this subsystem disabled (via low gain)
        config = {
            "name": f"lesion_{subsystem.lower()}",
            "n_subsystems": 8,
            "n_dims": 4,
            "n_timesteps": 300,
            # Set the lesioned subsystem to very low gain
            **{f"{subsystem.lower().replace(' ', '_')}_gain": 0.01},
        }
        
        result = runner.run_config(config, seed=qrng_seed)
        all_results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "lesion_studies",
        "total_runs": len(all_results),
        "elapsed_time": elapsed_time,
        "qrng_seed_used": bool(qrng_seed),
        "results": all_results
    }
    
    summary_file = runner.output_dir / f"lesion_study_summary_{int(time.time())}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("LESION STUDIES COMPLETE")
    print('='*60)
    
    return summary


# ============================================================================
# QRNG Data Integration
# ============================================================================

def load_qrng_data(qrng_path: Optional[str] = None) -> Dict[str, Any]:
    """Load QRNG data for seeding or analysis."""
    if qrng_path is None:
        # Default to the qrng-analysis-toolkit repository
        default_paths = [
            Path(__file__).parent.parent / "qrng-analysis-toolkit" / "data" / "qrng_streams",
            Path("/Users/akbon/OneDrive/Documents/GitHub/qrng-analysis-toolkit/data/qrng_streams"),
            Path("C:/Users/akbon/OneDrive/Documents/GitHub/qrng-analysis-toolkit/data/qrng_streams"),
        ]
        
        for path in default_paths:
            if path.exists():
                qrng_path = str(path)
                break
    
    if qrng_path is None or not Path(qrng_path).exists():
        print(f"Warning: QRNG data path not found: {qrng_path}")
        return {"files": [], "count": 0}
    
    qrng_dir = Path(qrng_path)
    json_files = list(qrng_dir.glob("*.json"))
    
    # Load sample files for analysis
    samples = []
    for json_file in json_files[:10]:  # Sample first 10 files
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                samples.append({
                    "filename": json_file.name,
                    "data": data,
                    "size": json_file.stat().st_size
                })
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return {
        "path": qrng_path,
        "files": [f.name for f in json_files],
        "count": len(json_files),
        "samples": samples
    }


def get_qrng_seed(qrng_data: Dict[str, Any]) -> int:
    """Get a seed value from QRNG data."""
    if not qrng_data.get("samples"):
        return int(time.time() * 1000) % (2**32)
    
    # Use first sample's data as seed source
    sample = qrng_data["samples"][0]
    data = sample["data"]
    
    if isinstance(data, dict):
        # Extract numeric values and combine
        total = 0
        for key, value in data.items():
            if isinstance(value, (int, float)):
                total += int(abs(value) * 1000000) % (2**32)
        return total if total > 0 else int(time.time() * 1000) % (2**32)
    elif isinstance(data, list):
        # Use first numeric value
        for item in data:
            if isinstance(item, (int, float)):
                return int(abs(item) * 1000000) % (2**32)
    
    return int(time.time() * 1000) % (2**32)


# ============================================================================
# Batch Runner Class
# ============================================================================

class BatchSimulationRunner:
    """Runs batch simulations with various configurations."""
    
    def __init__(self, output_dir: str = "outputs/batch", qrng_data: Optional[Dict] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qrng_data = qrng_data or {}
        self.results = []
        
    def run_config(self, config: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        """Run a single simulation configuration."""
        config_name = config.get("name", "unnamed")
        
        # Prepare command arguments
        cmd = [sys.executable, "fourD_slice_sim.py"]
        
        # Add parameters based on config - using actual simulator CLI arguments
        param_map = {
            "n_subsystems": "--n-subsystems",
            "n_dims": "--n-dims",
            "n_timesteps": "--n-timesteps",
            "basin_ambiguity_threshold": "--basin-ambiguity-threshold",
            "basin_pull_strength": "--basin-pull-strength",
            # Subsystem gains
            "perception_gain": "--perception-gain",
            "language_gain": "--language-gain",
            "planning_gain": "--planning-gain",
            "emotion_gain": "--emotion-gain",
            "memory_gain": "--memory-gain",
            "motor_gain": "--motor-gain",
            "attention_gain": "--attention-gain",
            "executive_gain": "--executive-gain",
        }
        
        for key, arg_name in param_map.items():
            if key in config:
                cmd.extend([arg_name, str(config[key])])
        
        # Add seed if provided
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # Set output directory
        output_file = self.output_dir / f"{config_name}_{int(time.time() * 1000)}.csv"
        cmd.extend(["--output", str(output_file)])
        
        print(f"\n{'='*60}")
        print(f"Running configuration: {config_name}")
        print(f"Parameters: {config}")
        print(f"Output file: {output_file}")
        print(f"Command: {' '.join(cmd)}")
        print('='*60)
        
        # Run simulation
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per run
            )
            
            elapsed_time = time.time() - start_time
            
            output_data = {
                "config_name": config_name,
                "parameters": config,
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "elapsed_time": elapsed_time,
                "stdout": result.stdout[:1000] if result.stdout else "",  # Limit output
                "stderr": result.stderr[:1000] if result.stderr else "",
                "output_file": str(output_file),
            }
            
            self.results.append(output_data)
            
            if result.returncode == 0:
                print(f"[OK] Completed in {elapsed_time:.2f}s")
                print(result.stdout[:500])
            else:
                print(f"[FAIL] Failed with return code {result.returncode}")
                print(result.stderr[:500])
                
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            output_data = {
                "config_name": config_name,
                "parameters": config,
                "status": "timeout",
                "elapsed_time": elapsed_time,
                "error": "Simulation timed out after 5 minutes"
            }
            self.results.append(output_data)
            print(f"[TIMEOUT] Timed out after {elapsed_time:.2f}s")
        except Exception as e:
            output_data = {
                "config_name": config_name,
                "parameters": config,
                "status": "error",
                "error": str(e)
            }
            self.results.append(output_data)
            print(f"[ERROR] Error: {e}")
        
        return output_data
    
    def run_all_subsystem_configs(self, seed: Optional[int] = None) -> List[Dict]:
        """Run all subsystem configurations."""
        results = []
        for config in SUBSYSTEM_CONFIGS:
            result = self.run_config(config, seed=seed)
            results.append(result)
        return results
    
    def run_all_basin_configs(self, seed: Optional[int] = None) -> List[Dict]:
        """Run all basin configurations."""
        results = []
        for config in BASIN_CONFIGS:
            result = self.run_config(config, seed=seed)
            results.append(result)
        return results
    
    def run_tuned_configs(self, seed: Optional[int] = None) -> List[Dict]:
        """Run tuned configurations."""
        results = []
        for config in TUNED_CONFIGS:
            result = self.run_config(config, seed=seed)
            results.append(result)
        return results
    
    def run_parameter_sweep(self, param_name: str, values: List[Any], 
                           base_config: Optional[Dict] = None,
                           seed: Optional[int] = None) -> List[Dict]:
        """Run a parameter sweep for a single parameter."""
        if base_config is None:
            base_config = {"name": f"sweep_{param_name}"}
        
        results = []
        for value in values:
            config = base_config.copy()
            config[param_name] = value
            result = self.run_config(config, seed=seed)
            results.append(result)
        return results
    
    def run_full_sweep(self, qrng_seed: bool = True) -> Dict[str, Any]:
        """Run a full parameter sweep across all combinations."""
        print("\n" + "="*60)
        print("FULL PARAMETER SWEEP")
        print("="*60)
        
        start_time = time.time()
        all_results = []
        
        # Get seed from QRNG if requested
        seed = get_qrng_seed(self.qrng_data) if qrng_seed else None
        
        # Run subsystem configs
        print("\n--- Running Subsystem Configurations ---")
        results = self.run_all_subsystem_configs(seed=seed)
        all_results.extend(results)
        
        # Run basin configs
        print("\n--- Running Basin Configurations ---")
        results = self.run_all_basin_configs(seed=seed)
        all_results.extend(results)
        
        # Run tuned configs
        print("\n--- Running Tuned Configurations ---")
        results = self.run_tuned_configs(seed=seed)
        all_results.extend(results)
        
        # Run parameter sweeps
        print("\n--- Running Parameter Sweeps ---")
        for param_name, values in PARAMETER_SWEEPS.items():
            print(f"\nSweeping {param_name}...")
            results = self.run_parameter_sweep(param_name, values, seed=seed)
            all_results.extend(results)
        
        elapsed_time = time.time() - start_time
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(all_results),
            "elapsed_time": elapsed_time,
            "qrng_seed_used": qrng_seed and bool(self.qrng_data.get("samples")),
            "results": all_results
        }
        
        summary_file = self.output_dir / f"batch_summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE")
        print(f"Total runs: {len(all_results)}")
        print(f"Elapsed time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        print(f"Summary saved to: {summary_file}")
        print('='*60)
        
        return summary
    
    def run_config_matrix(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run the full configuration matrix test with (n_subsystems, n_dims) combinations."""
        print("\n" + "="*60)
        print("CONFIGURATION MATRIX TEST")
        print("="*60)
        
        start_time = time.time()
        all_results = []
        
        # Get seed from QRNG if available
        qrng_seed = get_qrng_seed(self.qrng_data) if self.qrng_data.get("samples") else None
        
        for combo in CONFIG_MATRIX["combinations"]:
            print(f"\n--- Testing: {combo['name']} ---")
            result = self.run_config(combo, seed=qrng_seed)
            all_results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "config_matrix",
            "total_runs": len(all_results),
            "elapsed_time": elapsed_time,
            "qrng_seed_used": bool(qrng_seed),
            "results": all_results
        }
        
        summary_file = self.output_dir / f"config_matrix_summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("CONFIG MATRIX COMPLETE")
        print('='*60)
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a text report of all results."""
        lines = [
            "=" * 70,
            "BATCH SIMULATION REPORT",
            f"Generated: {datetime.now().isoformat()}",
            f"Total runs: {len(self.results)}",
            "=" * 70,
            ""
        ]
        
        # Count by status
        status_counts = {}
        for result in self.results:
            status = result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        lines.append("Status Summary:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"  {status}: {count}")
        
        lines.append("")
        lines.append("-" * 70)
        lines.append("Detailed Results:")
        lines.append("-" * 70)
        
        for i, result in enumerate(self.results, 1):
            lines.append(f"\n{i}. {result.get('config_name', 'unnamed')}")
            lines.append(f"   Status: {result.get('status', 'unknown')}")
            if result.get("parameters"):
                lines.append(f"   Parameters: {result['parameters']}")
            if result.get("elapsed_time"):
                lines.append(f"   Time: {result['elapsed_time']:.2f}s")
            if result.get("error"):
                lines.append(f"   Error: {result['error']}")
        
        return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run batch simulations with various configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_batch_simulations.py                    # Run full sweep
  python run_batch_simulations.py --qrng-data PATH   # Use QRNG data for seeding
  python run_batch_simulations.py --tuned-only       # Only run tuned configs
  python run_batch_simulations.py --sweep-param n_subsystems --sweep-values 2,3,4,5,6,7,8
  python run_batch_simulations.py --config-matrix    # Run full config matrix test
  python run_batch_simulations.py --lesion-study     # Run lesion studies on all subsystems
        """
    )
    
    parser.add_argument(
        "--qrng-data",
        type=str,
        default=None,
        help="Path to QRNG data directory for seeding"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--tuned-only",
        action="store_true",
        help="Only run tuned configurations"
    )
    
    parser.add_argument(
        "--sweep-param",
        type=str,
        default=None,
        help="Parameter name to sweep (e.g., n_subsystems)"
    )
    
    parser.add_argument(
        "--sweep-values",
        type=str,
        default=None,
        help="Comma-separated values for the sweep (e.g., 2,3,4,5,6,7,8)"
    )
    
    parser.add_argument(
        "--config-matrix",
        action="store_true",
        help="Run full configuration matrix test"
    )
    
    parser.add_argument(
        "--lesion-study",
        action="store_true",
        help="Run lesion studies on all subsystems"
    )
    
    args = parser.parse_args()
    
    # Load QRNG data if specified
    qrng_data = None
    if args.qrng_data or True:  # Always try to load QRNG data
        print("Loading QRNG data...")
        qrng_data = load_qrng_data(args.qrng_data)
        print(f"Found {qrng_data.get('count', 0)} QRNG files")
    
    # Create runner
    runner = BatchSimulationRunner(
        output_dir=args.output_dir,
        qrng_data=qrng_data
    )
    
    # Run specified task
    if args.lesion_study:
        # Run lesion studies on all subsystems
        print("\nRunning comprehensive lesion studies...")
        results = run_lesion_studies(runner, LESION_SUBSYSTEMS)
    elif args.config_matrix:
        # Run full configuration matrix test
        print("\nRunning configuration matrix test...")
        results = runner.run_config_matrix()
    elif args.sweep_param and args.sweep_values:
        # Single parameter sweep
        param_name = args.sweep_param
        values = [v.strip() for v in args.sweep_values.split(",")]
        
        print(f"\nRunning single sweep: {param_name} = {values}")
        results = runner.run_parameter_sweep(param_name, values)
    elif args.tuned_only:
        # Only tuned configs
        print("\nRunning only tuned configurations...")
        results = runner.run_tuned_configs()
    else:
        # Full sweep
        results = runner.run_full_sweep(qrng_seed=True)
    
    # Generate and save report
    if isinstance(results, dict):
        summary = results
    else:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(runner.results),
            "results": runner.results
        }
    
    report_file = Path(args.output_dir) / f"report_{int(time.time())}.txt"
    with open(report_file, 'w') as f:
        f.write(runner.generate_report())
    
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()