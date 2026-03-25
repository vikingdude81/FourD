#!/usr/bin/env python3
"""
GPU-Accelerated Consciousness Ensemble Simulation
==================================================

Runs 10K–500K simultaneous consciousness simulations on CUDA,
sweeping a 5D parameter grid and extracting compact "consciousness
signatures" from each trajectory. Produces a full phase diagram
of all possible conscious states the S³ dual-geometry model permits.

Hardware targets:
    - RTX 5090 (32 GB): up to ~500K simultaneous beings
    - RTX 3050 (8 GB):  up to ~100K simultaneous beings
    - Multi-GPU: splits batch across devices automatically

Requires: PyTorch with CUDA support, numpy, pandas, matplotlib
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import os
import argparse
from itertools import product
from pathlib import Path
from sklearn.cluster import KMeans


# ============================================================================
# MANIFOLD GEOMETRY (computed once on CPU, uploaded to GPU)
# ============================================================================

def generate_fibonacci_s3(n):
    """Generate n approximately-uniform points on S³ (Fibonacci lattice)."""
    phi = (1 + np.sqrt(5)) / 2
    points = np.zeros((n, 4))
    for i in range(n):
        t1 = np.arccos(1 - 2 * (i + 0.5) / n)
        t2 = 2 * np.pi * ((i * phi) % 1.0)
        t3 = 2 * np.pi * ((i * phi * phi) % 1.0)
        points[i, 0] = np.sin(t1) * np.sin(t2) * np.cos(t3)
        points[i, 1] = np.sin(t1) * np.sin(t2) * np.sin(t3)
        points[i, 2] = np.sin(t1) * np.cos(t2)
        points[i, 3] = np.cos(t1)
    norms = np.linalg.norm(points, axis=1, keepdims=True) + 1e-8
    return points / norms


def derive_macro_basins(micro_points, n_macro=24):
    """Cluster micro points into macro basins."""
    km = KMeans(n_clusters=n_macro, random_state=0, n_init=20)
    km.fit(micro_points)
    centers = km.cluster_centers_.copy()
    norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    return centers / norms


# Subsystem preference matrix (8 subsystems × 4 dims)
PREFERENCE_MATRIX = np.array([
    [+1.0,  0.0,  0.0,  0.0],   # Motor Control
    [ 0.0, +1.0,  0.0,  0.0],   # Planning
    [ 0.0,  0.0, +1.0,  0.0],   # Attention
    [ 0.0,  0.0,  0.0, +1.0],   # Memory
    [-1.0, +1.0,  0.0,  0.0],   # Emotion
    [ 0.0, -1.0, +1.0,  0.0],   # Social
    [ 0.0,  0.0, -1.0, +1.0],   # Intuition
    [+1.0,  0.0,  0.0, -1.0],   # Aesthetic
], dtype=np.float32)

# Normalize preferences (constant)
_norms = np.linalg.norm(PREFERENCE_MATRIX, axis=1, keepdims=True)
_norms = np.maximum(_norms, 1e-8)
PREFERENCE_MATRIX_NORMED = PREFERENCE_MATRIX / _norms

SUBSYSTEM_NAMES = [
    'Motor Control', 'Planning', 'Attention', 'Memory',
    'Emotion', 'Social', 'Intuition', 'Aesthetic'
]


# ============================================================================
# BATCH CONSCIOUSNESS ENGINE (PyTorch / CUDA)
# ============================================================================

class BatchConsciousnessEngine:
    """
    Runs N consciousness simulations in parallel on GPU.

    All state is batched: (N, ...) tensors. One call to step() advances
    every being by one timestep. No Python loops over beings.

    Per-config parameters are broadcast across the batch dimension
    so different beings can have different steering_strength, alpha_pull, etc.
    """

    def __init__(self, N, configs, device='cuda:0'):
        """
        Args:
            N: number of simultaneous simulations
            configs: dict of per-simulation parameters, each (N,) array:
                steering_strength, alpha_pull, fatigue_rate,
                exploration_noise, beta_macro
            device: torch device string
        """
        self.N = N
        self.device = torch.device(device)
        self.dim = 4
        self.n_subsystems = 8

        # --- Upload manifold geometry (shared across all beings) ---
        micro_np = generate_fibonacci_s3(600)
        macro_np = derive_macro_basins(micro_np, 24)
        self.macro_centers = torch.tensor(macro_np, dtype=torch.float32, device=self.device)  # (24, 4)
        self.n_macro = 24

        # --- Upload preference matrix ---
        self.prefs = torch.tensor(
            PREFERENCE_MATRIX_NORMED, dtype=torch.float32, device=self.device
        )  # (8, 4)

        # --- Per-simulation parameters (N,) ---
        def _to_dev(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        self.steering_strength = _to_dev(configs['steering_strength'])   # (N,)
        self.alpha_pull        = _to_dev(configs['alpha_pull'])           # (N,)
        self.fatigue_rate      = _to_dev(configs['fatigue_rate'])         # (N,)
        self.exploration_noise = _to_dev(configs['exploration_noise'])    # (N,)
        self.beta_macro        = _to_dev(configs['beta_macro'])           # (N,)

        # Fixed parameters
        self.recovery_rate = 0.025
        self.floor_value   = 0.05
        self.novelty_weight = 0.6

        # --- State tensors ---
        # Random initial direction on S³
        init = torch.randn(N, 4, device=self.device)
        self.u_t = F.normalize(init, dim=1)               # (N, 4)
        self.fatigue = torch.zeros(N, 8, device=self.device)  # (N, 8)

        # Basin dwell tracking
        self.basin_dwell = torch.zeros(N, dtype=torch.int32, device=self.device)
        self.current_basin = torch.full((N,), -1, dtype=torch.int32, device=self.device)

        # Previous state for curvature
        self.u_prev = self.u_t.clone()

        # --- History accumulators for signature extraction ---
        # We store per-step scalars in GPU tensors and extract signatures at the end
        self.max_steps = configs.get('timesteps', 1000)
        self._init_history()

    def _init_history(self):
        """Allocate history buffers on GPU."""
        N, T = self.N, self.max_steps
        dev = self.device
        self.hist_clarity        = torch.zeros(N, T, device=dev)
        self.hist_conflict       = torch.zeros(N, T, device=dev)
        self.hist_curvature      = torch.zeros(N, T, device=dev)
        self.hist_speed          = torch.zeros(N, T, device=dev)
        self.hist_integration    = torch.zeros(N, T, device=dev)
        self.hist_differentiation= torch.zeros(N, T, device=dev)
        self.hist_inner_outer    = torch.zeros(N, T, device=dev)
        self.hist_path_coherence = torch.zeros(N, T, device=dev)
        self.hist_dominant_sub   = torch.zeros(N, T, dtype=torch.int32, device=dev)
        self.hist_perc_mode      = torch.zeros(N, T, dtype=torch.int32, device=dev)
        self.hist_macro_basin    = torch.zeros(N, T, dtype=torch.int32, device=dev)
        self.hist_clarity_rate   = torch.zeros(N, T, device=dev)
        self.hist_force_mags     = torch.zeros(N, T, 8, device=dev)
        self.hist_clarity_decomp = torch.zeros(N, T, 8, device=dev)
        # Trajectory snapshots for attractor analysis (sample every K steps)
        self.traj_sample_interval = max(1, self.max_steps // 100)
        n_samples = self.max_steps // self.traj_sample_interval + 1
        self.hist_trajectory = torch.zeros(N, n_samples, 4, device=dev)
        self.traj_sample_idx = 0
        # Direction history for path coherence (rolling window)
        self.dir_history = torch.zeros(N, 10, 4, device=dev)
        self.dir_history_len = 0
        self.step_count = 0

    @torch.no_grad()
    def step(self):
        """Advance all N simulations by one timestep. No gradients needed."""
        t = self.step_count
        N = self.N
        u = self.u_t  # (N, 4)

        # ================================================================
        # STAGE 1: Subsystem competition & tangent force field
        # ================================================================

        # --- Compute influences: dot(u_t, pref_i) for each subsystem ---
        # prefs: (8, 4), u: (N, 4) → influences: (N, 8)
        influences = torch.einsum('nd,sd->ns', u, self.prefs)  # (N, 8)
        influences = 0.5 + 0.3 * influences

        # --- Apply fatigue and competition ---
        effective = influences * torch.exp(-self.fatigue)    # (N, 8)
        noise = self.exploration_noise.unsqueeze(1) * torch.randn(N, 8, device=self.device)
        effective = effective + noise
        effective = torch.clamp(effective, min=self.floor_value)

        # Divisive normalization → activities
        activities = effective / (effective.sum(dim=1, keepdim=True) + 1e-8)  # (N, 8)

        # --- Update fatigue ---
        fat_rate = self.fatigue_rate.unsqueeze(1)  # (N, 1)
        self.fatigue = self.fatigue + fat_rate * activities
        # Secondary fatigue for above-equal-share subsystems
        equal_share = 1.0 / self.n_subsystems
        excess = (activities - equal_share).clamp(min=0.02) - 0.02
        self.fatigue = self.fatigue + 0.03 * excess
        # Recovery
        inactive_recovery = (1.0 - activities) * self.recovery_rate
        self.fatigue = (self.fatigue - inactive_recovery).clamp(0.0, 3.0)

        # --- Tangent force field on S³ ---
        # prefs: (8, 4), u: (N, 4)
        # radial component: (N, 8) = dot(pref_i, u) for each subsystem
        radial = torch.einsum('sd,nd->ns', self.prefs, u)  # (N, 8)
        # forces[n, s, d] = prefs[s, d] - radial[n, s] * u[n, d]
        forces = self.prefs.unsqueeze(0) - radial.unsqueeze(2) * u.unsqueeze(1)  # (N, 8, 4)

        # --- Activity-weighted resultant ---
        activity_force = torch.einsum('ns,nsd->nd', activities, forces)  # (N, 4)

        # --- Novelty force (rested subsystems attract) ---
        rest_scores = torch.exp(-self.fatigue)     # (N, 8)
        novelty_force = torch.einsum('ns,nsd->nd', rest_scores, forces)  # (N, 4)
        mean_rest = rest_scores.mean(dim=1, keepdim=True)  # (N, 1)
        forces_sum = forces.sum(dim=1) / self.n_subsystems  # (N, 4)
        novelty_force = novelty_force - mean_rest * forces_sum

        # --- Blend ---
        nw = self.novelty_weight
        drive = (1.0 - nw) * activity_force + nw * novelty_force  # (N, 4)

        # --- Tangent-projected noise ---
        raw_noise = self.exploration_noise.unsqueeze(1) * torch.randn(N, 4, device=self.device)
        # Project to tangent: noise - (noise·u)*u
        noise_radial = (raw_noise * u).sum(dim=1, keepdim=True)
        drive = drive + raw_noise - noise_radial * u

        # --- Geodesic update ---
        ss = self.steering_strength.unsqueeze(1)  # (N, 1)
        new_u = u + ss * drive
        self.u_prev = u.clone()
        self.u_t = F.normalize(new_u, dim=1)

        # ================================================================
        # STAGE 2: Macro reconciliation
        # ================================================================

        # Soft assignment: softmax(beta * dot(u, macro_centers))
        macro_sim = torch.einsum('nd,md->nm', self.u_t, self.macro_centers)  # (N, 24)
        beta = self.beta_macro.unsqueeze(1)  # (N, 1)
        macro_weights = F.softmax(beta * macro_sim, dim=1)  # (N, 24)

        # Weighted macro field
        macro_field = torch.einsum('nm,md->nd', macro_weights, self.macro_centers)  # (N, 4)
        macro_field = F.normalize(macro_field, dim=1)

        # Basin dwell tracking
        dominant_basin = macro_weights.argmax(dim=1).int()  # (N,)
        same = (dominant_basin == self.current_basin)
        self.basin_dwell = torch.where(same, self.basin_dwell + 1, torch.zeros_like(self.basin_dwell))
        self.current_basin = dominant_basin

        # Basin escape (vectorized probabilistic)
        escape_mask = self.basin_dwell > 25
        if escape_mask.any():
            escape_prob = (0.05 * (self.basin_dwell.float() - 25.0)).clamp(0.0, 0.3)
            do_escape = (torch.rand(N, device=self.device) < escape_prob) & escape_mask
            if do_escape.any():
                n_escape = do_escape.sum().item()
                # Pick random target basins
                target_idx = torch.randint(0, self.n_macro, (n_escape,), device=self.device)
                target_dirs = self.macro_centers[target_idx]  # (n_esc, 4)
                u_esc = self.u_t[do_escape]  # (n_esc, 4)
                # Tangent direction toward target
                tangent = target_dirs - (target_dirs * u_esc).sum(dim=1, keepdim=True) * u_esc
                tn = tangent.norm(dim=1, keepdim=True).clamp(min=1e-6)
                tangent = tangent / tn
                # Great-circle step of 0.4 rad
                escape_dirs = torch.cos(torch.tensor(0.4)) * u_esc + torch.sin(torch.tensor(0.4)) * tangent
                self.u_t[do_escape] = F.normalize(escape_dirs, dim=1)
                self.basin_dwell[do_escape] = 0

        # Pull toward macro field
        macro_tangent = macro_field - (macro_field * self.u_t).sum(dim=1, keepdim=True) * self.u_t
        ap = self.alpha_pull.unsqueeze(1)  # (N, 1)
        self.u_t = F.normalize(self.u_t + ap * macro_tangent, dim=1)

        # ================================================================
        # METRICS (all batched)
        # ================================================================
        u = self.u_t

        # --- Conflict angle between top-2 forces ---
        top2 = activities.topk(2, dim=1).indices  # (N, 2)
        f0 = forces[torch.arange(N, device=self.device), top2[:, 0]]  # (N, 4)
        f1 = forces[torch.arange(N, device=self.device), top2[:, 1]]  # (N, 4)
        cos_conflict = (f0 * f1).sum(dim=1) / (f0.norm(dim=1) * f1.norm(dim=1) + 1e-8)
        conflict_angle = torch.acos(cos_conflict.clamp(-1, 1))

        # --- Clarity ---
        resultant = torch.einsum('ns,nsd->nd', activities, forces)  # (N, 4)
        clarity = resultant.norm(dim=1)

        # --- Clarity decomposition ---
        r_norm = clarity.unsqueeze(1).clamp(min=1e-10)
        r_hat = resultant / r_norm  # (N, 4)
        # Per-subsystem contribution: activities[s] * dot(forces[s], r_hat)
        clarity_decomp = activities * torch.einsum('nsd,nd->ns', forces, r_hat)  # (N, 8)

        # --- Curvature ---
        dot_curv = (self.u_prev * u).sum(dim=1).clamp(-1, 1)
        curvature = torch.acos(dot_curv)

        # --- Speed (from heading = u[:, :2]) ---
        heading_mag = u[:, :2].norm(dim=1)
        speed = 0.4 + (1.5 - 0.4) * heading_mag

        # --- Inner/outer ratio ---
        inner = u[:, 2:4].norm(dim=1)
        outer = u[:, 0:2].norm(dim=1) + 1e-8
        inner_outer = inner / outer

        # --- Perception mode (vectorized) ---
        range_param = u[:, 2].abs()
        focus_param = u[:, 3].abs()
        # 0=exploration, 1=threat-lock, 2=vigilant, 3=internal
        mode = torch.where(
            (range_param > 0.5) & (focus_param < 0.5), torch.tensor(0, device=self.device),
            torch.where(
                (range_param < 0.5) & (focus_param > 0.5), torch.tensor(1, device=self.device),
                torch.where(
                    (range_param > 0.5) & (focus_param > 0.5), torch.tensor(2, device=self.device),
                    torch.tensor(3, device=self.device)
                )
            )
        )

        # --- Integration (bell-shaped on effective basin count) ---
        neff = 1.0 / (macro_weights.pow(2).sum(dim=1) + 1e-8)
        target_neff = 8.0
        sigma = 3.0
        integration = torch.exp(-((neff - target_neff) ** 2) / (2 * sigma ** 2))

        # --- Differentiation (normalized entropy) ---
        max_entropy = np.log(self.n_macro)
        log_w = torch.log(macro_weights + 1e-8)
        current_entropy = -(macro_weights * log_w).sum(dim=1)
        differentiation = current_entropy / max_entropy

        # --- Path coherence (rolling window of last 10 direction changes) ---
        if t > 0:
            diff = u - self.u_prev  # (N, 4) latest direction change
            idx = min(t - 1, 9)
            # Shift window and insert
            if self.dir_history_len < 10:
                self.dir_history[:, self.dir_history_len] = diff
                self.dir_history_len += 1
            else:
                self.dir_history = torch.roll(self.dir_history, -1, dims=1)
                self.dir_history[:, 9] = diff
            # Compute mean consecutive alignment
            if self.dir_history_len >= 3:
                w = self.dir_history[:, :self.dir_history_len]  # (N, L, 4)
                a = w[:, :-1]  # (N, L-1, 4)
                b = w[:, 1:]   # (N, L-1, 4)
                na = a.norm(dim=2).clamp(min=1e-8)
                nb = b.norm(dim=2).clamp(min=1e-8)
                align = (a * b).sum(dim=2) / (na * nb)
                path_coherence = align.mean(dim=1)
            else:
                path_coherence = torch.zeros(N, device=self.device)
        else:
            path_coherence = torch.zeros(N, device=self.device)

        # --- Clarity rate ---
        if t > 0:
            clarity_rate = clarity - self.hist_clarity[:, t - 1]
        else:
            clarity_rate = torch.zeros(N, device=self.device)

        # --- Force magnitudes ---
        force_mags = forces.norm(dim=2)  # (N, 8)

        # --- Dominant subsystem ---
        dominant_sub = activities.argmax(dim=1).int()

        # ================================================================
        # STORE HISTORY
        # ================================================================
        if t < self.max_steps:
            self.hist_clarity[:, t]         = clarity
            self.hist_conflict[:, t]        = conflict_angle
            self.hist_curvature[:, t]       = curvature
            self.hist_speed[:, t]           = speed
            self.hist_integration[:, t]     = integration
            self.hist_differentiation[:, t] = differentiation
            self.hist_inner_outer[:, t]     = inner_outer
            self.hist_path_coherence[:, t]  = path_coherence
            self.hist_dominant_sub[:, t]    = dominant_sub
            self.hist_perc_mode[:, t]       = mode
            self.hist_macro_basin[:, t]     = dominant_basin
            self.hist_clarity_rate[:, t]    = clarity_rate
            self.hist_force_mags[:, t]      = force_mags
            self.hist_clarity_decomp[:, t]  = clarity_decomp

            # Trajectory samples
            if t % self.traj_sample_interval == 0:
                self.hist_trajectory[:, self.traj_sample_idx] = u.clone()
                self.traj_sample_idx += 1

        self.step_count += 1

    def run(self, steps=None):
        """Run full simulation for all beings."""
        if steps is None:
            steps = self.max_steps
        t0 = time.time()
        for t in range(steps):
            self.step()
            if (t + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (t + 1) * self.N / elapsed
                print(f"    Step {t+1}/{steps} | {rate:.0f} being-steps/sec | "
                      f"{elapsed:.1f}s elapsed")
        elapsed = time.time() - t0
        total_steps = steps * self.N
        print(f"  Completed {self.N:,} beings × {steps} steps = {total_steps:,} "
              f"being-steps in {elapsed:.1f}s ({total_steps/elapsed:.0f}/sec)")

    def extract_signatures(self):
        """
        Extract a compact consciousness signature from each being's history.

        Returns: (N, n_features) tensor on CPU
        """
        T = self.step_count
        N = self.N

        # Slice history to actual steps
        clarity     = self.hist_clarity[:, :T]
        conflict    = self.hist_conflict[:, :T]
        curvature   = self.hist_curvature[:, :T]
        speed       = self.hist_speed[:, :T]
        integ       = self.hist_integration[:, :T]
        diff        = self.hist_differentiation[:, :T]
        inner_outer = self.hist_inner_outer[:, :T]
        path_coh    = self.hist_path_coherence[:, :T]
        clar_rate   = self.hist_clarity_rate[:, :T]
        dom_sub     = self.hist_dominant_sub[:, :T]
        perc_mode   = self.hist_perc_mode[:, :T]
        macro_basin = self.hist_macro_basin[:, :T]
        force_mags  = self.hist_force_mags[:, :T]
        clar_decomp = self.hist_clarity_decomp[:, :T]

        sigs = []

        # --- Clarity metrics ---
        sigs.append(clarity.mean(dim=1))                                  # 0: mean clarity
        sigs.append(clarity.max(dim=1).values)                            # 1: max clarity
        sigs.append(clarity.std(dim=1))                                   # 2: clarity volatility
        # Sustained clarity: fraction of steps above 75th percentile
        p75 = clarity.quantile(0.75, dim=1)
        high_clarity = (clarity > p75.unsqueeze(1)).float()
        sigs.append(high_clarity.mean(dim=1))                             # 3: high clarity fraction

        # Clarity persistence: lag-1 autocorrelation
        c_mean = clarity.mean(dim=1, keepdim=True)
        c_centered = clarity - c_mean
        c_var = c_centered.pow(2).mean(dim=1)
        c_autocorr = (c_centered[:, :-1] * c_centered[:, 1:]).mean(dim=1) / (c_var + 1e-10)
        sigs.append(c_autocorr)                                           # 4: clarity persistence

        # --- Conflict & dynamics ---
        sigs.append(conflict.mean(dim=1))                                 # 5: mean conflict angle
        sigs.append(curvature.mean(dim=1))                                # 6: mean curvature
        sigs.append(speed.mean(dim=1))                                    # 7: mean speed
        sigs.append(speed.std(dim=1))                                     # 8: speed variance

        # --- Direction snap rate (fraction of steps with >90° direction changes) ---
        traj = self.hist_trajectory[:, :self.traj_sample_idx]
        if traj.shape[1] >= 2:
            dot_seq = (traj[:, :-1] * traj[:, 1:]).sum(dim=2).clamp(-1, 1)
            angle_seq = torch.acos(dot_seq)
            snap_rate = (angle_seq > (np.pi / 2)).float().mean(dim=1)
        else:
            snap_rate = torch.zeros(N, device=self.device)
        sigs.append(snap_rate)                                            # 9: direction snap rate

        # --- Integration / Differentiation ---
        sigs.append(integ.mean(dim=1))                                    # 10: mean integration
        sigs.append(diff.mean(dim=1))                                     # 11: mean differentiation
        sigs.append(inner_outer.mean(dim=1))                              # 12: mean inner/outer ratio
        sigs.append(path_coh.mean(dim=1))                                 # 13: mean path coherence

        # --- Perception mode entropy (how evenly modes are used) ---
        mode_counts = torch.zeros(N, 4, device=self.device)
        for m in range(4):
            mode_counts[:, m] = (perc_mode == m).float().sum(dim=1)
        mode_probs = mode_counts / (T + 1e-8)
        mode_entropy = -(mode_probs * torch.log(mode_probs + 1e-8)).sum(dim=1)
        sigs.append(mode_entropy)                                         # 14: perception mode entropy

        # Mode stickiness: fraction of steps that stay in same mode
        if T > 1:
            mode_same = (perc_mode[:, 1:] == perc_mode[:, :-1]).float().mean(dim=1)
        else:
            mode_same = torch.ones(N, device=self.device)
        sigs.append(mode_same)                                            # 15: mode stickiness

        # --- Subsystem dominance entropy ---
        sub_counts = torch.zeros(N, 8, device=self.device)
        for s in range(8):
            sub_counts[:, s] = (dom_sub == s).float().sum(dim=1)
        sub_probs = sub_counts / (T + 1e-8)
        sub_entropy = -(sub_probs * torch.log(sub_probs + 1e-8)).sum(dim=1)
        sigs.append(sub_entropy)                                          # 16: dominance entropy

        # --- Macro basin transitions ---
        if T > 1:
            basin_switches = (macro_basin[:, 1:] != macro_basin[:, :-1]).float().sum(dim=1)
        else:
            basin_switches = torch.zeros(N, device=self.device)
        sigs.append(basin_switches / T)                                   # 17: basin transition rate

        # --- Force magnitude spread (std across subsystems, averaged over time) ---
        force_std = force_mags.std(dim=2).mean(dim=1)
        sigs.append(force_std)                                            # 18: force magnitude spread

        # --- Alliance symmetry (correlation between paired decompositions) ---
        # Memory-Social (3,5), Planning-Aesthetic (1,7), Attention-Emotion (2,4), Motor-Intuition (0,6)
        alliance_pairs = [(3, 5), (1, 7), (2, 4), (0, 6)]
        pair_corrs = []
        for a, b in alliance_pairs:
            ca = clar_decomp[:, :, a]  # (N, T)
            cb = clar_decomp[:, :, b]  # (N, T)
            ca_m = ca - ca.mean(dim=1, keepdim=True)
            cb_m = cb - cb.mean(dim=1, keepdim=True)
            cov = (ca_m * cb_m).mean(dim=1)
            sa = ca_m.pow(2).mean(dim=1).sqrt()
            sb = cb_m.pow(2).mean(dim=1).sqrt()
            corr = cov / (sa * sb + 1e-10)
            pair_corrs.append(corr)
        alliance_sym = torch.stack(pair_corrs, dim=1).mean(dim=1)
        sigs.append(alliance_sym)                                         # 19: alliance symmetry

        # --- Effective attractor dimensionality (PCA on trajectory samples) ---
        # Participation ratio PR = tr(C)² / tr(C²) — no eigendecomposition needed
        if traj.shape[1] >= 4:
            traj_centered = traj - traj.mean(dim=1, keepdim=True)
            # Batched covariance: (N, 4, 4)
            cov = torch.einsum('nti,ntj->nij', traj_centered, traj_centered) / (traj.shape[1] - 1)
            tr_c = torch.diagonal(cov, dim1=1, dim2=2).sum(dim=1)    # tr(C)
            cov_sq = torch.bmm(cov, cov)                              # C²
            tr_c2 = torch.diagonal(cov_sq, dim1=1, dim2=2).sum(dim=1) # tr(C²)
            eff_dim = tr_c.pow(2) / (tr_c2 + 1e-10)
            eff_dim = eff_dim.clamp(1.0, 4.0)
            sigs.append(eff_dim)                                          # 20: effective dimensionality
        else:
            sigs.append(torch.ones(N, device=self.device) * 2.0)

        # --- Lyapunov-like divergence estimate ---
        # Use trajectory samples: for pairs separated by ~2 samples,
        # check if they diverge over the next 2 samples
        if traj.shape[1] >= 10:
            n_samp = traj.shape[1]
            # Compare point i to point i+3, check divergence at i+1 vs i+4
            i_pts = traj[:, :n_samp-5]     # starting points
            j_pts = traj[:, 3:n_samp-2]    # nearby-ish points
            i_next = traj[:, 1:n_samp-4]
            j_next = traj[:, 4:n_samp-1]
            d0 = (i_pts - j_pts).norm(dim=2)
            d1 = (i_next - j_next).norm(dim=2)
            # Fraction where d1 > d0 (diverging)
            valid = d0 > 0.005
            diverging = ((d1 > d0) & valid).float()
            total_valid = valid.float().sum(dim=1).clamp(min=1)
            lyap_proxy = diverging.sum(dim=1) / total_valid
            sigs.append(lyap_proxy)                                       # 21: lyapunov proxy
        else:
            sigs.append(torch.ones(N, device=self.device) * 0.5)

        return torch.stack(sigs, dim=1).cpu()  # (N, n_features)


SIGNATURE_NAMES = [
    'mean_clarity', 'max_clarity', 'clarity_volatility', 'high_clarity_frac',
    'clarity_persistence', 'mean_conflict', 'mean_curvature', 'mean_speed',
    'speed_variance', 'direction_snap_rate', 'mean_integration',
    'mean_differentiation', 'mean_inner_outer', 'mean_path_coherence',
    'perc_mode_entropy', 'mode_stickiness', 'dominance_entropy',
    'basin_transition_rate', 'force_mag_spread', 'alliance_symmetry',
    'effective_dimensionality', 'lyapunov_proxy',
]


# ============================================================================
# PHASE CARTOGRAPHER
# ============================================================================

class PhaseCartographer:
    """
    Sweeps a 5D parameter grid and builds a complete phase diagram
    of consciousness regimes.
    """

    # Default parameter ranges
    PARAM_RANGES = {
        'steering_strength': np.linspace(0.05, 0.80, 12),
        'alpha_pull':        np.linspace(0.00, 0.15, 10),
        'fatigue_rate':      np.linspace(0.00, 0.30, 10),
        'exploration_noise': np.linspace(0.00, 0.20, 8),
        'beta_macro':        np.linspace(0.50, 15.0, 8),
    }

    def __init__(self, device='cuda:0', steps_per_sim=1000):
        self.device = device
        self.steps_per_sim = steps_per_sim
        # Check available VRAM
        if torch.cuda.is_available():
            dev_idx = int(device.split(':')[1]) if ':' in device else 0
            props = torch.cuda.get_device_properties(dev_idx)
            self.vram_gb = props.total_memory / (1024 ** 3)
            self.gpu_name = props.name
        else:
            self.vram_gb = 0
            self.gpu_name = 'CPU'

    def estimate_batch_size(self):
        """Estimate max batch size that fits in VRAM with safety margin."""
        # Per-simulation GPU memory (rough estimate):
        # State: ~256 bytes
        # History: ~22 floats * T * 4 bytes + 8*T*4*2 ≈ (22 + 16) * T * 4 = 152 * T bytes
        # Trajectory: ~100 * 4 * 4 = 1600 bytes
        # Working buffers during step: ~5 * (8+4) * 4 ≈ 240 bytes
        bytes_per_sim = 256 + 152 * self.steps_per_sim + 1600 + 240
        # Use 70% of VRAM (leave room for PyTorch overhead)
        usable = self.vram_gb * 0.70 * (1024 ** 3)
        max_batch = int(usable / bytes_per_sim)
        return max(1000, min(max_batch, 500_000))

    def build_parameter_grid(self, param_ranges=None):
        """
        Build full Cartesian grid of parameter configurations.

        Returns: dict of (total_configs,) arrays
        """
        if param_ranges is None:
            param_ranges = self.PARAM_RANGES

        param_names = list(param_ranges.keys())
        param_values = [param_ranges[k] for k in param_names]

        # Full Cartesian product
        grid_points = list(product(*param_values))
        n_total = len(grid_points)
        print(f"  Parameter grid: {' × '.join(str(len(v)) for v in param_values)} = {n_total:,} configurations")

        configs = {}
        for i, name in enumerate(param_names):
            configs[name] = np.array([g[i] for g in grid_points], dtype=np.float32)

        return configs, n_total

    def run_sweep(self, param_ranges=None, output_dir='outputs/phase_cartography'):
        """
        Run the full parameter sweep in batches.

        Returns: DataFrame with parameters + consciousness signatures
        """
        configs, n_total = self.build_parameter_grid(param_ranges)
        max_batch = self.estimate_batch_size()
        print(f"  GPU: {self.gpu_name} ({self.vram_gb:.1f} GB)")
        print(f"  Max batch size: {max_batch:,}")
        print(f"  Steps per simulation: {self.steps_per_sim}")

        # Split into batches
        n_batches = (n_total + max_batch - 1) // max_batch
        print(f"  Batches needed: {n_batches}")

        all_signatures = []
        all_params = {k: [] for k in configs}

        os.makedirs(output_dir, exist_ok=True)
        sweep_start = time.time()

        for batch_idx in range(n_batches):
            start = batch_idx * max_batch
            end = min(start + max_batch, n_total)
            batch_size = end - start

            print(f"\n  === Batch {batch_idx + 1}/{n_batches}: "
                  f"beings {start:,}–{end-1:,} ({batch_size:,}) ===")

            # Slice configs for this batch
            batch_configs = {k: v[start:end] for k, v in configs.items()}
            batch_configs['timesteps'] = self.steps_per_sim

            # Run batch
            torch.cuda.empty_cache()
            engine = BatchConsciousnessEngine(batch_size, batch_configs, device=self.device)
            engine.run(self.steps_per_sim)

            # Extract signatures
            sigs = engine.extract_signatures()  # (batch_size, n_features) CPU tensor
            all_signatures.append(sigs.numpy())

            for k in configs:
                all_params[k].append(batch_configs[k])

            # Free GPU memory
            del engine
            torch.cuda.empty_cache()

        total_time = time.time() - sweep_start
        print(f"\n  Total sweep: {n_total:,} beings × {self.steps_per_sim} steps "
              f"in {total_time:.1f}s")

        # Assemble results
        all_sigs = np.concatenate(all_signatures, axis=0)  # (n_total, n_features)
        result = {}
        for k in configs:
            result[k] = np.concatenate(all_params[k])
        for i, name in enumerate(SIGNATURE_NAMES):
            result[name] = all_sigs[:, i]

        df = pd.DataFrame(result)
        out_path = os.path.join(output_dir, 'phase_cartography_results.csv')
        df.to_csv(out_path, index=False)
        print(f"  Results saved to {out_path}")

        return df

    def visualize_phase_diagram(self, df, output_dir='outputs/phase_cartography',
                                 default_config=None):
        """
        Generate 2D phase diagram slices through the 5D parameter space.

        For each pair of parameters, aggregates across the other 3 dimensions.
        """
        os.makedirs(output_dir, exist_ok=True)
        params = list(self.PARAM_RANGES.keys())
        features_to_plot = [
            'mean_clarity', 'clarity_persistence', 'effective_dimensionality',
            'lyapunov_proxy', 'perc_mode_entropy', 'dominance_entropy',
            'mean_conflict', 'basin_transition_rate', 'alliance_symmetry',
        ]

        if default_config is None:
            default_config = {
                'steering_strength': 0.3,
                'alpha_pull': 0.03,
                'fatigue_rate': 0.08,
                'exploration_noise': 0.05,
                'beta_macro': 4.0,
            }

        # For each pair of parameters, fix others at default and make heatmap
        n_pairs = len(params) * (len(params) - 1) // 2
        fig_rows = len(features_to_plot)
        fig_cols = n_pairs

        pair_list = [(i, j) for i in range(len(params)) for j in range(i + 1, len(params))]

        for feat_name in features_to_plot:
            fig, axes = plt.subplots(1, len(pair_list), figsize=(4 * len(pair_list), 3.5))
            if len(pair_list) == 1:
                axes = [axes]
            fig.suptitle(f'Phase Diagram: {feat_name}', fontsize=14, fontweight='bold')

            for ax_idx, (pi, pj) in enumerate(pair_list):
                p1_name, p2_name = params[pi], params[pj]
                other_params = [p for p in params if p not in (p1_name, p2_name)]

                # Filter to rows near default for other params
                mask = np.ones(len(df), dtype=bool)
                for op in other_params:
                    vals = self.PARAM_RANGES[op]
                    # Find closest grid value to default
                    default_val = default_config[op]
                    closest = vals[np.argmin(np.abs(vals - default_val))]
                    mask &= np.isclose(df[op].values, closest, atol=1e-4)

                subset = df[mask]
                if len(subset) < 4:
                    axes[ax_idx].text(0.5, 0.5, 'Insufficient data', transform=axes[ax_idx].transAxes)
                    continue

                pivot = subset.groupby([p1_name, p2_name])[feat_name].mean().reset_index()
                try:
                    heatmap_data = pivot.pivot(index=p2_name, columns=p1_name, values=feat_name)
                    im = axes[ax_idx].imshow(
                        heatmap_data.values, aspect='auto', origin='lower',
                        extent=[
                            heatmap_data.columns.min(), heatmap_data.columns.max(),
                            heatmap_data.index.min(), heatmap_data.index.max()
                        ],
                        cmap='viridis'
                    )
                    axes[ax_idx].set_xlabel(p1_name.replace('_', ' '))
                    axes[ax_idx].set_ylabel(p2_name.replace('_', ' '))
                    # Mark default config point
                    axes[ax_idx].plot(
                        default_config[p1_name], default_config[p2_name],
                        'r*', markersize=12, markeredgecolor='white', markeredgewidth=0.5
                    )
                    plt.colorbar(im, ax=axes[ax_idx], fraction=0.046)
                except Exception:
                    axes[ax_idx].text(0.5, 0.5, 'Pivot failed', transform=axes[ax_idx].transAxes)

            plt.tight_layout()
            fig_path = os.path.join(output_dir, f'phase_{feat_name}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fig_path}")

        # --- Summary: find Goldilocks zone ---
        self._find_goldilocks(df, output_dir, default_config)

    def _find_goldilocks(self, df, output_dir, default_config):
        """Find the parameter region that maximizes a composite flourishing score."""
        # Flourishing = high clarity + high persistence + high mode entropy
        #             + bounded chaos (lyapunov ~0.5-0.7) + high dimensionality
        # Normalize each feature to [0, 1] across the sweep
        feats = ['mean_clarity', 'clarity_persistence', 'perc_mode_entropy',
                 'dominance_entropy', 'effective_dimensionality']
        normed = {}
        for f in feats:
            fmin, fmax = df[f].min(), df[f].max()
            if fmax > fmin:
                normed[f] = (df[f] - fmin) / (fmax - fmin)
            else:
                normed[f] = pd.Series(0.5, index=df.index)

        # Lyapunov: optimal around 0.5-0.7 (edge of chaos)
        lyap = df['lyapunov_proxy']
        lyap_score = 1.0 - 2.0 * (lyap - 0.6).abs()
        lyap_score = lyap_score.clip(0, 1)
        normed['lyapunov_proxy'] = lyap_score

        # Composite
        df_copy = df.copy()
        df_copy['flourishing'] = sum(normed[f] for f in feats + ['lyapunov_proxy']) / (len(feats) + 1)

        # Top 10 configurations
        top10 = df_copy.nlargest(10, 'flourishing')
        params = list(self.PARAM_RANGES.keys())

        report = []
        report.append("=" * 70)
        report.append("GOLDILOCKS ZONE: Top 10 Flourishing Configurations")
        report.append("=" * 70)
        report.append(f"\nDefault config scores: {df_copy.loc[0, 'flourishing'] if len(df_copy) > 0 else 'N/A':.4f}")
        report.append(f"\n{'Rank':>4}  {'Flourish':>8}  " + "  ".join(f"{p:>12}" for p in params))
        report.append("-" * 70)
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            line = f"{rank:4d}  {row['flourishing']:8.4f}  "
            line += "  ".join(f"{row[p]:12.4f}" for p in params)
            report.append(line)

        report.append(f"\n\nDefault config: " +
                      ", ".join(f"{k}={v}" for k, v in default_config.items()))

        report_text = "\n".join(report)
        print(report_text)

        report_path = os.path.join(output_dir, 'goldilocks_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        # Save full results with flourishing score
        full_path = os.path.join(output_dir, 'phase_cartography_results_scored.csv')
        df_copy.to_csv(full_path, index=False)
        print(f"  Full scored results: {full_path}")


# ============================================================================
# MULTI-GPU SUPPORT
# ============================================================================

def run_multi_gpu(param_ranges=None, steps=1000, output_dir='outputs/phase_cartography'):
    """
    Split parameter sweep across all available CUDA devices.
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("No CUDA devices available")

    print(f"\n{'='*60}")
    print(f"  MULTI-GPU PHASE CARTOGRAPHY")
    print(f"{'='*60}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    carto = PhaseCartographer(device='cuda:0', steps_per_sim=steps)
    configs, n_total = carto.build_parameter_grid(param_ranges)

    # Split configs across GPUs proportional to VRAM
    vrams = []
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vrams.append(props.total_memory)
    total_vram = sum(vrams)
    splits = [int(n_total * v / total_vram) for v in vrams]
    # Ensure all configs are assigned
    splits[-1] = n_total - sum(splits[:-1])

    print(f"  Work split: {splits}")

    all_dfs = []
    offset = 0
    for gpu_idx in range(n_gpus):
        n_this = splits[gpu_idx]
        if n_this <= 0:
            continue
        device = f'cuda:{gpu_idx}'
        print(f"\n  --- GPU {gpu_idx} ({torch.cuda.get_device_properties(gpu_idx).name}): "
              f"{n_this:,} configurations ---")

        gpu_configs = {k: v[offset:offset + n_this] for k, v in configs.items()}
        gpu_configs['timesteps'] = steps

        carto_gpu = PhaseCartographer(device=device, steps_per_sim=steps)
        # Inline batch run for this GPU's configs
        max_batch = carto_gpu.estimate_batch_size()
        n_batches = (n_this + max_batch - 1) // max_batch

        gpu_sigs = []
        gpu_params = {k: [] for k in configs}

        for batch_idx in range(n_batches):
            b_start = batch_idx * max_batch
            b_end = min(b_start + max_batch, n_this)
            batch_size = b_end - b_start

            batch_configs = {k: v[b_start:b_end] for k, v in gpu_configs.items()}
            batch_configs['timesteps'] = steps

            torch.cuda.empty_cache()
            engine = BatchConsciousnessEngine(batch_size, batch_configs, device=device)
            engine.run(steps)
            sigs = engine.extract_signatures()
            gpu_sigs.append(sigs.numpy())
            for k in configs:
                gpu_params[k].append(batch_configs[k])
            del engine
            torch.cuda.empty_cache()

        # Build DataFrame for this GPU
        gpu_all_sigs = np.concatenate(gpu_sigs, axis=0)
        result = {}
        for k in configs:
            result[k] = np.concatenate(gpu_params[k])
        for i, name in enumerate(SIGNATURE_NAMES):
            result[name] = gpu_all_sigs[:, i]
        all_dfs.append(pd.DataFrame(result))

        offset += n_this

    # Combine all GPUs
    df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'phase_cartography_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Combined results ({len(df):,} configs): {out_path}")
    return df


# ============================================================================
# SINGLE-CONFIG VALIDATION RUN
# ============================================================================

def run_validation(device='cuda:0', steps=500):
    """
    Run a single simulation with default config on GPU and print metrics
    for comparison with the CPU v2 simulation.
    """
    print(f"\n{'='*60}")
    print(f"  VALIDATION RUN (single being, default config)")
    print(f"{'='*60}")

    configs = {
        'steering_strength': np.array([0.3], dtype=np.float32),
        'alpha_pull': np.array([0.03], dtype=np.float32),
        'fatigue_rate': np.array([0.08], dtype=np.float32),
        'exploration_noise': np.array([0.05], dtype=np.float32),
        'beta_macro': np.array([4.0], dtype=np.float32),
        'timesteps': steps,
    }

    engine = BatchConsciousnessEngine(1, configs, device=device)
    engine.run(steps)

    # Print summary metrics for comparison
    T = engine.step_count
    clarity = engine.hist_clarity[0, :T].cpu().numpy()
    conflict = engine.hist_conflict[0, :T].cpu().numpy()
    curvature = engine.hist_curvature[0, :T].cpu().numpy()
    speed = engine.hist_speed[0, :T].cpu().numpy()
    integ = engine.hist_integration[0, :T].cpu().numpy()
    diff = engine.hist_differentiation[0, :T].cpu().numpy()

    print(f"\n  Metrics (compare to CPU v2 output):")
    print(f"    Mean clarity:         {clarity.mean():.4f}")
    print(f"    Max clarity:          {clarity.max():.4f}")
    print(f"    Mean conflict angle:  {np.degrees(conflict.mean()):.1f}°")
    print(f"    Mean curvature:       {curvature.mean():.4f} rad/step")
    print(f"    Mean speed:           {speed.mean():.4f}")
    print(f"    Mean integration:     {integ.mean():.4f}")
    print(f"    Mean differentiation: {diff.mean():.4f}")

    # Dominant subsystem distribution
    dom = engine.hist_dominant_sub[0, :T].cpu().numpy()
    print(f"\n  Subsystem dominance:")
    for s in range(8):
        count = (dom == s).sum()
        print(f"    {SUBSYSTEM_NAMES[s]:20s}: {count:4d} ({100*count/T:.1f}%)")

    # Perception modes
    modes = engine.hist_perc_mode[0, :T].cpu().numpy()
    mode_names = ['exploration', 'threat-lock', 'vigilant', 'internal']
    print(f"\n  Perception modes:")
    for m in range(4):
        count = (modes == m).sum()
        print(f"    {mode_names[m]:15s}: {count:4d} ({100*count/T:.1f}%)")

    sigs = engine.extract_signatures()
    print(f"\n  Consciousness signature ({len(SIGNATURE_NAMES)} features):")
    for i, name in enumerate(SIGNATURE_NAMES):
        print(f"    {name:30s}: {sigs[0, i].item():.4f}")

    return engine


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated Consciousness Phase Cartography',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate GPU engine matches CPU (single being)
  python gpu_ensemble_sim.py --mode validate

  # Quick sweep (small grid, fast)
  python gpu_ensemble_sim.py --mode sweep --preset quick

  # Full sweep with default grid
  python gpu_ensemble_sim.py --mode sweep

  # Full sweep on all GPUs
  python gpu_ensemble_sim.py --mode sweep --multi-gpu

  # Custom sweep
  python gpu_ensemble_sim.py --mode sweep --steps 2000 --grid-size 15
        """
    )
    parser.add_argument('--mode', choices=['validate', 'sweep', 'visualize'],
                        default='sweep', help='Run mode')
    parser.add_argument('--device', default='cuda:0', help='CUDA device')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Timesteps per simulation')
    parser.add_argument('--preset', choices=['quick', 'medium', 'full', 'mega'],
                        default='medium', help='Grid resolution preset')
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Override grid points per parameter')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Use all available GPUs')
    parser.add_argument('--output', default='outputs/phase_cartography',
                        help='Output directory')
    parser.add_argument('--visualize-from', default=None,
                        help='Path to existing results CSV for visualization only')

    args = parser.parse_args()

    if args.mode == 'validate':
        run_validation(device=args.device, steps=args.steps)
        return

    if args.mode == 'visualize':
        csv_path = args.visualize_from or os.path.join(args.output, 'phase_cartography_results.csv')
        df = pd.read_csv(csv_path)
        carto = PhaseCartographer(device=args.device, steps_per_sim=args.steps)
        carto.visualize_phase_diagram(df, output_dir=args.output)
        return

    # --- Sweep mode ---
    # Build parameter grid based on preset
    grid_sizes = {
        'quick':  {'ss': 5, 'ap': 4, 'fr': 4, 'en': 3, 'bm': 3},
        'medium': {'ss': 10, 'ap': 8, 'fr': 8, 'en': 6, 'bm': 6},
        'full':   {'ss': 12, 'ap': 10, 'fr': 10, 'en': 8, 'bm': 8},
        'mega':   {'ss': 16, 'ap': 14, 'fr': 14, 'en': 10, 'bm': 10},
    }

    sizes = grid_sizes[args.preset]
    if args.grid_size:
        sizes = {k: args.grid_size for k in sizes}

    param_ranges = {
        'steering_strength': np.linspace(0.05, 0.80, sizes['ss']),
        'alpha_pull':        np.linspace(0.00, 0.15, sizes['ap']),
        'fatigue_rate':      np.linspace(0.00, 0.30, sizes['fr']),
        'exploration_noise': np.linspace(0.00, 0.20, sizes['en']),
        'beta_macro':        np.linspace(0.50, 15.0, sizes['bm']),
    }

    n_total = 1
    for v in param_ranges.values():
        n_total *= len(v)
    print(f"\n{'='*60}")
    print(f"  CONSCIOUSNESS PHASE CARTOGRAPHY")
    print(f"{'='*60}")
    print(f"  Preset: {args.preset}")
    print(f"  Grid: {' × '.join(str(len(v)) for v in param_ranges.values())} = {n_total:,} configs")
    print(f"  Steps per sim: {args.steps}")
    print(f"  Total being-steps: {n_total * args.steps:,}")

    if args.multi_gpu:
        df = run_multi_gpu(param_ranges=param_ranges, steps=args.steps,
                           output_dir=args.output)
    else:
        carto = PhaseCartographer(device=args.device, steps_per_sim=args.steps)
        df = carto.run_sweep(param_ranges=param_ranges, output_dir=args.output)

    # Auto-visualize
    carto_viz = PhaseCartographer(device=args.device, steps_per_sim=args.steps)
    carto_viz.visualize_phase_diagram(df, output_dir=args.output)

    print(f"\n{'='*60}")
    print(f"  PHASE CARTOGRAPHY COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
