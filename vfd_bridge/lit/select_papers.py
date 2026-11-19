#!/usr/bin/env python3
"""Select top papers for VFD sweep based on user's pillars."""
import json

master_path = "C:/Users/akbon/OneDrive/Documents/GitHub/FourD/vfd_bridge/lit/master.json"

with open(master_path, 'r') as f:
    papers = json.load(f)

# Manual selection based on relevance to user's pillars:
# - Laplacian spectra of regular polytopes/highly symmetric graphs
# - Spectral filtering / mode cancellation / dimensionality reduction on graphs
# - Coupled-oscillator or resonance spectra on network topologies
# - Spectral entropy / complexity scaling of graph families
# - Synchronizability of small-world/scale-free networks

# Select papers that directly address these topics
selected = [
    # Coupled oscillators + Laplacian (direct match to user's VFD)
    ("2609.10386", "Making the Graph Laplacian Physical: Multiscale Coarse-Graining in Electrical Oscillator Networks",
     "Directly connects graph Laplacian to physical oscillator networks with measurable modes - matches user's coupled-oscillator model"),
    
    # Spectral filtering / mode reduction
    ("2609.26895", "The Laplacian S_{n,n} conjecture is true",
     "Deals with Laplacian spectrum structure and eigenvalue constraints on graphs"),
    
    # Highly symmetric graphs
    ("2609.14110", "Torsion detection in clique complexes is conditionally QMA_1-hard",
     "Connects Laplacian spectra to topological features (torsion) - relates to highly symmetric structures"),
    
    # Spectral entropy / complexity
    ("2611.21670", "Resistance-regular graphs need not be regular",
     "Connects spectral properties (Laplacian pseudoinverse) to graph regularity - spectral geometry"),
    
    # Network dynamics via spectrum
    ("2609.22909", "Spectral Characterization and Network Systems Dynamics of Zero-Divisor Topologies",
     "Uses Laplacian spectrum for consensus dynamics, reliability, transport - matches mode selection theme"),
    
    # Synchronizability / eigenratio
    ("2607.14168", "Tight Wavelet Frames on Graphs via Isometric Group Embedding",
     "Connects graph spectra to tight frames and group symmetries - highly symmetric networks"),
    
    # Spectral theory of special graphs
    ("2611.06981", "The Spectral Theory of Cographs: A Survey, Organized by Graph Matrix",
     "Comprehensive survey of spectral graph theory including Laplacian - provides theoretical background"),
    
    # Dimensionality reduction
    ("2611.04995", "Exact Selective Inference for Spectral Clustering via Piecewise-Constant Graph Perturbation",
     "Connects Laplacian spectrum to dimensionality reduction via spectral clustering"),
    
    # Scale-free / complex networks
    ("2610.24431", "Low-Bit Distributed Constrained Optimization with Application to Optimal Scheduling and Allocation",
     "Uses graph Laplacian spectrum for network optimization - applies to scale-free networks"),
    
    # Spectral geometry
    ("2607.03388", "The Equality Cases For the Laplacian Conjecture of Brouwer",
     "Deals with Laplacian spectrum constraints on graphs - spectral geometry of graphs"),
    
    # Regular polytopes/highly symmetric
    ("2608.16129", "Equality Cases for the Face-Degree Majorization Theorem on Simplicial Complexes",
     "Extends spectral majorization to simplicial complexes - highly symmetric structures"),
    
    # Graph signal processing
    ("2611.26963", "Spectral Entropy of Graph Signal Processing",
     "Directly addresses spectral entropy of graphs - matches user's spectral entropy pillar"),
    
    # Normal modes on graphs
    ("2606.20943", "Quantum Dust from the Curse of Dimensionality",
     "Connects spectral properties to dimensionality and mode structure"),
    
    # Spectral complexity
    ("2605.24640", "Laplacian Spectrum of the Weakly Zero-Divisor Graph of a Finite Commutative Ring",
     "Analyzes Laplacian spectrum of structured graphs - spectral theory"),
    
    # Additional relevant paper on eigenratio/synchronizability
    ("2605.17951", "Graph-Theoretic Detection of Hilbert Space Fragmentation",
     "Connects graph structure to Hilbert space - relevant for mode analysis"),
]

# Sort by ID and select top 15
selected.sort(key=lambda x: x[0])
top_15 = selected[:15]

print(f"Selected {len(top_15)} papers:")
for paper_id, title, why in top_15:
    print(f"  {paper_id}: {title}")
    print(f"    Why: {why}")
    print()

# Write to file for reference
with open("C:/Users/akbon/OneDrive/Documents/GitHub/FourD/vfd_bridge/lit/selected_papers.json", 'w') as f:
    json.dump(top_15, f, indent=2)

print(f"Saved selection to selected_papers.json")
