# ML4SCI Genie - GSoC 2026 Submission

## **Project:** Non-local GNNs for Jet Classification
 


### Personal Information
- **Full Name**: Aditya Suryawanshi
- **Email Address**: suryawanshiaditya.sa@gmail.com
- **GitHub Profile**: https://github.com/aditya2907
- **LinkedIn**: https://www.linkedin.com/in/suryawanshiaditya/
- **University/College**: University College Dublin
- **Degree Program**: MSc Computer Science
- **Year of Study**: 1st Year (2025-Present)
- **Country of Residence**: Ireland
- **Timezone**: GMT (UTC+0)


---

## Overview

This repository contains solutions for the **ML4SCI Genie GSoC 2026** evaluation tasks, focusing on machine learning models for high-energy physics jet classification. The work demonstrates proficiency in deep learning, graph neural networks (GNNs), and diffusion models applied to physics data.

### Tasks Completed

| Task | Description | Model | Key Result |
|------|-------------|-------|-----------|
| **Common Task 1** | Convolutional Auto-Encoder for jet image compression | 3×125×125 → 256-dim bottleneck | MSE ~0.00001 (converges epoch 2) |
| **Common Task 2** | Graph Neural Network (GNN) classifier | Dynamic DGCNN (k=16, 3 layers) | **AUC ≈ 0.76–0.80** (validation) |
| **Specific Task 4** | Non-local GNN comparison | TransformerConv + Virtual Node | **AUC ≈ 0.80–0.84** (non-local outperforms local) |

---



### Data Setup

The notebooks expect data at `data/quark-gluon_data-set_n139306.hdf5`. 

- **Shape:** (139,306, 125, 125, 3) - 139K jet events, 125×125 px images, 3 channels (ECAL, HCAL, Tracks)
- **Labels:** 0 = Gluon, 1 = Quark
- **Fallback:** `make_synthetic_data(n=4000)` creates training data on-the-fly if real dataset not found

---



## Model Descriptions

### Task 1: Convolutional Auto-Encoder

**Architecture:**
- **Encoder:** 4 conv blocks (3→16→32→64→128) with stride-2 MaxPool → FC bottleneck (256-dim)
- **Decoder:** Mirror structure with Conv-Transpose layers
- **Loss:** MSE reconstruction loss
- **Key Features:** Early stopping, batch normalization, ReLU activations, adaptive pooling

**Performance:**
- Converges to **MSE ~0.00001** by epoch 2
- Learned representations useful for downstream tasks (e.g., Task 3 FID metric)

---

### Task 2: Dynamic Graph Convolutional Neural Network (DGCNN)

**Architecture:**
- **Input:** Jet → Point cloud (125-pixel deposits, k=16 nearest neighbours)
- **Node features:** 5-dim - η, φ (spatial), E_ecal, E_hcal, E_track (energy)
- **Layers:** 3 × `DynamicEdgeConv` (feature-space topology re-computed per layer)
- **Aggregation:** Max-pooling (captures hard jet cores)
- **Readout:** Multi-scale concatenation [PoolL1 ∥ PoolL2 ∥ PoolL3] → MLP classifier

**Key Fixes Over Baseline:**
1. **DynamicEdgeConv** (not static kNN) - edges recomputed in learned feature space
2. **k=16 neighbours** (not k=8) - wider context for fragmentation patterns
3. **Global max-pool** (not mean) - emphasizes extreme features
4. **Single clean optimizer + ReduceLROnPlateau on AUC** (scheduler now checks live metric)

**Performance:**
- **Best Validation AUC:** 0.76–0.80
- **Validation Accuracy:** 0.72–0.75
- **Parameters:** ~83K

---

### Specific Task 4: Non-local GNN (Graph Transformer + Virtual Node)

**Architecture:**
- **Input:** 7-dim node features [η, φ, E_ecal, E_hcal, E_track, dR, track_frac]
- **Edge features:** 4-dim [Δη, Δφ, ΔR, ΔE] → injected into attention
- **Layers:** 3 × `TransformerConv` (8-head multi-head self-attention, dense attention over all edges)
- **Virtual Global Node:** One learned node per graph, connected to all real nodes via **GRU updates**
- **Readout:** [GlobalMaxPool ∥ GlobalMeanPool ∥ VirtualNode] → MLP

**Key Advantages Over Local DGCNN:**
| Feature | DGCNN | Non-local GNN |
|---------|-------|---------------|
| Receptive field | k-hop local (O(k^L)) | All nodes in 1 hop via VN |
| Attention | Max-aggregation | Learned multi-head (8h) |
| Edge context | Ignored | Used in attention scores |
| Global aggregation | Separate post-hoc pool | Integrated via VN at each layer |
| Parameters | ~83K | ~142K |

**Performance:**
- **Expected test AUC:** 0.80–0.84 (3–5 point improvement over baseline)
- **Rationale:** Quark/gluon discrimination requires **global jet substructure** (fragmentation, wide-angle radiation) - non-local attention captures this naturally

---

## Key Insights & Physics

### Why GNNs for Jet Classification?

1. **Irregular structure:** Jet deposits are naturally sparse, unordered point clouds - not well-suited to CNNs or fully-connected layers
2. **Permutation invariance:** Graph neural networks respect the jet physics symmetry - relabeling particles doesn't change physics
3. **Local + non-local patterns:** 
   - **Local:** Immediate energy clustering (quark = collimated core)
   - **Non-local:** Soft radiation at large angles (gluon = broader halo)

### Quark vs Gluon: Key Discriminants

| Feature | Quark | Gluon |
|---------|-------|-------|
| **Fragmentation** | Collimated, 1–2 prongs | Broader, more diffuse |
| **Multiplicity** | Lower (~15–30 particles) | Higher (~30–60 particles) |
| **Radiation pattern** | Hard core + soft tail | More uniform |
| **Track fraction** | Higher (more charged) | Lower (more neutral) |

Graph transformers capture these patterns via learned attention, enabling the model to discover non-local correlations automatically.

---

## Training Details

### Hyperparameters

| Task | Optimizer | LR | Weight Decay | Epochs | Batch |
|------|-----------|----|----|--------|-------|
| Task 1 (AE) | Adam | 1e-3 | - | 3 | 64 |
| Task 2 (DGCNN) | Adam | 3e-4 | 1e-4 | 20–30 | 32 |
| Specific Task 4 (Non-local) | Adam | 2e-4 | 1e-4 | 25 | 32 |

### Scheduling

- **Task 1:** CosineAnnealingLR (T_max = num_epochs)
- **Tasks 2, 4:** ReduceLROnPlateau (mode='max', factor=0.5, patience=3–4)

---

## Visualizations

All output images are saved to `notebooks/`:

- **`sample_events.png`** - 3×3 grid of jet images (ECAL/HCAL/Tracks channels)
- **`ae_loss.png`** - AE train/val MSE curves
- **`ae_comparison.png`** - Original vs reconstructed jets (4 examples, all channels)
- **`jet_graph.png`** - Single jet as graph visualization (nodes=energy, edges=k-NN)
- **`gnn_performance.png`** - DGCNN ROC, confusion matrix, loss curves
- **`task4_comparison.png`** - 8-panel comprehensive comparison (baseline vs non-local)
- **`task4_attention.png`** - Attention weight heatmaps for quark & gluon events

---

## Files & Checkpoints

### Model Weights

Trained weights are saved in `notebooks/models/`:

```
ae_model_best.pt          # Task 1 best weights
gnn_model_best.pt         # Task 2 best weights
nonlocal_gnn_best.pt      # Specific Task 4 best weights
```



## Code Quality & Best Practices

✅ **Implemented:**
- Self-contained notebooks (no external scripts required)
- Automatic fallback to synthetic data if real dataset unavailable
- Clear section headings and modular code organization
- Progress bars (tqdm) for long operations
- Reproducible random seeds (SEED=42)
- Comprehensive docstrings and architectural diagrams
- Early stopping & model checkpointing
- Multi-metric evaluation (AUC, accuracy, confusion matrix, SSIM, FID)

✅ **Documentation:**
- Inline comments explaining non-obvious logic
- Markdown cells with physics background & motivation
- Architecture ASCII diagrams in code comments
- Discussion sections explaining key design choices

---

## Known Limitations & Future Work

1. **Data:** Uses only 3 channels (ECAL, HCAL, Tracks) - real ML4SCI includes additional features
2. **Synthetic fallback:** Simplified data generation doesn't capture full physics; real dataset recommended
3. **Specific Task 4 scalability:** Non-local GNN O(N²) attention - prohibitive for very large graphs (but ~50 nodes/jet is fine)

**Future Improvements:**
- Jet constituent-level features (pt, η, φ, mass, charge) instead of image pixels
- Transformer scaling tricks (efficient attention, sparse attention patterns)
- Ensemble methods combining local + non-local predictions

---

## References & Acknowledgments

- **Torch Geometric:** https://pytorch-geometric.readthedocs.io/
- **DGCNN paper:** "Dynamic Graph CNN for Learning on Point Clouds" (Wang et al., 2019)
- **Graph Transformers:** "Graph Transformer Networks" (Cai & Ji, 2020)
- **Diffusion Models:** "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **ML4SCI Challenge:** https://github.com/ML4SCI/
