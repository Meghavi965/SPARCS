# SPARCS: Single-Pass Adaptive Risk Topology and Certifiable Security for Real-Time LLM Middleware

Official reference implementation, benchmark suites, and quantization pipeline for the manuscript:  
**"SPARCS: Single-Pass Adaptive Risk Topology and Certifiable Security for Real-Time LLM Middleware"**  


**Authors:** M. V. Vyas, M. H. Mehta (SVIT College, Gujarat Technological University)  
**Repository:** [https://github.com/Meghavi965/SPARCS](https://github.com/Meghavi965/SPARCS)  
**License:** [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)

---

## 1. Overview & Architectural Pipeline

SPARCS addresses the security-latency trade-off in Large Language Model (LLM) middleware by replacing heavy multi-stage sequential evaluation loops ($\mathcal{O}(K)$ LLM calls) with a single-pass ($\mathcal{O}(1)$ ) disentangled manifold architecture:

* **Phase I: Single-Pass Disentangled Feature Extraction (SPDMD):** A single forward pass over a spectral-normalized transformer encoder (`microsoft/deberta-v3-base`) simultaneously derives pooled continuous embeddings ($E_P \in \mathbb{R}^{768}$) and intent classification logits ($z_P \in \mathbb{R}^2$) without secondary model roundtrips (<15 ms).
* **Phase II: Parallel Inbound Risk Tiers ($L_1 - L_4$):** Evaluates privacy entity density ($L_1$, Microsoft Presidio NER), latent intent classification ($L_2$, dual-headed Softmax), angular semantic manifold divergence against a tenant policy centroid $\mu_\pi$ ($L_3$), and context length saturation ($L_4$).
* **Theoretical Robustness Guarantee (Theorem 1):** PyTorch Spectral Normalization enforces a certified local $K$-Lipschitz continuity bound on the composite decision score $\Delta S \le w_3 \frac{K}{\pi \Vert{}\mu_\pi\Vert{}_2} + w_2 L_\sigma \Vert{}\delta\Vert{}_2$.
* **Phase IV: Closed-Loop Forensic Outbound Defense ($L_5$):** Employs an $\mathcal{O}(1)$ Aho-Corasick automaton tracking dynamic session canaries ($\kappa$) across Raw UTF-8, Base64, Hexadecimal, and Rot13 representations to guarantee 0.0% system prompt leakage.

---

## 2. Hardware & Environment Requirements

### Recommended Production / Replication Setup
* **Operating System:** Linux (Ubuntu 20.04/22.04 LTS tested)
* **Python Version:** 3.10+
* **Training & Full GPU Benchmarks:** NVIDIA GPU with CUDA Compute Capability $\ge$ 8.0, minimum 16 GB VRAM (e.g., NVIDIA T4, RTX 3090, or A100).
* **Quantized CPU Inference:** Multi-core x86_64 CPU (tested on Intel Xeon Silver 4214R; ~35 ms median latency under INT8 ONNX).
* **System RAM:** 16 GB minimum (32 GB recommended for full entity extraction pipelines).

> **Resource Constraints & Free-Tier Cloud Environments (e.g., Colab / Kaggle T4):**  
> Training loops and the full dataset calibration pipeline involve loading transformer attention weights alongside spaCy/Presidio NER engines. On shared free-tier instances (typically capped at 12 GB host RAM), simultaneous extraction can trigger host memory termination (SIGKILL / OOM). For resource-constrained validation, use the provided smoke test script (`scripts/train_smoke.sh`) or direct evaluation routines.

---

## 3. Installation & Setup

Clone the repository and install dependencies in an isolated virtual environment:

```bash
# Clone the repository
git clone [https://github.com/Meghavi965/SPARCS.git](https://github.com/Meghavi965/SPARCS.git)
cd SPARCS

# Create and activate a virtual environment
python3 -m venv sparcs_env
source sparcs_env/bin/activate

# Upgrade packaging tools and install in editable mode
pip install --upgrade pip setuptools wheel
pip install -e .
