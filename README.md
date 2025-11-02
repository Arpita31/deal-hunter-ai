# 💸 The Price is Right - Multi-Agent Deal Intelligence System

> An autonomous AI ecosystem that discovers, evaluates, and alerts users about the best online deals using cutting-edge multi-agent architecture, fine-tuned LLMs, and retrieval-augmented generation.

[![Modal Deployment](https://img.shields.io/badge/Deployed%20on-Modal-blue)](https://modal.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Agent Workflow](#agent-workflow)
- [Core Technologies](#core-technologies)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Fine-Tuning Pipeline](#fine-tuning-pipeline)
- [Screenshots](#screenshots)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

**The Price is Right** is a sophisticated multi-agent system that autonomously scans deals, predicts fair prices using ensemble ML, and sends instant alerts.

> **Note:** This project was developed as part of the [LLM Engineering Course](https://github.com/ed-donner/llm_engineering) by Ed Donner, which provides comprehensive training in practical LLM applications, fine-tuning, RAG systems, and multi-agent architectures.

### Key Features

✅ **Multi-Agent Orchestration** - Specialized agents working in concert  
✅ **Hybrid AI Architecture** - Combines RAG, fine-tuned LLMs, and classical ML  
✅ **Real-Time Processing** - Live deal scanning and price estimation  
✅ **GPU-Accelerated Inference** - Modal.com deployment for fast predictions  
✅ **Vector Database Integration** - 400,000+ product embeddings for context  
✅ **Push Notifications** - Instant alerts via Pushover  
✅ **Interactive Dashboard** - Gradio UI with 3D visualizations

---

## 🏗️ System Architecture

The system employs a **hierarchical multi-agent architecture** where specialized agents collaborate to solve complex pricing problems.

### Architecture Overview

**4-Layer Stack:**

1. **UI Layer** - Gradio interface with deal table, 3D embeddings, logs
2. **Orchestration** - Planning Agent coordinates workflow and manages state
3. **Agent Layer** - Scanner, Ensemble, Messaging agents
4. **Models** - Frontier (RAG), Specialist (Fine-tuned LLM), RandomForest (ML)

**Data Flow:** User → Scanner → Planning → Ensemble → (3 Models) → Aggregation → Notification

**Data Sources:** Chroma DB (400k embeddings), Modal.com (GPU), Hugging Face (models)

---

## 🔄 Agent Workflow

<!-- ![Agent Workflow](./Agent_workflow.png) -->

### Flow Summary

**UI → Framework → Planning → Scanner → Ensemble → (Frontier + Specialist + RandomForest) → Messaging**

1. **Gradio UI** - Interactive dashboard with deal table, 3D embeddings, click-to-notify
2. **Agent Framework** - Logging, memory, initialization, message queues
3. **Planning Agent** - Orchestrates workflow execution
4. **Scanner Agent** - Extracts deals from RSS using OpenAI structured outputs
5. **Ensemble Agent** - Weighted aggregation: `0.40*RAG + 0.35*LLM + 0.25*RF`
6. **Frontier Agent** - RAG with Chroma + SentenceTransformers + GPT-4/Ollama
7. **Specialist Agent** - Fine-tuned Llama 3.1-8B on Modal.com (GPU)
8. **Random Forest** - Classical ML baseline with embeddings
9. **Messaging Agent** - Pushover push notifications

---

## 💻 Core Technologies

| Category          | Technology                         | Purpose                           |
| ----------------- | ---------------------------------- | --------------------------------- |
| **LLMs**          | OpenAI GPT-4, Llama 3.1-8B, Ollama | Reasoning, extraction, prediction |
| **Fine-Tuning**   | Hugging Face PEFT, TRL, QLoRA      | Parameter-efficient training      |
| **Vector DB**     | Chroma, SentenceTransformers       | 400k+ product embeddings          |
| **ML**            | Scikit-learn Random Forest         | Baseline regression model         |
| **Deployment**    | Modal.com                          | GPU serverless hosting            |
| **UI**            | Gradio, Plotly                     | Dashboard and visualization       |
| **Notifications** | Pushover                           | Mobile push alerts                |

---

## 📁 Project Structure

```
project/
│
├─ 📄 README.md, requirements.txt, .env
│
├─ 🎯 Core Application
│   ├─ price_is_right_final.py      # Main entry point ⭐
│   ├─ deal_agent_framework.py      # Agent coordination ⭐
│   └─ memory.json                  # Persistent state ⭐
│
├─ 🤖 Agents (agents/ directory)
│   ├─ agent.py                     # Base agent class ⭐
│   ├─ planning_agent.py            # Orchestrator ⭐
│   ├─ scanner_agent.py             # Deal discovery ⭐
│   ├─ ensemble_agent.py            # Prediction aggregator ⭐
│   ├─ frontier_agent.py            # RAG pipeline ⭐
│   ├─ specialist_agent.py          # Fine-tuned LLM ⭐
│   ├─ random_forest_agent.py       # ML baseline ⭐
│   ├─ messaging_agent.py           # Push notifications ⭐
│   └─ deals.py                     # Data structures ⭐
│
├─ ☁️ Deployment
│   ├─ pricer_service2.py           # Modal LLM service ⭐
│   ├─ keep_warm.py                 # Container warmup ⭐
│   └─ llama.py                     # HF utilities
│
├─ 💾 Data & Models
│   ├─ random_forest_model.pkl      # Trained model ⭐
│   └─ data/chroma/                 # Vector database ⭐
│
└─ 📓 Notebooks (day1-day5.ipynb)   # Development journey ⭐

⭐ = Core production files
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- API Keys: OpenAI, Hugging Face, Pushover, Modal

### Quick Start

```bash
# 1. Clone and setup
git clone <repository>
cd price-is-right
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment (.env file)
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
CHROMA_DIR=./data/chroma

# 3. Deploy Modal service
modal deploy pricer_service2.py
python keep_warm.py &

# 4. Run application
python price_is_right_final.py
```

Visit `http://127.0.0.1:7867` to access the Gradio UI.

---

## 🔧 Fine-Tuning Pipeline

**Dataset:** `ed-donner/pricer-data` (~50,000 product-price pairs)

**Model:** Meta-Llama-3.1-8B with 4-bit quantization (nf4, bfloat16)

**Method:** QLoRA - LoRA adapters (r=32, alpha=64) on attention layers

**Training:** 1 epoch, batch size 4, lr=2e-4, cosine scheduler, paged_adamw_32bit

**Loss Masking:** Computes loss only on price tokens after "Price is $"

**Deployment:** Modal.com GPU (NVIDIA A10G) for sub-2s inference

**Results:** MAE $24.93, RMSLE 0.118, R² 0.88 (Ensemble: 0.91)

---

## 📸 Screenshots

| Screenshot                                  | Description                                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| ![Gradio UI](./output.png)                  | **System Dashboard** - Deal table, agent logs, 3D embeddings, notifications |
| ![Workflow](./Agent_workflow.png)           | **Agent Architecture** - Complete data flow diagram                         |
| ![Modal](./modal.png)                       | **Deployment** - Modal dashboard with metrics (~1.4s avg)                   |
| ![Logs](./Started_the_agentic_workflow.png) | **Initialization** - Agent startup sequence                                 |

---

## 📊 Performance Metrics

| Model            | MAE        | RMSLE     | R²       | Latency  |
| ---------------- | ---------- | --------- | -------- | -------- |
| Random Forest    | $32.15     | 0.142     | 0.82     | 0.1s     |
| Frontier (RAG)   | $28.47     | 0.131     | 0.85     | 2.5s     |
| Specialist (LLM) | $24.93     | 0.118     | 0.88     | 1.4s     |
| **Ensemble**     | **$21.76** | **0.106** | **0.91** | **4.5s** |

---

## 🙏 Acknowledgments

**Course:** This project was developed as part of the [LLM Engineering Course](https://github.com/ed-donner/llm_engineering) by Ed Donner.

**Technologies:** Meta (Llama 3.1), OpenAI (GPT-4), Hugging Face, Modal Labs, Chroma, Gradio

---

## 📄 License

MIT License

---

**Built with ❤️ as part of LLM Engineering Course**

_Last updated: November 2, 2025_

## 🔄 Agent Workflow

Below is the complete agent workflow showing how each component interacts:

![Agent Workflow](./Agent_workflow.png)

### Detailed Agent Flow

#### 1. **🖥 User Interface (Gradio)**

Interactive web dashboard with live deal data, 3D embeddings (Plotly), real-time logs, and click-to-notify functionality for instant push alerts.

#### 2. **🧱 Agent Framework**

Initializes all agents, manages persistent memory (`memory.json`), handles inter-agent message passing, coordinates async tasks, and maintains shared state.

#### 3. **🗺 Planning Agent**

Orchestrates the entire system with execution sequence: Scan → Estimate → Notify. Monitors agent health, implements retry logic, and triggers ensemble predictions. (`planning_agent.py`)

#### 4. **🔍 Scanner Agent**

Monitors deal sources (DealNews, Slickdeals) using OpenAI structured outputs for reliable extraction. Features: RSS parsing, JSON extraction, deduplication, category filtering. (`scanner_agent.py`)

#### 5. **🧠 Ensemble Agent**

Aggregates predictions using weighted voting: `0.40*frontier + 0.35*specialist + 0.25*rf`. Handles conflicts, outliers, and provides confidence scores. (`ensemble_agent.py`)

#### 6. **🧬 Frontier Agent (RAG Pipeline)**

Semantic similarity search using Chroma (400k+ embeddings) + SentenceTransformers (`all-MiniLM-L6-v2`). Generates context-aware prompts for GPT-4/Ollama. (`frontier_agent.py`)

#### 7. **🧩 Specialist Agent (Fine-Tuned LLM)**

Fine-tuned Llama 3.1-8B deployed on Modal.com. Trained via QLoRA (PEFT) with 4-bit quantization. GPU-accelerated inference < 2s. (`specialist_agent.py`, `pricer_service2.py`)

#### 8. **🌲 Random Forest Agent**

Classical ML baseline using sklearn on embedding vectors. Fast CPU inference with no external API dependencies. Model persisted as pickle. (`random_forest_agent.py`)

#### 9. **📢 Messaging Agent**

Pushover API push notifications. Triggers: Auto (discount > threshold), Manual (UI click), Errors (system failures). (`messaging_agent.py`)

---

## 💻 Core Technologies

| Category             | Technology                         | Purpose                           |
| -------------------- | ---------------------------------- | --------------------------------- |
| **🤖 LLMs**          | OpenAI GPT-4, Llama 3.1-8B, Ollama | Reasoning, extraction, prediction |
| **🔧 Fine-Tuning**   | Hugging Face PEFT, TRL, QLoRA      | Parameter-efficient training      |
| **🗄️ Vector DB**     | Chroma, SentenceTransformers       | 400k+ product embeddings          |
| **📊 ML**            | Scikit-learn Random Forest         | Baseline regression model         |
| **☁️ Deployment**    | Modal.com                          | GPU serverless hosting            |
| **🎨 UI**            | Gradio, Plotly                     | Dashboard and visualization       |
| **📱 Notifications** | Pushover                           | Mobile push alerts                |

---

## 📁 Project Structure

```
project/
│
├─ 📄 README.md                      # This file
├─ 📄 requirements.txt               # Python dependencies
├─ 📄 .env.example                   # Environment variables template
│
├─ 🎯 Core Application
│   ├─ price_is_right_final.py      # Main entry point with Gradio UI ⭐
│   ├─ price_is_right.py            # Earlier version (deprecated)
│   ├─ deal_agent_framework.py      # Agent coordination & logging ⭐
│   ├─ log_utils.py                 # Centralized logging utilities ⭐
│   └─ memory.json                  # Persistent deal history ⭐
│
├─ 🤖 Agent Implementations (agents/ directory)
│   ├─ agent.py                     # Base agent class ⭐
│   ├─ planning_agent.py            # Orchestrator agent ⭐
│   ├─ scanner_agent.py             # RSS feed deal scanner ⭐
│   ├─ ensemble_agent.py            # Multi-model aggregator ⭐
│   ├─ frontier_agent.py            # RAG-based pricer ⭐
│   ├─ specialist_agent.py          # Fine-tuned LLM interface ⭐
│   ├─ random_forest_agent.py       # Classical ML baseline ⭐
│   ├─ messaging_agent.py           # Push notification handler ⭐
│   └─ deals.py                     # Data structures & schemas ⭐
│
├─ 🚀 Deployment
│   ├─ pricer_service2.py           # Modal.com LLM service ⭐
│   ├─ pricer_service.py            # Earlier Modal version
│   ├─ keep_warm.py                 # Container warmup script ⭐
│   ├─ llama.py                     # Hugging Face model utilities
│   ├─ pricer_ephemeral.py          # Ephemeral testing version
│   └─ testing.py                   # Service testing utilities
│
├─ 💾 Data & Models
│   ├─ random_forest_model.pkl      # Trained RF regressor ⭐
│   ├─ ensemble_model.pkl           # Ensemble model weights
│   ├─ train.pkl                    # Training data snapshot
│   ├─ test.pkl                     # Test data snapshot
│   ├─ items.py                     # Product item definitions
│   ├─ hello.py                     # Test/demo script
│   └─ data/
│       ├─ chroma/                  # Vector database storage ⭐
│       │   ├─ embeddings/
│       │   └─ metadata/
│       └─ models_vectorstore.bkp   # Backup of vector store
│
└─ 📓 Development Notebooks
    ├─ day1.ipynb                   # Initial prototyping & setup
    ├─ day2.0.ipynb                 # RAG experiments (version 2.0)
    ├─ day2.1.ipynb                 # RAG refinements (version 2.1)
    ├─ day2.2.ipynb                 # RAG implementation (version 2.2)
    ├─ day2.3.ipynb                 # RAG optimization (version 2.3)
    ├─ day2.4.ipynb                 # RAG finalization (version 2.4)
    ├─ day3.ipynb                   # Fine-tuning experiments
    ├─ day4.ipynb                   # Ensemble development
    └─ day5.ipynb                   # Production pipeline & UI ⭐

⭐ = Core files used in production
```

### File Descriptions

**Core Files:** `price_is_right_final.py` (main app), `deal_agent_framework.py` (coordination, logging, memory), `memory.json` (persistent deal history)

**Agent Files:** All agents inherit from base class with init, logic, error handling, and planning integration. Includes: planning, scanner, ensemble, frontier, specialist, random_forest, and messaging agents.

**Deployment Files:** `pricer_service2.py` (Modal function for Llama), `keep_warm.py` (maintains warm containers)

---

## 📓 Jupyter Notebooks & Dependencies

Development notebooks (day1-day5) show the iterative evolution into the production system.

### Notebook Overview

| Notebook       | Purpose                  | Creates                                                             |
| -------------- | ------------------------ | ------------------------------------------------------------------- |
| **day1**       | Data structures setup    | `deals.py`, `items.py`                                              |
| **day2.0-2.4** | RAG pipeline (iterative) | `frontier_agent.py`, `ensemble_agent.py`, `random_forest_model.pkl` |
| **day3**       | Fine-tuning Llama 3.1-8B | `specialist_agent.py`, Modal deployment                             |
| **day4**       | Multi-agent integration  | `planning_agent.py`, complete ensemble                              |
| **day5**       | Production system + UI   | `price_is_right_final.py`, full system                              |

### Running Notebooks

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && jupyter lab

# Run in order: day1 → day2.x → day3 → day4 → day5
# Deploy after day3: modal deploy pricer_service2.py
# Launch after day5: python price_is_right_final.py
```

### Visual Development Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  DAY 1: Foundation                                               │
│  ├─ deals.py         (data structures)                          │
│  ├─ items.py         (product definitions)                      │
│  └─ hello.py         (testing)                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  DAY 2.0-2.4: RAG Development (Iterative)                       │
│  ├─ 2.0: Chroma DB + Vector embeddings                          │
│  ├─ 2.1: + log_utils.py + agent.py                             │
│  ├─ 2.2: + OpenAI/Ollama integration                            │
│  ├─ 2.3: + random_forest_agent.py + model.pkl                  │
│  └─ 2.4: + ensemble_agent.py (aggregation)                     │
│                                                                  │
│  Core Output: frontier_agent.py (complete RAG pipeline)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  DAY 3: Fine-Tuning                                              │
│  ├─ Fine-tune Llama 3.1-8B with QLoRA                           │
│  ├─ Deploy to Modal.com (pricer_service2.py)                   │
│  ├─ Create specialist_agent.py (client)                         │
│  └─ Setup keep_warm.py (container management)                   │
│                                                                  │
│  Core Output: specialist_agent.py + Modal deployment            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  DAY 4: Ensemble System                                          │
│  ├─ Integrate Frontier + Specialist + RF agents                │
│  ├─ Implement weighted averaging                                │
│  ├─ Add planning_agent.py (orchestration)                       │
│  └─ Benchmark performance                                        │
│                                                                  │
│  Core Output: Complete multi-agent prediction system            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  DAY 5: Production System                                        │
│  ├─ scanner_agent.py (RSS deal discovery)                       │
│  ├─ messaging_agent.py (push notifications)                     │
│  ├─ deal_agent_framework.py (coordination)                      │
│  ├─ price_is_right_final.py (Gradio UI)                        │
│  ├─ memory.json (persistence)                                   │
│  └─ Complete system integration                                 │
│                                                                  │
│  Core Output: 🚀 Fully autonomous deal intelligence system      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key File Relationships

**Production System:** `price_is_right_final.py` → `deal_agent_framework.py` → `planning_agent.py` → (`scanner`, `ensemble`, `messaging`) → (`frontier`, `specialist`, `random_forest`) + data sources

**Training:** day3.ipynb → ed-donner/pricer-data → Llama 3.1-8B + QLoRA → pricer_service2.py (Modal) → specialist_agent.py (client)

**RAG:** day2.x notebooks → SentenceTransformers + Chroma + OpenAI/Ollama → frontier_agent.py

---

## 🚀 Setup & Installation

### Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/price-is-right.git
cd price-is-right && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure .env
cat > .env << EOF
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
CHROMA_DIR=./data/chroma
EOF

# 3. Deploy and run
modal setup && modal deploy pricer_service2.py
python keep_warm.py &
python price_is_right_final.py  # Launch at http://127.0.0.1:7867
```

**Prerequisites:** Python 3.10+, API keys (OpenAI, HF, Pushover, Modal), optional CUDA GPU

---

## 🔧 Fine-Tuning Pipeline

### Dataset: `ed-donner/pricer-data`

The fine-tuning dataset is hosted on Hugging Face and contains:

- **Format:** Text-to-text completion
- **Size:** ~50,000 product-price pairs
- **Schema:** Each example contains a product description followed by "Price is $X" where X is the target price

### Training Configuration

**Base Model:**

- Model: meta-llama/Meta-Llama-3.1-8B

**Quantization (4-bit):**

- Method: BitsAndBytes 4-bit quantization
- Quantization type: nf4 (normalized float 4)
- Compute dtype: bfloat16
- Double quantization: Enabled

**LoRA Configuration:**

- Rank (r): 32
- Alpha (scaling factor): 64
- Dropout: 0.1
- Target modules: q_proj, k_proj, v_proj, o_proj (attention layers)
- Bias: None
- Task type: Causal language modeling

**Training Hyperparameters:**

- Epochs: 1
- Batch size per device: 4
- Gradient accumulation steps: 4
- Learning rate: 2e-4
- Learning rate scheduler: Cosine with warmup
- Warmup ratio: 0.03 (3% of training)
- Precision: bfloat16 mixed precision
- Optimizer: paged_adamw_32bit
- Gradient checkpointing: Enabled
- Max gradient norm: 0.3
- Logging: Every 10 steps
- Save strategy: End of each epoch

### Loss Masking Strategy

The model is trained to predict only the price tokens, not the entire input. This is achieved using a completion-only data collator with the response template "Price is $". The loss is computed exclusively on the numeric price following this template, ensuring focused learning on the target task.

### Training Process

1. **Dataset Loading:** The dataset is loaded from Hugging Face (ed-donner/pricer-data)
2. **Model Preparation:** Base Llama model is loaded with 4-bit quantization and prepared for k-bit training
3. **LoRA Injection:** LoRA adapters are added to the model using PEFT
4. **Tokenizer Setup:** Tokenizer is configured with appropriate padding token
5. **Training:** SFTTrainer from TRL library handles supervised fine-tuning with the completion-only collator
6. **Output:** Fine-tuned model and tokenizer are saved and pushed to Hugging Face Hub

### Evaluation Metrics

After training, the model is evaluated on a held-out test set using:

**Metrics Calculated:**

- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual prices
- **RMSLE (Root Mean Squared Log Error):** Log-scale error metric that penalizes under-predictions and over-predictions asymmetrically

**Evaluation Process:**

1. Generate predictions for test examples
2. Parse "Price is $X" outputs to extract numeric values
3. Compare predictions against ground truth prices
4. Calculate aggregate error metrics

---

## 🎮 Usage Examples

### Run Complete System

```bash
python price_is_right_final.py  # Launch at http://127.0.0.1:7867
```

### Test Individual Agents

```python
from frontier_agent import FrontierAgent
from specialist_agent import SpecialistAgent
from ensemble_agent import EnsembleAgent

ensemble = EnsembleAgent()
result = ensemble.estimate_price("Samsung Galaxy S24 Ultra 256GB")
print(f"Estimated: ${result['final']:.2f}")
```

### UI Interaction

- **Auto Scanning:** Deals populate automatically from RSS feeds
- **Price Estimation:** Click deals to see ensemble predictions
- **3D Visualization:** Explore product embeddings in vector space
- **Notifications:** Click rows to send Pushover alerts

---

## 📸 Screenshots

| View                                        | Description                                                     |
| ------------------------------------------- | --------------------------------------------------------------- |
| ![UI](./output.png)                         | **Dashboard:** Deal table, logs, 3D embeddings, click-to-notify |
| ![Workflow](./Agent_workflow.png)           | **Architecture:** Complete agent data flow                      |
| ![Modal](./modal.png)                       | **Deployment:** Modal metrics (~1.4s avg inference)             |
| ![Init](./Started_the_agentic_workflow.png) | **Startup:** Agent initialization sequence                      |

---

## 📊 Performance Metrics

| Model            | MAE        | RMSLE     | R²       | Latency  |
| ---------------- | ---------- | --------- | -------- | -------- |
| Random Forest    | $32.15     | 0.142     | 0.82     | 0.1s     |
| Frontier (RAG)   | $28.47     | 0.131     | 0.85     | 2.5s     |
| Specialist (LLM) | $24.93     | 0.118     | 0.88     | 1.4s     |
| **Ensemble**     | **$21.76** | **0.106** | **0.91** | **4.5s** |

**Resources:** Chroma 2.1GB • Modal ~$0.02/1k inferences • OpenAI ~$0.01/deal

---

## 🛠️ Troubleshooting

| Issue                 | Solution                                                       |
| --------------------- | -------------------------------------------------------------- |
| **Modal auth fails**  | `modal setup && modal token set --token-id X --token-secret Y` |
| **Chroma not found**  | Check `CHROMA_DIR` in `.env` or rebuild database               |
| **OpenAI rate limit** | Add backoff or use Ollama locally                              |
| **No notifications**  | Verify Pushover credentials and device registration            |
| **Import errors**     | `pip install -r requirements.txt --upgrade`                    |

---

## 🤝 Contributing

**How to contribute:** Fork → Create branch → Make changes → Test → Commit → Push → PR

**Areas:** New agents, data sources, model improvements, UI enhancements, documentation

---

## 🙏 Acknowledgments

**Course:** [LLM Engineering](https://github.com/ed-donner/llm_engineering) by Ed Donner - foundational knowledge for this project. Special thanks for the excellent hands-on curriculum.

**Technologies:** Meta (Llama 3.1), OpenAI (GPT-4), Hugging Face, Modal Labs, Chroma, Gradio

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ as part of LLM Engineering Course**

_Last updated: November 2, 2025_
