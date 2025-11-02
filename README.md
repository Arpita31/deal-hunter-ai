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
