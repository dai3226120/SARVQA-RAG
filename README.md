# SARVQA-RAG 🛰️

**SAR Remote Sensing Visual Question Answering with Retrieval-Augmented Generation**

A research system that combines multi-modal Vision-Language Models (VLMs) with a novel **Membership Degree Hybrid Retrieval** mechanism to answer natural language questions about SAR (Synthetic Aperture Radar) remote sensing imagery.

---

## ✨ Key Features

- **Multi-VLM Support** — Integrates Doubao Seed 2.0 Mini and InternVL3.5-8B as vision-language backbones via OpenAI-compatible APIs.
- **Three-Stage Hybrid RAG Retrieval** — A novel "membership degree" pipeline that learns from historically successful retrievals:
  - **Stage 0** — Optional RAG vector retrieval over a general remote sensing knowledge corpus.
  - **Stage 1** — Membership degree calculation: computes weighted similarity + correctness scores against a semantic cache of previously successful Q&A pairs. If membership exceeds a threshold, associated slices are returned.
  - **Stage 2** — Fallback to standard vector similarity search over SAR QA slices.
- **ReAct Agent with Tool Calling** — Built on LangChain/LangGraph, the agent can invoke retrieval tools, weather queries, and location services during reasoning.
- **Interactive Streamlit Web UI** — Upload SAR images and ask questions in natural language through a chat interface.
- **Full Benchmark Pipeline** — Automated 3-step evaluation: parallel prediction → multi-metric scoring (Cosine, BLEU-1~4, ROUGE-L, METEOR) → analysis & visualization.
- **Modular RAG Subsystem** — Cleanly separated into abstract base classes, vector stores, retrieval services, membership calculators, and data builders.

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                      │
│                   (agent/mainapp.py)                     │
├─────────────────────────────────────────────────────────┤
│              ReAct Agent (LangChain/LangGraph)            │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │ Doubao Seed  │  │  InternVL    │  │  Tool Calls  │  │
│   │   (VLM)      │  │   (VLM)      │  │              │  │
│   └──────────────┘  └──────────────┘  └──────┬───────┘  │
├───────────────────────────────────────────────┼─────────┤
│                 RAG Subsystem                  │         │
│  ┌────────────────────────────────────────────┐│         │
│  │       MembershipHybridService              ││         │
│  │  Stage 0 → Stage 1 → Stage 2              ││         │
│  │  (RAG)    (Membership) (Vector Search)    ││         │
│  └──────────┬──────────┬─────────────────────┘│         │
│             │          │                       │         │
│  ┌──────────▼──┐ ┌─────▼──────────┐           │         │
│  │ ChromaDB    │ │ Semantic Cache │           │         │
│  │ (Slices +   │ │ (Logs +        │           │         │
│  │  Knowledge) │ │  Membership)   │           │         │
│  └─────────────┘ └────────────────┘           │         │
└───────────────────────────────────────────────┼─────────┘
                                                │
                    ┌───────────────────────────▼─────────┐
                    │       Benchmark Pipeline            │
                    │  Predict → Benchmark → Analyze      │
                    └─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SARVQA-RAG-trae/
├── agent/                          # Main agent application
│   ├── mainapp.py                  # Streamlit web UI entry point
│   ├── mainagent.py                # Doubao Seed + membership RAG
│   ├── mainagent_rscsv.py          # Doubao Seed + slice-only RAG
│   ├── mainagent_internVL.py       # InternVL + membership RAG
│   ├── mainagent_internVL_rscsv.py # InternVL + slice-only RAG
│   ├── tools/
│   │   ├── agent_tools.py          # Tool definitions (rag_summarize, rag_rscsv, etc.)
│   │   └── middleware.py           # Tool monitoring & latency tracking
│   ├── rag/                        # RAG subsystem
│   │   ├── base/                   # Abstract base classes
│   │   ├── builders/               # Data builders (slice clustering, knowledge chunking)
│   │   ├── core/                   # Config, ChromaDB manager, MD5 store
│   │   ├── services/               # Retrieval services (hybrid, slice, knowledge)
│   │   ├── membership/             # Membership degree calculation
│   │   ├── stores/                 # Vector store implementations
│   │   └── eval/                   # Evaluation utilities
│   ├── data/                       # VQA datasets, knowledge corpus, slice tables
│   └── chroma_db/                  # ChromaDB persistent vector storage
├── benchmark/                      # Benchmark evaluation pipeline
│   ├── main_eval.py                # Main evaluation entry (3-step pipeline)
│   ├── core/                       # Predictor, benchmarker, analyzer, metrics
│   ├── models/                     # API client wrappers
│   └── result/                     # Evaluation result outputs
├── config/                         # YAML configuration files
│   ├── model.yml                   # Model names, endpoints, parameters
│   ├── chroma.yml                  # RAG subsystem parameters
│   ├── eval.yml                    # Evaluation parameters
│   ├── agent.yml                   # Agent external data paths
│   └── prompts.yml                 # Prompt template paths
├── model/                          # Model factory
│   └── factory.py                  # ChatTongyi, DashScope, HuggingFace, Doubao, InternVL
├── prompts/                        # Prompt templates (system, RAG, report, agent)
├── utils/                          # Utilities (config, logging, file, path, text)
├── dataset_analysis/               # Dataset statistical analysis scripts
├── dataset_split/                  # Train/val/test dataset splits
└── logs/                           # Agent run logs
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** >= 3.10
- **API Keys** for:
  - Doubao Seed (ByteDance Ark) — primary VLM
  - DashScope (Alibaba Cloud) — embeddings & chat model
  - DeepSeek (optional) — alternative LLM
- **SAR Image Dataset** — placed at a configured path

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd SARVQA-RAG-trae

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install core dependencies
pip install langchain langchain-community langchain-chroma langchain-openai langgraph
pip install chromadb dashscope streamlit pandas numpy pyyaml python-dotenv
pip install scikit-learn k-means-constrained nltk matplotlib openpyxl Pillow
pip install langchain-huggingface sentence-transformers

# Install the project in editable mode
pip install -e agent/
```

### Environment Setup

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=sk-your-deepseek-key
DASHSCOPE_API_KEY=sk-your-dashscope-key
DOUBAO_SEED_API_KEY=ark-your-doubao-key
DOUBAO_SEED_TEMPERATURE=0.7
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your-langsmith-key
```

---

## ⚙️ Configuration

All configuration is managed through YAML files in the `config/` directory:

| File | Purpose |
|------|---------|
| `model.yml` | Model names, API endpoints, timeouts, temperature |
| `chroma.yml` | RAG parameters: k, thresholds, weights, clustering, chunking |
| `eval.yml` | Evaluation dataset paths, metrics, concurrency |
| `agent.yml` | Agent external data paths |
| `prompts.yml` | Paths to prompt template files |

**Key RAG Parameters** (`config/chroma.yml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 1 | Number of top results |
| `top_p` | 9 | Top-P sampling for retrieval |
| `slice_k` | 50 | Slice retrieval candidates |
| `membership_k` | 50 | Membership cache candidates |
| `w1` | 0.9 | Similarity weight in membership score |
| `w2` | 0.1 | Correctness weight in membership score |
| `fit_threshold` | 0.65 | Membership degree threshold |
| `enable_rag_context` | false | Enable Stage 0 RAG knowledge retrieval |

---

## 📖 Usage

### 1. Interactive Web UI

Launch the Streamlit app for interactive SAR visual question answering:

```bash
streamlit run agent/mainapp.py
```

The UI opens in your browser. Upload a SAR image and ask questions in natural language (Chinese or English). The agent will reason step-by-step, calling RAG retrieval tools as needed.

### 2. Command Line Agent

Run specific agent variants directly:

```bash
# Doubao Seed VLM + membership hybrid retrieval
python agent/mainagent.py

# Doubao Seed VLM + slice-only retrieval (no membership)
python agent/mainagent_rscsv.py

# InternVL VLM + membership hybrid retrieval  
python agent/mainagent_internVL.py

# InternVL VLM + slice-only retrieval
python agent/mainagent_internVL_rscsv.py
```

### 3. Benchmark Evaluation

Run the full 3-step evaluation pipeline:

```bash
python benchmark/main_eval.py
```

This will:
1. **Predict** — Run the selected model on the validation dataset with configurable concurrency
2. **Benchmark** — Compute similarity metrics (Cosine, BLEU-1~4, ROUGE-L, METEOR)
3. **Analyze** — Generate visualizations and statistical reports

Select the model variant by editing the `MODEL_KEY` variable in `benchmark/main_eval.py` (line 61):

| `MODEL_KEY` | Description |
|-------------|-------------|
| `doubao-seed` | Direct Doubao API (no agent) |
| `internVL` | Direct InternVL API (no agent) |
| `agent-text-doubao-seed` | Agent + Doubao + membership RAG |
| `agent-text-doubao-seed_rscsv` | Agent + Doubao + slice-only RAG |
| `agent-text-internVL` | Agent + InternVL + membership RAG |
| `agent-text-internVL_rscsv` | Agent + InternVL + slice-only RAG |

Results are saved to `benchmark/result/<timestamp>/`.

### 4. Building the Vector Database

The vector database is built automatically on first run, or can be rebuilt manually:

```python
from agent.rag.builders.slice_builder import SliceBuilder
from agent.rag.builders.knowledge_builder import KnowledgeBuilder

# Build SAR QA slices with KMeans-constrained clustering
slice_builder = SliceBuilder()
slice_builder.build()

# Build knowledge corpus chunks
knowledge_builder = KnowledgeBuilder()
knowledge_builder.build()
```

MD5-based incremental update detection prevents unnecessary rebuilding.

---

## 🧪 Membership Degree Hybrid Retrieval

The core innovation of this project is the **Membership Degree** mechanism:

```
μ(query) = w₁ × similarity(query, cache) + w₂ × correctness(prediction, ground_truth)
```

Where:
- **similarity** — cosine similarity between query embedding and cached successful queries
- **correctness** — BLEU + overlap score between the VLM's prediction and the ground truth answer
- **w₁, w₂** — configurable weights (default: 0.9, 0.1)

When `μ > fit_threshold`, the system retrieves the associated answer slices from the cache, providing the agent with relevant historical context. This allows the system to **learn from past successful retrievals** and improve answer quality over time.

The semantic cache is persisted in ChromaDB and synchronized with a CSV feedback log for transparency and analysis.

---

## 🔧 Tech Stack

| Category | Technology |
|----------|------------|
| Agent Framework | LangChain, LangGraph |
| Vision-Language Models | Doubao Seed 2.0 Mini, InternVL3.5-8B |
| Embeddings | DashScope text-embedding-v4, all-MiniLM-L6-v2 (HuggingFace) |
| Vector Database | ChromaDB |
| Web UI | Streamlit |
| Clustering | K-Means Constrained (scikit-learn) |
| Evaluation Metrics | Cosine Similarity, BLEU, ROUGE-L, METEOR |
| Configuration | YAML + python-dotenv |

---

## 📄 License

This project is for research purposes. Please contact the author for licensing information.

---

## 🤝 Citation

If you use this work in your research, please cite:

```bibtex
@software{sarvqa_rag_trae,
  title     = {SARVQA-RAG: SAR Remote Sensing Visual Question Answering with Retrieval-Augmented Generation},
  year      = {2025},
  note      = {Research project featuring membership degree hybrid retrieval for SAR VQA}
}
```
