# 📚 TAIHU: Modular Document Discovery System

**Taihu** is a modular, benchmark-driven document retrieval and generation framework for scholarly and domain-specific search. It supports hybrid search over dense and sparse embeddings, structural filtering via relational engines (e.g., Elasticsearch, SQLite), and LLM-based response generation.

---

## 🧠 System Architecture

### 📌 Modular Dependency Diagram

![Architecture Diagram](./assets/TaihuMDD.drawio%20(6).svg)

This system is designed as a clean Directed Acyclic Graph (DAG) of modular components. Dependencies are managed through interfaces, and inheritance is explicitly visualized to encourage extensibility and testability.

### 🔧 Core Component Layers

| Layer | Description |
|-------|-------------|
| 🟥 **Execution Layer** | Entry points for full workflow orchestration: `SearchApp`, `SearchAppEvaluator`, and `Benchmark`. |
| 🟨 **Manager Layer** | Classes like `HybridManager` and `MonolithManager` control composition of relational and vector search engines. |
| 🟦 **Engine Layer** | `SearchEngine` implementations for Milvus, Elasticsearch, SQLite. All support a shared `search()` interface. |
| 🟩 **Embedding Layer** | `DenseEmbedder` and `SparseEmbedder` interfaces with concrete implementations like `BGEM3Embedder`. |
| 🟫 **Library Layer** | Manages document storage and retrieval by ID. Supports in-memory, file-based, or database-backed storage. |
| ⬜ **LLM Layer** | Prompts are formatted via `PromptBuilder` and generated using `LLMBuilder` (e.g., LLaMA3-8B). |

> 💡 *This layered structure allows any module to be swapped or extended independently.*

---

## 🧪 Benchmark-Driven Development

Taihu includes a benchmark suite defined in `.jsonl` format, containing queries and their expected relevant document IDs. The `SearchAppEvaluator` measures:

- Top-k retrieval accuracy
- Precision/Recall/F1
- Exact-match metrics

This enables reproducible comparisons across embedding models, search engines, and prompt formats.

---

## 🎯 Design Principles

1. **Modular and Extensible**  
   Interfaces like `SearchEngine`, `Embedder`, and `Sampler` are used throughout — making experimentation safe and localized.

2. **Hybrid Search Ready**  
   Combines keyword-based filtering (e.g., Elasticsearch) with vector similarity (e.g., Milvus), optionally reranked by LLMs.

3. **Embedding-Agnostic**  
   Supports both dense (`MiniLM`, `E5`, etc.) and sparse (`BGE-M3`) models via interchangeable embedders.

4. **LLM-Augmented**  
   Uses LLMs to generate or refine responses from top-k hits, with clean prompting strategies.

5. **Storage-Agnostic**  
   Document storage (`Library`) can be swapped between memory, file, or SQL depending on the environment.

---

## 🧱 File Index & Diagram Color Legend

| Color        | Category                         | Examples                                    |
|--------------|----------------------------------|---------------------------------------------|
| 🟦 Blue      | Interfaces / Engines             | `SearchEngine`, `MilvusSearchEngine`, etc.  |
| 🟨 Yellow    | Abstract Base Managers           | `BaseManager`, `MonolithManager`            |
| 🟩 Green     | Data / Document Entity           | `Document`, `Filter`, `MetaData`            |
| 🟥 Red       | Execution & Evaluation Modules   | `SearchApp`, `Benchmark`, `SearchAppEvaluator` |
| 🟪 Gray      | Operation Flow (flowchart)       | Raw data → Embedder → Index → Search → LLM |

---

## 🛠️ Elasticsearch Setup (Secured, Local Only)

```bash
# Step 1: Download
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.0.2-linux-x86_64.tar.gz

# Step 2: Extract
tar -xzf elasticsearch-9.0.2-linux-x86_64.tar.gz
mv elasticsearch-9.0.2 elasticsearch

# Step 3: Start
cd elasticsearch
bin/elasticsearch

# Elasticsearch will print a password and start on https://localhost:9200
