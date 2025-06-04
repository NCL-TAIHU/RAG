# 📚 TAIHU: Modular Document Discovery System

Taihu is a modular, benchmark-driven document retrieval and generation system designed for scholarly and domain-specific search. It integrates dense and sparse embeddings, hybrid search via Milvus, structural metadata queries using ElasticSearch, and LLM-based response generation.

---

## 🧠 System Architecture

### 📌 Modular Dependency Diagram

![Architecture Diagram](./assets/TaihuMDD.drawio%20(3).svg)

This system is composed of the following main modules:

- **SearchApp**: Core controller that interfaces with embedder modules, data loaders, filters, and LLMs.
- **Sampler**: Provides sampling logic over datasets for benchmarking and evaluation.
- **DataLoader**: Responsible for loading dataset documents and converting them into `Document` objects.
- **Embedder Interface**: Abstracts both dense and sparse embedding backends:
  - `DenseEmbedder` → e.g., `AutoModelEmbedder`
  - `SparseEmbedder` → e.g., `MilvusBGEM3Embedder`
- **CollectionBuilder & CollectionManager**: Manages the creation and indexing of Milvus vector databases.
- **Filter**: Optional preprocessing filters (e.g., `ElasticSearchFilter`) to narrow candidate sets before embedding.
- **PromptBuilder**: Formats top-k results into LLM-ready prompts.
- **LLMBuilder**: Loads the instruction-tuned LLM (e.g., `meta-llama/Llama-3.1-8B-Instruct`) for generation.
- **Benchmark**: Evaluates retrieval performance using a set of ground truth (query, document) pairs.

---

## 📊 Benchmark-Driven Development

TaihuMDD includes a benchmark suite that validates search quality using a `.jsonl` benchmark file of query-answer pairs.

The `SearchAppEvaluator` evaluates precision, recall, and exact-match against known ground truths — enabling rapid iteration and measurable progress.

---

## 🎯 Design Goals

1. **Modular & Testable**: Interfaces for embedders, samplers, and filters encourage easy extension and unit testing.
2. **Embedding-Agnostic**: Plug-and-play support for dense (`MiniLM`) and sparse (`BGE-M3`) embedding models.
3. **Benchmark-Centric**: Reproducible evaluation pipeline to guide improvements.
4. **Ready for Scale**: Milvus vector DB support and dataset abstractions enable scaling to millions of documents.
5. **LLM-Augmented**: Response generation is powered by instruction-tuned LLMs with customizable prompts.

---

## 🛠️ File Index (Color-Coded from Diagram)

| Color        | Type                            | Description                                |
|--------------|----------------------------------|--------------------------------------------|
| 🟦 Blue      | Interface/Implementation        | Core functionality (e.g., Embedder, Filter)|
| 🟨 Yellow    | Base Classes / Abstract Layer   | Shared behavior interfaces (e.g., `Sampler`)|
| 🟩 Green     | Data / Configuration            | Classes for config/data (e.g., `Document`) |
| 🟥 Red       | Execution / External Systems    | LLM, Elasticsearch, Milvus                 |
| 🟪 Gray      | Operation Flow (flowchart)      | Shows how data flows from input to response|

---

## 📌 TODO / Future Work

- [ ] Implement Samplers for sampling Benchmark
- [ ] Implement Metadata filtering using ElasticSearch in ElasticSearchFilter

---

## 📂 Diagram Source

- Architecture source editable via [draw.io](https://draw.io)
- File: `TaihuMDD.drawio.svg` or `.png` in repo

---

## 📄 License

MIT License © 2025 Jerome Tze-Hou Hsu
