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
```

# Adaptive Hybrid Search with Monitoring-Guided Knob Tuning

## Overview

This system is built on the principle that **adaptability** and **observability** together enable intelligent retrieval. By exposing a set of tunable parameters — or _knobs_ — and instrumenting the system to monitor performance across data partitions, we enable it to learn which configurations work best under which conditions.

## Controllable Parameters

We expose several degrees of freedom in the hybrid search system:

- **Hybridization weight `α`**: controls the interpolation between dense and sparse retrieval engines.
- **Language mixture weights**: governs the balance between using English and Chinese fields in document scoring.

These knobs are adjustable at query time, serving as the primary control surface for retrieval behavior.

## Conditioning on Metadata Filters

Metadata fields such as `school`, `year`, `category`, and `domain` define natural partitions over the corpus. Each filter value (e.g., `school=NTU`) identifies a subset of documents. Given a query with filters, the system adjusts knobs based on the historical performance of those subsets.

## Monitoring Subset Distributions

For each partition, we log performance metrics such as:

- Retrieval latency per engine
- Precision@k under different `α` values
- Dense/sparse score distributions
- Result overlap across engines

This builds empirical performance distributions that help guide knob tuning.

## Subset Value and Confidence Estimation

Not all partitions are equally informative. We estimate the confidence that a partition meaningfully impacts retrieval performance:

- Use **ANOVA** or **permutation tests** to check for significant performance differences across values
- Compute **signal-to-noise ratio**:

  `weight_f = between_variance / (within_variance + ε)`

- Use `weight_f` as a confidence-weight for that partition’s influence

## Aggregating Partition Signals

A query may belong to multiple partitions (e.g., `school=NTU`, `category=CS`). The final knob value is computed as a weighted combination:

  `α_final = Σ_f (weight_f × α_f)`

Where:
- `f` is a filter partition (e.g., `school`)
- `weight_f` is the confidence in that partition
- `α_f` is the empirically optimal knob value for that subset

## Conclusion

By combining a highly configurable architecture with rich performance monitoring, we enable vision-guided control. Each monitored subset becomes a signal generator, and each knob a response. Over time, this framework allows the system to adapt to its data and users — not through static rules, but through learned inference.



# Adaptive Hybrid Search with Monitoring and Partition-Aware Knob Tuning

## Overview

This system is designed with the core belief that **monitorability** (vision) and **configurability** (mobility) together enable intelligent retrieval. By exposing many degrees of freedom (knobs) and observing the effect of those knobs across metadata partitions (e.g., school, domain, language), the system becomes capable of adjusting its behavior dynamically to improve over time.

We move beyond static pipelines to a feedback-driven, self-aware search infrastructure.

---

## Key Concepts

### 🎛️ Adjustable Knobs
- **Hybridization weight `α`**: interpolation between dense and sparse retrieval
- **Inter-Channel Weights**: English vs. Chinese field influence. Name v.s. Abstract embedding distance. 

### 🔍 Informative Context (Partition Signals)
- **Metadata filters**: school, domain, category, year
- **Performance metrics**: precision@k, MRR, overlap, latency

---

## Partition-Aware Adaptivity

For each query, filters induce one or more **subsets**. Each subset has a history of performance (monitored via WandB), and this data informs the best knob setting.

We compute a weighted combination:

```math
α_{final} = Σ_f (w_f × α_f)
```

Where:
- `f` = metadata field (e.g., `school`)
- `w_f` = confidence that the field is predictive
- `α_f` = best known setting for that field’s value

---

## Monitoring Infrastructure

### Logged per-query via WandB:
- Query ID, filters
- Routing decision and knob values
- Rank of clicked document (if any)
- Engine score distributions
- Retrieval latency

This supports:
- Per-partition dashboards
- A/B testing of knob configurations
- Observability for tuning decisions

---

## Ground Truth and Feedback

### Why static ground truth isn't enough:
- Benchmarks don't reflect the diversity of real-world query intents
- Controlled labels lack user interaction nuance

### Our alternative:
- Use **click rank** or **reciprocal rank** as proxy rewards
- Track whether knob adjustment moves clicked docs higher
- Adapt over time based on logged reward signals (bandits, empirical updates)

---

## Benchmark Integration: LitSearch and Adaptive Knob Tuning

We use **LitSearch** not to evaluate dynamic behavior, but to **demonstrate the importance and tunability of the hybridization knob `α`**:

- LitSearch contains static queries and relevance labels — no user feedback.
- This allows us to run controlled experiments showing that **different partitions (e.g., domains) have different optimal `α` values**.
- We show that **globally static `α` is suboptimal**, and per-partition tuning leads to better MRR/nDCG.

💡 This proves that `α` is an **optimizable variable** that affects search quality.

In contrast, our **adaptive system is the mechanism to optimize `α` in real-world settings**, where user feedback is available but ground truth isn't fixed. It learns `α` over time using monitored metrics like click rank and partition-specific trends.

---

## Publication Worthiness

This work is valuable and publishable because:
- It presents a general, monitorable framework for adaptive hybrid search
- It provides a method to learn where (and how) to adjust knobs
- It bridges the gap between benchmark IR and dynamic real-world systems

Possible venues: SIGIR, WSDM, ECIR, NeurIPS Datasets & Benchmarks.

---

## Summary Philosophy

> Adjust knobs based on what you know.  
> Monitoring gives you vision.  
> Configurability gives you motion.  
> Together, they enable learning-driven search that gets to the right place — not by chance, but by design.


## Challenges
How do we abstract "document" since experimentation in different corpuses can be generalizable but each document can be different. Can it be a functional datatype that satisfies certain conditions? 
 - Interface defines metadata fields and embeddable content. 
 - potentially have multilingual version. 
 - languages as "channels" 
How does same attributes differing by languages differ from different metadata attributes? 
Only content makes sense for translation, no need to translate keywords. 
For content, we can introduce channels, and a function that converts the original data into something in that channel. Not necessarily paraphrase, but also a prompted paraphrasing LLM may also be a viable option. 
How we abstract it would depend on the usage. 

