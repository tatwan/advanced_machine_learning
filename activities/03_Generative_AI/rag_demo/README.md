# Retrieval-Augmented Generation (RAG)

## 📚 Overview

This module covers **RAG systems** that combine LLMs with external knowledge retrieval. Learn to build, evaluate, and optimize RAG pipelines using LangChain and LlamaIndex.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Build RAG pipelines with **LangChain**
- ✅ Implement RAG using **LlamaIndex**
- ✅ Connect RAG to **SQL databases**
- ✅ **Evaluate RAG systems** with metrics
- ✅ Use different cloud backends (OpenAI, Bedrock)

---

## 📂 Module Structure

```
rag_demo/
├── README.md (this file)
├── 01_RAG_with_Langchain.ipynb
├── 02_RAG_with_Langchain_Bedrock.ipynb
├── 02_RAG_with_Langchain_Bedrock_Students.ipynb
├── 03_RAG_with_LlamaIndex.ipynb
├── 04_RAG_with_LlamaIndex_SQL.ipynb
├── 05_RAG_evaluation.ipynb
├── pdfs/ (Document corpus)
├── vector_db/ (Vector storage)
├── Chinook.db (SQL database for demos)
├── golden_dataset.csv (Evaluation data)
└── contextualize_prompt.txt, qa_prompt.txt
```

---

## 🔄 Recommended Learning Path

### **Part 1: LangChain RAG** (3-4 hours)

1. `01_RAG_with_Langchain.ipynb` - Core RAG concepts
2. `02_RAG_with_Langchain_Bedrock_Students.ipynb` - AWS integration

### **Part 2: LlamaIndex** (2-3 hours)

3. `03_RAG_with_LlamaIndex.ipynb` - Alternative framework
4. `04_RAG_with_LlamaIndex_SQL.ipynb` - Database integration

### **Part 3: Evaluation** (2 hours)

5. `05_RAG_evaluation.ipynb` - Metrics and optimization

---

## 🔍 Topics Covered

| Topic | Description |
|-------|-------------|
| **Document Loading** | PDF, text ingestion |
| **Chunking** | Text splitting strategies |
| **Embeddings** | Vector representations |
| **Vector Stores** | Similarity search |
| **Retrieval** | Semantic search |
| **Generation** | LLM response with context |
| **Evaluation** | RAGAS, faithfulness metrics |

---

## 🛠️ Technical Requirements

```python
langchain, llama-index, chromadb, openai, boto3, ragas
```

---

## 🔗 Related Modules

- **Prerequisites**: [LLMs Intro](../llms_intro/)
- **Next**: [Agentic AI](../agentic_ai/)

---

*Module Difficulty: Intermediate*  
