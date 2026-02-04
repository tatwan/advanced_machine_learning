# LLM Fine-Tuning

## 📚 Overview

This module covers **fine-tuning Large Language Models** for domain-specific tasks. Learn both OpenAI's fine-tuning API and open-source approaches.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Prepare **training data** in JSONL format
- ✅ Fine-tune models using **OpenAI's API**
- ✅ Fine-tune **open-source LLMs** with HuggingFace
- ✅ Evaluate fine-tuned model performance
- ✅ Understand when fine-tuning vs. prompting is appropriate

---

## 📂 Module Structure

```
llms_fine_tuning/
├── README.md (this file)
├── openai/
│   ├── fine_tuning_plan.md
│   ├── openai_fine_tuning_demo.ipynb
│   ├── medical_qa_train.jsonl
│   └── medical_qa_valid.jsonl
└── open_source/
    └── Fine_Tuning_LLM_Healthcare.ipynb
```

---

## 🔄 Recommended Learning Path

### **Part 1: OpenAI Fine-Tuning** (2-3 hours)

1. Review `openai/fine_tuning_plan.md`
2. `openai/openai_fine_tuning_demo.ipynb`
   - Data preparation
   - Job creation and monitoring
   - Using fine-tuned model

### **Part 2: Open-Source Fine-Tuning** (3-4 hours)

3. `open_source/Fine_Tuning_LLM_Healthcare.ipynb`
   - HuggingFace Transformers
   - LoRA/QLoRA techniques
   - Local fine-tuning

---

## 🔍 Methods Covered

| Method | Platform | Best For |
|--------|----------|----------|
| **OpenAI Fine-Tuning** | Cloud | Production, ease of use |
| **Full Fine-Tuning** | Local/Cloud | Maximum control |
| **LoRA** | Local | Memory-efficient adaptation |
| **QLoRA** | Local | Low-resource fine-tuning |

---

## 🛠️ Technical Requirements

```python
openai, transformers, peft, bitsandbytes
```

---

## 🔗 Related Modules

- **Prerequisites**: [LLMs Intro](../llms_intro/)
- **Next**: [RAG Demo](../rag_demo/)

---

*Module Difficulty: Intermediate to Advanced*  
