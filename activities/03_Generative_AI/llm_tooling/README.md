# LLM Tooling & Function Calling

## 📚 Overview

This module covers **function calling** and **tool use** with LLMs. Learn to extend LLM capabilities by connecting them to external tools and databases.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Implement **function calling** with OpenAI
- ✅ Use **LangChain** for structured tool integration
- ✅ Connect LLMs to **databases** and APIs
- ✅ Build **tool-augmented** LLM applications

---

## 📂 Module Structure

```
llm_tooling/
├── README.md (this file)
├── 01_function_calling_openai.ipynb
├── 02_demo_LangChain.ipynb
├── 03_function_calling_langchain.ipynb
├── city_tour.db (Demo database)
├── flight_data.csv
└── fun_facts.csv
```

---

## 🔄 Recommended Learning Path

### **Part 1: OpenAI Function Calling** (2 hours)

1. `01_function_calling_openai.ipynb`
   - JSON schema for functions
   - Parallel function calls
   - Response handling

### **Part 2: LangChain Integration** (2-3 hours)

2. `02_demo_LangChain.ipynb` - LangChain basics
3. `03_function_calling_langchain.ipynb` - Advanced tool use

---

## 🔍 Topics Covered

| Topic | Description |
|-------|-------------|
| **Function Calling** | Structured tool invocation |
| **Tool Definitions** | JSON schema specifications |
| **LangChain Tools** | Pre-built and custom tools |
| **Database Integration** | SQL queries via LLM |

---

## 🛠️ Technical Requirements

```python
openai, langchain, sqlite3
```

---

## 🔗 Related Modules

- **Prerequisites**: [LLMs Intro](../llms_intro/)
- **Next**: [Agentic AI](../agentic_ai/)

---

*Module Difficulty: Intermediate*  
*Estimated Time: 4-5 hours total*
