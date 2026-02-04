# Tokenization

## 📚 Overview

This module covers **tokenization fundamentals** for NLP and LLMs. Learn how text is converted to tokens for model processing.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Understand different **tokenization strategies**
- ✅ Apply **BPE, WordPiece, and SentencePiece**
- ✅ Analyze **vocabulary size** trade-offs
- ✅ Debug tokenization issues

---

## 📂 Module Structure

```
tokenization/
├── README.md (this file)
├── tokenization_tutorial.ipynb
└── the-verdict.txt (Sample text)
```

---

## 🔄 Learning Path

### **Complete Tutorial** (2 hours)

Work through `tokenization_tutorial.ipynb`:

1. **Tokenization Basics**: Words, subwords, characters
2. **BPE**: Byte-Pair Encoding
3. **WordPiece**: BERT's tokenizer
4. **SentencePiece**: Language-agnostic
5. **Practical Considerations**: Context length, costs

---

## 🔍 Topics Covered

| Method | Used By | Description |
|--------|---------|-------------|
| **BPE** | GPT | Merge frequent pairs |
| **WordPiece** | BERT | Maximize likelihood |
| **SentencePiece** | T5, LLaMA | Unigram model |

---

## 🛠️ Technical Requirements

```python
tiktoken, transformers, sentencepiece
```

---

## 🔗 Related Modules

- **Related**: [NLP](../natural_language_processing/), [HuggingFace](../intro_huggingface/)

---

*Module Difficulty: Beginner to Intermediate*  
*Estimated Time: 2 hours total*
