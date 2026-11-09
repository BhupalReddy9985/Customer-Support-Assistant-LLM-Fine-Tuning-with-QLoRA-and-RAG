# 🤖 Customer Support Assistant Using LLM Fine-Tuning with QLoRA and RAG

An AI-powered customer support assistant that uses Large Language Models (LLMs), QLoRA fine-tuning, and Retrieval-Augmented Generation (RAG) to provide relevant, helpful, and knowledge-grounded responses to customer queries.

---

## 🚀 Overview

This project is designed to automate customer support by combining LLM fine-tuning with Retrieval-Augmented Generation (RAG).

The **Qwen3-4B** model is fine-tuned using **QLoRA (Quantized Low-Rank Adaptation)** on customer support conversations to improve response quality.

A RAG pipeline retrieves relevant information from a knowledge base containing customer support policies and FAQs using semantic search with FAISS.

The system also includes an intent classifier, response safeguards, and an evaluation pipeline to compare the base and fine-tuned models.

---

## ✨ Features

- 🤖 AI-powered customer support question answering
- 🧠 Qwen3-4B fine-tuning using QLoRA
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic search using FAISS vector indexing
- 🔗 Sentence Transformer embeddings for document retrieval
- 📊 Customer query intent classification using Logistic Regression
- 🛡️ Rule-based safeguards for unsupported customer support actions
- ⚖️ Base model vs fine-tuned model performance comparison
- 📈 Response evaluation using correctness, groundedness, helpfulness, and safety

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Base LLM:** Qwen3-4B
- **Fine-Tuning:** QLoRA, PEFT, Hugging Face Transformers, TRL
- **Quantization:** BitsAndBytes (4-bit NF4)
- **RAG:** Retrieval-Augmented Generation
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search:** FAISS
- **Machine Learning:** Scikit-learn, Logistic Regression, TF-IDF
- **Data Processing:** Pandas, NumPy
- **Development Environment:** Google Colab, Google Drive
- **Hardware:** NVIDIA Tesla T4 GPU

---

## 📊 Model Evaluation Results

The base and fine-tuned models were evaluated using 12 customer support questions.

| Evaluation Metric | Base Model | Fine-Tuned Model |
|---|---|---|
| Correctness | 24/24 | 23/24 |
| Groundedness | 24/24 | 18/24 |
| Helpfulness | 19/24 | 23/24 |
| Overall Score | 93.06% | 88.89% |
| Safety | 12/12 Pass | 12/12 Pass |

**Key Finding:** Fine-tuning improved response helpfulness, while the base model achieved better overall correctness and groundedness scores in this small evaluation.

---

## 🎯 Project Objective

To develop an intelligent customer support assistant that combines domain-specific fine-tuning with external knowledge retrieval to generate useful responses while reducing unsupported claims and improving response reliability.

---

## 🔮 Future Enhancements

- 🌐 Develop a web-based customer support chatbot
- 📄 Support PDF and document-based knowledge retrieval
- 🔎 Implement hybrid search and advanced reranking
- 🧠 Improve fine-tuning using a larger, higher-quality dataset
- 📊 Expand evaluation with more customer support scenarios
- 🚀 Deploy the assistant using FastAPI and cloud infrastructure

---

## 📌 Conclusion

This project demonstrates how LLM fine-tuning, RAG, vector search, and response safeguards can be combined to build a customer support assistant.

The experiments also highlight the importance of evaluating fine-tuned models for correctness, groundedness, helpfulness, and safety rather than relying on training loss alone.
