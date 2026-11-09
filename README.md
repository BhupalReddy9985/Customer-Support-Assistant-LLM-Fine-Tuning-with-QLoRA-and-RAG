# 🤖 Customer Support Assistant Using LLM Fine-Tuning with QLoRA and RAG

An AI-powered customer support assistant that uses Large Language Models (LLMs), QLoRA fine-tuning, and Retrieval-Augmented Generation (RAG) to provide relevant, helpful, and knowledge-grounded responses to customer queries.

---

## 🚀 Overview

This project automates customer support by combining LLM fine-tuning with Retrieval-Augmented Generation (RAG).

The **Qwen3-4B** model is fine-tuned using **QLoRA (Quantized Low-Rank Adaptation)** on customer support conversations to improve response quality.

A RAG pipeline retrieves relevant information from a knowledge base of customer support policies and FAQs using semantic search with FAISS.

The system also includes an intent classifier, response safeguards, and an evaluation pipeline that compares the base and fine-tuned models.

---

## 🧭 Architecture

```
Bitext dataset ─► clean & dedupe ─► grouped train / val / test split
                                            │
                                            ▼
         Qwen3-4B (4-bit NF4) + LoRA adapters ─► SFT training ─► saved adapter
                                                                      │
Knowledge base ─► MiniLM embeddings ─► FAISS index                    │
                          │                                           │
   user question ─► intent classifier ─► top-k retrieval ─► prompt ─► base / fine-tuned answer
                                                                      │
                                                          rule-based safeguards ─► final reply
```

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

## 🗂️ Dataset

- **Source:** [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (26,872 rows with `instruction`, `response`, `category`, and `intent`)
- **Cleaning:** normalized whitespace, removed blank and duplicate question-response pairs, handled template placeholders such as `{{Order Number}}`
- **After cleaning:** 13,831 rows
- **Split:** grouped by normalized question text so the same question never appears in more than one split

| Split | Rows |
|-------|------|
| Train | 11,062 |
| Validation | 1,380 |
| Test | 1,389 |

> Check the dataset card for license terms before reusing the data.

---

## ⚙️ Fine-Tuning Configuration

| Setting | Value |
|---------|-------|
| Base model | Qwen/Qwen3-4B |
| Quantization | 4-bit NF4, fp16 compute |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable parameters | 16,515,072 of 4,038,983,168 (0.41%) |
| Max sequence length | 512 |
| Training subset | 1,500 examples (200 validation) |
| Epochs | 1 |
| Batch size / gradient accumulation | 1 / 4 |
| Learning rate | 2e-4 |
| Final training loss | 0.503 |
| Training time | about 25 minutes on a Tesla T4 |

---

## 📚 RAG Component

- **Knowledge base:** 72 records describing a fictional sample store (refund, cancellation, delivery, and similar policies)
- **Embeddings:** `all-MiniLM-L6-v2` (384 dimensions), stored in a FAISS index
- **Retrieval:** the top-3 documents are inserted into the prompt as context
- **Example:** the query "How many days do I have to request a refund?" retrieves the refund-policy record first (similarity about 0.78)

---

## 📊 Model Evaluation Results

The base and fine-tuned models were evaluated using 12 customer support questions.

| Evaluation Metric | Base Model | Fine-Tuned Model |
|-------------------|------------|------------------|
| Correctness       | 24/24      | 23/24            |
| Groundedness      | 24/24      | 18/24            |
| Helpfulness       | 19/24      | 23/24            |
| Overall Score     | 93.06%     | 88.89%           |
| Safety            | 12/12 Pass | 12/12 Pass       |

**Key Finding:** Fine-tuning improved response helpfulness, while the base model achieved better overall correctness and groundedness scores in this small evaluation.

> ⚠️ This is a manual evaluation on only 12 questions. It shows the evaluation method and early trends, not general model accuracy.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `Customer_Support_Assistant_LLM_Fine_Tuning_with_QLoRA_and_RAG_.ipynb` | Main Colab notebook |
| `combined_retrieval_knowledge_base.csv` | Knowledge base used for retrieval |
| `sample_customer_faq.csv` | Sample FAQ data |
| `sample_ecommerce_knowledge_base.csv` | Sample e-commerce knowledge base |
| `base_vs_finetuned_rag_comparison.csv` | Base vs fine-tuned answer comparison |
| `post_safeguard_manual_evaluation.csv` | Manual evaluation after adding safeguards |

---

## ▶️ How to Run

1. Open the notebook in **Google Colab** and choose **Runtime → Change runtime type → T4 GPU**.
2. Install dependencies:
   ```bash
   pip install transformers peft trl bitsandbytes datasets accelerate scikit-learn sentence-transformers faiss-cpu
   ```
3. Some cells load files from Google Drive (such as the saved LoRA adapter and knowledge base). Update the `/content/drive/MyDrive/...` paths to your own locations, or upload the CSV files from this repository.
4. Run the notebook from top to bottom.

---

## ⚠️ Limitations

- Fine-tuning used a 1,500-example subset for one epoch to fit a free GPU.
- The evaluation set has only 12 questions, scored manually.
- The knowledge base describes a fictional store, not a real company.
- The fine-tuned model scored lower than the base model on groundedness, so retrieval prompts and training data need further work.
- The notebook may depend on earlier session state and Drive files, so it may not run cleanly with a single Run All.

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

---
