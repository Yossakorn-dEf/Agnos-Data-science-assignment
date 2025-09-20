#  Agnos Data Science Assignment

This repository contains solutions for the **Agnos Data Science Candidate Assignment**.  
It includes implementations for both **Task 1 (RAG Chatbot with Private Dataset)** and **Task 2 (AI Symptom Recommender)**.

---

##  Task 1: RAG Chatbot with Private Dataset +  Task 2: AI Symptom Recommender

### Task 1 Overview
A chatbot built using **Retrieval-Augmented Generation (RAG)** to answer questions based on the **Agnos Health Forum**.  
The pipeline is: scrape forum → clean & chunk text → embed with SentenceTransformer → FAISS index → query with LLM → generate response with citations.  

**Architecture**

flowchart LR
U[User Question] --> E[Embed Query]
subgraph KB[Knowledge Base]
H[HTML + META] --> P[Parser + Chunk] --> V[Embeddings]
V --> F[(FAISS Index)]
end
E --> R[Retrieve Top-K]
R --> C[Compose Context + Citations]
C --> L[LLM Generate Answer]
L --> A[Final Answer + Sources]


Embedding model: BAAI/bge-small-en-v1.5
Vector DB: FAISS (L2 search)
LLM: microsoft/Phi-3-mini-4k-instruct
Chat UI: Gradio

Tasks1/
├─ notebooks/
│  ├─ 01_scrape_forum.ipynb
│  ├─ 02_ingest_build_index.ipynb
│  ├─ 03_chat_gradio.ipynb
│  └─ 04_auto_scraper_update.ipynb
├─ code/
│  └─ ingest_build_index.py
├─ requirements.txt
└─ .gitignore

Example Q&A
  Question: เยื่อหุ้มหัวใจอักเสบมีอาการอย่างไร?
  Answer: ผู้ป่วยมักมีอาการแน่นหน้าอก เจ็บหน้าอก โดยเฉพาะเวลาหายใจเข้าลึก [1], [2]

### Task 2 Overview

A **symptom recommendation system** that suggests co-occurring symptoms given patient information (gender, age, initial symptoms).  
The model combines statistical association rules with similarity measures and Bayesian priors.

**Workflow**
1. Load dataset (ai_symptom_picker.csv)
2. Preprocess data
3. Compute co-occurrence
4. Association metrics: Support, Confidence, Lift
5. Similarity (Jaccard index, KNN-based)
6. Priors by gender and age (Bayesian)
7. Final scoring (weighted combination)
8. Evaluate with Precision@K, Recall@K, MAP@K, Coverage

Tasks2/
├─ data/
│  └─ ai_symptom_picker.csv
├─ notebooks/
│  └─ symptom_recommender.ipynb
├─ results/
│  ├─ evaluation_metrics.json
│  └─ top_symptoms.png
├─ requirements.txt
└─ README.md
Example Evaluation
{
  "Precision@K": 0.8054,
  "Recall@K": 0.8054,
  "MAP@K": 0.5791,
  "Coverage": 0.5281,
  "EvaluatedPairs": 185
}

###How to Run

Task 1 (RAG Chatbot)
cd Tasks1
pip install -r requirements.txt
jupyter notebook notebooks/03_chat_gradio.ipynb

Task 2 (Symptom Recommender)
cd Tasks2
pip install -r requirements.txt
jupyter notebook notebooks/symptom_recommender.ipynb

