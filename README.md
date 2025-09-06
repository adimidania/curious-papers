# Curious Papers: Text Processing & AI Agent System

This repository is a personal project exploring natural language processing, information extraction, and building a small AI agent. The goal is to experiment with processing academic papers and creating tools that summarize and extract insights from them.

## 📋 Project Overview

The project consists of three main components:

1. **Data Preparation & Exploration** – Processing and analyzing ArXiv ML papers
2. **Information Extraction & Summarization** – Extracting entities and summarizing text
3. **Agentic System** – Building SciDigest AI, a research assistant prototype

## 🗂️ Repository Structure

```
curious-papers/
├── 1- Data Preparation & Exploration.ipynb    # Data preprocessing and EDA
├── 2- Information Extraction & Summarization.ipynb    # NLP tasks implementation
├── 3- Agentic System Design.ipynb            # AI agent development
├── data/
│   └── sample_df.csv                         # Sample dataset (1,000 papers)
└── README.md                                 # Project documentation
```

## 📊 Part 1: Data Preparation & Exploration

### Dataset

* **Source**: [ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) from HuggingFace
* **Original Size**: \~100,000 machine learning papers
* **Sample Size**: 1,000 papers (randomly sampled)
* **Fields**: Title, Abstract

### Data Preprocessing Pipeline

1. **Text Cleaning**:

   * Lowercasing
   * Tokenization using NLTK
   * Stopword removal
   * Punctuation and symbol filtering
   * Lemmatization using spaCy

2. **Processing Function**:

   ```python
   def preprocess_text(doc):
       # Lowercase, tokenize, remove stopwords, lemmatize
       # Returns clean, processed text
   ```

### Exploratory Data Analysis

* Document length distributions for raw and processed abstracts
* Top 20 frequent words (excluding stopwords)
* Named Entity Recognition (NER) on sample abstracts
* Visualizations: Histograms and bar plots with matplotlib and seaborn

### Key Insights

* Most abstracts have 100–200 words
* Common ML terms dominate the vocabulary: "model", "algorithm", "network", "learning"
* Entity extraction highlights CARDINAL numbers, organizations, and dates

## 🔍 Part 2: Information Extraction & Summarization

### Entity Extraction Methods

1. **Rule-Based Extraction (Regex)**

   * Dates, numbers/metrics, email addresses
2. **Named Entity Recognition (spaCy)**

   * Entity types: PERSON, ORG, GPE, DATE, CARDINAL, ORDINAL
   * Batch processing for efficiency

### Text Summarization

* **Model**: `facebook/bart-large-cnn` (Transformers pipeline)
* **Input Handling**: Truncated to 1024 tokens
* **Result**: Concise summaries reducing text length by \~70–80% while keeping key concepts

## 🤖 Part 3: SciDigest AI – Research Assistant Prototype

### Concept

A small AI agent designed to help process and summarize research papers quickly.

### Core Capabilities

* **Entity Extraction**: Key concepts, authors, organizations, datasets, methods, and metrics
* **Summarization**: Concise, coherent summaries for rapid understanding

### Technical Implementation

* **LLM Backend**: Google Gemini 1.5 Flash
* **Function Calling**: Integrated spaCy NER and BART summarization
* **System Prompt**: Research-focused, structured instructions

```python
functions = {
    "spacy_ner_extraction": spacy_ner_extraction,
    "abstractive_summarization": abstractive_summarization
}
```

### Demonstration

* Accurate entity identification
* Coherent summarization
* Structured, readable results

## 🛠️ Technical Stack

* **Data Processing**: pandas, numpy
* **NLP**: spaCy, NLTK, transformers
* **Visualization**: matplotlib, seaborn
* **ML/AI**: HuggingFace Transformers, Google GenerativeAI
* **Development**: Jupyter Notebooks, Google Colab

### Models

* **spaCy**: `en_core_web_sm` for NER and lemmatization
* **BART**: `facebook/bart-large-cnn` for summarization
* **Gemini**: `gemini-1.5-flash` for AI agent

## 🚀 Usage Instructions

1. **Setup**:

```bash
pip install datasets pandas nltk spacy transformers google-generativeai
python -m spacy download en_core_web_sm
```

2. **Run Notebooks**: Execute in order (1 → 2 → 3)
3. **Data Access**: `data/sample_df.csv`

## 🔗 References

* [ML-ArXiv-Papers Dataset](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)
* [BART Model](https://huggingface.co/facebook/bart-large-cnn)
* [spaCy Documentation](https://spacy.io/)
* [Google Gemini API](https://ai.google.dev/)

Do you want me to do that?
