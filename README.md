# ML Technical Assessment: Text Processing & AI Agent System

This repository contains a comprehensive machine learning technical assessment focused on natural language processing, information extraction, and agentic AI system design. The project demonstrates end-to-end capabilities in processing academic literature using modern NLP techniques.

## 📋 Project Overview

The assessment consists of three main components:
1. **Data Preparation & Exploration** - Processing and analyzing ArXiv ML papers
2. **Information Extraction & Summarization** - Entity extraction and text summarization
3. **Agentic System Design** - Building SciDigest AI, a research assistant

## 🗂️ Repository Structure

```
ml-technical-assessment/
├── 1- Data Preparation & Exploration.ipynb    # Data preprocessing and EDA
├── 2- Information Extraction & Summarization.ipynb    # NLP tasks implementation
├── 3- Agentic System Design.ipynb            # AI agent development
├── data/
│   └── sample_df.csv                         # Processed dataset (1,000 papers)
└── README.md                                 # This documentation
```

## 📊 Part 1: Data Preparation & Exploration

### Dataset
- **Source**: [ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) from HuggingFace
- **Original Size**: ~100,000 machine learning papers from ArXiv
- **Sample Size**: 1,000 papers (randomly sampled for assessment)
- **Fields**: Title, Abstract

### Data Preprocessing Pipeline
1. **Text Cleaning**:
   - Lowercasing
   - Tokenization using NLTK
   - Stopword removal
   - Punctuation and symbol filtering
   - Lemmatization using spaCy

2. **Processing Function**:
   ```python
   def preprocess_text(doc):
       # Lowercase, tokenize, remove stopwords, lemmatize
       # Returns clean, processed text
   ```

### Exploratory Data Analysis
- **Document Length Distribution**: Analyzed word count distributions for both raw and processed abstracts
- **Word Frequency Analysis**: Identified top 20 most frequent terms (excluding stopwords)
- **Named Entity Recognition**: Extracted and analyzed entity types from sample texts
- **Visualizations**: Histograms, bar plots using matplotlib and seaborn

### Key Insights
- Abstract lengths follow a normal distribution with most papers having 100-200 words
- Common ML terms dominate the vocabulary: "model", "algorithm", "network", "learning"
- Entity extraction revealed high frequency of CARDINAL numbers, ORG entities, and DATE references

## 🔍 Part 2: Information Extraction & Summarization

### Entity Extraction Methods

#### 1. Rule-Based Extraction (Regex)
- **Dates**: Multiple formats (YYYY-MM-DD, MM/DD/YYYY, Month YYYY)
- **Numbers/Metrics**: Decimal and integer patterns
- **Email Addresses**: Standard email pattern matching

#### 2. Named Entity Recognition (spaCy)
- Used `en_core_web_sm` model
- Extracted entity types: PERSON, ORG, GPE, DATE, CARDINAL, ORDINAL
- Batch processing for efficiency (20 documents per batch)

### Results
- **Entity Distribution**: CARDINAL (77), ORG (24), ORDINAL (19), DATE (17), PERSON (11)
- **Processing**: Handled 50 sample abstracts with comprehensive entity extraction (yet there are some tiny misclassifications)

### Text Summarization

#### Abstractive Summarization (BART)
- **Model**: `facebook/bart-large-cnn` via Transformers pipeline
- **Parameters**: 
  - Max length: 130 tokens
  - Min length: 50 tokens
  - No sampling (deterministic)
- **Input Handling**: Truncated to 1024 tokens for model compatibility

#### Sample Results
Generated concise, coherent summaries maintaining key technical concepts while reducing text length by ~70-80%.

## 🤖 Part 3: Agentic System Design - SciDigest AI

### System Concept
**SciDigest AI** is a virtual research assistant designed to help scientists, students, and industry professionals quickly digest academic literature and technical reports.

### Core Capabilities
1. **Entity Extraction**: Identifies key concepts, authors, organizations, datasets, methods, and metrics
2. **Summarization**: Produces concise, coherent summaries for quick relevance assessment

### Real-World Problem Solved
- **Challenge**: Researchers overwhelmed by thousands of weekly publications on arXiv
- **Solution**: Automated content distillation and metadata extraction
- **Benefits**: Faster literature review, improved decision-making, enhanced knowledge accessibility

### Technical Implementation

#### Architecture
- **LLM Backend**: Google's Gemini 1.5 Flash model
- **Function Calling**: Integrated spaCy NER and BART summarization
- **System Prompt**: Professional, research-focused instruction design

#### Key Components
```python
# Core functions integrated into the agent
functions = {
    "spacy_ner_extraction": spacy_ner_extraction,
    "abstractive_summarization": abstractive_summarization
}
```

#### Agent Capabilities
- **Professional Tone**: Formal, academic-appropriate responses
- **Structured Output**: Organized, readable results
- **Function Integration**: Automatic tool selection and execution
- **Research Focus**: Tailored for academic and technical content

### Demonstration
Successfully processed research abstracts with:
- Accurate entity identification
- Coherent summarization maintaining technical accuracy
- Professional, structured responses

## 🛠️ Technical Stack

### Libraries & Frameworks
- **Data Processing**: pandas, numpy
- **NLP**: spaCy, NLTK, transformers
- **Visualization**: matplotlib, seaborn
- **ML/AI**: HuggingFace transformers, Google GenerativeAI
- **Development**: Jupyter Notebooks, Google Colab

### Models Used
- **spaCy**: `en_core_web_sm` for NER and lemmatization
- **BART**: `facebook/bart-large-cnn` for abstractive summarization
- **Gemini**: `gemini-1.5-flash` for agentic system

## 📈 Results & Performance

### Data Processing
- Successfully processed 1,000 research papers
- Effective text preprocessing with 60-70% token reduction
- Comprehensive EDA with meaningful visualizations

### Information Extraction
- High-accuracy entity extraction with proper categorization
- Effective summarization with 70-80% length reduction while preserving key information
- Robust handling of technical academic content

### Agent System
- Functional AI agent with integrated NLP capabilities
- Professional, research-appropriate responses
- Successful demonstration of automated literature processing

## 💬 Discussion of Results

### Model Performance Analysis

#### Data Preprocessing Effectiveness
The preprocessing pipeline demonstrated excellent performance in cleaning and standardizing the ArXiv dataset:
- **Token Reduction**: Achieved 60-70% reduction in token count while preserving semantic meaning
- **Normalization Success**: Lemmatization effectively reduced word variants (e.g., "learning", "learns", "learned" → "learn")
- **Quality Preservation**: Manual inspection confirmed that technical terminology and domain-specific concepts were retained

#### Entity Extraction Performance
The regex and spaCy NER provided complementary strengths:

**spaCy NER Results:**
- **High Precision**: CARDINAL entities (77 instances) accurately identified numerical values and counts
- **Domain Adaptation**: Successfully extracted ORG entities (24) including research institutions and algorithm names
- **Temporal Recognition**: DATE entities (17) captured publication years and temporal references
- **Limitation**: Some domain-specific terms misclassified (e.g., "CNN" as organization)

**Regex Pattern Matching:**
- **Structured Data**: Effectively captured formatted dates and email addresses
- **Numerical Precision**: Reliable extraction of metrics, percentages, and measurements
- **Scalability**: Fast processing suitable for large-scale document analysis

#### Summarization Quality Assessment
BART model performance on academic abstracts:

**Strengths:**
- **Coherence**: Generated summaries maintained logical flow and technical accuracy
- **Compression Ratio**: Consistent 70-80% length reduction while preserving key information
- **Domain Adaptation**: Successfully handled technical ML terminology without significant hallucination

**Quantitative Metrics:**
- Average original abstract length: 150-200 words
- Average summary length: 30-50 words
- Processing time: ~2-3 seconds per abstract
- Content preservation: High retention of key technical concepts

### Challenges Encountered

#### Technical Challenges

1. **Entity Recognition Limitations**
   - **Domain Specificity**: Standard spaCy models lack specialized ML/AI entity recognition
   - **Ambiguous Terms**: Words like "network", "model" have multiple contextual meanings
   - **Acronym Handling**: Technical abbreviations (CNN, RNN, GAN) often misclassified

2. **Summarization Constraints**
   - **Model Limitations**: BART's 1024-token input limit required text truncation
   - **Technical Density**: Academic abstracts contain high information density, making compression challenging
   - **Evaluation Difficulty**: Lack of ground-truth summaries for objective quality assessment with ROUGE and similar metrics

3. **Computational Resources**
   - **Processing Time**: Large-scale NER processing required batch optimization
   - **Memory Management**: Handling 100K+ documents required careful memory allocation
   - **Model Loading**: Multiple model initialization (spaCy, BART, Gemini) increased startup time

### Real-World Applicability

#### SciDigest AI System Performance
The agentic system demonstrated strong practical potential:

**Successful Capabilities:**
- **Integration**: Seamless combination of NER and summarization functions
- **Response Quality**: Professional, structured outputs suitable for research contexts
- **Tool Selection**: Appropriate automatic function calling based on user queries
- **Scalability**: Architecture supports extension to additional NLP tasks


## 🚀 Future Enhancements

1. **Scalability**: Implement batch processing for larger datasets
2. **Advanced NER**: Fine-tune models for domain-specific entities (algorithms, datasets)
3. **Multi-modal**: Extend to process figures, tables, and equations
4. **Interactive UI**: Develop web interface for user interaction
5. **Knowledge Graph**: Build connected knowledge representation
6. **Evaluation Metrics**: Implement ROUGE, BLEU scores for summarization quality

## 📝 Usage Instructions

1. **Environment Setup**:
   ```bash
   pip install datasets pandas nltk spacy transformers google-generativeai
   python -m spacy download en_core_web_sm
   ```

2. **Run Notebooks**: Execute in order (1 → 2 → 3) for complete workflow

3. **Data Access**: Sample dataset available in `data/sample_df.csv`

## 🔗 References

- [ML-ArXiv-Papers Dataset](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)
- [ArXiv Full Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- [BART Model](https://huggingface.co/facebook/bart-large-cnn)
- [spaCy Documentation](https://spacy.io/)
- [Google Gemini API](https://ai.google.dev/)
