# Automated Mortgage Loan System

A comprehensive AI-powered system for automating and analyzing mortgage loan applications using Retrieval-Augmented Generation (RAG), embeddings, and document processing.

## Project Overview

This project implements an intelligent mortgage loan processing system that:
- **Extracts** text from PDFs, DOCX files, and images (with OCR support)
- **Chunks** documents into manageable segments for efficient retrieval
- **Embeds** text using sentence transformers for semantic search
- **Stores** embeddings in a vector database (Chroma)
- **Retrieves** relevant documents based on natural language queries
- **Generates** accurate answers using FLAN-T5 language model

## Project Structure

```
Automated-Mortgage-Loan-System/
├── Embedding/
│   ├── vector_embedding.ipynb                  # Main RAG pipeline notebook
│   ├── Documents/                              # Input documents folder
│   └── Output/                                 # Generated embeddings and outputs
├── FineTuning/                                 # (Coming Soon) Model fine-tuning module
├── MCP/                                        # (Coming Soon) Model Context Protocol integration
├── .venv/                                      # Python virtual environment
├── requirements.txt                            # Project dependencies
└── README.md                                   # This file
```

## Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)
- Tesseract OCR (for PDF/image processing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Automated-Mortgage-Loan-System
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate      # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR:**
   - **Windows:** Download installer from [github.com/UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux:** `sudo apt-get install tesseract-ocr poppler-utils`
   - **Mac:** `brew install tesseract poppler`

### Usage

#### Option 1: Jupyter Notebook (Recommended for exploration)
```bash
jupyter notebook Embedding/vector_embedding.ipynb
```

#### Option 2: Python Script
```bash
python Embedding/vector_embedding.ipynb
```

**Note:** The main notebook (`vector_embedding.ipynb`) is designed to work in both local Jupyter environments and Google Colab.

#### Option 3: Google Colab (Cloud-based, no local setup needed)
Open the `.ipynb` file directly in [Google Colab](https://colab.research.google.com/)

## Configuration

### Chunk Size
Modify chunk parameters in the chunking function:
```python
def chunk_text(text, chunk_size=150, overlap=25):
    # chunk_size: words per chunk (default: 150)
    # overlap: words shared between consecutive chunks (default: 25)
```

**Guidelines:**
- **150-300 words:** Specific queries, detailed retrieval
- **300-500 words:** Balanced accuracy and coverage
- **500+ words:** Broad context, fewer chunks

### Embedding Model
Change the embedding model in cell 3:
```python
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"   # High accuracy
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Faster alternative
```

### Generation Model
Adjust the language model for answer synthesis:
```python
GEN_MODEL_NAME = "google/flan-t5-base"  # Use 'flan-t5-large' for GPU systems
```

## Key Features

### Document Processing
- **PDF Extraction:** Text and OCR-based (handles flattened PDFs)
- **DOCX Support:** Paragraph extraction
- **Image Support:** OCR with preprocessing (contrast adjustment, denoising)
- **Auto-chunking:** Configurable overlap for context preservation

### Vector Storage
- **Chroma DB:** Persistent local vector database
- **Semantic Search:** Find documents by meaning, not keywords
- **Top-K Retrieval:** Configurable retrieval count

### RAG Pipeline
- **Context Assembly:** Combines retrieved chunks with queries
- **Prompt Engineering:** System prompts for reliable answers
- **Provenance Tracking:** Source attribution for generated answers
- **Token Budgeting:** Efficient context truncation

### Quality Assurance
- **QA Model:** Optional exact-span extraction (SQuAD v2)
- **Distance Metrics:** Relevance scoring for retrieved documents
- **Metadata Tracking:** Document source and chunk indexing

## Example Usage

```python
from rag_pipeline import rag_answer

# Ask a question
question = "What are the eligibility criteria for a mortgage loan?"
result = rag_answer(question, top_k=3)

print("Answer:", result["answer"])
print("Sources:", result["provenance"])
```

## Upcoming Modules

### FineTuning (Coming Soon)
- Fine-tune embedding models on mortgage-specific data
- Optimize answer generation with domain-specific examples
- Evaluate and benchmark RAG system performance

### MCP (Coming Soon)
- Model Context Protocol integration
- Advanced tool calling capabilities
- Enhanced multi-step reasoning for complex queries

## Performance Tips

1. **GPU Acceleration:** Use GPU for faster embeddings and generation
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Batch Processing:** Adjust batch size in embedding cell for memory efficiency

3. **Caching:** Reuse computed embeddings when possible

4. **Model Size:** Use smaller models for faster inference, larger for accuracy

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tesseract not found | Install Tesseract and set correct path in cell 3 |
| Out of memory | Reduce batch size or use smaller embedding model |
| Slow embeddings | Use `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2` |
| Poor retrieval | Adjust chunk size or change embedding model |

## Author

**BlessedForever04**

## 🤝 Contributing

Contributions are welcome! Please follow standard Git workflow:
1. Create a feature branch
2. Make changes
3. Commit with clear messages
4. Push and create a pull request

## Contact

For questions or issues, please open a GitHub issue.

---

