# Superstore Sales RAG System

This project is a Retrieval-Augmented Generation (RAG) system built to analyze a real-world sales dataset using a local LLM (Llama 3.2 3B), ChromaDB, and sentence-transformers.

## Setup Instructions

### Prerequisites
- **Python 3.13**
- **Ollama** installed on your system (for running the local LLM)

### 1. Environment Setup
Create a virtual environment and install the required dependencies:

```bash
# Create the virtual environment
python3.13 -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. LLM Setup
Make sure Ollama is installed and running in the background. If not installed, on Linux you can install it using:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

You can check https://ollama.com/download/linux for details. Then pull the required Llama 3.2 3B model:

```bash
ollama pull llama3.2:3b
```
