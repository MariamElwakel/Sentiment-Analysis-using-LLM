# Multilingual Sentiment Analysis (Arabic + English)

LLM-based sentiment analysis system that classifies Arabic and English reviews with optional reasoning explanations.

The project includes:
- Batch processing pipeline
- Structured JSON outputs
- Interactive Streamlit chat interface

---

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

---

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

### 3. Add Your OpenAI API Key

Create a `.env` file in the project root directory and add:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Running the Project

### Run Batch Processing

```bash
python main.py
```

This will:
- Read the input review files  
- Classify sentiment  
- Generate JSON output files inside the `outputs/` folder  

---

### Run the Streamlit App

```bash
streamlit run app.py
```

After running the command, open the local URL shown in the terminal to use the sentiment chat interface.

---
