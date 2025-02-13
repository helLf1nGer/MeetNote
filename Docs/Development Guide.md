# MeetNote Development Guide

## Table of Contents

- [MeetNote Development Guide](#meetnote-development-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
  - [Project Structure](#project-structure)

## Introduction

This guide is designed to help developers contribute to the MeetNote project, particularly focusing on the dev-combiner-testing branch. This branch is dedicated to improving and expanding our combiner methods, which are crucial for merging transcription and diarization results.

## Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/helLf1nGer/meetnote.git
   cd meetnote
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install groq sentence-transformers
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add:
   ```bash
   HUGGING_FACE_AUTH_TOKEN=your_huggingface_token
   GROQ_API_KEY=your_groq_api_key
   ```

5. Install additional development tools:
   ```bash
   pip install pytest black isort mypy
   ```

## Project Structure

```
meetnote/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── adaptive_combiner.py
│   │   ├── adaptive_rule_combiner.py
│   │   ├── combiner_testing.py
│   │   ├── config_manager.py
│   │   ├── data_manager.py
│   │   ├── groq_api_helper.py
│   │   ├── groq_llm_combiner.py
│   │   ├── local_llama_tiny_combiner.py
│   │   ├── output_generator.py
│   │   ├── rate_limiter.py
│   │   ├── result_combiner.py
│   │   ├── semantic_combiner.py
│   │   ├── semantic_combiner_adaptive.py
│   │   ├── semantic_combiner_enhanced.py
│   │   ├── semantic_flow_combiner.py
│   │   ├── simple_combiner.py
│   │   ├── two_stage_llm_combiner.py
│   │   └── weighted_combiner.py
│   ├── main.py
│   └── dev_main.py
├── tests/
├── Docs/
├── Config/
├── .env
├── requirements.txt
└── README.md
```

