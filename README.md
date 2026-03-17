# CogniX

### Constructive Learning with a Multi-Agent AI System for Open-Ended Question Support

------------------------------------------------------------------------

## Overview

**CogniX** is a multi-agent, Retrieval-Augmented Generation (RAG)
powered AI web application designed to help students improve answers to
open-ended questions through constructive, real-time feedback.

Unlike traditional AI grading tools that provide static scores or final
answers, CogniX focuses on **guiding the learning process**, encouraging
reflection, and fostering critical thinking.

This project was designed and implemented as an individual AI-driven
educational research system addressing key limitations in current AI
feedback models.
<br>
<p align="center">
  <a href="https://youtu.be/J7KggRkwXy8" target="_blank" rel="noopener noreferrer">
    <img 
      src="https://img.youtube.com/vi/J7KggRkwXy8/hqdefault.jpg" 
      alt="ElectroMart Demo Video" 
      width="640" 
      style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);"
    />
    <br><br>
    <strong style="font-size: 1.3em;">▶️ Watch Demo Video</strong>
  </a>
</p>

------------------------------------------------------------------------

# Problem Statement

Students often struggle with open-ended exam questions due to:

-   Stress and time pressure
-   Difficulty organizing thoughts
-   Lack of clarity about examiner expectations
-   Limited detailed feedback outside exams

Most AI systems:

-   Focus on grading rather than improvement
-   Provide static, generic feedback
-   Rely on a single LLM
-   Lack contextual awareness of lecture materials
-   Do not adapt to learner progress

------------------------------------------------------------------------

# Solution

CogniX introduces a **multi-agent AI architecture** that:

-   Retrieves contextual lecture material
-   Extracts questions automatically
-   Evaluates student answers
-   Generates constructive improvement guidance
-   Applies reflection over generated feedback
-   Encourages iterative learning

Instead of revealing correct answers directly, CogniX guides students
toward discovering improvements themselves --- aligning with
constructive learning principles.

------------------------------------------------------------------------

# System Architecture

Frontend (Next.js)\
⬇\
FastAPI Backend\
⬇\
Orchestrator Agent\
⬇\
Specialized AI Agents\
⬇\
ChromaDB (Vector Store) + GPT-4o-mini

------------------------------------------------------------------------

## Multi-Agent Architecture

### 1️⃣ Orchestrator Agent

-   Controls overall session flow
-   Manages retry logic
-   Maintains session memory
-   Routes between agents
-   Controls feedback iteration

### 2️⃣ Context Agent

-   Parses uploaded PDFs (lecture notes)
-   Chunks and embeds text
-   Stores embeddings in ChromaDB
-   Retrieves relevant context using cosine similarity

### 3️⃣ Question Agent

-   Extracts numbered questions from PDFs
-   Filters valid exam-style questions
-   Prepares structured question context

### 4️⃣ Evaluation Agent

-   Compares student answer against retrieved lecture material and
    reference answers
-   Identifies knowledge gaps, missing key points, and structural
    issues
-   Generates scoring alignment

### 5️⃣ Reflection Agent

-   Reviews Evaluation Agent output
-   Improves clarity and constructiveness
-   Removes overly direct answers
-   Enhances guidance quality

------------------------------------------------------------------------

# Agentic Workflow

1.  Student uploads lecture notes, question paper, and optional
    reference answers
2.  Context is embedded into ChromaDB
3.  Student selects a question
4.  Student submits an answer
5.  Evaluation Agent analyzes response
6.  Reflection Agent refines feedback
7.  Constructive guidance returned
8.  Student improves answer
9.  Process repeats until understanding improves

------------------------------------------------------------------------

# RAG Implementation

-   Embedding model: `intfloat/e5-base-v2`
-   Vector DB: ChromaDB
-   Similarity metric: Cosine similarity
-   LLM: GPT-4o-mini

Instead of hallucinating answers, the model retrieves semantic context
before evaluation, improving accuracy and constructiveness.

------------------------------------------------------------------------

# Experimental Results

| Metric                      | Single LLM | CogniX |
|-----------------------------|------------|--------|
| Answer Scoring Alignment    | 81%        | 91%    |
| Constructiveness            | 72%        | 90%    |
| Accuracy & Relevance        | 78%        | 90%    |

CogniX demonstrated measurable improvements across all feedback quality metrics.

------------------------------------------------------------------------

# Testing Strategy

-   **Unit Testing** - Individual agent validation
-   **Integration Testing** - Agent-to-agent data flow
-   **System Testing** - End-to-end workflow validation
-   **Regression Testing** - Ensured updates did not break feedback
    logic

------------------------------------------------------------------------

# Tech Stack

### AI & NLP

-   GPT-4o-mini
-   HuggingFace Embeddings (intfloat/e5-base-v2)
-   RAG Architecture
-   Prompt Engineering
-   Multi-Agent Orchestration

### Backend

-   Python
-   FastAPI
-   ChromaDB
-   PyPDFLoader

### Frontend

-   Next.js
-   React
-   Streaming chat interface

------------------------------------------------------------------------

# How to Run

### Backend

``` bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

``` bash
cd frontend
npm install
npm run dev
```

------------------------------------------------------------------------

# About This Implementation

This project demonstrates:

-   Multi-agent system design
-   RAG architecture implementation
-   AI evaluation pipeline engineering
-   Prompt optimization
-   Vector database integration
-   Frontend-backend orchestration
-   Research-driven AI experimentation
