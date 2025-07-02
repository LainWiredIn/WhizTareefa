# WhizTareefa - Smart PDF Query & Response System with Web Search

## Overview

WhizTareefa is your personal knowledge assistant, designed to handle complex queries by transforming static PDFs into dynamic, conversational experiences. With WhizTareefa, you can upload documents, ask questions, and receive smart, contextually accurate responses powered by retrieval-augmented generation (RAG) and Google Generative AI. For added flair, the system converts text responses into speech, and even performs web searches to deliver comprehensive and up-to-date answers.

## Features
- **PDF Upload & Analysis**: Upload PDFs, and WhizTareefa extracts the text for answering your queries.
- **Intelligent RAG System**: Retrieves relevant information from the document and formulates an AI-driven response.
- **Text-to-Speech**: Listen to responses with the built-in text-to-speech feature using API from sarvam.ai.
- **Web Search Enhancement**: Integrates web results into the answers for broader context.
- **Engaging AI Templates**: Responses are crafted using refined, engaging templates designed to guide clear communication.

## Core Technologies
- FastAPI: The web framework used to build the API.
- Sentence Transformers: Used to create embeddings for text chunks from the PDF.
- Annoy (Approximate Nearest Neighbors): For fast and memory-efficient nearest neighbor search.
- Google Gemini: Used for generating natural language responses.
- PyPDF2: PDF text extraction.
- sarvam.ai/text-to-speech: API used for converting text into speech.
- SerpAPI: Used for web searching capabilities to fetch additional context.
- bs4: Beautiful Soup for scraping webpages for additional information

## API Endpoints

1. ```/upload-pdf/```
- POST method that allows users to upload a PDF file. The text from the PDF is extracted and stored for further use in the RAG system.
- Input is a valid .pdf file.
- Output is a JSON response with status and success message.
2. ```/ask/```
- POST method that handles user queries. It uses the RAG system to retrieve relevant chunks from the PDF (if it has been uploaded) and generates a context-aware answer. The response is also converted to audio.
- Input is a JSON object with the user's query.
- Output is a response with the answer and an audio of the response playable via a player.
3. ```/search-web/```
- POST method that accepts a query and retrieves detailed web content using SerpAPI.
- Integrates the results with the RAG system and generates a response.
- Input is a JSON object containing the query.
- Output is the generated response based on web search results, with an audio player to play out the response.
4. ```/```
- GET method for displaying a simple HTML template.

## Setup and Execution

1. Installing dependencies
- We suggest creating a virtual environment with Python3.10 or above, installing the required dependencies via the following:
```
pip install fastapi uvicorn PyPDF2 annoy sentence-transformers playsound google-generativeai serpapi requests bs4
```
- An error you mite see during the installation of playsound is probably related to Python 3.13, which is currently in development and not yet fully supported by many Python packages (including playsound). Use Python 3.10 or 3.11 instead of 3.13 for your virtual environment.
2. Running the Application
- First clone the repository:
```
git clone https://github.com/LainWiredIn/WhizTareefa.git
```
- Once inside the repository, activate your environment and use the following command:
```
uvicorn fastAPI_gemini_backend:app --reload
```
- The application should now be accessible at ```http://127.0.0.1:8000```. Change the port number in the code if you are already using 8000 for some other application.
## File Structure
```
.
├── fastAPI_gemini_backend.py   # The main application file
├── templates/                  # Directory for HTML templates
│   └── index.html              # Homepage template
├── static/                     # Directory for static files (CSS, JS, audio files, etc.)
│   └── audios/                 # Directory for generated audio files
└── uploaded_files/             # Directory for uploaded PDFs

```


