import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import PyPDF2
import os
from serpapi import GoogleSearch

import requests
from bs4 import BeautifulSoup
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import base64
from playsound import playsound

app = FastAPI()

os.makedirs("static/audios", exist_ok=True)

# mounting static files (for future CSS, JS, files)
app.mount("/static", StaticFiles(directory="static"), name="static")

# set up templates (for index.html)
templates = Jinja2Templates(directory="templates")

embedder = SentenceTransformer('all-MiniLM-L6-v2')


import requests

def text_to_speech(text):
    url = "https://api.sarvam.ai/text-to-speech"
    if len(text)>450:
        text = text[:450]
    payload = {
        "inputs": [text],
        "target_language_code": "en-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.2,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }

    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": "b6f6afd0-ed46-46fd-a6fb-49b46c6fc1c4"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    if response.status_code == 200:
        audio_data = response.json().get('audios', [])[0]
        if audio_data:

            audio_bytes = base64.b64decode(audio_data)
            audio_file_path = f"static/audios/output.wav"

            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(audio_bytes)
            
            print("Audio saved as output.wav")
            return audio_file_path
        else:
            print("No audio data returned.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def chunk_text(text, chunk_size=100):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_annoy_index(chunks, embedder, n_trees=10):
    dim = embedder.get_sentence_embedding_dimension()
    
    annoy_index = AnnoyIndex(dim, 'angular')
    
    chunk_embeddings = []
    
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk)
        chunk_embeddings.append(embedding)
        annoy_index.add_item(i, embedding)
    
    annoy_index.build(n_trees)  # Higher n_trees gives better accuracy, but slower queries
    return annoy_index, chunk_embeddings

def retrieve_relevant_chunks(query, annoy_index, chunks, top_k=5):
    query_embedding = embedder.encode([query])[0]  # Get the query embedding
    nearest_indices = annoy_index.get_nns_by_vector(query_embedding, top_k)  # Find nearest neighbors
    
    return [chunks[i] for i in nearest_indices]

def configure_genai(api_key):
    genai.configure(api_key=api_key)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    generation_config = {
        "temperature": 0.7,  # Adjusted for more engaging conversation
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 512  # Shorter responses for ongoing chat
    }
    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model

model = configure_genai("AIzaSyBxzuL9gFINm3GtnA0SlqWwwtB1ZPVMJY0")

# 6. Define the RAG template
template = """
You are a well-regarded and intelligent teacher, who likes to not tell the students the answers directly but likes them to do some work. The student has asked the following question:

{}

They have also provided the following context from the textbook:

{}

If the student's query can be answered with the provided context, answer it. Always start your answer with the following token: [ANSWER] 

Else, tell them that the text is not related to the problem. Always start that response with [CONTEXT]
"""

def generate_response(model, prompt):
    try:
        response = model.generate_content(prompt)
        return response.text if response else "I'm sorry, I didn't understand that."
    except ResourceExhausted:
        print("Resource exhausted, retrying...")
        time.sleep(30)
        return generate_response(model, prompt)
    except Exception as e:
        print(f"Error occurred: {e}")
        return "I'm sorry, I encountered an error."

# 7. RAG System: Generate responses based on query and retrieved chunks
def generate_rag_response(query):
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, annoy_index, chunks)
    
    # Format the template with the query and retrieved chunks
    prompt = template.format(query, "\n\n".join(relevant_chunks))
    # print("these are the chunks I found: ", relevant_chunks)
    try:
        response = model.generate_content(prompt)
        return response.text if response else "I'm sorry, I didn't understand that."
    except ResourceExhausted:
        print("Resource exhausted, retrying...")
        time.sleep(30)
        return generate_response(model, prompt)
    except Exception as e:
        print(f"Error occurred: {e}")
        return "I'm sorry, I encountered an error."

def search_web(query):
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": "e815a79bab92dd5b328dfa8eadf0bc145d38377068926c587276be27e584b4e7"  # Insert your SerpAPI key here
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    x = 5
    return organic_results[:x]  # limit to top x results for brevity and for not depleting the free web search API

def extract_full_content_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = " ".join([para.get_text() for para in paragraphs[:5]])  # Limit to first 5 paragraphs
            return content
        else:
            return None
    except requests.RequestException as e:
        print(f"Failed to retrieve content from {url}: {e}")
        return None

def retrieve_full_web_content_and_links(query):
    web_results = search_web(query)
    detailed_results = []

    for result in web_results:
        content = extract_full_content_from_url(result['link'])
        if content:
            detailed_results.append({
                'title': result['title'],
                'link': result['link'],
                'content': content
            })

    return detailed_results

# queries = ["What did fantano think of the latest Voidz album, Like all before you? How does it compare to his view of their last record, Virtue?"]

template_0 = """
You are an intelligent teacher who students can freely talk to and ask things about. The student said the following to you:

{}

If the utterance is a greeting or a simple message asking you how you are doing, answer it acordingly. The response should always start with [GREETING] and always end with "What would you like to know about today?".

If the utterance is a question that needs to be answered, output [QUESTION]. Do not output anything else.

"""

template_2 = """
You are an intelligent teacher, who likes to go beyond the textbook when teaching your students. Here is the question the student has asked:

{}

{}

Referencing all the original articles, succintly answer the student's query using the web results. Feel free to include any additional non-conflicting information that could act as cool facts. 

Please only answer in sentences and paragraphs. There should be no bullet points. Write as if you were speaking in a continuous manner.
"""

template_3 = """
You are the best google searcher in the world. You have expertise in phrasing queries to Google Search so that it outputs the best possible search results. Your friend wants your help. He is looking for the most relevant search results for the following query: {}

Help him by making his query better. He will be using the modified query you provide him to search on several search engines. Please do not omit any information from his query.

ONLY OUTPUT THE QUERY. 
"""

def refine_query(query):
    refined_prompt = template_3.format(query)
    model = configure_genai("AIzaSyAKo6gfAKiKQ46c7NSnGQ-FUXloscyZ_wY")
    refined_response = generate_response(model, refined_prompt)
    return refined_response

##################################################################################

from pydantic import BaseModel

class QueryModel(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.filename.endswith('.pdf'):
        pdf_path = f"uploaded_files/{file.filename}"
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        global document_text
        document_text = extract_text_from_pdf(pdf_path)
        
        return {"status": "success", "message": "PDF uploaded and processed successfully."}
    else:
        return {"status": "error", "message": "Please upload a valid PDF file."}

@app.post("/ask/")
async def ask(query_model: QueryModel):
    # Extract text and create annoy index
    first_prompt = template_0.format(query_model.query)
    first_response = generate_response(model, first_prompt)
    if "[QUESTION]" in first_response:
        try:
            if document_text:
                global annoy_index, chunk_embeddings, chunks
                chunks = chunk_text(document_text)
                annoy_index, chunk_embeddings = create_annoy_index(chunks, embedder)
                refined_query = refine_query(query_model.query)
                response = generate_rag_response(refined_query)
                audio_url = text_to_speech(response)
                print("SARVAM COOKED:", audio_url)
                return {"response": response, "audio_url": audio_url}
        except NameError:
            response_without_rag = generate_response(model, query_model.query)
            audio_url = text_to_speech(response_without_rag)
            print("SARVAM COOKED:", audio_url)
            if audio_url:
                return {"response": response_without_rag, "audio_url": f"/static/audios/{os.path.basename(audio_url)}"}
            else:
                return {"response": response_without_rag, "audio_url": None}
    else:
        greeting = first_response.replace("[GREETING]", "").strip()
        audio_url = text_to_speech(greeting)
        print("SARVAM COOKED:", audio_url)
        if audio_url:
            return {"response": greeting, "audio_url": f"/static/audios/{os.path.basename(audio_url)}"}
        else:
            return {"response": greeting, "audio_url": None}

@app.post("/search-web/")
async def search_web_endpoint(query_model: QueryModel):
    print(f"Searching the web for more information on: {query_model.query}")
    detailed_web_results = retrieve_full_web_content_and_links(query_model.query)

    web_content = ""
    for result in detailed_web_results:
        web_content += f"Title: {result['title']}\nLink: {result['link']}\nContent: {result['content']}\n\n"

    extended_prompt = f"Here is more context and information from the web:\n\n{web_content}"
    final_prompt = template_2.format(query_model.query, extended_prompt)
    model = configure_genai("AIzaSyAwuW7ZvbYb7ygDqxWGDLjUSgfq8hbZ1x0")
    final_response = generate_response(model, final_prompt)
    audio_url = text_to_speech(final_response)
    if audio_url:
        return {"response": final_response, "audio_url": f"/static/audios/{os.path.basename(audio_url)}"}
    else:
        return {"response": final_response, "audio_url": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

###################################################################################
# for query in queries:
#     query = refine_query(query)
#     print("Here is the refined query: ", query)
#     response = generate_rag_response(query)
#     print(f"Query: {query}")
#     print(f"Response: {response}")
    
#     # prompting user if they want more information
#     user_input = input("Would you like to know more about this topic from the web? (yes/no): ").strip().lower()
    
#     if user_input == "yes":
#         print(f"Searching the web for more information on: {query}")
#         detailed_web_results = retrieve_full_web_content_and_links(query)

#         web_content = ""
#         for result in detailed_web_results:
#             web_content += f"Title: {result['title']}\nLink: {result['link']}\nContent: {result['content']}\n\n"

#         extended_prompt = f"Here is more context and information from the web:\n\n{web_content}"
#         # print(extended_prompt)
#         # THIS SEEMS TO BE WORKING
#         final_prompt = template_2.format(query, extended_prompt)
#         model = configure_genai("AIzaSyAwuW7ZvbYb7ygDqxWGDLjUSgfq8hbZ1x0")
#         final_response = generate_response(model, final_prompt)
#         print(f"Wanna know what I found on the web? {final_response}")
