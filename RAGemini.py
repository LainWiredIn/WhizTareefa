import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import PyPDF2

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Parse PDF and extract text
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

pdf_path = "iesc111.pdf"
document_text = extract_text_from_pdf(pdf_path)

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(document_text)

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

annoy_index, chunk_embeddings = create_annoy_index(chunks, embedder)

def retrieve_relevant_chunks(query, annoy_index, chunks, top_k=5):
    query_embedding = embedder.encode([query])[0]  # Get the query embedding
    nearest_indices = annoy_index.get_nns_by_vector(query_embedding, top_k)  # Find nearest neighbors
    
    return [chunks[i] for i in nearest_indices]

genai.configure(api_key="AIzaSyBxzuL9gFINm3GtnA0SlqWwwtB1ZPVMJY0")

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

generation_config = {
    "temperature": 0.1,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

# 6. Define the RAG template
template = """
You are given a user query and relevant chunks of text from a document.

Query: {}
Relevant Text: {}

Generate a response using the query and the relevant text.
"""

# 7. RAG System: Generate responses based on query and retrieved chunks
def generate_rag_response(query):
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, annoy_index, chunks)
    
    # Format the template with the query and retrieved chunks
    prompt = template.format(query, "\n\n".join(relevant_chunks))
    print("these are the chunks I found: ", relevant_chunks)
    model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config, safety_settings=safety_settings)
    response = model.generate_content(prompt)
    
    return response.text


queries = ["Why are the ceilings of cinema halls curved?"]
for query in queries:
    response = generate_rag_response(query)
    print(f"Query: {query}")
    print(f"Response: {response}")


