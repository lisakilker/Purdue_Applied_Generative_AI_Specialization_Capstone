import os
import pandas as pd
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from string import Template
import streamlit as st
from striprtf.striprtf import rtf_to_text

# Paths to data files (pdf's are in a separate folder)
current_dir = os.path.dirname(os.path.abspath("__file__"))
pdf_folder = os.path.join(current_dir, "pdfs")
csv_file_path = os.path.join(current_dir, "sales_data.csv")
key_file_path = os.path.join(current_dir, "openai_key.rtf")

# Load OpenAI API Key
try:
    with open(key_file_path, "r") as file:
        raw_content = file.read()
        openai_api_key = rtf_to_text(raw_content).strip()
        openai.api_key = openai_api_key

# Error handling
except FileNotFoundError:
    raise FileNotFoundError(f"ERROR: API key file not found: {key_file_path}")
except Exception as e:
    raise RuntimeError(f"ERROR: Could not load OpenAI API key: {e}")

# Initialize ChromaDB client
client = chromadb.Client()

# Memory system to retain conversation history
memory = []

# Step 1: Load PDFs and CSV
def load_data():
    pdf_chunks = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Loading PDF: {pdf_file}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            pdf_chunks.extend(chunks)

    # Error handling
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"ERROR: CSV file not found: {csv_file_path}")

    print(f"SUCCESS: CSV file found: {csv_file_path}")
    csv_data = pd.read_csv(csv_file_path)
    print("CSV loaded successfully!")
    print(csv_data.head())
    
    return pdf_chunks, csv_data

# Step 2: Store Data in ChromaDB with Batch Processing
def store_data_in_chromadb(pdf_chunks, csv_data, batch_size=100):
    collection = client.get_or_create_collection("knowledge_base")

    print("Storing PDF chunks...")
    for i in range(0, len(pdf_chunks), batch_size):
        batch = pdf_chunks[i : i + batch_size]
        documents = [chunk.page_content for chunk in batch]
        metadatas = [{"chunk_id": f"pdf_chunk_{i+j+1}"} for j in range(len(batch))]
        ids = [f"pdf_chunk_{i+j+1}" for j in range(len(batch))]
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print("Storing CSV rows...")
    for i in range(0, len(csv_data), batch_size):
        batch = csv_data.iloc[i : i + batch_size]
        documents = [row.to_string() for _, row in batch.iterrows()]
        metadatas = [{"row_id": f"csv_row_{i+j+1}"} for j in range(len(batch))]
        ids = [f"csv_row_{i+j+1}" for j in range(len(batch))]
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print("Knowledge base stored in ChromaDB!")

# Step 3: Retrieve Data from ChromaDB
def retrieve_data_from_chromadb(query, n_results=3):
    collection = client.get_collection("knowledge_base")
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    retrieved_docs = results["documents"][0] if results["documents"] else []
    return retrieved_docs

# Step 4: Generate AI Response
def generate_ai_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if needed
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant specializing in business intelligence."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    # Error handling
    except Exception as e:
        return f"ERROR: Generating response: {e}"

# Step 5: Answer Query
def answer_query(user_query):
    retrieved_docs = retrieve_data_from_chromadb(user_query)
    if not retrieved_docs:
        return "Sorry, I couldn't find any relevant information."

    # Generate AI response based on retrieved documents
    context = "\n".join(retrieved_docs)
    prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer clearly and concisely:"
    return generate_ai_response(prompt)

# Step 6: Streamlit UI
def streamlit_ui():
    st.title("Your Virtual AI Assistant: How can I help?")
    
    # User Query Section
    st.subheader("Ask a Question: ")
    user_query = st.text_input("Enter your query: ", "")
    
    if user_query:
        response = answer_query(user_query)
        st.text_area("Response: ", response, height=200)
    
    # Visualization Section
    st.subheader("Data Visualizations")
    
    # Placeholder buttons for easy search
    if st.button("Show Sales Trends Over Time"):
        st.write("Sales trends visualization coming soon!")
    
    if st.button("Show Product Performance Comparison"):
        st.write("Product performance visualization coming soon!")

# Runs main
if __name__ == "__main__":
    pdf_chunks, csv_data = load_data()
    store_data_in_chromadb(pdf_chunks, csv_data)
    streamlit_ui()