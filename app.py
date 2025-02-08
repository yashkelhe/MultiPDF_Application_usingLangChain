import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ ERROR: Google API key is missing. Check your .env file.")
    st.stop()  # Stop execution if API key is missing

genai.configure(api_key=api_key)

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure text is not None
                text += page_text + "\n"
    return text

# Split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Store vector embeddings locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Corrected model name
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Ensure directory exists before saving
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    
    vector_store.save_local("faiss_index")


# Load the conversational model
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, making sure to provide all relevant details. 
    If the answer is not in the provided context, say: "answer is not available in the context". 
    Do not fabricate information.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user questions and generate responses

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Corrected model name
    index_path = "faiss_index/index.faiss"

    if not os.path.exists(index_path):
        st.error("❌ ERROR: FAISS index not found. Please upload and process PDFs first.")
        return  

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        context_text = "\n\n".join([doc.page_content for doc in docs])
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        if isinstance(response, dict) and response:
            response_text = response.get("output_text") or response.get("answer") or "❌ ERROR: No valid response received."
            st.write("Reply: ", response_text)
        else:
            st.error("❌ ERROR: Unexpected response format from Gemini.")

    except Exception as e:
        st.error(f"❌ ERROR: {str(e)}")

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini 💁")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():  # Ensure there is extracted text
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! Ready to answer questions.")
                else:
                    st.error("❌ ERROR: No text extracted from PDFs. Please check your files.")

    # Check if FAISS index exists before allowing questions
    if os.path.exists("faiss_index/index.faiss"):
        user_question = st.text_input("Ask a question about the uploaded PDF(s)")
        if user_question:
            user_input(user_question)
    else:
        st.warning("⚠️ Please upload and process PDFs before asking questions.")

if __name__ == "__main__":
    main()
