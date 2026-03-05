import os
import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Functions for PDF and DB (Same as before) ---
@st.cache_resource # This tells Streamlit to only run this once!
def process_pdf_and_setup_db(pdf_path):
    # Read PDF
    images = convert_from_path(pdf_path)
    full_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        full_text += f"\n--- Page {i + 1} ---\n{text}"
        
    # Chunk and Embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(full_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create Local DB
    vector_db = Chroma.from_texts(texts=chunks, embedding=embeddings)
    return vector_db

# --- 2. Setup Conversational AI ---
def get_conversational_chain(vector_db):
    llm = Ollama(model="llama3") 
    
    # The prompt now includes a placeholder for chat history
    system_prompt = (
        "You are a helpful study assistant. Use the retrieved context to answer the question. "
        "If the answer is not in the context, honestly say 'I don't have enough information'. "
        "Never make up information. "
        "\n\nContext: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) 
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# --- 3. Streamlit User Interface ---
st.title("📚 Personal Study Assistant")
st.write("Ask questions based *only* on the provided handwritten notes!")

# Set up the database behind the scenes
pdf_file = "sample_notes.pdf" # Make sure this file is in your folder
if os.path.exists(pdf_file):
    with st.spinner("Reading handwritten notes... This might take a minute."):
        db = process_pdf_and_setup_db(pdf_file)
        qa_system = get_conversational_chain(db)
else:
    st.error(f"Please put a file named '{pdf_file}' in the project folder!")
    st.stop()

# Initialize Chat Memory in Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Chat Input Box
user_query = st.chat_input("Ask a question about your notes...")

if user_query:
    # 1. Show the user's question
    st.chat_message("user").write(user_query)
    
    # 2. Get the AI's answer, passing in the chat history
    with st.spinner("Thinking..."):
        response = qa_system.invoke({
            "input": user_query,
            "chat_history": st.session_state.chat_history
        })
        
        answer = response["answer"]
        sources = response["context"]
        
        # 3. Show the answer and sources
        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("View Sources"):
                for doc in sources:
                    st.info(f"Snippet: {doc.page_content[:150]}...")
            
    # 4. Save the conversation to memory
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=answer))