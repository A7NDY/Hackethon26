import os
import time
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

# --- 1. Functions for PDF and DB ---
@st.cache_resource 
def process_pdf_and_setup_db(pdf_path):
    """Reads the PDF, extracts text, stores it, and calculates detailed stats."""
    total_start_time = time.time()
    
    # STEP 1: OCR PHASE
    ocr_start_time = time.time()
    images = convert_from_path(pdf_path)
    num_pages = len(images)
    
    full_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        full_text += f"\n--- Page {i + 1} ---\n{text}"
        
    ocr_time = round(time.time() - ocr_start_time, 2)
    
    num_words = len(full_text.split())
    num_chars = len(full_text)
    num_lines = len(full_text.split('\n'))
    avg_words_per_page = round(num_words / num_pages) if num_pages > 0 else 0
    
    # STEP 2: DB & EMBEDDING PHASE
    db_start_time = time.time()
    chunk_size = 500
    chunk_overlap = 50
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(full_text)
    num_chunks = len(chunks)
    
    embedding_model = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_db = Chroma.from_texts(texts=chunks, embedding=embeddings)
    
    db_time = round(time.time() - db_start_time, 2)
    total_time = round(time.time() - total_start_time, 2)
    
    stats = {
        "pages": num_pages, "words": num_words, "chars": num_chars,
        "lines": num_lines, "avg_words": avg_words_per_page,
        "ocr_time": ocr_time, "chunks": num_chunks, "chunk_size": chunk_size,
        "db_time": db_time, "total_time": total_time, "embedding_model": embedding_model
    }
    
    return vector_db, stats

# --- 2. Setup Conversational AI ---
def get_conversational_chain(vector_db):
    """Sets up the local Llama 3 AI to answer questions using the database."""
    llm = Ollama(model="llama3") 
    
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

# --- 3. Streamlit User Interface & Pastel Theme ---
st.set_page_config(page_title="Study Assistant", page_icon="📚")

st.markdown("""
<style>
    .stApp { background-color: #FFFBF0; }
    html, body, [class*="css"], h1, h2, h3, p { color: #4A4A4A !important; }
    section[data-testid="stFileUploadDropzone"] { background-color: #E6F4F1; border: 2px dashed #A2CFFE; border-radius: 10px; }
    div[data-testid="stChatInput"] { background-color: #F3E8FF; border-radius: 10px; border: 1px solid #D8B4E2; }
    div[data-testid="chat-message-user"] { background-color: #D6EAF8; border-radius: 15px; padding: 10px; }
    div[data-testid="chat-message-assistant"] { background-color: #F5EEF8; border-radius: 15px; padding: 10px; }
    .streamlit-expanderHeader { background-color: #FFFFFF; border-radius: 10px; color: #4A4A4A !important; border: 1px solid #E0E0E0; }
    div[data-testid="stMetricValue"] { color: #5D3FD3 !important; font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Personal Study Assistant")
st.write("Upload your handwritten notes and ask questions based *only* on them!")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    temp_pdf_path = "temp_uploaded_notes.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner("Reading handwriting and building vector database..."):
        db, doc_stats = process_pdf_and_setup_db(temp_pdf_path)
        qa_system = get_conversational_chain(db)
        
    st.success("Notes processed successfully!")
    
    # --- DISPLAYING THE DETAILED STATS ---
    with st.expander("📊 View Document Intelligence Dashboard"):
        st.markdown("#### 📝 OCR & Extraction Details")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Pages Read", doc_stats["pages"])
        with col2: st.metric("Total Words", doc_stats["words"])
        with col3: st.metric("Characters", doc_stats["chars"])
        with col4: st.metric("OCR Time", f"{doc_stats['ocr_time']}s")
        
        st.markdown("#### 🧠 AI & Database Details")
        col5, col6, col7, col8 = st.columns(4)
        with col5: st.metric("Searchable Chunks", doc_stats["chunks"])
        with col6: st.metric("Chunk Size", f"{doc_stats['chunk_size']} chars")
        with col7: st.metric("Embedding Time", f"{doc_stats['db_time']}s")
        with col8: st.metric("Total Time", f"{doc_stats['total_time']}s")
        
    st.divider() 

    # --- CHAT INTERFACE ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Re-draw the chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    user_query = st.chat_input("Ask a question about your notes...")

    if user_query:
        st.chat_message("user").write(user_query)
        
        with st.spinner("Analyzing question and fetching OCR matches..."):
            ask_start_time = time.time()
            
            # GET ACCURACY/CONFIDENCE SCORES
            # We manually search the database first to see the math behind the scenes
            retrieval_results = db.similarity_search_with_score(user_query, k=3)
            
            # GET LLM ANSWER
            response = qa_system.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            
            answer_time = round(time.time() - ask_start_time, 2)
            answer = response["answer"]
            
            with st.chat_message("assistant"):
                st.write(answer)
                
                # SHOW QUESTION-LEVEL ANALYTICS
                with st.expander("⚙️ Question Analytics & OCR Matches (Proof)"):
                    st.write(f"⏱️ **Generation Time:** {answer_time} seconds")
                    st.write("🔍 **OCR Database Matches:**")
                    
                    for i, (doc, distance) in enumerate(retrieval_results):
                        # Convert Chroma's mathematical distance into a 0-100% confidence score
                        confidence_score = max(0.0, round(100 - (distance * 50), 1))
                        
                        st.info(f"**Match #{i+1} | AI Confidence: {confidence_score}%**\n\n*\"{doc.page_content[:250]}...\"*")
                
        # Save to memory
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))
else:
    st.info("👆 Please upload a PDF document to get started!")