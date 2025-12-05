import streamlit as st
import os
import tempfile
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow importing from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.rag_service import RAGService

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Lecture RAG Bot", page_icon="📚", layout="wide")

# Custom CSS for professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #41444e;
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: transparent;
        border: none;
    }
    
    [data-testid="stChatMessageContent"] {
        background-color: #1e212b;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #41444e;
    }

    /* User message specific */
    [data-testid="stChatMessage"][data-testid="user"] [data-testid="stChatMessageContent"] {
        background-color: #2b303b;
        border-color: #4a4e5a;
    }

    /* Header styling */
    h1 {
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.2);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1e212b;
        padding: 20px;
        border-radius: 10px;
        border: 1px dashed #41444e;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_service" not in st.session_state:
    st.session_state.rag_service = RAGService()

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    
    st.header("Upload Lecture")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    st.divider()
    
    if st.button("Clear Conversation", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Main Content
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://img.icons8.com/fluency/96/books.png", width=80)
with col2:
    st.title("Lecture RAG Bot")
    st.markdown("Upload a lecture PDF and ask questions based **only** on its content.")

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
    st.stop()

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Only process if not already processed or if file changed (simple check)
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
             with st.spinner("Processing PDF... This may take a moment."):
                st.session_state.rag_service.process_pdf(tmp_file_path)
                st.session_state.current_file = uploaded_file.name
                
                st.toast("PDF Processed successfully!", icon="✅")

        # Chat Interface
        st.divider()
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the lecture..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.rag_service.get_answer(prompt)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"An error occurred during generation: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except:
                pass

else:
    st.info("👈 Please upload a PDF file in the sidebar to start.")
