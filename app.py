import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Lecture RAG Bot", layout="wide")

st.title("📚 Lecture RAG Bot")
st.markdown("Upload a lecture PDF and ask questions based **only** on its content.")

# Sidebar for API Key and File Upload
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.header("Upload Lecture")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
    st.stop()

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with st.spinner("Processing PDF..."):
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Create Vector Store
            # Note: In production, you might want to persist this
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            retriever = vectorstore.as_retriever()

            st.success("PDF Processed successfully!")

        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the lecture..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # RAG Chain
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                
                # Strict Context Prompt
                system_prompt = (
                    "You are a helpful assistant for answering questions about lecture material. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If the answer is not present in the context, strictly state: "
                    "'I cannot answer this based on the provided lecture context.' "
                    "Do not use outside knowledge."
                    "\n\n"
                    "{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

else:
    st.info("Please upload a PDF file to start.")
