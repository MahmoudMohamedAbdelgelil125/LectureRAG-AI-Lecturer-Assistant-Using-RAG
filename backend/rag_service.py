import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGService:
    def __init__(self):
        self.retriever = None

    def process_pdf(self, file_path):
        """
        Loads a PDF, splits it, and creates a vector store retriever.
        """
        # Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create Vector Store
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever()
        return True

    def get_answer(self, question):
        """
        Generates an answer for the given question using the initialized retriever.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Please process a PDF first.")

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
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": question})
        return response["answer"]
