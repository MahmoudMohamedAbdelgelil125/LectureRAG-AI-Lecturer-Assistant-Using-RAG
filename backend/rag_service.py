import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate

class RAGService:
    def __init__(self):
        self.retriever = None
        self.rag_chain = None
        self.vectorstore = None

    def process_pdf(self, file_path: str) -> bool:
        """
        Loads a PDF, splits it into chunks, and creates a vector store retriever.
        """
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Create Vector Store
            # Note: In a real production app, we might want to persist this or use a proper DB.
            # For this session-based demo, in-memory is fine.
            self.vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            self.retriever = self.vectorstore.as_retriever()

            # Prepare the Retrieval QA chain
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

            system_prompt = (
                "You are a helpful assistant for answering questions about lecture material. "
                "Use the following pieces of retrieved context to answer the question. "
                "If the answer is not present in the context, strictly state: "
                "'I cannot answer this based on the provided lecture context.'"
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("user", "{question}"),
                ]
            )

            self.rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=self.retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template}
            )

            return True
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False

    def get_answer(self, question: str) -> str:
        """
        Generates an answer for the given question using the initialized retriever.
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Please process a PDF first.")

        response = self.rag_chain.run(question)
        return response
    
    def cleanup(self):
        """
        Cleanup resources if needed.
        """
        if self.vectorstore:
            self.vectorstore.delete_collection()

