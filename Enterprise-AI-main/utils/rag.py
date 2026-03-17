from typing import List, Optional
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


class SimpleQAChain:
    """
    Minimal QA chain that mimics LangChain's RetrievalQA .invoke() API
    so the rest of the app code works without langchain.chains.
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def invoke(self, inputs):
        # Support both {"query": "..."} and plain string
        if isinstance(inputs, dict):
            question = inputs.get("query") or inputs.get("question") or ""
        else:
            question = str(inputs)

        question = question.strip()
        if not question:
            return {"result": "", "source_documents": []}

        # 1. Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question)

        # 2. Build a simple prompt using the retrieved context
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            "You are an AI assistant that answers questions using ONLY the context provided.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer clearly and concisely:"
        )

        # 3. Call the LLM
        resp = self.llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp))

        return {
            "result": answer,
            "source_documents": docs,
        }


def build_vectorstore(
    docs: List[Document],
    persist_dir: str = "vectorstore",
) -> Optional[FAISS]:
    """
    Build a FAISS vector store from documents.
    """
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?"],
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)

    return vectorstore


def load_vectorstore(persist_dir: str = "vectorstore") -> Optional[FAISS]:
    """
    Load an existing FAISS vectorstore from disk, if it exists.
    """
    if not os.path.exists(persist_dir):
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vs = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vs
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None


def build_qa_chain(vectorstore: FAISS) -> SimpleQAChain:
    """
    Build our custom QA chain that exposes .invoke()
    just like LangChain's RetrievalQA.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return SimpleQAChain(retriever, llm)
