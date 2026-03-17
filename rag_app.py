import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import io
import csv
import google.generativeai as genai
import pdfplumber
from docx import Document as DocxDocument

# ============ VECTOR DATABASE (Endee) ============
class EndeeVectorDB:
    """Vector Database inspired by Endee for semantic search"""
    def __init__(self, db_path="./endee_db"):
        self.db_path = db_path
        self.documents = []
        self.embeddings = []
        os.makedirs(db_path, exist_ok=True)
        self.load_db()

    def load_db(self):
        db_file = os.path.join(self.db_path, "data.json")
        emb_file = os.path.join(self.db_path, "embeddings.npy")
        if os.path.exists(db_file) and os.path.exists(emb_file):
            with open(db_file, "r") as f:
                self.documents = json.load(f)
            self.embeddings = np.load(emb_file).tolist()

    def save_db(self):
        db_file = os.path.join(self.db_path, "data.json")
        emb_file = os.path.join(self.db_path, "embeddings.npy")
        with open(db_file, "w") as f:
            json.dump(self.documents, f)
        if self.embeddings:
            np.save(emb_file, np.array(self.embeddings))

    def add(self, doc_id, vector, metadata):
        self.documents.append({"id": doc_id, "metadata": metadata})
        self.embeddings.append(vector)
        self.save_db()

    def search(self, query_vector, top_k=3):
        if not self.embeddings:
            return []
        query = np.array(query_vector)
        scores = []
        for i, emb in enumerate(self.embeddings):
            emb = np.array(emb)
            score = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb))
            scores.append((score, i))
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            results.append({
                "score": float(score),
                "metadata": self.documents[idx]["metadata"],
                "id": self.documents[idx]["id"]
            })
        return results

    def get_doc_count(self):
        return len(self.documents)

    def clear_db(self):
        self.documents = []
        self.embeddings = []
        self.save_db()


# ============ GEMINI MODEL (cached — fetched once) ============
@st.cache_resource
def get_gemini_model(api_key: str):
    """Resolve and cache the best available Gemini model once per session."""
    import time, re
    genai.configure(api_key=api_key)
    preferred = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash-lite",
        "gemini-1.0-pro",
    ]
    try:
        available = {
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        }
        ordered = [m for m in preferred if m in available] or list(available)
    except Exception:
        ordered = preferred
    return ordered  # return list; pick first that works at call time


# ============ AI ANSWER GENERATION ============
def generate_answer(query, context, api_key, placeholder=None):
    """Stream answer using Google Gemini AI."""
    import time, re

    prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

    genai.configure(api_key=api_key)
    models_to_try = get_gemini_model(api_key)

    for model_name in models_to_try:
        try:
            gemini = genai.GenerativeModel(model_name)
            if placeholder:
                # streaming — write chunks live into the placeholder
                full = ""
                for chunk in gemini.generate_content(prompt, stream=True):
                    if chunk.text:
                        full += chunk.text
                        placeholder.markdown(full + "▌")
                placeholder.markdown(full)
                return full
            else:
                return gemini.generate_content(prompt).text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                match = re.search(r"retry.*?(\d+)\s*s", err, re.IGNORECASE)
                wait = min(int(match.group(1)) if match else 10, 15)
                time.sleep(wait)
                continue
            if "404" in err or "not found" in err.lower():
                continue
            return f"Error: {err}"

    return "⚠️ All Gemini models are rate-limited. Please wait a minute and try again."


# ============ FILE TEXT EXTRACTION ============
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()

    if name.endswith((".txt", ".md", ".rst")):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            pages = [page.extract_text(x_tolerance=2, y_tolerance=2) or "" for page in pdf.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())

    if name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if name.endswith(".csv"):
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        return "\n".join(", ".join(row) for row in csv.reader(io.StringIO(content)))

    if name.endswith(".json"):
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        return json.dumps(json.loads(content), indent=2)

    return uploaded_file.read().decode("utf-8", errors="ignore")


# ============ INITIALIZE ============
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_db():
    return EndeeVectorDB(db_path="./endee_knowledge_base")


# ============ MAIN APP ============
def main():
    st.set_page_config(page_title="RAG System with Endee", page_icon="🤖", layout="wide")

    st.title("🤖 RAG System - Powered by Endee Vector Database")
    st.markdown("*Retrieval Augmented Generation with AI-powered answers*")

    model = load_model()
    db = get_db()

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input("Enter Gemini API Key:", type="password",
                                 help="Get free key from https://aistudio.google.com/app/apikey")

        st.divider()
        st.header("📊 Database Stats")
        st.metric("Total Documents", db.get_doc_count())

        st.divider()
        st.header("📚 About")
        st.info("""
        **How it works:**
        1. Upload documents to Endee DB
        2. Ask a question
        3. Endee finds relevant docs
        4. Gemini AI generates answer
        """)

        st.divider()
        st.header("🔧 Tech Stack")
        st.write("- **Vector DB**: Endee")
        st.write("- **Embeddings**: Sentence Transformers")
        st.write("- **LLM**: Google Gemini")
        st.write("- **Frontend**: Streamlit")

        if st.button("🗑️ Clear Database"):
            db.clear_db()
            st.success("Database cleared!")
            st.rerun()

    # ---- TABS ----
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "🔍 Search & Ask", "💬 Chat"])

    # ---- TAB 1: UPLOAD DOCUMENTS ----
    with tab1:
        st.header("📤 Upload Your Knowledge Base")

        # Sample data button
        if st.button("📋 Load Sample Data"):
            sample_docs = [
                "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
                "Python is the most popular programming language for data science and machine learning applications.",
                "Neural networks are computing systems inspired by biological neural networks in the human brain.",
                "Deep learning is a subset of machine learning that uses multiple layers of neural networks to analyze data.",
                "Natural Language Processing (NLP) is a field of AI that helps computers understand and process human language.",
                "TensorFlow and PyTorch are the two most popular deep learning frameworks used by researchers and developers.",
                "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data.",
                "Transfer learning allows a model trained on one task to be reused for a different but related task.",
                "Computer vision is an AI field that enables computers to interpret and understand visual information from images and videos.",
                "Reinforcement learning is a type of machine learning where an agent learns to make decisions by receiving rewards or penalties.",
                "Data preprocessing involves cleaning, transforming, and organizing raw data before using it for machine learning.",
                "GPT and BERT are popular transformer-based models used for natural language understanding and generation."
            ]
            with st.spinner("Adding sample documents to Endee..."):
                for i, doc in enumerate(sample_docs):
                    embedding = model.encode(doc).tolist()
                    db.add(doc_id=f"sample_{i}", vector=embedding, metadata={"text": doc})
            st.success(f"✅ Added {len(sample_docs)} sample documents to Endee!")
            st.balloons()
            st.rerun()

        st.divider()

        # Manual input
        documents_input = st.text_area(
            "Enter documents (one per line):",
            height=200,
            placeholder="Enter your documents here...\nOne document per line..."
        )

        uploaded_file = st.file_uploader(
            "Or upload a file (txt, pdf, docx, csv, json, md):",
            type=["txt", "pdf", "docx", "csv", "json", "md", "rst"]
        )
        if uploaded_file:
            documents_input = extract_text(uploaded_file)
            if documents_input:
                st.success(f"✅ File uploaded: `{uploaded_file.name}`")

        if st.button("🚀 Add to Endee Database", type="primary"):
            if documents_input:
                docs = [doc.strip() for doc in documents_input.split("\n") if doc.strip()]
                with st.spinner("Adding documents to Endee..."):
                    for i, doc in enumerate(docs):
                        embedding = model.encode(doc).tolist()
                        db.add(
                            doc_id=f"doc_{i}_{hash(doc) % 10000}",
                            vector=embedding,
                            metadata={"text": doc}
                        )
                st.success(f"✅ Added {len(docs)} documents to Endee!")
                st.balloons()
            else:
                st.warning("⚠️ Please enter some documents first!")

    # ---- TAB 2: SEARCH & ASK ----
    with tab2:
        st.header("🔍 Search & Get AI Answers")

        query = st.text_input("Enter your question:", placeholder="What is machine learning?")
        top_k = st.slider("Number of documents to retrieve:", 1, 10, 3)

        if query:
            if db.get_doc_count() == 0:
                st.warning("⚠️ No documents in database! Go to 'Upload Documents' tab first.")
            else:
                # Step 1: Retrieve from Endee
                with st.spinner("🔍 Searching Endee database..."):
                    query_embedding = model.encode(query).tolist()
                    results = db.search(query_embedding, top_k=top_k)

                if results:
                    # Show retrieved documents
                    st.subheader("📄 Retrieved Documents from Endee")
                    for i, result in enumerate(results, 1):
                        score = result["score"]
                        text = result["metadata"]["text"]
                        with st.expander(f"📄 Result {i} - Similarity: {score:.2%}"):
                            st.write(text)

                    # Step 2: Generate AI Answer
                    st.divider()
                    st.subheader("🤖 AI-Generated Answer")

                    context = "\n".join([r["metadata"]["text"] for r in results])

                    if api_key:
                        st.subheader("🤖 AI-Generated Answer")
                        placeholder = st.empty()
                        generate_answer(query, context, api_key, placeholder=placeholder)
                    else:
                        st.warning("⚠️ Enter your Gemini API key in the sidebar to get AI-generated answers!")
                        st.info(f"**Retrieved Context:**\n\n{context}")

    # ---- TAB 3: CHAT ----
    with tab3:
        st.header("💬 Chat with Your Documents")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask anything about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                if db.get_doc_count() == 0:
                    response = "⚠️ No documents in database. Please upload documents first in the 'Upload Documents' tab."
                    st.markdown(response)
                elif not api_key:
                    query_embedding = model.encode(prompt).tolist()
                    results = db.search(query_embedding, top_k=3)
                    context = "\n".join([f"• {r['metadata']['text']}" for r in results])
                    response = f"📄 **Retrieved from Endee:**\n\n{context}\n\n⚠️ *Add Gemini API key in sidebar for AI answers.*"
                    st.markdown(response)
                else:
                    query_embedding = model.encode(prompt).tolist()
                    results = db.search(query_embedding, top_k=3)
                    context = "\n".join([r["metadata"]["text"] for r in results])
                    placeholder = st.empty()
                    response = generate_answer(prompt, context, api_key, placeholder=placeholder)

            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()