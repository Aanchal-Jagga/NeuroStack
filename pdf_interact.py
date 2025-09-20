# pdf_interact.py
import uuid
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
import re
import requests

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

class PDFProcessor:
    """PDF processing and chunking"""
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()

    def create_chunks(self, text, pdf_file):
        """Split into fixed-size overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "metadata": {"source": getattr(pdf_file, "name", "uploaded.pdf")}
                })
            start += self.chunk_size - self.chunk_overlap
        return chunks


class RAGSystem:
    """RAG system using Gemma3 + ChromaDB"""
    def __init__(self, persist_dir="./chroma_db"):
        self.db = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            model_name="nomic-embed-text"
        )
        self.collection = self._setup_collection()

    def _setup_collection(self):
        try:
            return self.db.get_collection(
                name="gemma3_documents",
                embedding_function=self.embedding_fn
            )
        except:
            return self.db.create_collection(
                name="gemma3_documents",
                embedding_function=self.embedding_fn
            )

    def add_pdf(self, pdf_file):
        processor = PDFProcessor()
        text = processor.read_pdf(pdf_file)
        chunks = processor.create_chunks(text, pdf_file)
        self.add_documents(chunks)
        return len(chunks)

    def add_documents(self, chunks, batch_size=5):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

    def query_documents(self, query, n_results=3):
        results = self.collection.query(query_texts=[query], n_results=n_results)

        documents = results.get("documents", [])
        contexts = []
        for doc_list in documents:
            for doc in doc_list:
                if doc.strip():
                    contexts.append(doc)

        context_text = "\n\n".join(contexts[:n_results])
        return {"documents": documents, "context": context_text}

    def generate_response(self, query, n_results=5):
        retrieved = self.query_documents(query, n_results=n_results)
        context_text = retrieved["context"]

        if not context_text.strip():
            return {"answer": "I don't know", "context": context_text, "documents": retrieved["documents"]}

        prompt = f"""
        Answer the following question using only the context provided. 
        If the context does not contain the answer, reply with "I don't know".

        Context:
        {context_text}

        Question:
        {query}

        Answer in complete sentences:
        """

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3:270m", "prompt": prompt, "stream": False},
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                # Different Ollama versions return "response" or "message"
                answer = (
                    data.get("response")
                    or (data.get("message", {}).get("content") if isinstance(data.get("message"), dict) else None)
                    or "I don't know"
                )
            else:
                answer = f"Error: {response.text}"
        except Exception as e:
            answer = f"Error generating response: {str(e)}"

        return {"answer": answer.strip(), "context": context_text, "documents": retrieved["documents"]}
