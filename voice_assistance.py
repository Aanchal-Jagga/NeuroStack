import os
import tempfile
import whisper
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from elevenlabs import ElevenLabs
import sounddevice as sd
import soundfile as sf
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()  # this loads your .env file




class DocumentProcessor:
    """Handles document loading, processing, and vector store creation."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def load_documents(self, directory: str) -> List[Document]:
        """Load PDF, TXT, and MD documents from a directory."""
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        }
        documents = []
        for ext, loader in loaders.items():
            try:
                documents.extend(loader.load())
                print(f"Loaded {ext} documents")
            except Exception as e:
                print(f"Error loading {ext} documents: {e}")
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for embedding."""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document], persist_directory: str) -> Chroma:
        """Create or load a Chroma vector store."""
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print(f"Loading existing vector store from {persist_directory}")
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        else:
            print(f"Creating new vector store at {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)
            vector_store = Chroma.from_documents(documents=documents,embedding_function=self.embeddings, persist_directory=persist_directory)
            vector_store.persist()
        return vector_store


class VoiceGenerator:
    """Handles text-to-speech using ElevenLabs."""

    def __init__(self, api_key: str):
        self.client = ElevenLabs(api_key=api_key)
        # store mapping: name → voice_id
        self.available_voices = {
            "Rachel": "aD6riP1btT197c6dACmy",
            
            # add other voices here if you want
            # "Bella": "xxxx", "Josh": "yyyy"
        }
        self.default_voice = "Rachel"

    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        """Generate TTS audio file and return path."""
        voice_name = voice_name or self.default_voice
        voice_id = self.available_voices.get(voice_name)

        if not voice_id:
            print(f"Voice '{voice_name}' not found in available_voices")
            return None

        try:
            response = self.client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                text=text
            )

            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            for chunk in response:  # stream comes in chunks
                temp_file.write(chunk)
            temp_file.close()

            return temp_file.name
        except Exception as e:
            print(f"Error generating voice: {e}")
            return None



class VoiceAssistantRAG:
    """Gemma3 RAG voice assistant."""

    def __init__(self, elevenlabs_api_key: str, gemma_model: str = "gemma3:270m"):
        self.whisper_model = whisper.load_model("base")
        # ❌ removed format="json" so LLM can output natural text
        self.llm = ChatOllama(model=gemma_model, temperature=0, timeout=120)
        self.vector_store = None
        self.qa_chain = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)

    def setup_vector_store(self, vector_store: Chroma):
        """Initialize QA chain with vector store."""
        self.vector_store = vector_store
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            verbose=False
        )

    def record_audio(self, duration: int = 5):
        """Record audio from microphone."""
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        return recording

    def transcribe_audio(self, audio_array):
        """Transcribe audio array using Whisper."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        try:
            sf.write(temp_file.name, audio_array, self.sample_rate)
            text = self.whisper_model.transcribe(temp_file.name)["text"]
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
        return text.strip()

    def generate_response(self, query: str) -> str:
        """Generate response using RAG QA chain with fallback to plain LLM."""
        if not self.qa_chain:
            return self.llm.invoke(query).content  # fallback directly to LLM

        result = self.qa_chain.invoke({"question": query})
        answer = (
            result.get("answer")
            or result.get("result")  # ✅ handle both keys
            or ""
        ).strip()

        # If RAG gives empty/echo answer, fallback to LLM
        if not answer or answer.lower() in [query.lower(), query.lower().rstrip("?")]:
            print("⚠️ RAG had no useful context, falling back to plain LLM.")
            return self.llm.invoke(query).content

        return answer


    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        """Convert text to speech audio file path."""
        return self.voice_generator.generate_voice_response(text, voice_name)
