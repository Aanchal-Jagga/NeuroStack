# main.py
import os
import tempfile
import streamlit as st
# from streamlit_float import *
import pandas as pd
import io
from docx import Document
import matplotlib.pyplot as plt
from finance_report import run_financial_report
from news_summariser import NewsArticleSummarizer
from yt_summariser import YoutubeVideoSummarizer
from voice_assistance import VoiceAssistantRAG, DocumentProcessor
from pdf_interact import RAGSystem, PDFProcessor
import base64
# from finance_report import graph  # import the prebuilt graph
from finance_report import AgentState 
from sentiment import SentimentAnalyzer

# from floating_chatbot import floating_chatbot_ui
# from chatbot_agent import chatbot_ui
from fetchers import fetch_comments
from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/NeuroStack/.env")
# ---------------- API Key ----------------
ELEVENLABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
if not ELEVENLABS_API_KEY:
    st.error("Please set ELEVEN_LABS_API_KEY in your environment variables")
    st.stop()


# ---------------- Main UI ----------------
def run_ui():
    st.set_page_config(page_title="NeuroStack", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🧠 NeuroStack</h1>", unsafe_allow_html=True)

    # chatbot_ui()
    
    menu = st.sidebar.selectbox("Choose a Feature", [
        "YouTube Summarizer",
        "News Summarizer",
        "PDF Chat",
        "Voice Assistant",
        "Finance Report",
        "Sentiment & Topic Classifier",
        
    ])

    
    
    # ---------------- YouTube Summarizer ----------------
    if menu == "YouTube Summarizer":
        st.subheader("🎬 YouTube Video Summarization")
        url = st.text_input("Enter YouTube Video URL:")
        if url and st.button("Summarize"):
            try:
                summarizer = YoutubeVideoSummarizer()
                result = summarizer.summarize(url)
                summary_text = None
                if isinstance(result.get("summary"), dict):
                    if "output_text" in result["summary"]:
                        import json
                        try:
                            summary_dict = json.loads(result["summary"]["output_text"])
                            summary_text = summary_dict.get("text", result["summary"]["output_text"])
                        except:
                            summary_text = result["summary"]["output_text"]
                    elif "text" in result["summary"]:
                        summary_text = result["summary"]["text"]
                elif isinstance(result.get("summary"), str):
                    summary_text = result.get("summary")

                st.subheader("Video Title:")
                st.write(result.get("title", "Unknown Title"))
                if summary_text:
                    st.subheader("📝 Summary")
                    st.info(summary_text)
                else:
                    st.warning("No summary could be extracted.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ---------------- News Summarizer ----------------
    elif menu == "News Summarizer":
        st.subheader("📰 News Article Summarization")
        url = st.text_input("Enter news article URL:")
        if url:
            summarizer = NewsArticleSummarizer()
            result = summarizer.summarize(url)
            summary_text = None
            if isinstance(result.get("summary"), dict):
                if "output_text" in result["summary"]:
                    import json
                    try:
                        summary_dict = json.loads(result["summary"]["output_text"])
                        summary_text = summary_dict.get("text", result["summary"]["output_text"])
                    except:
                        summary_text = result["summary"]["output_text"]
                elif "text" in result["summary"]:
                    summary_text = result["summary"]["text"]
            if summary_text:
                st.markdown("### 📝 Summary")
                st.info(summary_text)
            else:
                st.warning("No summary text could be extracted.")

    # ---------------- PDF Chat ----------------
    elif menu == "PDF Chat":
        st.title("PDF Interaction")

        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        if "rag_system" not in st.session_state:
            st.session_state.rag_system = RAGSystem()

        pdf_file = st.file_uploader("Upload PDF", type="pdf")

        if pdf_file and pdf_file.name not in st.session_state.processed_files:
            with st.spinner("Processing PDF..."):
                n_chunks = st.session_state.rag_system.add_pdf(pdf_file)
                st.session_state.processed_files.add(pdf_file.name)
                st.success(f"Processed {pdf_file.name} ({n_chunks} chunks) ✅")

        if st.session_state.processed_files:
            st.markdown("---")
            query = st.text_input("🔍 Ask a Question:")
            if query:
                with st.spinner("Generating response..."):
                    results = st.session_state.rag_system.generate_response(query, n_results=5)
                    st.markdown("### 📝 Answer:")
                    st.write(results["answer"])

                    with st.expander("View Source Passages"):
                        for idx, doc_list in enumerate(results["documents"], 1):
                            for doc in doc_list:
                                st.markdown(f"**Passage {idx}:**")
                                st.info(doc.strip())

                
        # ---------------- Knowledge Base Setup ----------------
    elif menu == "Setup Knowledge Base":
        st.header("📚 Knowledge Base Setup")
        doc_processor = DocumentProcessor()

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                temp_dir = tempfile.mkdtemp()
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                try:
                    documents = doc_processor.load_documents(temp_dir)
                    processed_docs = doc_processor.process_documents(documents)
                    vector_store = doc_processor.create_vector_store(processed_docs, "vector_store")
                    st.session_state.vector_store = vector_store
                    st.success(f"Processed {len(processed_docs)} document chunks!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                finally:
                    for f in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, f))
                    os.rmdir(temp_dir)
    # ---------------- Voice Assistant ----------------
    elif menu == "Voice Assistant":
        st.header("🎤 Voice Assistant (Gemma3)")

        # --- Step 1: Setup Knowledge Base ---
        st.subheader("📚 Upload Knowledge Base")
        doc_processor = DocumentProcessor()

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                temp_dir = tempfile.mkdtemp()
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                try:
                    documents = doc_processor.load_documents(temp_dir)
                    processed_docs = doc_processor.process_documents(documents)
                    vector_store = doc_processor.create_vector_store(processed_docs, "vector_store")
                    st.session_state.vector_store = vector_store
                    st.success(f"Processed {len(processed_docs)} document chunks! ✅")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                finally:
                    for f in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, f))
                    os.rmdir(temp_dir)

        # --- Step 2: Initialize Assistant ---
        if "vector_store" in st.session_state:
            if "assistant" not in st.session_state:
                st.session_state.assistant = VoiceAssistantRAG(ELEVENLABS_API_KEY)
                st.session_state.assistant.setup_vector_store(st.session_state.vector_store)

            assistant = st.session_state.assistant

            # Voice selection
            selected_voice = st.sidebar.selectbox(
                "Select Voice", assistant.voice_generator.available_voices
            )

            # --- Step 3: Record Voice Query ---
            st.subheader("🎙️ Ask a Question by Voice")
            duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

            if st.button("Start Recording"):
                with st.spinner(f"Recording {duration}s..."):
                    audio_data = assistant.record_audio(duration)
                    st.session_state.audio_data = audio_data
                    st.success("Recording complete! ✅")

            if st.button("Process Recording"):
                if "audio_data" not in st.session_state:
                    st.warning("Please record audio first!")
                else:
                    with st.spinner("Transcribing..."):
                        query = assistant.transcribe_audio(st.session_state.audio_data)
                        st.write("You said:", query)

                    with st.spinner("Generating response..."):
                        response = assistant.generate_response(query)
                        st.write("Response:", response)

                        audio_file = assistant.text_to_speech(response, selected_voice)
                        if audio_file:
                            st.audio(audio_file)
                            os.remove(audio_file)
                        else:
                            st.error("❌ Failed to generate voice response")
        else:
            st.info("👉 Please upload and process documents first to build the knowledge base.")

    # ---------------- Other Modules ----------------
    # ---------------- Finance Report ----------------
    elif menu == "Finance Report":
        st.title("📊 AI-Powered Financial Report Generator")

        uploaded_file = st.file_uploader("Upload your CSV financial data", type=["csv"])

        # Preview
        df_preview = None
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                df_preview = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                if df_preview.empty:
                    st.error("❌ CSV has no data rows. Please add some.")
                else:
                    st.subheader("📂 Preview of Uploaded Data")
                    st.dataframe(df_preview.head())
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")

        task = st.text_area("Describe your task (e.g., 'Generate report for Q2 2025')")
        competitors = st.text_input(
            "Enter competitor company names (comma-separated)",
            "google,nvidia"
        )
        max_revisions = st.slider("Max report revision cycles", min_value=1, max_value=5, value=2)

        if st.button("🚀 Generate Report"):
            if uploaded_file is None:
                st.error("Please upload a CSV file before generating the report.")
            elif not task:
                st.error("Please describe the task for the report.")
            else:
                try:
                    uploaded_file.seek(0)
                    csv_str = uploaded_file.getvalue().decode("utf-8")
                    state: AgentState = {
                        "task": task,
                        "competitors": [c.strip() for c in competitors.split(",") if c.strip()],
                        "csv_file": io.StringIO(csv_str),
                        "financial_data": "",
                        "analysis": "",
                        "comparison": "",
                        "report": "",
                        "content": [],
                        "revision_number": 0,
                        "max_revisions": max_revisions,
                        "current_step": [],
                    }

                    final_state = run_financial_report(state)

                    # Display report
                    report_text = final_state.get("report", "⚠️ No report generated.")
                    st.subheader("📑 Financial Report")
                    st.text_area("Report", report_text, height=350)

                    # Display plots
                    plots = final_state.get("content", [])
                    if plots:
                        st.subheader("📊 Financial Charts")
                        tabs = st.tabs([name for name, _ in plots])
                        for tab, (name, buf) in zip(tabs, plots):
                            with tab:
                                st.image(buf, caption=f"{name} Trend", use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error generating report: {e}")


    # ---------------- Sentiment & Review Analyzer ----------------
    elif menu == "Sentiment & Topic Classifier":
        st.subheader("💬 Sentiment & Review Analyzer")
        url = st.text_input("Enter product/video link:")
        max_comments = st.slider("Max comments to analyze", 10, 200, 50)

        if url and st.button("Analyze"):
            comments = fetch_comments(url)[:max_comments]
            if not comments:
                st.warning("No comments/reviews found for this link.")
            else:
                analyzer = SentimentAnalyzer()

                st.subheader("📊 Overall Analysis")
                progress_bar = st.progress(0)
                results_container = st.empty()

                # Initialize cumulative counts
                cumulative_summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
                details = []

                # Use spinner while analyzing
                with st.spinner("Analyzing comments..."):
                    batch_size = 5
                    total = len(comments)

                    for i in range(0, total, batch_size):
                        batch_comments = comments[i:i+batch_size]
                        batch_result = analyzer.analyze(batch_comments)
                        details.extend(batch_result["details"])

                        # Update cumulative counts
                        for r in batch_result["details"]:
                            label = r.get("label", "").capitalize()
                            if label in cumulative_summary:
                                cumulative_summary[label] += 1

                        # Update progress
                        progress = min((i + batch_size) / total, 1.0)
                        progress_bar.progress(progress)

                        # Update current cumulative summary
                        results_container.markdown(
                            f"**Processed {min(i+batch_size, total)}/{total} comments** | "
                            f"**Current Summary:** {cumulative_summary}"
                        )

                final_result = {"summary": cumulative_summary, "details": details}
                st.success("✅ Analysis complete!")

                # ---------------- Display final summary table ----------------
                st.markdown("### 📊 Sentiment Summary")
                summary_df = pd.DataFrame(
                    list(final_result["summary"].items()),
                    columns=["Sentiment", "Count"]
                )
                st.table(summary_df.style.apply(
                    lambda x: ['color: green' if x['Sentiment']=="Positive" 
                            else 'color: red' if x['Sentiment']=="Negative" 
                            else 'color: gray' for i in x], axis=1
                ))

                # ---------------- Display top comments with color coding ----------------
                st.subheader("📝 Sample Comments with Sentiments (Top 10)")
                color_map = {"Positive": "green", "Negative": "red", "Neutral": "gray"}

                for r in final_result.get("details", [])[:10]:  # Show only first 10 comments
                    label = r.get("label", "Unknown").capitalize()
                    comment = r.get("comment", "")
                    color = color_map.get(label, "black")
                    st.markdown(
                        f"<span style='color:{color}; font-weight:bold'>{label}</span> - {comment}",
                        unsafe_allow_html=True
                    )
    # elif menu == "Sentiment & Topic Classifier":
    #     classify_sentiment()
    # elif menu == "Recommender System":
    #     recommend_something()
    # elif menu == "Agent Chatbot":
    #     ai_agent_chatbot()
    

        
if __name__ == "__main__":
    run_ui()
