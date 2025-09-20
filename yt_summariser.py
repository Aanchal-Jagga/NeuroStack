import os
import yt_dlp
import whisper
import streamlit as st
from typing import Optional, Tuple, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate


class YoutubeVideoSummarizer:
    def __init__(self, model_name: str = "gemma3:270m"):
        """
        Initialize with Gemma3 via Ollama and Whisper for transcription
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            format="json",
            timeout=120,
        )
        self.whisper_model = whisper.load_model("base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
        )

        # Ensure FFmpeg is found
        self.ffmpeg_path = os.path.join(os.getcwd(), "FFMpeg", "bin")
        os.environ["PATH"] = self.ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

    def download_audio(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Download YouTube video and extract audio"""
        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "ffmpeg_location": self.ffmpeg_path,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "outtmpl": "downloads/%(title)s.%(ext)s",
            }
            os.makedirs("downloads", exist_ok=True)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = os.path.splitext(ydl.prepare_filename(info))[0] + ".mp3"
                return audio_path, info.get("title", "Unknown Title")
        except Exception as e:
            st.error(f"Error downloading video: {e}")
            return None, None

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def create_documents(self, text: str, title: str) -> List[Document]:
        texts = self.text_splitter.split_text(text)
        return [Document(page_content=t, metadata={"title": title}) for t in texts]

    def summarize(self, url: str, summary_type: str = "detailed") -> dict:
        audio_path, video_title = self.download_audio(url)
        if not audio_path:
            return {"error": "Failed to download video"}

        transcript = self.transcribe(audio_path)
        docs = self.create_documents(transcript, video_title)

        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
{text}
DETAILED SUMMARY:"""
            combine_prompt_template = """Combine the summaries into one detailed summary:
{text}
FINAL DETAILED SUMMARY:"""
        else:
            map_prompt_template = """Write a concise summary of the following text:
{text}
CONCISE SUMMARY:"""
            combine_prompt_template = """Combine the summaries into one concise summary:
{text}
FINAL CONCISE SUMMARY:"""

        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        summary = chain.invoke(docs)

        # Clean up audio
        try:
            os.remove(audio_path)
        except Exception as e:
            st.warning(f"Could not remove audio file: {e}")

        return {
            "title": video_title,
            "summary": summary,
            "full_transcript": transcript,
            "url": url,
            "model_info": {"name": self.model_name},
        }


# ---------------- Streamlit UI ---------------- #
def main():
    st.title("🎥 YouTube Video Summarizer (Gemma3 + Whisper)")
    url = st.text_input("Enter YouTube URL:")
    
    if st.button("Summarize"):
        if not url:
            st.warning("Please enter a valid YouTube URL")
        else:
            summarizer = YoutubeVideoSummarizer(model_name="gemma3:270m")
            with st.spinner("Processing video..."):
                result = summarizer.summarize(url, summary_type="detailed")

            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader(f"📌 Title: {result['title']}")
                st.write(f"**Model Used:** {result['model_info']['name']}")
                st.write("### 📝 Summary")
                st.write(result["summary"])

                with st.expander("📜 Full Transcript"):
                    st.write(result["full_transcript"])


if __name__ == "__main__":
    main()
