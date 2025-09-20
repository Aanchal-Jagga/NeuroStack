import os
import streamlit as st
from typing import Optional
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document

from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


class NewsArticleSummarizer:
    def __init__(self, model_name: str = "gemma3:270m"):
        """
        Initialize the summarizer using Ollama with Gemma.
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            format="json",
            timeout=120,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
        )

    def fetch_article(self, url: str) -> Optional[Article]:
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            st.error(f"Error fetching article: {e}")
            return None

    def create_documents(self, text: str) -> list[Document]:
        texts = self.text_splitter.split_text(text)
        return [Document(page_content=t) for t in texts]

    def summarize(self, url: str, summary_type: str = "detailed") -> dict:
        article = self.fetch_article(url)
        if not article:
            return {"error": "Failed to fetch article"}

        docs = self.create_documents(article.text)

        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
            "{text}"
            DETAILED SUMMARY:"""
            combine_prompt_template = """Combine the summaries into one detailed summary:
            "{text}"
            FINAL DETAILED SUMMARY:"""
        else:
            map_prompt_template = """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:"""
            combine_prompt_template = """Combine the summaries into one concise summary:
            "{text}"
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

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": str(article.publish_date),
            "summary": summary,
            "url": url,
            "model_info": {"name": self.model_name},
        }


# ---------------- Streamlit UI ---------------- #
def main():
    st.title("📰 News Article Summarizer (Gemma3)")

    url = st.text_input("Enter a news article URL:")

    option = st.radio("Choose summary type:", ["detailed", "concise"], horizontal=True)

    if st.button("Summarize"):
        if not url:
            st.warning("Please enter a valid news URL")
        else:
            summarizer = NewsArticleSummarizer(model_name="gemma3:270m")
            with st.spinner("Summarizing article..."):
                result = summarizer.summarize(url, summary_type=option)

            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader(f"📌 Title: {result['title']}")
                if result["authors"]:
                    st.write(f"👤 Authors: {', '.join(result['authors'])}")
                if result["publish_date"]:
                    st.write(f"📅 Published: {result['publish_date']}")
                st.write(f"**Model Used:** {result['model_info']['name']}")

                st.write("### 📝 Summary")
                st.write(result["summary"])

                with st.expander("📄 Original Article Text"):
                    article = summarizer.fetch_article(url)
                    if article:
                        st.write(article.text)


if __name__ == "__main__":
    main()
