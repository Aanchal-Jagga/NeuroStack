# comment_fetcher.py
import os
import re
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def fetch_youtube_comments(video_url):
    """
    Fetch all top-level comments from a YouTube video.
    Returns a list of comment texts.
    """
    # Extract video ID
    video_id_match = re.search(r"(?:v=|youtu\.be/)([\w-]+)", video_url)
    if not video_id_match:
        print("Invalid YouTube URL")
        return []
    video_id = video_id_match.group(1)

    if not YOUTUBE_API_KEY:
        print("Missing YOUTUBE_API_KEY")
        return []

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        comments = []

        # Initial request
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # max allowed per request
            textFormat="plainText"
        )

        while request:
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            # Get next page if exists
            request = youtube.commentThreads().list_next(request, response)

        return comments
    except Exception as e:
        print(f"Error fetching YouTube comments: {e}")
        return []

def fetch_amazon_reviews(product_url, max_results=50):
    """
    Fetch Amazon product reviews.
    Returns a list of review texts (top max_results only).
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(product_url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        reviews = [rev.get_text().strip() for rev in soup.find_all("span", {"data-hook": "review-body"})]
        return reviews[:max_results]
    except Exception as e:
        print(f"Error fetching Amazon reviews: {e}")
        return []

def fetch_comments(url):
    """
    Unified function to fetch comments/reviews from YouTube or Amazon.
    Returns a list of strings.
    """
    if "youtube.com" in url or "youtu.be" in url:
        return fetch_youtube_comments(url)
    elif "amazon." in url:
        return fetch_amazon_reviews(url)
    else:
        print("URL not supported for comment fetching")
        return []
