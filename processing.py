import os
import tempfile
import trafilatura
from api_clients import (
    fetch_top_news, summarize_text, rate_credibility, 
    summarize_all_articles, extract_perspectives_from_articles,
    generate_followup_questions, transcribe_audio_groq, SERP_API_KEY
)

def extract_article(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded: return None
    return trafilatura.extract(downloaded, include_comments=False)

def run_full_pipeline(query, context=None):
    articles, error = fetch_top_news(query, SERP_API_KEY, num_results=6)
    if error: return {"error": error}
    if not articles: return {"error": "No news found."}

    processed_articles = []
    perspectives_pool = []
    processed_sources = set()

    for art in articles:
        source = art.get("source", {}).get("name", "Unknown")
        if source in processed_sources: continue
        
        text = extract_article(art.get("link"))
        if not text: continue

        summary = summarize_text(text)
        if not summary: continue

        article_data = {
            "source": source,
            "url": art.get("link"),
            "title": art.get("title", ""),
            "summary": summary.strip(),
            "credibility": rate_credibility(source),
            "thumbnail": art.get("thumbnail")
        }

        if len(processed_articles) < 4:
            processed_articles.append(article_data)
        else:
            perspectives_pool.append(article_data)
        processed_sources.add(source)

    combined_summary = summarize_all_articles(processed_articles)
    perspectives = extract_perspectives_from_articles(processed_articles + perspectives_pool)
    followups = generate_followup_questions(combined_summary, context=context)

    return {
        "summary": combined_summary,
        "articles": processed_articles,
        "perspectives": perspectives,
        "followup_questions": followups
    }