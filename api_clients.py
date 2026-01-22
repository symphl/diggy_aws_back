import os
import requests
import json
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
SERP_API_KEY = os.getenv("SERP_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

def fetch_top_news(query, serp_api_key, num_results=6):
    """Fetches top news articles using SerpApi."""
    print(f"Attempting to fetch news for query: '{query}'...")
    # Append major news sources to query to avoid local/irrelevant results
    # Using 'site:' operator within Google News search to prioritize top global outlets
    # Excluded hard paywalls (WSJ, Bloomberg, NYT) to improve extraction success
    trusted_sites = (
        "site:bbc.com OR site:cnn.com OR site:reuters.com OR site:theguardian.com OR "
        "site:cnbc.com OR site:apnews.com OR site:aljazeera.com OR site:npr.org OR "
        "site:cbsnews.com OR site:abcnews.go.com OR site:nbcnews.com OR site:usatoday.com OR "
        "site:politico.com OR site:foxnews.com"
    )
    refined_query = f"{query} ({trusted_sites})"

    params = {
        "engine": "google_news",
        "q": refined_query,
        "gl": "us",
        "hl": "en",
        "api_key": serp_api_key
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        if "news_results" in results and results["news_results"]:
            print("SerpApi news fetch: successful")
            return results["news_results"][:num_results], None
        else:
            # Fallback: if restricted search fails, try original query without site filters
            print("Restricted search yielded no results. Retrying with original query...")
            params["q"] = query
            search = GoogleSearch(params)
            results = search.get_dict()
            if "news_results" in results and results["news_results"]:
                print("SerpApi fallback news fetch: successful")
                return results["news_results"][:num_results], None
            else:
                error_message = results.get("error", "No news_results found.")
                print(f"SerpApi news fetch: fail. Error: {error_message}")
                return [], error_message
    except Exception as e:
        print(f"SerpApi news fetch: fail. An exception occurred: {e}")
        return [], str(e)

def debug_groq_request(payload, timeout=30):
    """
    Send payload to Groq endpoint, print debug info on failure and return parsed parsed JSON on success.
    """
    if not GROQ_API_KEY:
        print("debug_groq_request: GROQ_API_KEY not set.")
        return None

    try:
        r = requests.post(GROQ_URL, headers=GROQ_HEADERS, json=payload, timeout=timeout)
    except Exception as e:
        print("debug_groq_request: network error:", e)
        return None

    # Always log status & a short preview of the response for debugging
    print(f"Groq response status: {r.status_code}")
    # Try to show response text (trim to avoid console blowup)
    resp_text_preview = (r.text[:2000] + "...") if len(r.text) > 2000 else r.text
    print("Groq response preview:", resp_text_preview)

    if r.status_code != 200:
        # Return None on non-200 so caller can fallback
        return None

    try:
        return r.json()
    except Exception as e:
        print("debug_groq_request: failed to parse json:", e)
        return None


def summarize_text(text, model="llama-3.1-8b-instant"):
    """
    Safer Groq summarizer:
    - smaller model
    - truncate article text aggressively
    - debug logged responses
    - fallback to None if Groq fails
    """
    if not GROQ_API_KEY:
        print("Missing GROQ_API_KEY")
        return None

    # Aggressive truncation to avoid token/context problems
    safe_text = text.strip().replace("\n", " ")[:3000]

    prompt = (
        "You are a neutral news summarizer. Provide a concise factual summary under 80 words. "
        "Include main event, key people involved, and outcome. No opinions.\n\n"
        f"Article:\n{safe_text}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a short, factual news summarizer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.15,
        "max_tokens": 250
    }

    resp = debug_groq_request(payload, timeout=25)
    if not resp:
        print("summarize_text: Groq returned error (see above).")
        return None

    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("summarize_text: cannot read choice:", e)
        return None

def rate_credibility(source, model="llama-3.1-8b-instant"):
    if not GROQ_API_KEY:
        return "N/A"

    prompt = f"Rate the credibility (0-100) of news source '{source}'. Return only the number."

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 6
    }

    resp = debug_groq_request(payload, timeout=10)
    if not resp:
        return "N/A"

    try:
        raw = resp["choices"][0]["message"]["content"].strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        return digits or raw
    except:
        return "N/A"


def summarize_all_articles(articles, model="llama-3.1-8b-instant"):
    if not articles:
        return None

    # Use short per-article summaries where possible; enforce short length
    snippets = []
    for a in articles:
        s = (a.get("summary") or "")[:600]
        if s:
            snippets.append(s)
    combined = "\n\n".join(snippets)[:4000]

    prompt = (
        "Synthesize these short article summaries into a structured news summary following this exact format:\n\n"
        "1. A contextual introduction paragraph setting the scene for about 100 words.\n"
        "2. 3-4 bullet points highlighting the most important details.\n"
        "3. A concluding paragraph summarizing the overall implication.\n\n"
        "IMPORTANT: Do NOT use headings like 'Contextual Intro' or 'Key Points'. Just provide the text directly.\n\n"
        f"{combined}"
    )

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": "You are an unbiased news synthesizer. You provide clean, header-free summaries."},
                     {"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 600
    }

    resp = debug_groq_request(payload, timeout=30)
    if not resp:
        print("summarize_all_articles: Groq error.")
        return None

    try:
        return resp["choices"][0]["message"]["content"].strip()
    except:
        return None


def generate_followup_questions(combined_summary, n_questions=5, model="llama-3.1-8b-instant", context=None):
    if not combined_summary:
        return []

    safe_summary = combined_summary[:2000]
    
    prompt = ""
    if context:
        prompt += f"Previous context:\n{context}\n\n"

    prompt += f"Suggest {n_questions} short follow-up questions (why/how/what-if) about this summary:\n\n{safe_summary}"

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": "You propose curiosity-driven questions."},
                     {"role": "user", "content": prompt}],
        "temperature": 0.35,
        "max_tokens": 200
    }

    resp = debug_groq_request(payload, timeout=20)
    if not resp:
        return []

    try:
        text = resp["choices"][0]["message"]["content"]
        questions = [q.strip("•-1234567890. ") for q in text.split("\n") if q.strip()]
        return questions[:n_questions]
    except:
        return []

def extract_event_location(text, model="llama-3.1-8b-instant"):
    """
    Extracts the primary real-world location of an event from a block of text.
    """
    if not GROQ_API_KEY:
        print("Missing GROQ_API_KEY")
        return None

    # Reduce text size to keep the prompt focused and save tokens
    safe_text = text.strip().replace("\n", " ")[:2000]

    prompt = (
        "From the following text, identify the primary real-world location (city, state, country) of the main event described. "
        "Return only the location name. If no specific location is mentioned, return 'N/A'.\n\n"
        f"Text: {safe_text}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an AI assistant that extracts locations from text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 60
    }

    resp = debug_groq_request(payload, timeout=20)
    if not resp:
        print("extract_event_location: Groq returned error.")
        return None

    try:
        location = resp["choices"][0]["message"]["content"].strip()
        return location if location.upper() != 'N/A' else None
    except Exception as e:
        print(f"extract_event_location: cannot read choice: {e}")
        return None


def test_groq_connection():
    print("Testing Groq connectivity...")
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Say hello in one sentence." }],
        "max_tokens": 20,
        "temperature": 0.0
    }
    resp = debug_groq_request(payload, timeout=10)
    if resp and "choices" in resp:
        print("Groq test OK:", resp["choices"][0]["message"]["content"])
    else:
        print("Groq test FAILED. See debug output above.")



def extract_keywords(text, model="llama-3.1-8b-instant"):
    """
    Extracts keywords from a block of text using the Groq API.
    """
    if not GROQ_API_KEY:
        print("Missing GROQ_API_KEY")
        return None

    prompt = (
        "You are an AI assistant that extracts the most important keywords from a block of text. "
        "Please provide the top 3-5 keywords, separated by commas.\n\n"
        f"Text: {text}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an AI assistant that extracts keywords from text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }

    resp = debug_groq_request(payload, timeout=20)
    if not resp:
        print("extract_keywords: Groq returned error (see above).")
        return None

    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"extract_keywords: cannot read choice: {e}")
        return None

def describe_image(image_base64, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    if not GROQ_API_KEY:
        print("describe_image: Missing GROQ_API_KEY")
        return "Error: GROQ_API_KEY not set."

    # Build the message payload the Groq API expects (image + text prompt)
    prompt_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "The uploaded image contains a real news scene. Describe the scene concisely, focusing on objective details and likely context."
            },
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    }

    payload = {
        "model": model,
        "messages": [prompt_message],
        "temperature": 0.4,
        "max_tokens": 400
    }

    # send request and debug/log the response
    resp = debug_groq_request(payload, timeout=30)
    if not resp:
        print("describe_image: Groq returned error (see debug output above).")
        return None

    try:
        content = resp["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"describe_image: failed to parse Groq response: {e}")
        return None


# --- NEW FUNCTIONS: perspective extraction & follow-up answering ---

def extract_perspectives_from_articles(articles, model="llama-3.1-8b-instant"):
    """
    Given a list of articles (dicts with 'source','summary','url','title' optional),
    ask the LLM to enumerate distinct societal perspectives present in the reporting,
    and provide a 3-4 line concise summary for each. If a perspective appears in an article,
    include the article link.
    Returns: list of dicts: { "perspective": str, "summary": str, "articles": [url,...] }
    """
    if not GROQ_API_KEY or not articles:
        return []

    # Build a compact input combining source + summary + url
    snippets = []
    for a in articles:
        title = a.get("title") or ""
        src = a.get("source") or ""
        summary = a.get("summary") or ""
        url = a.get("url") or ""
        snippets.append(f"Source: {src}\nTitle: {title}\nSummary: {summary}\nURL: {url}")

    prompt = (
        "You are a neutral analyst. From the following list of news article summaries, "
        "identify the distinct **societal perspectives** (e.g., public safety, economic impact, "
        "humanitarian concerns, legal/constitutional issues, technological implications, "
        "ethnic/religious tension, labor/workforce stress). For each perspective, output:\n\n"
        "1) Perspective name (one short phrase)\n"
        "2) 3-4 line concise summary explaining 'what is this perspective' in the context of the news\n"
        "3) An 'interesting_fact' related to this specific perspective (a statistic, historical precedent, or surprising detail)\n"
        "4) If any of the articles mention or support this perspective, include their URLs in a list.\n\n"
        "Return valid JSON only. The output must be a JSON array of objects with keys: perspective, summary, interesting_fact, articles (array of strings). "
        "Do not include markdown formatting, code blocks, or conversational text.\n\n"
        "Articles:\n\n" + "\n\n---\n\n".join(snippets)
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert news analyst that extracts societal perspectives."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.15,
        "max_tokens": 800
    }

    resp = debug_groq_request(payload, timeout=30)
    if not resp:
        return []

    try:
        raw = resp["choices"][0]["message"]["content"].strip()
        
        # Robust JSON extraction: look for the outer-most brackets
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1:
            raw = raw[start:end+1]

        # Try to parse JSON from LLM output; if it isn't strict JSON, attempt a best-effort parse.
        import json
        try:
            parsed = json.loads(raw)
            # Ensure expected structure
            out = []
            for p in parsed:
                out.append({
                    "perspective": p.get("perspective") or p.get("name") or "",
                    "summary": p.get("summary") or "",
                    "interesting_fact": p.get("interesting_fact") or "",
                    "articles": p.get("articles") or []
                })
            return out
        except Exception:
            # Fallback: return the raw text, and attribute all articles to it so it isn't hidden
            all_urls = [a.get("url") for a in articles if a.get("url")]
            return [{"perspective": "Analysis", "summary": raw, "interesting_fact": "", "articles": all_urls}]
    except Exception:
        return []


def answer_followup(question, context=None, model="llama-3.1-8b-instant"):
    """
    Answer a follow-up question using the saved context (combined summary or article snippets).
    Returns a concise answer (3-5 sentences).
    """
    if not GROQ_API_KEY:
        return "N/A (GROQ key not set)"

    prompt = "Answer the question concisely (3-4 lines)."
    if context:
        # keep context reasonably sized
        safe_ctx = context[:3000]
        prompt += f"\n\nContext:\n{safe_ctx}"
    prompt += f"\n\nQuestion: {question}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You answer follow-up news questions concisely and factually."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 300
    }

    resp = debug_groq_request(payload, timeout=20)
    if not resp:
        return "Error: Could not generate answer."

    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Error: Failed to parse answer."
