# Run:
# pip install streamlit pandas numpy pyspellchecker scikit-learn rank-bm25 sentence-transformers langchain-community
# python -m streamlit run app_standalone.py

import re
import streamlit as st
import pandas as pd
import numpy as np
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ── QUERY PROCESSOR ───────────────────────────────────────────────
ABBREVIATIONS = pd.read_csv("abbreviations_list.csv")
ABBREVIATIONS = dict(zip(ABBREVIATIONS['Abbreviation'], ABBREVIATIONS['Expansion']))

_spell = SpellChecker()
with open('technical_terms.txt', 'r', encoding='utf-8') as f:
    technical_terms = [line.strip() for line in f if line.strip()]
_spell.word_frequency.load_words(technical_terms)


def preprocess_query(query: str) -> dict:
    original = query.strip()

    is_phrase = False
    working = original
    if (working.startswith('"') and working.endswith('"')) or \
       (working.startswith("'") and working.endswith("'")):
        is_phrase = True
        working = working[1:-1].strip()

    tokens = working.split()
    expanded_tokens = []
    for token in tokens:
        clean = token.strip(".,!?;:()")
        if clean in ABBREVIATIONS:
            expanded_tokens.append(ABBREVIATIONS[clean])
        elif clean.upper() in ABBREVIATIONS:
            expanded_tokens.append(ABBREVIATIONS[clean.upper()])
        else:
            expanded_tokens.append(token)
    working = " ".join(expanded_tokens)

    working = working.lower()

    words = working.split()
    corrections = {}
    corrected_words = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if len(clean_word) <= 2 or clean_word.isdigit():
            corrected_words.append(word)
            continue
        if clean_word in _spell:
            corrected_words.append(word)
        else:
            suggestion = _spell.correction(clean_word)
            if suggestion and suggestion != clean_word:
                corrections[clean_word] = suggestion
                corrected_words.append(word.replace(clean_word, suggestion))
            else:
                corrected_words.append(word)

    processed = " ".join(corrected_words)

    return {
        "original": original,
        "processed": processed,
        "is_phrase": is_phrase,
        "corrections": corrections,
    }


# ── LLM AUGMENTATION (LangChain + Ollama) ────────────────────────

@st.cache_resource
def load_llm():
    """Load the local Ollama LLM via LangChain. Cached so it only loads once."""
    from langchain_community.llms import Ollama
    return Ollama(model="llama3.2:1b")


def rewrite_query(llm, query: str) -> str:
    """
    Query Rewriting: ask the LLM to produce a cleaner, more specific
    version of the user query before it goes into retrieval.
    Returns the rewritten query string.
    """
    prompt = (
        "You are a search query optimizer. "
        "Rewrite the following query to be clearer and more specific for document retrieval. "
        "Return ONLY the rewritten query — no explanation, no quotes, no extra text.\n\n"
        f"Original query: {query}\n"
        "Rewritten query:"
    )
    try:
        result = llm.invoke(prompt).strip()
        # Safety: if the LLM returns something very long or empty, fall back
        if not result or len(result) > 300:
            return query
        return result
    except Exception:
        return query


def summarize_results(llm, query: str, results: list) -> str:
    """
    Result Summarization: after retrieving top-10 documents, ask the LLM
    to write a short, useful summary of what was found.
    Returns the summary string.
    """
    doc_lines = "\n".join(
        f"{r['rank']}. [{r['topic']}] {r['title']} — {r['snippet'][:120].strip()}..."
        for r in results
    )
    prompt = (
        f'A user searched for: "{query}"\n\n'
        f"The top retrieved documents are:\n{doc_lines}\n\n"
        "Write a 2-3 sentence summary of what these results collectively cover. "
        "Be concise and informative. Do not list the documents — synthesize them."
    )
    try:
        return llm.invoke(prompt).strip()
    except Exception as e:
        return f"(Summary unavailable: {e})"


# ── RETRIEVAL MODELS ──────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


class VSMRetriever:
    def __init__(self, documents: list):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            stop_words='english', max_features=50000, sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        print(f"  [VSM] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def retrieve(self, query: str, is_phrase: bool = False, top_k: int = 10):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        if is_phrase:
            phrase_lower = query.lower()
            for i, doc in enumerate(self.documents):
                if phrase_lower in doc.lower():
                    scores[i] *= 2.0
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class BM25Retriever:
    def __init__(self, documents: list):
        self.documents = documents
        self.tokenized_docs = [_tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"  [BM25] Indexed {len(self.tokenized_docs)} documents")

    def retrieve(self, query: str, is_phrase: bool = False, top_k: int = 10):
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        if is_phrase:
            phrase_lower = query.lower()
            for i, doc in enumerate(self.documents):
                if phrase_lower in doc.lower():
                    scores[i] *= 2.0
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class EmbeddingRetriever:
    def __init__(self, documents: list):
        self.documents = documents
        print("  [Embedding] Loading sentence-transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        truncated = [doc[:2000] for doc in documents]
        print(f"  [Embedding] Encoding {len(truncated)} documents...")
        self.doc_embeddings = self.model.encode(
            truncated, show_progress_bar=True, batch_size=64, convert_to_numpy=True,
        )
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.doc_embeddings = self.doc_embeddings / norms
        print(f"  [Embedding] Embeddings shape: {self.doc_embeddings.shape}")

    def retrieve(self, query: str, is_phrase: bool = False, top_k: int = 10):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb)
        scores = np.dot(self.doc_embeddings, query_emb.T).flatten()
        if is_phrase:
            phrase_lower = query.lower()
            for i, doc in enumerate(self.documents):
                if phrase_lower in doc.lower():
                    scores[i] *= 1.5
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class RetrievalSystem:
    def __init__(self, corpus_path: str = "corpus.csv"):
        print("=" * 60)
        print("Initializing Retrieval System")
        print("=" * 60)
        self.df = pd.read_csv(corpus_path)
        self.df['text'] = self.df['text'].fillna('')
        self.df['title'] = self.df['title'].fillna('')
        self.documents = (self.df['title'] + " " + self.df['text']).tolist()
        print(f"\nCorpus loaded: {len(self.documents)} documents")
        print(f"Topics: {self.df['topic'].unique().tolist()}\n")

        print("Initializing VSM (TF-IDF)...")
        self.vsm = VSMRetriever(self.documents)
        print("\nInitializing BM25...")
        self.bm25 = BM25Retriever(self.documents)
        print("\nInitializing Embedding model...")
        self.embedding = EmbeddingRetriever(self.documents)
        print("\n" + "=" * 60)
        print("All models ready!")
        print("=" * 60)

    def search(self, query, model_name="VSM", is_phrase=False, top_k=10):
        model_map = {"VSM": self.vsm, "BM25": self.bm25, "Embedding": self.embedding}
        retriever = model_map.get(model_name, self.vsm)
        raw_results = retriever.retrieve(query, is_phrase=is_phrase, top_k=top_k)

        results = []
        for rank, (doc_idx, score) in enumerate(raw_results, start=1):
            row = self.df.iloc[doc_idx]
            text = str(row['text'])
            snippet = text[:200].strip()
            if len(text) > 200:
                snippet += "..."
            results.append({
                "rank": rank, "doc_index": doc_idx,
                "title": str(row['title']), "url": str(row['url']),
                "snippet": snippet, "score": round(score, 6),
                "topic": str(row['topic']), "content_type": str(row['content_type']),
            })
        return results


# ── STREAMLIT APP ─────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-Powered IR System",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTextInput > div > div > input { border-radius: 20px; }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-title {
        color: #007bff;
        font-size: 1.2rem;
        font-weight: bold;
        text-decoration: none;
    }
    .result-title:hover { text-decoration: underline; }
    .result-snippet { color: #495057; margin: 10px 0; }
    .result-meta { font-size: 0.85rem; color: #6c757d; }
    .score-badge {
        background-color: #e9ceec;
        color: #6f42c1;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
    }
    .topic-badge {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 5px;
    }
    .summary-box {
        background-color: #f0f7ff;
        border-left: 4px solid #0056b3;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 24px;
        font-size: 0.97rem;
        color: #1a1a2e;
    }
    .rewrite-box {
        background-color: #fff8e1;
        border-left: 4px solid #f0a500;
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.93rem;
        color: #4a3800;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    return RetrievalSystem("corpus.csv")


def main():
    # ── Sidebar ──────────────────────────────────────────────────
    st.sidebar.title("⚙️ IR System Settings")

    with st.spinner("Initializing system and loading models..."):
        system = load_system()

    st.sidebar.success("System Ready!")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Corpus Statistics")
    st.sidebar.write(f"Total Documents: {len(system.df)}")
    st.sidebar.write(f"Topics: {len(system.df['topic'].unique())}")
    st.sidebar.bar_chart(system.df['topic'].value_counts())

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 LLM Augmentation")

    # Toggle for Query Rewriting
    use_rewriting = st.sidebar.toggle(
        "Query Rewriting",
        value=False,
        help="Send the query to a local LLM (Ollama) to rewrite it into a cleaner, "
             "more specific form before retrieval."
    )

    # Result summarization is always available (shown after results)
    show_summary = st.sidebar.toggle(
        "Result Summarization",
        value=False,
        help="After retrieval, ask the LLM to generate a short synthesis of the top-10 results."
    )

    if use_rewriting or show_summary:
        st.sidebar.caption("Using local Ollama · llama3.2:1b")

    # ── Main ─────────────────────────────────────────────────────
    st.title("🚀 Advanced Information Retrieval System")
    st.markdown("Search through 500+ heterogeneous documents using VSM, BM25, or Neural Embeddings.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_input = st.text_input(
            "",
            placeholder="Enter your search query here (e.g., 'NLP transformers' or '\"quantum computing\"')"
        )
    with col2:
        model_choice = st.selectbox("Retrieval Model", ["VSM", "BM25", "Embedding"])

    if query_input:
        # Step 1 — Standard preprocessing (spell correction, abbreviation expansion)
        processed_data = preprocess_query(query_input)
        query_for_retrieval = processed_data['processed']

        if processed_data['corrections']:
            correction_str = ", ".join(
                [f"'{k}' → '{v}'" for k, v in processed_data['corrections'].items()]
            )
            st.info(f"✨ Auto-corrected: {correction_str}")

        if processed_data['is_phrase']:
            st.caption("🔍 Performing exact phrase matching boost.")

        # Step 2 — Optional LLM Query Rewriting
        if use_rewriting:
            with st.spinner("🤖 Rewriting query with LLM..."):
                llm = load_llm()
                rewritten = rewrite_query(llm, query_for_retrieval)

            if rewritten.lower() != query_for_retrieval.lower():
                st.markdown(
                    f'<div class="rewrite-box">🔄 <b>LLM rewrote your query:</b> '
                    f'<i>"{query_for_retrieval}"</i> → <b>"{rewritten}"</b></div>',
                    unsafe_allow_html=True
                )
                query_for_retrieval = rewritten
            else:
                st.caption("🤖 LLM kept the query unchanged.")

        # Step 3 — Retrieval
        with st.spinner(f"Retrieving results using {model_choice}..."):
            results = system.search(
                query=query_for_retrieval,
                model_name=model_choice,
                is_phrase=processed_data['is_phrase'],
                top_k=10
            )

        st.subheader(f"Top 10 Results for: '{query_for_retrieval}'")

        if not results:
            st.warning("No documents found for this query.")
            return

        # Step 4 — Optional LLM Result Summarization
        if show_summary:
            with st.spinner("🤖 Summarizing results with LLM..."):
                llm = load_llm()
                summary = summarize_results(llm, query_for_retrieval, results)
            st.markdown(
                f'<div class="summary-box">📝 <b>AI Summary:</b> {summary}</div>',
                unsafe_allow_html=True
            )

        # Step 5 — Display results
        for res in results:
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <a href="{res['url']}" class="result-title" target="_blank">{res['title']}</a>
                    <div class="result-snippet">{res['snippet']}</div>
                    <div class="result-meta">
                        <span class="topic-badge">{res['topic']}</span>
                        <span class="topic-badge">{res['content_type']}</span>
                        Rank: <b>#{res['rank']}</b> |
                        Score: <span class="score-badge">{res['score']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("👋 Enter a query above to start searching!")
        st.subheader("Capabilities:")
        cols = st.columns(3)
        cols[0].markdown(
            "### 🛠️ Processing\n"
            "- Case Insensitivity\n"
            "- Abbreviation Expansion\n"
            "- Spelling Correction\n"
            "- Phrase Matching"
        )
        cols[1].markdown(
            "### 🧠 Models\n"
            "- **VSM**: Classic TF-IDF\n"
            "- **BM25**: Probabilistic Ranking\n"
            "- **Embeddings**: Semantic Search"
        )
        cols[2].markdown(
            "### 🤖 LLM Augmentation\n"
            "- **Query Rewriting**: LLM clarifies query\n"
            "- **Result Summarization**: LLM synthesizes top-10"
        )


if __name__ == "__main__":
    main()
