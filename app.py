"""
================================================================================
MIS 542 — Spam Detection Web App
King Fahd University of Petroleum & Minerals  |  Semester 252 (2026)

Team: Mohammed Sharaf, Mohammed Al-Garni, Abdulaziz Aldekhiel,
      Rayan Alsurayhi, Ibrahim Abaalkhail

This is the WEB version of the project's spam detector. It uses the same
pipeline as the original Python script (TF-IDF + 4 classifiers + SMOTE) but
exposes it through a public URL so the instructor can paste any email and
get a SPAM / NOT SPAM verdict with confidence %.

Run locally:        streamlit run app.py
Deploy to web:      push to GitHub -> share.streamlit.io
================================================================================
"""

import re
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from imblearn.over_sampling import SMOTE


# -----------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MIS 542 — Spam Email Validator",
    page_icon="📧",
    layout="centered",
)


# -----------------------------------------------------------------------------
# Text preprocessing — identical to the original project script
# -----------------------------------------------------------------------------
STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","you're","you've",
    "you'll","you'd","your","yours","yourself","yourselves","he","him","his",
    "himself","she","she's","her","hers","herself","it","it's","its","itself",
    "they","them","their","theirs","themselves","what","which","who","whom",
    "this","that","that'll","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing","a",
    "an","the","and","but","if","or","because","as","until","while","of","at",
    "by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more","most","other",
    "some","such","no","nor","not","only","own","same","so","than","too","very",
    "s","t","can","will","just","don","don't","should","should've","now","d",
    "ll","m","o","re","ve","y","ain","aren","aren't","couldn","couldn't",
    "didn","didn't","doesn","doesn't","hadn","hadn't","hasn","hasn't","haven",
    "haven't","isn","isn't","ma","mightn","mightn't","mustn","mustn't","needn",
    "needn't","shan","shan't","shouldn","shouldn't","wasn","wasn't","weren",
    "weren't","won","won't","wouldn","wouldn't"
}
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    """Same six-step cleaning as the original script."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)        # hyperlinks
    text = re.sub(r"\d+", " ", text)                      # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)                 # special chars
    text = re.sub(r"\s+", " ", text).strip()              # whitespace
    tokens = [
        STEMMER.stem(tok)
        for tok in text.split()
        if tok not in STOP_WORDS and len(tok) > 1
    ]
    return " ".join(tokens)


# -----------------------------------------------------------------------------
# Train pipeline — cached so it runs ONCE per app session (~10–20 s on first load)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training spam detection models on 5,572 messages…")
def train_pipeline():
    df_raw = pd.read_csv("data/spam_dataset.csv")
    df = df_raw[["class", "message"]].copy()
    df = df.dropna(subset=["message"]).reset_index(drop=True)
    df["clean_message"] = df["message"].apply(clean_text)
    df["label"] = (df["class"] == "spam").astype(int)

    # IMPORTANT: pass X as a plain Python list of strings.
    # scikit-learn 1.7+ rejects object-dtype numpy arrays in train_test_split.
    X_text = df["clean_message"].astype(str).tolist()
    y = df["label"].to_numpy()

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    candidates = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, mdl in candidates.items():
        mdl.fit(X_train_sm, y_train_sm)
        y_pred = mdl.predict(X_val)
        results[name] = {
            "model":     mdl,
            "accuracy":  accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall":    recall_score(y_val, y_pred, zero_division=0),
            "f1":        f1_score(y_val, y_pred, zero_division=0),
        }

    # Pick the model with the highest F1 on the spam class as the deployed one
    best_name = max(results, key=lambda n: results[n]["f1"])

    return {
        "vectorizer":  vectorizer,
        "results":     results,
        "best_name":   best_name,
        "n_total":     len(df),
        "n_spam":      int(df["label"].sum()),
        "n_valid":     int((df["label"] == 0).sum()),
    }


def predict_message(raw_msg: str, pipeline):
    cleaned = clean_text(raw_msg)
    if not cleaned:
        return None
    vec = pipeline["vectorizer"].transform([cleaned])
    model = pipeline["results"][pipeline["best_name"]]["model"]
    label = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    return {
        "label":      "SPAM" if label == 1 else "NOT SPAM",
        "is_spam":    label == 1,
        "p_spam":     float(proba[1]),
        "p_valid":    float(proba[0]),
        "confidence": float(proba[label]),
        "cleaned":    cleaned,
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
pipeline = train_pipeline()
best = pipeline["results"][pipeline["best_name"]]

st.title("📧 Spam Email Validation Tool")
st.caption(
    "MIS 542 — Data Mining for Business Analytics  |  "
    "King Fahd University of Petroleum & Minerals  |  Semester 252"
)

# Top metrics row — instructor sees model + accuracy at a glance
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model in use",   pipeline["best_name"])
c2.metric("Accuracy",       f"{best['accuracy']*100:.2f}%")
c3.metric("Precision (spam)", f"{best['precision']*100:.2f}%")
c4.metric("Recall (spam)",  f"{best['recall']*100:.2f}%")

st.divider()

st.subheader("Paste an email or SMS message below")
default_text = ""
email_text = st.text_area(
    "Message",
    value=default_text,
    height=200,
    placeholder="Paste the full email body here…",
    label_visibility="collapsed",
)

col_a, col_b = st.columns([1, 1])
classify_clicked = col_a.button("🔎 Classify message", type="primary", use_container_width=True)
clear_clicked    = col_b.button("🧹 Clear",            use_container_width=True)

if clear_clicked:
    st.rerun()

if classify_clicked:
    if not email_text.strip():
        st.warning("Please paste a message first.")
    else:
        result = predict_message(email_text, pipeline)
        if result is None:
            st.warning(
                "After cleaning, the message has no usable words "
                "(only stop-words / numbers / links). Try a longer message."
            )
        else:
            if result["is_spam"]:
                st.error(
                    f"## 🚫 SPAM\n"
                    f"### Confidence: **{result['p_spam']*100:.2f}%**"
                )
            else:
                st.success(
                    f"## ✅ NOT SPAM (Valid)\n"
                    f"### Confidence: **{result['p_valid']*100:.2f}%**"
                )

            st.progress(result["confidence"], text=f"Model confidence: {result['confidence']*100:.2f}%")

            with st.expander("🔬 Show technical details"):
                st.markdown(f"**Detection model:** {pipeline['best_name']}")
                st.markdown(
                    f"**P(spam)** = {result['p_spam']*100:.2f}%  &nbsp;|&nbsp;  "
                    f"**P(valid)** = {result['p_valid']*100:.2f}%"
                )
                st.markdown("**Cleaned message (after preprocessing):**")
                st.code(result["cleaned"] or "(empty)", language="text")

st.divider()

with st.expander("📊 All four models — performance on validation set (after SMOTE)"):
    rows = []
    for name, r in pipeline["results"].items():
        rows.append({
            "Model":          name,
            "Accuracy":       f"{r['accuracy']*100:.2f}%",
            "Precision (spam)": f"{r['precision']*100:.2f}%",
            "Recall (spam)":   f"{r['recall']*100:.2f}%",
            "F1 (spam)":       f"{r['f1']*100:.2f}%",
        })
    perf_df = pd.DataFrame(rows)
    st.dataframe(perf_df, hide_index=True, use_container_width=True)
    st.caption(
        f"Trained on {pipeline['n_total']:,} messages "
        f"({pipeline['n_valid']:,} valid + {pipeline['n_spam']:,} spam). "
        f"80/20 stratified split. SMOTE applied to the training side. "
        f"Deployed model: **{pipeline['best_name']}** (highest F1 on spam class)."
    )

with st.expander("🧪 Two ready-made examples (from the project report)"):
    ex1 = "Hi mate, are we still meeting at the cafeteria for lunch at 1 pm?"
    ex2 = ("CONGRATULATIONS! You have WON a $1000 Walmart gift card. "
           "Click http://bit.ly/claim-now to claim before midnight. "
           "Reply STOP to opt out.")
    st.markdown("**Example 1 (expected: NOT SPAM):**")
    st.code(ex1, language="text")
    st.markdown("**Example 2 (expected: SPAM):**")
    st.code(ex2, language="text")
    st.caption("Copy either example into the box above and click *Classify message*.")

st.caption("Built with Streamlit · scikit-learn · imbalanced-learn · NLTK")

st.markdown(
    """
    <div style="text-align:center; margin-top:30px; padding:20px;
                border-top:2px solid #1F4E79;">
        <h3 style="margin-bottom:14px; letter-spacing:2px;"><b>TEAM 8</b></h3>
        <p style="font-size:16px; line-height:1.9; margin:0;">
            <b>MOHAMMED SHARAF</b><br>
            <b>MOHAMMED AL-GARNI</b><br>
            <b>ABDULAZIZ ALDEKHIEL</b><br>
            <b>RAYAN ALSURAYHI</b><br>
            <b>IBRAHIM ABAALKHAIL</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
