# MIS 542 — Spam Email Validation Tool (Web)

Live URL version of the MIS 542 spam-detection project.
Paste any email/SMS into the box → get **SPAM / NOT SPAM** with **confidence %**.

- **Same pipeline** as the original Python script
  (TF-IDF + KNN, Naive Bayes, Logistic Regression, Random Forest, with SMOTE)
- **Best model auto-selected** by F1 score on the spam class (typically Logistic Regression or Random Forest)
- Trained on 5,572 SMS messages from the project dataset

---

## ▶️ Run locally (test before deploying)

```bash
cd spam_web_app
pip install -r requirements.txt
streamlit run app.py
```

Opens automatically at <http://localhost:8501>

---

## 🌐 Deploy to a public URL (free, ~5 minutes)

The professor wanted a **clickable URL link**. Easiest path is **Streamlit Community Cloud** — free, no credit card.

### Step 1 — Push this folder to a new GitHub repo

```bash
cd spam_web_app
git init
git add .
git commit -m "MIS 542 spam detection web app"
git branch -M main
git remote add origin https://github.com/<your-github-username>/mis542-spam-detector.git
git push -u origin main
```

(Create the empty repo on github.com first, then run the commands.)

### Step 2 — Deploy on Streamlit Cloud

1. Go to <https://share.streamlit.io/>
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo: `<your-github-username>/mis542-spam-detector`
5. Branch: `main`  ·  Main file path: `app.py`
6. Click **Deploy**

After ~2 minutes you get a public URL like:

```
https://<your-github-username>-mis542-spam-detector.streamlit.app
```

That's the link to send to the professor. ✅

---

## 📁 Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web interface |
| `requirements.txt` | Python dependencies (Streamlit Cloud auto-installs these) |
| `data/spam_dataset.csv` | The 5,572-message training dataset |
| `.gitignore` | Keeps caches/venv out of the repo |

---

## 👥 Team

Mohammed Sharaf · Mohammed Al-Garni · Abdulaziz Aldekhiel · Rayan Alsurayhi · Ibrahim Abaalkhail

King Fahd University of Petroleum & Minerals — Semester 252 (2026)
