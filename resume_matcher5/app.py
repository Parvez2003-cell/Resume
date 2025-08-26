# app.py
import streamlit as st
import sqlite3, hashlib, os, io, re
from docx import Document
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ---------------------------
#  Config / Styling
# ---------------------------
st.set_page_config(page_title="Resume Parser & Job Matcher", layout="wide")

st.markdown("""
<style>
/* Google-like clean layout */
body { font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial; }
.header {
  display:flex; align-items:center; gap:16px;
}
.title { font-size:28px; font-weight:700; }
.subtitle { color: #6c757d; margin-top:-6px; font-size:13px; }

.card {
  background: #fff; border-radius:12px; padding:20px; box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

.skill-badge {
  display:inline-block; margin:4px 6px; padding:6px 10px; background:#f1f5f9; border-radius:999px; color:#0f172a; font-weight:600;
}

.small-muted { color:#6c757d; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
#  Database (users)
# ---------------------------
DB_PATH = "users.db"
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                 )""")
    conn.commit()
    return conn

conn = init_db()

# ---------------------------
#  Password hashing helpers
# ---------------------------
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + pwdhash.hex()

def verify_password(stored: str, provided: str) -> bool:
    if not stored or len(stored) < 32:
        return False
    salt_hex = stored[:32]
    stored_hash = stored[32:]
    new_hash = hashlib.pbkdf2_hmac('sha256', provided.encode('utf-8'), bytes.fromhex(salt_hex), 100000).hex()
    return new_hash == stored_hash

# ---------------------------
#  Auth helpers
# ---------------------------
def register_user(username: str, password: str) -> (bool, str):
    try:
        hashed = hash_password(password)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True, "Registered successfully."
    except Exception as e:
        return False, f"Error: {str(e)}"

def authenticate_user(username: str, password: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if not row:
        return False
    return verify_password(row[0], password)
     


# ---------------------------
#  Resume parsing helpers
# ---------------------------
def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        text = []
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        return ""

def extract_text_from_docx_bytes(b: bytes) -> str:
    try:
        doc = Document(io.BytesIO(b))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        return ""

def parse_resume(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    content = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(content)
    elif name.endswith(".docx"):
        return extract_text_from_docx_bytes(content)
    else:
        # treat uploaded as plain text if any
        try:
            return content.decode('utf-8')
        except:
            return ""

# ---------------------------
#  Skill extraction
# ---------------------------
SKILLS_DB = [
    "python","java","c++","c","c#","javascript","typescript","react","angular","vue",
    "node","express","django","flask","sql","mysql","postgresql","mongodb","redis",
    "aws","azure","gcp","docker","kubernetes","git","html","css","bootstrap",
    "tensorflow","keras","pytorch","scikit-learn","pandas","numpy","data analysis",
    "machine learning","deep learning","nlp","computer vision","rest api",
    "testing","selenium","agile","leadership","communication","problem solving",
    "devops","ci/cd","spark","hadoop","spark","matlab","r"
]

def extract_skills(text: str) -> dict:
    text_low = text.lower()
    skills_count = {}
    for skill in SKILLS_DB:
        # match whole words or common forms
        pattern = r'\b' + re.escape(skill) + r'\b'
        matches = re.findall(pattern, text_low)
        if matches:
            skills_count[skill] = len(matches)
    # sort descending by count
    sorted_skills = dict(sorted(skills_count.items(), key=lambda x: x[1], reverse=True))
    return sorted_skills

# ---------------------------
#  Summarization (simple extractive)
# ---------------------------
def generate_summary(text: str, max_sentences: int = 3) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return "No text found in resume."
    # split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences[:max_sentences])
    try:
        # TF-IDF on sentences
        vect = TfidfVectorizer(stop_words='english')
        X = vect.fit_transform(sentences)
        # score sentences by sum of TF-IDF weights
        scores = X.sum(axis=1).A1
        # pick top indices
        top_idx = scores.argsort()[-max_sentences:][::-1]
        # preserve original order
        top_idx_sorted = sorted(top_idx)
        summary = " ".join([sentences[i] for i in top_idx_sorted])
        return summary
    except Exception:
        # fallback: first few sentences
        return " ".join(sentences[:max_sentences])

# ---------------------------
#  Job matching logic
# ---------------------------
# Example job database (you can expand)
JOBS_DB = [
    {
        "title": "Data Scientist",
        "description": "Build ML models, analyze data, use Python, pandas, numpy, scikit-learn, SQL, deep learning frameworks.",
        "skills": ["python","pandas","numpy","scikit-learn","sql","machine learning","deep learning","tensorflow","keras"]
    },
    {
        "title": "Backend Developer (Python)",
        "description": "Develop REST APIs using Django or Flask. Work with PostgreSQL, Docker, AWS and CI/CD.",
        "skills": ["python","django","flask","rest api","postgresql","docker","aws","ci/cd"]
    },
    {
        "title": "Frontend Developer",
        "description": "Create responsive UIs using JavaScript, React, HTML, CSS. Optimize performance and accessibility.",
        "skills": ["javascript","react","html","css","bootstrap"]
    },
    {
        "title": "DevOps Engineer",
        "description": "Maintain CI/CD pipelines, Docker, Kubernetes, cloud infra (AWS/Azure/GCP), monitoring and automation.",
        "skills": ["docker","kubernetes","aws","azure","gcp","ci/cd","devops"]
    },
    {
        "title": "Business Analyst",
        "description": "Analyze business requirements, SQL, data visualization, stakeholder communication, reporting.",
        "skills": ["sql","data analysis","communication","excel","reporting"]
    }
]

def compute_text_similarity(target_text: str, other_text: str) -> float:
    try:
        vect = TfidfVectorizer(stop_words='english')
        tf = vect.fit_transform([target_text, other_text])
        sim = cosine_similarity(tf[0], tf[1])[0][0]
        return float(sim)
    except Exception:
        return 0.0

def match_jobs(text: str, skills_dict: dict) -> list:
    matches = []
    resume_skills = set(skills_dict.keys())
    for job in JOBS_DB:
        req_skills = set(job["skills"])
        matched = list(resume_skills.intersection(req_skills))
        missing = list(req_skills.difference(resume_skills))
        # skill overlap score
        if len(req_skills) == 0:
            skill_overlap = 0.0
        else:
            skill_overlap = len(matched) / len(req_skills)
        # textual similarity
        txt_sim = compute_text_similarity(text, job["description"])
        # combined fit (weights can be tuned)
        fit = 0.7 * skill_overlap + 0.3 * txt_sim
        score = int(round(fit * 100))
        explanation = f"Matched: {', '.join(matched) if matched else 'None'}. Missing: {', '.join(missing) if missing else 'None'}. Text similarity: {txt_sim:.2f}"
        matches.append({
            "title": job["title"],
            "score": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "explanation": explanation
        })
    # sort descending
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches

# ---------------------------
#  Radar chart helper
# ---------------------------
def plot_skill_radar(skills_dict: dict, top_n: int = 6):
    if not skills_dict:
        st.info("No skills found to plot.")
        return
    items = list(skills_dict.items())[:top_n]
    labels = [k.title() for k, v in items]
    counts = [v for k, v in items]
    max_count = max(counts) if counts else 1
    # normalize to 0-100
    values = [int((c / max_count) * 100) for c in counts]
    # close radar
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='Skill Strength'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100])
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
#  Streamlit App Layout
# ---------------------------
def main():
    # header
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="header"><div class="title">Resume Parser & Job Matcher</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Upload your resume (PDF/DOCX) and get a summary, skill radar and job-fit explanation.</div>', unsafe_allow_html=True)
    with col2:
        st.empty()

    # sidebar for login / register
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    st.sidebar.markdown("## Account")
    if not st.session_state.logged_in:
        mode = st.sidebar.radio("Choose:", ["Login","Register"])
        if mode == "Register":
            new_user = st.sidebar.text_input("Username", key="reg_user")
            new_pwd = st.sidebar.text_input("Password", type="password", key="reg_pwd")
            if st.sidebar.button("Create account"):
                if not new_user or not new_pwd:
                    st.sidebar.error("Enter username and password.")
                else:
                    ok, msg = register_user(new_user.strip(), new_pwd)
                    if ok:
                        st.sidebar.success(msg + " You can log in now.")
                    else:
                        st.sidebar.error(msg)
        else:
            user = st.sidebar.text_input("Username", key="login_user")
            pwd = st.sidebar.text_input("Password", type="password", key="login_pwd")
            if st.sidebar.button("Log in"):
                if authenticate_user(user.strip(), pwd):
                    st.session_state.logged_in = True
                    st.session_state.username = user.strip()
                    st.sidebar.success(f"Logged in as {st.session_state.username}")
                else:
                    st.sidebar.error("Invalid credentials.")
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.experimental_rerun()

    st.markdown("---")

    if not st.session_state.logged_in:
        st.info("Please log in or register from the left panel to use the app.")
        return

    # App main area: upload resume
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload Resume")
        st.markdown('<div class="small-muted">Supported: .pdf and .docx</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose your resume file", type=["pdf","docx"])
        analyze_btn = st.button("Analyze Resume")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and analyze_btn:
        # Parse, extract skills, summarize, match jobs — show spinners for UX
        with st.spinner("Parsing resume..."):
            text = parse_resume(uploaded_file)
        if not text or len(text.strip()) < 30:
            st.warning("Couldn't extract text from resume — make sure it's a valid PDF or DOCX with selectable text.")
            return

        with st.spinner("Extracting skills..."):
            skills = extract_skills(text)  # dict skill->count

        with st.spinner("Generating summary..."):
            summary = generate_summary(text)

        with st.spinner("Matching to jobs..."):
            matches = match_jobs(text, skills)

        # Layout: Summary + Skills + Radar + Matches
        left, right = st.columns([2,1])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Auto-generated Summary")
            st.write(summary)
            st.markdown("*Top resume lines (preview):*")
            st.text_area("Resume Preview", value="\n".join(text.strip().splitlines()[:20]), height=180)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
            st.subheader("Job Matches")
            for m in matches:
                st.markdown(f"{m['title']}** — *{m['score']}% fit*")
                st.markdown(f"<div class='small-muted'>{m['explanation']}</div>", unsafe_allow_html=True)
                st.progress(m['score'])
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Skills")
            if skills:
                for s, c in skills.items():
                    st.markdown(f"<span class='skill-badge'>{s.title()} ({c})</span>", unsafe_allow_html=True)
            else:
                st.info("No recognized skills found.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
            st.subheader("Skill Radar")
            plot_skill_radar(skills)
            st.markdown('</div>', unsafe_allow_html=True)

    elif not uploaded_file:
        st.info("Upload a resume to begin analysis.")

if __name__ == "__main__":
    main()