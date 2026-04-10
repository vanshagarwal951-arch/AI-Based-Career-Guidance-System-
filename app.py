import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime
import json
import os

# ---------------------------------
# 1. Extended Dataset (India-relevant careers added)
data = [
    {
        "job_title": "Data Scientist",
        "skills": "python, machine learning, statistics, data visualization",
        "education": "bachelor",
        "interests": "data, analytics, research",
        "experience": "mid",
        "description": "Analyzes data to extract insights and build predictive models."
    },
    {
        "job_title": "Software Engineer",
        "skills": "java, c++, algorithms, problem solving",
        "education": "bachelor",
        "interests": "coding, development, problem solving",
        "experience": "junior",
        "description": "Designs and develops software applications and systems."
    },
    {
        "job_title": "Graphic Designer",
        "skills": "photoshop, creativity, adobe illustrator, visual design",
        "education": "associate",
        "interests": "art, creativity, media",
        "experience": "entry",
        "description": "Creates visual concepts to communicate ideas."
    },
    {
        "job_title": "Project Manager",
        "skills": "leadership, communication, scheduling, budgeting",
        "education": "bachelor",
        "interests": "management, organization, planning",
        "experience": "senior",
        "description": "Oversees projects to ensure timely delivery within budget."
    },
    {
        "job_title": "Marketing Specialist",
        "skills": "seo, content creation, social media, communication",
        "education": "bachelor",
        "interests": "marketing, branding, communication",
        "experience": "mid",
        "description": "Develops strategies to promote products and brands."
    },
    {
        "job_title": "Cybersecurity Analyst",
        "skills": "network security, python, risk assessment, cryptography",
        "education": "bachelor",
        "interests": "security, technology, risk management",
        "experience": "mid",
        "description": "Protects an organization's computer systems and networks."
    },
    {
        "job_title": "Mechanical Engineer",
        "skills": "cad, thermodynamics, mechanics, problem solving",
        "education": "bachelor",
        "interests": "engineering, mechanics, design",
        "experience": "mid",
        "description": "Designs and tests mechanical devices and systems."
    },
    {
        "job_title": "Financial Analyst",
        "skills": "excel, finance, accounting, data analysis",
        "education": "bachelor",
        "interests": "finance, economics, data",
        "experience": "junior",
        "description": "Provides investment and financial recommendations."
    },
    {
        "job_title": "Teacher",
        "skills": "communication, patience, subject knowledge, mentoring",
        "education": "bachelor",
        "interests": "teaching, education, helping others",
        "experience": "mid",
        "description": "Educates and supports students in learning."
    },
    {
        "job_title": "UX Designer",
        "skills": "wireframing, user research, creativity, prototyping",
        "education": "bachelor",
        "interests": "design, user experience, psychology",
        "experience": "mid",
        "description": "Improves user satisfaction with products by enhancing usability."
    },
    # --- India-specific careers ---
    {
        "job_title": "IAS Officer (Civil Services)",
        "skills": "leadership, general knowledge, administration, communication, decision making",
        "education": "bachelor",
        "interests": "governance, public service, law, politics, social work",
        "experience": "entry",
        "description": "Works in Indian Administrative Service to manage government policies and public administration at district and state level."
    },
    {
        "job_title": "Chartered Accountant (CA)",
        "skills": "accounting, taxation, auditing, finance, tally, excel",
        "education": "bachelor",
        "interests": "finance, economics, law, numbers, business",
        "experience": "junior",
        "description": "Manages financial audits, taxation, and accounting for businesses and individuals under ICAI certification."
    },
    {
        "job_title": "Doctor (MBBS)",
        "skills": "biology, chemistry, patient care, diagnosis, anatomy",
        "education": "bachelor",
        "interests": "healthcare, medicine, helping others, research, science",
        "experience": "junior",
        "description": "Provides medical care, diagnosis, and treatment to patients in hospitals or clinics."
    },
    {
        "job_title": "Lawyer / Advocate",
        "skills": "communication, law, critical thinking, argumentation, research",
        "education": "bachelor",
        "interests": "law, justice, politics, writing, social issues",
        "experience": "junior",
        "description": "Represents clients in legal matters, drafts legal documents, and provides legal advice."
    },
    {
        "job_title": "Bank PO (Probationary Officer)",
        "skills": "quantitative aptitude, reasoning, communication, finance, english",
        "education": "bachelor",
        "interests": "banking, finance, government jobs, economics",
        "experience": "entry",
        "description": "Manages banking operations, customer services, and financial products in public sector banks."
    },
    {
        "job_title": "Software Developer (IT/TCS/Infosys)",
        "skills": "python, java, sql, problem solving, algorithms, communication",
        "education": "bachelor",
        "interests": "coding, technology, development, software",
        "experience": "junior",
        "description": "Develops and maintains software products in Indian IT service companies like TCS, Infosys, Wipro, etc."
    },
    {
        "job_title": "Entrepreneur / Startup Founder",
        "skills": "leadership, business planning, communication, marketing, finance, creativity",
        "education": "bachelor",
        "interests": "business, innovation, startups, management, risk taking",
        "experience": "mid",
        "description": "Builds and runs a new business or startup, managing all aspects from product to operations."
    },
    {
        "job_title": "Content Creator / YouTuber",
        "skills": "video editing, communication, creativity, social media, storytelling",
        "education": "highschool",
        "interests": "media, entertainment, creativity, teaching, art",
        "experience": "entry",
        "description": "Creates digital content on platforms like YouTube, Instagram, or podcasts to build an audience."
    },
    {
        "job_title": "Data Analyst",
        "skills": "excel, sql, python, data visualization, statistics",
        "education": "bachelor",
        "interests": "data, analytics, business, research",
        "experience": "junior",
        "description": "Collects, processes, and interprets data to help businesses make informed decisions."
    },
    {
        "job_title": "Architect",
        "skills": "autocad, design, creativity, mathematics, civil engineering",
        "education": "bachelor",
        "interests": "design, construction, art, engineering, planning",
        "experience": "mid",
        "description": "Plans and designs buildings, ensuring functionality, safety, and aesthetic appeal."
    },
]

df = pd.DataFrame(data)

# ---------------------------------
# 2. Data preparation
def combine_text_features(row):
    return f"{row['skills']} {row['education']} {row['interests']} {row['experience']}"

df['combined_features'] = df.apply(combine_text_features, axis=1)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined_features'])

le = LabelEncoder()
y = le.fit_transform(df['job_title'])

# ---------------------------------
# 3. Model training
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ---------------------------------
# 4. Streamlit UI
st.set_page_config(page_title="AI Career Recommender", page_icon="🎯", layout="centered")

st.title("🎯 AI-Based Career Path Recommender")
st.markdown("Fill in your profile below to get **AI-powered career recommendations** tailored for you.")
st.markdown("---")

# User inputs
col1, col2 = st.columns(2)

with col1:
    skills_input = st.text_input("💡 Your Skills (comma separated):", placeholder="e.g. python, communication")
    education_input = st.selectbox("🎓 Highest Education Level:",
                                   ['highschool', 'associate', 'bachelor', 'master', 'phd'])

with col2:
    interests_input = st.text_input("❤️ Your Interests (comma separated):", placeholder="e.g. data, research")
    experience_input = st.selectbox("💼 Experience Level:", ['entry', 'junior', 'mid', 'senior'])

st.markdown("---")

# ---------------------------------
# 5. Predict on button click
if st.button("🔍 Recommend Careers", use_container_width=True):

    if not skills_input.strip() or not interests_input.strip():
        st.warning("⚠️ Please fill in both Skills and Interests before recommending.")
    else:
        user_features = f"{skills_input} {education_input} {interests_input} {experience_input}"
        user_vector = vectorizer.transform([user_features])

        pred_probs = model.predict_proba(user_vector)[0]

        def text_to_vector(text):
            skill_vectorizer = CountVectorizer(vocabulary=vectorizer.vocabulary_)
            return skill_vectorizer.transform([text])

        user_skills_vec = text_to_vector(skills_input.lower())
        descriptions = df['skills'].str.lower().values
        career_skills_vec = vectorizer.transform(descriptions)
        similarities = cosine_similarity(user_skills_vec, career_skills_vec)[0]

        combined_scores = 0.7 * pred_probs + 0.3 * similarities
        top3_idx = combined_scores.argsort()[::-1][:3]

        st.subheader("🏆 Your Top 3 Career Matches")

        results_for_save = []

        for rank, idx in enumerate(top3_idx, start=1):
            score_percent = round(float(combined_scores[idx]) * 100, 1)
            career = df.iloc[idx]

            with st.container():
                st.markdown(f"### #{rank} — {career['job_title']}")

                # Score bar
                st.markdown(f"**Match Score: {score_percent}%**")
                st.progress(min(score_percent / 100, 1.0))

                st.write(career['description'])
                st.markdown(f"**🛠️ Required Skills:** {career['skills']}")
                st.markdown(f"**🎓 Typical Education:** {career['education'].capitalize()}")
                st.markdown(f"**❤️ Interests:** {career['interests']}")
                st.markdown(f"**💼 Experience Level:** {career['experience'].capitalize()}")
                st.markdown("---")

            results_for_save.append({
                "rank": rank,
                "career": career['job_title'],
                "match_score": f"{score_percent}%",
                "skills_required": career['skills'],
                "education": career['education'],
                "interests": career['interests'],
                "experience": career['experience'],
            })

        # ---------------------------------
        # 6. Save Results button
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_data = {
            "timestamp": timestamp,
            "user_input": {
                "skills": skills_input,
                "education": education_input,
                "interests": interests_input,
                "experience": experience_input,
            },
            "top_3_recommendations": results_for_save
        }

        save_json = json.dumps(save_data, indent=4)

        st.download_button(
            label="💾 Save My Results (JSON)",
            data=save_json,
            file_name=f"career_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# ---------------------------------
# 7. Sidebar
st.sidebar.title("ℹ️ How It Works")
st.sidebar.info("""
1. Enter your **skills** and **interests**
2. Select your **education** and **experience** level
3. Click **Recommend Careers**
4. Get your **Top 3 matches** with a % score
5. **Save your results** as a file anytime!

The AI uses a **Random Forest model** + **Cosine Similarity** to find the best career fit for you.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("🇮🇳 Includes careers relevant to **Indian students** — Civil Services, CA, Bank PO, and more!")
