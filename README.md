# 🎯 AI-Based Career Path Recommender

A machine learning web app that recommends the **best career paths** based on your skills, interests, education level, and experience — built using Python and Streamlit.

---

## 💡 Why I Built This

As a student, I always found it confusing to pick the right career path. There are so many options but no clear guidance. So I decided to build an AI-powered tool that takes your profile as input and tells you which careers are the best match — with a percentage score.

I also made sure to include careers relevant to **Indian students** like Civil Services (IAS), CA, Bank PO, and more, since most tools online are focused on Western job markets.

---

## 🚀 Features

- ✅ AI-powered career recommendations using Machine Learning
- ✅ **Match Score (%)** shown for each recommended career
- ✅ **Top 3 career matches** with full details
- ✅ **Save Results** — download your recommendations as a JSON file
- ✅ Includes **India-specific careers** — IAS, CA, Bank PO, Doctor, Lawyer, and more
- ✅ Clean two-column UI built with Streamlit

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web interface |
| Scikit-learn | ML model (Random Forest + Cosine Similarity) |
| Pandas | Data handling |
| NumPy | Numerical operations |

---

## ⚙️ How to Run Locally

### 1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Based-Career-Guidance-System.git
cd AI-Based-Career-Guidance-System
```

### 2. Install required packages
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in your browser
Visit 👉 `http://localhost:8501`

---

## 📸 How to Use

1. Enter your **skills** (e.g. python, communication, excel)
2. Select your **education level**
3. Enter your **interests** (e.g. data, finance, teaching)
4. Select your **experience level**
5. Click **"Recommend Careers"**
6. View your **Top 3 matches with % score**
7. Hit **"Save My Results"** to download them!

---

## 🧠 How the AI Works

- All career profiles (skills, education, interests, experience) are **combined into text features**
- A **CountVectorizer** converts text into numerical format
- A **Random Forest Classifier** learns which career fits which profile
- **Cosine Similarity** measures how closely your skills match each career's requirements
- Final score = `0.7 × ML probability + 0.3 × similarity score`
- Top 3 highest scoring careers are shown

---

## 📁 Project Structure

```
AI-Based-Career-Guidance-System/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🇮🇳 Careers Included

General: Data Scientist, Software Engineer, UX Designer, Project Manager, Marketing Specialist, Cybersecurity Analyst, Mechanical Engineer, Financial Analyst, Teacher, Architect, Data Analyst, Graphic Designer, Entrepreneur, Content Creator

India-specific: IAS Officer, Chartered Accountant (CA), Doctor (MBBS), Lawyer/Advocate, Bank PO, Software Developer (IT/TCS/Infosys)

---

## 📄 License

Open source — free to use for learning and educational purposes.
