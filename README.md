# 🤖 Smart Business Assistant Chatbot

A Streamlit-based AI chatbot designed to automate business interactions, answer frequently asked questions, track orders, and provide real-time analytics — powered by NLP and transformer models.

---

## 📌 Project Overview

The **Smart Business Assistant** is an interactive chatbot that can handle:
- Customer queries about business policies, hours, and order statuses
- Intelligent fallback responses using semantic search
- Real-time conversation analytics (intent frequency, confidence)
- Responsive chat UI for both desktop and mobile

---

## 🎯 Problem Statement

Businesses face a large number of repetitive queries which burden customer service teams. A smart assistant is needed to:
- Reduce support load
- Deliver 24/7 query handling
- Provide actionable insights

---

## ✅ Objectives

- Build a natural language chatbot using modern NLP models
- Deploy via Streamlit with professional UI/UX
- Visualize chat behavior using analytics

---

## 🧠 Features

| Feature                      | Description |
|-----------------------------|-------------|
| 💬 Interactive Chat         | Real-time, emoji-enhanced user interaction |
| 🧠 Intent Detection         | Uses `SentenceTransformer` for semantic similarity |
| 📊 Analytics Dashboard      | Tracks user queries, intents, confidence |
| 🧾 Fallback Mechanism       | Graceful response for low-confidence inputs |
| 📱 Responsive UI            | Optimized for both desktop and mobile |
| 🧠 Memory Support           | Remembers previous messages in session |

---

## ⚙️ Technologies Used

- **Frontend/UI**: Streamlit  
- **NLP Model**: `sentence-transformers` (`all-MiniLM-L6-v2`)  
- **ML Backend**: scikit-learn, TF-IDF, LogisticRegression  
- **Data & Visualization**: pandas, matplotlib, seaborn  
- **IDE**: VS Code, Jupyter Notebook  
- **Version Control**: Git & GitHub  

---

## 📂 Project Structure
'''
smart_business_assistant_capstone/
│
├── src/
│ ├── app.py # Streamlit UI code
│ ├── dialog_manager.py # Core chatbot logic
│
├── data/
│ ├── faqs.json # Optional FAQ fallback file
│
├── README.md # Project documentation
├── requirements.txt # List of required Python packages
└── Smart_Business_Assistant_Presentation.pptx

'''

---

## 🧪 Key Libraries & Functions

# Intent classification
from sentence_transformers import SentenceTransformer, util

# Streamlit chat
st.chat_input()
st.chat_message()
st.session_state

# ML classification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

---
📊 Analytics
Number of messages

Intent frequency distribution

Confidence score histogram

Logged via session and displayed in sidebar / analytics tab
---
🚀 How to Run
1. Clone the repo
   `git clone https://github.com/anjub004/smart-business-assistant.git
    cd smart-business-assistant`

2. Create a virtual environment
   `python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate`

3.Install dependencies
`pip install -r requirements.txt`

4. Run the app
   `streamlit run src/app.py`

---
💡 Future Enhancements
Add voice assistant integration (speech-to-text)

Support for multilingual queries (Hindi, Marathi, etc.)

Backend database for order tracking and user history

Live agent escalation for complex queries

---
🙋‍♀️ Author
Anju Barai

##### Email: anjubarai004@gmail.com
##### GitHub: https://github.com/Anjub004
---
📄 License
This project is for educational and academic use only.

`
Let me know if you want me to generate the actual `README.md` file for download or customize it with your GitHub link, email, or team name.
`
