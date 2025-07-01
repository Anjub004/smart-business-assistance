import streamlit as st
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Intent model setup
data = {
    "intent": ["greeting", "greeting", "goodbye", "ask_hours", "ask_hours", "order_status", "order_status"],
    "example": [
        "Hello",
        "Hi there",
        "Bye",
        "What time do you open?",
        "When are you open?",
        "Where is my order 12345?",
        "Track order ABCD"
    ]
}
df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(df["example"])
clf = LogisticRegression()
clf.fit(X_vec, df["intent"])

def predict_intent(text):
    v = vectorizer.transform([text])
    return clf.predict(v)[0]

def extract_order_id(text):
    match = re.search(r"\b[a-zA-Z0-9]{3,}\b", text)
    return match.group(0) if match else None

faq_data = {
    "What is your return policy?": "You can return products within 30 days.",
    "How do I track my order?": "Please provide your order ID.",
    "What time are you open?": "We are open from 9 AM to 9 PM every day."
}
model = SentenceTransformer('all-MiniLM-L6-v2')
faq_questions = list(faq_data.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

def semantic_faq_answer(user_question):
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)[0]
    top_idx = scores.argmax()
    return faq_data[faq_questions[top_idx]]

session = {}

def handle_message(text):
    global session
    if session.get("pending") == "order_status":
        oid = extract_order_id(text)
        if oid:
            session.pop("pending")
            return f"Thank you! Order {oid} is being processed."
        return "Still waiting for your order ID."

    intent = predict_intent(text)
    if intent == "greeting":
        return "Hello! How can I assist you?"
    elif intent == "ask_hours":
        return "We are open from 9 AM to 9 PM daily."
    elif intent == "order_status":
        oid = extract_order_id(text)
        if oid:
            return f"Order {oid} is currently in transit."
        session["pending"] = "order_status"
        return "Please provide your order ID."
    elif intent == "goodbye":
        return "Goodbye! Have a great day!"
    else:
        return semantic_faq_answer(text)

# Streamlit UI
st.title("🤖 Smart Business Assistant")
st.write("Chat with your AI-powered assistant below.")

user_input = st.text_input("You:", key="user_input")
if st.button("Send"):
    if user_input:
        response = handle_message(user_input)
        st.markdown(f"**Bot:** {response}")
