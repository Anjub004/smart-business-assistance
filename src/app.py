import sys, os
# ← CHANGED: ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.dialog_manager import DialogManager
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize
st.set_page_config(page_title="Smart Business Assistant", layout="wide")
dm = DialogManager()

# Init session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # (role, message, intent, confidence)

# Tabs
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Analytics"])

# 1️⃣ Chat Tab
with tab1:
    st.title("💬 Smart Business Assistant")
    st.markdown("Ask me about orders, hours, services, or anything else.")

    # Display past messages
    for role, message, intent, confidence in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input box
    if user_input := st.chat_input("Type your message..."):
        intent, confidence = dm.predict_with_confidence(user_input)
        response = dm.handle(user_input)

        # Store & show messages
        st.session_state.chat_history.append(("user", user_input, None, None))
        st.session_state.chat_history.append(("assistant", response, intent, confidence))

        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

# 2️⃣ Analytics Tab
with tab2:
    st.header("📊 Chatbot Analytics")

    # Filter assistant responses only
    df = pd.DataFrame(
        [x for x in st.session_state.chat_history if x[0] == "assistant"],
        columns=["Role", "Message", "Intent", "Confidence"]
    )

    if df.empty:
        st.info("No analytics yet. Start chatting in the first tab.")
    else:
        st.metric("Messages Sent", len(df))
        st.metric("Intents Used", df["Intent"].nunique())

        st.subheader("Intent Distribution")
        st.bar_chart(df["Intent"].value_counts())

        st.subheader("Confidence Levels")
        fig, ax = plt.subplots()
        sns.histplot(df["Confidence"], bins=10, kde=True, ax=ax, color="skyblue")
        ax.set_xlabel("Confidence")
        st.pyplot(fig)

        st.subheader("Conversation Log")
        st.dataframe(df.style.background_gradient(cmap="Greens", subset=["Confidence"]))
