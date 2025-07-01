
import random
from sentence_transformers import SentenceTransformer, util
import torch

class DialogManager:
    def __init__(self):
        self.intents = {
            "greeting": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "greetings", "hello there", "hi there", "howdy"
            ],
            "goodbye": [
                "bye", "goodbye", "see you", "see you later", "farewell",
                "talk to you soon", "bye bye", "take care"
            ],
            "ask_hours": [
                "what are your hours", "when are you open", "opening hours",
                "store hours", "what time do you open", "when do you close"
            ],
            "order_status": [
                "track my order", "order status", "where is my order",
                "how is my order", "when will my package arrive",
                "is my order shipped"
            ]
        }

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intent_sentences = []
        self.intent_labels = []

        for label, phrases in self.intents.items():
            for phrase in phrases:
                self.intent_sentences.append(phrase)
                self.intent_labels.append(label)

        self.embeddings = self.model.encode(self.intent_sentences, convert_to_tensor=True)
        self.threshold = 0.5  # semantic similarity threshold

    def predict_with_confidence(self, user_input):
        query_embedding = self.model.encode(user_input, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        predicted_intent = self.intent_labels[best_idx]
        return predicted_intent, best_score

    def handle(self, user_input):
        intent, confidence = self.predict_with_confidence(user_input)
        print(f"[DEBUG] '{user_input}' → Intent: {intent}, Confidence: {confidence:.2f}")

        if confidence < self.threshold:
            return "🤔 Sorry, I didn’t understand that. Can you rephrase?"


        if intent == "greeting":
            return random.choice([
                "👋 Hello! How can I help you today?",
                "Hi there! 😊 What can I assist you with?"
            ])
        elif intent == "goodbye":
            return random.choice([
                "👋 Goodbye! Have a great day!",
                "Take care! See you soon!"
            ])
        elif intent == "ask_hours":
            return "🕘 We're open from 9 AM to 9 PM, every day!"
        elif intent == "order_status":
            return "📦 Please provide your order ID to check its status."

        return "❓ I'm not sure how to help with that."
