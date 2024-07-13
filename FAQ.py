import nltk

nltk.download('stopwords', quiet=True)

import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
import threading
from ttkthemes import ThemedTk


# Load FAQs from a JSON file
def load_faqs(file_path='faqs.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {
            "What is the return policy?": "You can return any item within 30 days of purchase.",
            "How do I track my order?": "You can track your order using the tracking link sent to your email.",
            "What payment methods are accepted?": "We accept credit cards, PayPal, and bank transfers.",
        }


faqs = load_faqs()


def load_interactions(file_path='interactions.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []


def save_interactions(interactions, file_path='interactions.json'):
    with open(file_path, 'w') as file:
        json.dump(interactions, file)


interactions = load_interactions()

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))


def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.text not in string.punctuation]
    return " ".join(tokens)


preprocessed_faqs = {question: preprocess(question) for question in faqs.keys()}


def update_vectors():
    all_questions = list(preprocessed_faqs.values()) + [preprocess(interaction['question']) for interaction in
                                                        interactions]
    vectorizer.fit(all_questions)

    faq_vectors = vectorizer.transform(preprocessed_faqs.values())
    interaction_vectors = vectorizer.transform(
        [preprocess(interaction['question']) for interaction in interactions]) if interactions else []

    return faq_vectors, interaction_vectors


vectorizer = TfidfVectorizer()
faq_vectors, interaction_vectors = update_vectors()


def chatbot_response(query):
    query = preprocess(query)
    query_vector = vectorizer.transform([query])

    faq_similarities = cosine_similarity(query_vector, faq_vectors).flatten()
    best_faq_index = np.argmax(faq_similarities)

    interaction_similarities = cosine_similarity(query_vector,
                                                 interaction_vectors).flatten() if interactions else np.array([])
    best_interaction_index = np.argmax(interaction_similarities) if interactions else -1

    if faq_similarities[best_faq_index] > (interaction_similarities[best_interaction_index] if interactions else 0):
        if faq_similarities[best_faq_index] < 0.1:
            return "I'm sorry, I don't understand your question. Could you please rephrase it?"
        best_match = list(preprocessed_faqs.keys())[best_faq_index]
        response = faqs[best_match]
    else:
        if interactions and interaction_similarities[best_interaction_index] < 0.1:
            return "I'm sorry, I don't understand your question. Could you please rephrase it?"
        response = interactions[best_interaction_index][
            'answer'] if interactions else "I'm sorry, I don't understand your question. Could you please rephrase it?"

    interactions.append({'question': query, 'answer': response})
    save_interactions(interactions)
    update_vectors()

    return response


class EnhancedChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced FAQ Chatbot")
        master.geometry("800x600")
        master.minsize(600, 400)

        style = ttk.Style(master)
        style.theme_use("equilux")

        self.create_widgets()
        self.create_layout()

    def create_widgets(self):
        # Chat display
        self.chat_frame = ttk.Frame(self.master, padding="10")
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, width=80, height=25,
                                                      font=("Helvetica", 12))
        self.chat_display.tag_configure('user', foreground='#4CAF50', font=("Helvetica", 12, "bold"))
        self.chat_display.tag_configure('bot', foreground='#2196F3', font=("Helvetica", 12))
        self.chat_display.tag_configure('system', foreground='#9E9E9E', font=("Helvetica", 10, "italic"))
        self.chat_display.config(state='disabled')

        # User input
        self.input_frame = ttk.Frame(self.master, padding="10")
        self.user_input = ttk.Entry(self.input_frame, font=("Helvetica", 12), width=50)
        self.user_input.bind("<Return>", self.send_message)
        self.send_button = ttk.Button(self.input_frame, text="Send", command=self.send_message)

        # Control buttons
        self.button_frame = ttk.Frame(self.master, padding="10")
        self.clear_button = ttk.Button(self.button_frame, text="Clear Chat", command=self.clear_chat)
        self.faq_button = ttk.Button(self.button_frame, text="Show FAQs", command=self.show_faqs)
        self.exit_button = ttk.Button(self.button_frame, text="Exit", command=self.on_closing)

    def create_layout(self):
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.chat_frame.grid(row=0, column=0, sticky="nsew")
        self.chat_frame.columnconfigure(0, weight=1)
        self.chat_frame.rowconfigure(0, weight=1)
        self.chat_display.grid(row=0, column=0, sticky="nsew")

        self.input_frame.grid(row=1, column=0, sticky="ew")
        self.input_frame.columnconfigure(0, weight=1)
        self.user_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.send_button.grid(row=0, column=1)

        self.button_frame.grid(row=2, column=0, sticky="ew")
        self.button_frame.columnconfigure(1, weight=1)
        self.clear_button.grid(row=0, column=0, padx=5)
        self.faq_button.grid(row=0, column=1, padx=5)
        self.exit_button.grid(row=0, column=2, padx=5)

    def send_message(self, event=None):
        user_query = self.user_input.get()
        if user_query.strip() != "":
            time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.update_chat_display(f"You ({time_now}): {user_query}\n", 'user')
            self.update_chat_display("Chatbot is typing...\n", 'system')
            self.user_input.delete(0, tk.END)
            threading.Thread(target=self.process_response, args=(user_query, time_now)).start()

    def process_response(self, user_query, time_now):
        response = chatbot_response(user_query)
        self.master.after(1000, self.update_response, response, time_now)

    def update_response(self, response, time_now):
        self.chat_display.config(state='normal')
        self.chat_display.delete("end-2l", "end-1l")
        self.update_chat_display(f"Chatbot ({time_now}): {response}\n", 'bot')

    def update_chat_display(self, message, tag):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, message, tag)
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)

    def clear_chat(self):
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state='disabled')

    def show_faqs(self):
        faq_window = tk.Toplevel(self.master)
        faq_window.title("Frequently Asked Questions")
        faq_window.geometry("600x400")

        style = ttk.Style(faq_window)
        style.theme_use("equilux")

        faq_text = scrolledtext.ScrolledText(faq_window, wrap=tk.WORD, width=70, height=20, font=("Helvetica", 12))
        faq_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        for question, answer in faqs.items():
            faq_text.insert(tk.END, f"Q: {question}\n", 'question')
            faq_text.insert(tk.END, f"A: {answer}\n\n", 'answer')

        faq_text.tag_config('question', foreground='#4CAF50', font=("Helvetica", 12, "bold"))
        faq_text.tag_config('answer', foreground='#2196F3', font=("Helvetica", 12))
        faq_text.config(state='disabled')

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.master.destroy()


if __name__ == "__main__":
    root = ThemedTk(theme="equilux")
    app = EnhancedChatbotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()