import streamlit as st
import json
import speech_recognition as sr
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from gtts import gTTS
import os

# Load model & tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Load context from JSON file
data_path = "context.json"  # Update with your JSON file path
with open(data_path, "r") as f:
    context_data = json.load(f)
    context = context_data["context"]

# Function for voice input
def voice_input():
    st.write("Press the button and ask your question:")
    # Initialize speech recognizer
    recognizer = sr.Recognizer()

    # Capture voice input
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        # Convert voice input to text
        user_input = recognizer.recognize_google(
            audio, language="en-US")  # Set language to English
        st.write("You said:", user_input)

        # Process the user's question
        process_question(user_input)

    except sr.UnknownValueError:
        st.error("Sorry, could not understand audio input.")

def text_input():
    st.write("Enter your question below:")
    user_input = st.text_input("Question:")

    if st.button("Ask"):
        if user_input.strip() != "":
            process_question(user_input)

def process_question(user_input):
    QA_input = {'question': user_input, 'context': context}
    res = nlp(QA_input)
    st.write("Answer:", res['answer'])

    # Convert text to speech
    tts = gTTS(text=res['answer'], lang='en')
    tts.save("output.mp3")

    # Display audio output
    st.audio("output.mp3", format="audio/mp3", start_time=0)

    # Attempt to remove the audio file
    try:
        os.remove("output.mp3")
    except FileNotFoundError as e:
        st.error(f"Error removing audio file: {e}")

def main():
    st.title("Chatbot Web Application")
    st.write("Choose your input method:")

    # Get the current session state
    active_input_method = st.session_state.get("active_input_method", None)

    # User selects input method: voice or text
    st.write("### Input Method")
    col1, col2 = st.columns(2)
    with col1:
        voice_button = st.button("Voice", key="voice_button")
    with col2:
        text_button = st.button("Text", key="text_button")

    if voice_button:
        st.session_state.active_input_method = "voice"
        voice_input()
    elif text_button:
        st.session_state.active_input_method = "text"
        text_input()

if __name__ == "__main__":
    main()
