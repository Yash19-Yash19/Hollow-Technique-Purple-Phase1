import streamlit as st
import json
import pyttsx3
import speech_recognition as sr
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Initialize text-to-speech assistant
assistant = pyttsx3.init("sapi5")
voices = assistant.getProperty("voices")
assistant.setProperty("voice", voices[0].id)


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

# Function to speak


def speak(audio):
    print(audio)
    assistant.say(audio)
    assistant.runAndWait()

# Function to take command using speech recognition


def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.phrase_threshold = 0.3
        r.dynamic_energy_threshold = True
        r.operation_timeout = 5
        r.non_speaking_duration = 0.5
        r.dynamic_energy_adjustment = 2
        r.energy_threshold = 4000
        r.phrase_time_limit = 10
        r.timeout = 5

        audio = r.listen(source)

        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
            return query
        except Exception as e:
            print("Sorry, I could not understand that. Please try again.")
            return ""

# Define Streamlit app


def main():
    st.title("Chatbot Web Application")
    st.write("Enter your question below:")

    user_input = st.text_input("Question:")
    voice_input = st.checkbox("Voice Input")

    if voice_input:
        st.write("Speak your question...")
        user_input = take_command()
        st.write("You said:", user_input)

    if st.button("Ask"):
        if user_input.strip() != "":
            QA_input = {'question': user_input, 'context': context}
            res = nlp(QA_input)
            st.write("Answer:", res['answer'])

            # Voice Output
            voice_output = st.checkbox("Voice Output")
            if voice_output:
                speak("The answer is " + res['answer'])


if __name__ == "__main__":
    main()
