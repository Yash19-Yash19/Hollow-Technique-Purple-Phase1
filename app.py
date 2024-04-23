import streamlit as st
import json
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

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

# Define Streamlit app


def main():
    st.title("Chatbot Web Application")

    st.write("Enter your question below:")

    user_input = st.text_input("Question:")

    if st.button("Ask"):
        if user_input.strip() != "":
            QA_input = {
                'question': user_input,
                'context': context
            }
            res = nlp(QA_input)
            st.write("Answer:", res['answer'])


if __name__ == "__main__":
    main()
