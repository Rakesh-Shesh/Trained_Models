import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Specify the model repository on Hugging Face
model_id = "Rakesh44/HRPolicyQandA"

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

# Set the model to evaluation mode
model.eval()

# Function to generate answers based on a question
def generate_answers(question, max_length, num_return_sequences, top_k, top_p, temperature):
    input_text = f"Question: {question} Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )

    answers = [tokenizer.decode(output, skip_special_tokens=True).split("Answer:")[-1].strip() for output in outputs]
    return answers

# Sample questions to guide users
sample_questions = [
    "What are the best practices for remote work policies?",
    "How can we improve employee retention?",
    "What should be included in a performance appraisal process?",
    "How should HR handle workplace harassment complaints?",
    "What is the importance of diversity in hiring?",
    "How can we create an inclusive workplace culture?",
    "What are the legal requirements for employee benefits?",
    "How often should HR policies be reviewed and updated?"
]

# Streamlit app layout
st.title("HR Policy Q&A")

st.sidebar.header("Settings")
max_length = st.sidebar.slider("Max Length of Response", 50, 300, 150)
num_return_sequences = st.sidebar.slider("Number of Responses to Generate", 1, 5, 1)
top_k = st.sidebar.slider("Top-k Sampling", 0, 100, 50)
top_p = st.sidebar.slider("Top-p Sampling", 0.0, 1.0, 0.95)
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 1.0)

# User input for question
question = st.text_input("Enter your HR question:", "How should HR policies be changed?")

if st.button("Generate Answers"):
    if question:
        answers = generate_answers(question, max_length, num_return_sequences, top_k, top_p, temperature)
        st.subheader("Generated Answers:")
        for i, answer in enumerate(answers):
            st.write(f"Response {i + 1}: {answer}")
    else:
        st.write("Please enter a question.")

st.sidebar.header("Sample Questions")
for q in sample_questions:
    st.sidebar.write(f"- {q}")
