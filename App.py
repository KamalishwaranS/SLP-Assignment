import streamlit as st
# Function to calculate perplexity based on sentence length
def calculate_perplexity(sentence):
    length = len(sentence)
    if length > 70:
        perplexity = 8
    elif 30 < length <= 70:
        perplexity = 5
    elif 10 < length <= 30:
        perplexity = 3
    elif length <= 10:
        perplexity = 2
    else:
        perplexity = 1
    return perplexity
# Streamlit UI
st.title("Text Generation Quality")
# Input sentence
input_sentence = st.text_input("Enter a sentence:", value="This is a sample sentence.")
# Calculate perplexity
if st.button("Calculate Perplexity"):
    perplexity = calculate_perplexity(input_sentence)
    st.write("Perplexity of the sentence:", perplexity)

