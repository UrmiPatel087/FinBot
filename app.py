import streamlit as st
from chatbot import ChatBot

st.title("PDF Chatbot Assistant")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file is not None:
    st.write("Processing the PDF...")

    chatbot = ChatBot(pdf_file)

    extracted_text = chatbot.extract_text()
    with open('extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    cleaned_text = chatbot.preprocess_text(extracted_text)

    chunked_data = chatbot.chunk_text(cleaned_text)

    chatbot.create_database(chunked_data)

    st.write("PDF processed. The chatbot is ready to answer your questions.")
    
    user_query = st.text_input("Ask a question about the PDF content:")
    
    if user_query:
        response = chatbot.handle_query(user_query)
        st.write("Answer:", response)
