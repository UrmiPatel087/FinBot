# PDF Chatbot Assistant

This is a chatbot assistant that processes PDF files and answers questions based on the extracted content using OpenAI API.

## Features
- Upload a PDF and extract text
- Ask questions about the document content
- Uses OpenAI GPT model for query handling
- Stores processed content in a vector database (ChromaDB)

## Setup Instructions

### 1. Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo


2. Install dependencies:
python -m venv chatbot_openai
source chatbot_openai/bin/activate  
# On Windows use `chatbot_openai\Scripts\activate`
pip install -r requirements.txt

3. Set your OpenAI API Key as an environment variable:
export OPENAI_API_KEY="your-api-key-here"

4. Run the app:
streamlit run app.py

