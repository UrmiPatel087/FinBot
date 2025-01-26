import pdfplumber
import re
import pandas as pd
import io
import openai
import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

# Set OpenAI API key (Use environment variables for security)
openai.api_key = "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"

CHROMA_PATH = "chroma"
CONFIDENCE_THRESHOLD = 0.730365


class ChatBot:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def extract_text(self) -> str:
        extracted_text = ""
        pdf_file_bytes = io.BytesIO(self.pdf_file.getvalue())

        with pdfplumber.open(pdf_file_bytes) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = f"Page {i + 1}\n"
                extracted_text += page_number

                text = page.extract_text()
                if text:
                    extracted_text += "Text:\n" + text + "\n"
                else:
                    extracted_text += "No text extracted on this page.\n"

        return extracted_text

    def preprocess_text(self, text: str):
        text = re.sub(r'\s+', ' ', text).strip()  # Remove spaces and newlines
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        return text

    def chunk_text(self, cleaned_text: str) -> pd.DataFrame:
        page_pattern = re.compile(r"Page (\d+)")
        page_matches = list(page_pattern.finditer(cleaned_text))
        content_data = []

        for i in range(len(page_matches)):
            start_index = page_matches[i].end()
            end_index = page_matches[i + 1].start() if i + 1 < len(page_matches) else len(cleaned_text)
            page_content = cleaned_text[start_index:end_index].strip()
            page_num = int(page_matches[i].group(1))
            content_data.append([page_num, page_content])

        return pd.DataFrame(content_data, columns=["pageno", "content"])

    def create_database(self, chunked_data: pd.DataFrame):
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)

        documents = [
            Document(page_content=row['content'], metadata={"page_number": row['pageno']})
            for _, row in chunked_data.iterrows()
        ]

        db_exists = os.path.exists(CHROMA_PATH)
        db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=CHROMA_PATH) if not db_exists else Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

        db.add_documents(documents)
        db.persist()
        print(f"Database updated with {len(documents)} documents.")

    def classify_query(self, query_text):
        """Classify the query as 'General' or 'Data' using OpenAI API."""
        prompt = f"""
        You are a financial expert. Your task is to classify the following user query into one of two categories:  
        - 'Data': If the query pertains to financial data analysis, numbers, trends, ratios, or specific details that can be answered using financial statement data (e.g., profit and loss statement, balance sheets, income statements, cash flow statements).  
        - 'General': If the query is related to general conversations, definitions, or explanations that do not require financial data.

        Please classify the following query: "{query_text}"  
        Respond only with one of the two categories: 'Data' or 'General'. No additional explanation is needed.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify queries strictly according to the criteria provided."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0  # Set temperature to 0 to make responses deterministic (less creative)
        )

        category = response.choices[0].message['content'].strip().lower()
        return category


    def query_data(self, query_text):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
        results = db.similarity_search_with_relevance_scores(query_text, k=2)

        if not results or all(score < CONFIDENCE_THRESHOLD for _, score in results):
            return "I don't know the answer."

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt = f"Answer the following question based on this context:\n\n{context_text}\n\n{query_text}"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        return response.choices[0].message['content'].strip()

    def handle_query(self, query_text):
        query_type = self.classify_query(query_text)
        if query_type == "data":
            return self.query_data(query_text)
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query_text}],
                max_tokens=200
            )
            return response.choices[0].message['content'].strip()
