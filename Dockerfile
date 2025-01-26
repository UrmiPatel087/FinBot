# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Set environment variables (you can add more environment variables as needed)
# If you're using OpenAI API key as an environment variable, we set it here
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Step 5: Install dependencies
RUN python -m venv chatbot_openai
RUN /bin/bash -c "source /app/chatbot_openai/bin/activate && pip install --no-cache-dir -r requirements.txt"

# Step 6: Expose the port that Streamlit uses
EXPOSE 8501

# Step 7: Define the command to run the app
CMD /bin/bash -c "source /app/chatbot_openai/bin/activate && streamlit run /app/app.py"
