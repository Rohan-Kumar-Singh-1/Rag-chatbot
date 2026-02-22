# RAG Chatbot – deploy on Render (or any Docker host)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT at runtime; Streamlit must listen on 0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8501

CMD streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
