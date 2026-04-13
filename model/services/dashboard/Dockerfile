FROM python:3.10-slim

WORKDIR /app

# Copy and install requirements
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dashboard code
COPY services/dashboard/dashboard.py .
COPY services/config.py ./services/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]