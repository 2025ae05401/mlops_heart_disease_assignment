FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps -e .
# Train first so artifacts/model.joblib exists.
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn","heartml.api:app","--host","0.0.0.0","--port","8000"]
