# Usar una imagen base de Python 3
FROM python:3.8.19


WORKDIR /app

COPY . .
COPY .env .env

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "main.py"]
