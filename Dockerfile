# Use uma imagem base oficial do Python
FROM python:3.11-slim

# Define o diretório de trabalho no contêiner
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY src/ ./src
COPY app.py .

# Expõe a porta que o Streamlit usa
EXPOSE 8501

# Comando para executar a aplicação
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
