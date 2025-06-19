FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

# Copiar los archivos de entorno
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

# Copiar solo el script (no el modelo)
COPY ["homework.py", "./"]

# Ejecutar directamente el script al iniciar el contenedor
CMD ["python", "predict.py", "2023","5"]