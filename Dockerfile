FROM python:3.11-slim
WORKDIR /app
COPY environment.yml /app/environment.yml
RUN apt-get update && apt-get install -y build-essential git
RUN pip install --upgrade pip

minimal deps for quick runs; use conda locally for full env
RUN pip install numpy scipy matplotlib jupyterlab numba pandas torch
COPY . /app
CMD [ "bash" ]
