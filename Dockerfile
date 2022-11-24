FROM python:3.8.10-slim

COPY requirements.txt requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt

COPY . .

CMD python -m api
