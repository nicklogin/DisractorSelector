FROM python:3.8.10

COPY requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt
RUN python -m nltk.downloader punkt

COPY . .

CMD python -m distractor_generator
