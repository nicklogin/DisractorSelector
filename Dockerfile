FROM python:3.8.10

COPY . .

RUN python -m pip install -r requirements.txt
RUN python -m nltk.downloader punkt

CMD python -m distractor_generator
