FROM python:3.8

COPY . .

RUN python -m pip install requirements.txt

CMD python -m distractor_generator