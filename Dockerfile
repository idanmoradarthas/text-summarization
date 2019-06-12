FROM python:3.7
COPY wrappers /app/wrappers
COPY app.py /app
COPY properties.yaml /app
COPY requirements.txt /app
COPY sentence_handler.py /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
EXPOSE 8080
CMD python ./app.py