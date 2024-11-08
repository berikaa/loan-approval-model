FROM python:3.9-bullseye

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code/
COPY ./loan_model.pkl /code/loan_model.pkl

EXPOSE 8081

CMD ["uvicorn", "evaluation:app", "--host", "0.0.0.0", "--port", "8081", "--no-access-log"]
