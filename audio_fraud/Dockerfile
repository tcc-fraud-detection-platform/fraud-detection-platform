FROM python:3.7.16-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim nano wget curl apt-transport-https ca-certificates gnupg

WORKDIR /app/
COPY ./app /app/
RUN python -m pip install --upgrade pip
RUN pip3 install --no-input --requirement /app/requirements.txt

ENTRYPOINT [ "python3" ]
CMD [ "/app/app.py" ]