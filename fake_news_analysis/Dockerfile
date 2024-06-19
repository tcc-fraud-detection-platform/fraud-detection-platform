FROM python:3.9-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim nano wget curl apt-transport-https ca-certificates gnupg

WORKDIR /app/
COPY ./app /app/
RUN pip3 install --no-input --requirement /app/requirements.txt

ENTRYPOINT [ "python3" ]
CMD [ "/app/prediction.py" ]