FROM huggingface/transformers-pytorch-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y qt5-default

RUN mkdir -p workdir
COPY requirements.txt /workdir
COPY infer.py /workdir

WORKDIR /workdir

RUN pip install -r requirements.txt

CMD ["python3", "-u", "/workdir/infer.py"]