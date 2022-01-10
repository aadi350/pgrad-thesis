FROM python:3.8

LABEL Maintainer="aadi350"

RUN pip install numpy tensorflow
t
ADD ./src/runall.py .

CMD [ "python", "runall.py" ]
