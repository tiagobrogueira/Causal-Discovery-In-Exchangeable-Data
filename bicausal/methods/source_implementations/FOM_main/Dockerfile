FROM python:3.7-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt .
RUN pip config set global.index-url https://pypi.douban.com/simple
RUN pip install -r requirements.txt
COPY libs/gp_extras ./libs/gp_extras
RUN cd libs/gp_extras && python setup.py install

COPY . .
RUN chmod +x *.sh
# ENTRYPOINT [ "/bin/bash", "-C" ]