FROM jupyter/scipy-notebook

#RUN apt update
#ENV DEBIAN_FRONTEND noninteractive

#EXPOSE 8866/tcp

WORKDIR /home/jovyan

# USER root
# RUN apt-get update && apt-get install -y curl
# USER 1000

#RUN /usr/bin/python3 -m pip3 install --upgrade pip
RUN pip install torch
RUN pip install regex anywidget plotly nltk peft sentence_transformers transformers igraph pandas matplotlib openai tqdm
# RUN pip install fastlexrank

# Need to roll back versions (Not tested on docker yet)
RUN pip install --upgrade "pyzmq<25" "jupyter_client<8"
RUN pip install --upgrade "voila<0.5"

WORKDIR /home/jovyan/.jupyter
COPY docker/jupyter_notebook_config.py jupyter_notebook_config.py 

WORKDIR /opt/conda/lib/python3.11/site-packages/voila
COPY docker/app.py app.py

WORKDIR /opt/conda/share/jupyter/voila/templates/base/static
COPY docker/main.js main.js

WORKDIR /home/jovyan
COPY tweet_browser.ipynb tweet_browser.ipynb
COPY allCensus_sample.csv allCensus_sample.csv
COPY allCensus_sample_embeddings.csv allCensus_sample_embeddings.csv
COPY tweet_browser.py tweet_browser.py
COPY custom_widgets.py custom_widgets.py
COPY ui_modules.py ui_modules.py
COPY prompts.py prompts.py
COPY ai_summary.py ai_summary.py
COPY anywidget/ anywidget/
COPY images/ images/
RUN mkdir Data
WORKDIR /home/jovyan

# CMD ["voila", "tweet_browser.ipynb", "--Voila.ip=0.0.0.0 --Voila.tornado_settings=\"{'websocket_max_message_size': 209715200}\""]
CMD ["voila", "--Voila.ip=0.0.0.0", "tweet_browser.ipynb"]

#ENV NOTEBOOK_ARGS = " --no-browser --ServerApp.token='' "
# https://github.com/voila-dashboards/voila/issues/807

## docker build -t erikzhou2021/tweet_browser:latest .
## docker run -p 8866:8866 erikzhou2021/tweet_browser:latest