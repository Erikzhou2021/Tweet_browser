FROM jupyter/scipy-notebook

#RUN apt update
#ENV DEBIAN_FRONTEND noninteractive

#EXPOSE 8866/tcp

WORKDIR /home/jovyan

# USER root
# RUN apt-get update && apt-get install -y curl
# USER 1000

#RUN /usr/bin/python3 -m pip3 install --upgrade pip
RUN pip install -i https://test.pypi.org/simple/ tweet-browser-test --upgrade
RUN pip install bitarray regex plotly nltk umap-learn hdbscan leidenalg igraph

# Need to roll back versions (Not tested on docker yet)
RUN pip install --upgrade "pyzmq<25" "jupyter_client<8"
RUN pip install --upgrade "voila<0.5"

WORKDIR /home/jovyan/.jupyter
COPY jupyter_notebook_config.py jupyter_notebook_config.py 

WORKDIR /home/jovyan
COPY tweet_browser.ipynb tweet_browser.ipynb
COPY allCensus_sample.csv allCensus_sample.csv
COPY simple_tweet_browser.ipynb simple_tweet_browser.ipynb
COPY tweet_browser_with_search.ipynb tweet_browser_with_search.ipynb
COPY tweet_browser_simple_search.ipynb tweet_browser_simple_search.ipynb
RUN mkdir Data
COPY viewHistory.py Data/viewHistory.py
WORKDIR /home/jovyan

CMD ["voila", "--Voila.ip=0.0.0.0 --Voila.tornado_settings=\"{'websocket_max_message_size': 209715200}\" ", "simple_tweet_browser.ipynb"]

#ENV NOTEBOOK_ARGS = " --no-browser --ServerApp.token='' "
# New CMD was not tested yet
# https://github.com/voila-dashboards/voila/issues/807