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
RUN pip install voila bitarray regex plotly nltk umap-learn hdbscan leidenalg igraph

WORKDIR /home/jovyan/.jupyter
COPY jupyter_notebook_config.py jupyter_notebook_config.py 

WORKDIR /home/jovyan
COPY tweet_browser.ipynb tweet_browser.ipynb
COPY allCensus_sample.csv allCensus_sample.csv
COPY simple_tweet_browser.ipynb simple_tweet_browser.ipynb
COPY tweet_browser_with_search.ipynb tweet_browser_with_search.ipynb
COPY tweet_browser_simple_search.ipynb tweet_browser_simple_search.ipynb
RUN mkdir Data
WORKDIR /home/jovyan/Data
COPY viewHistory.py viewHistory.py
WORKDIR /home/jovyan

CMD ["voila", "--Voila.ip=0.0.0.0", "tweet_browser_simple_search.ipynb"]

#ENV NOTEBOOK_ARGS = " --no-browser --ServerApp.token='' "