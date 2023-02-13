FROM jupyter/scipy-notebook

#RUN apt update

EXPOSE 8866/tcp

WORKDIR /home/jovyan

#RUN /usr/bin/python3 -m pip3 install --upgrade pip
RUN pip install -i https://test.pypi.org/simple/ tweet-browser-test --upgrade
RUN pip install voila bitarray regex plotly nltk umap-learn hdbscan leidenalg igraph

WORKDIR /home/jovyan
COPY allCensus_sample.csv allCensus_sample.csv
COPY simple_tweet_browser.ipynb simple_tweet_browser.ipynb
COPY tweet_browser_with_search.ipynb tweet_browser_with_search.ipynb

#CMD ["voila", "simple_tweet_browser.ipynb"]