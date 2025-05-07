# Tweet Browser
A simple browser for analyzing Tweets and other social media datasets in .csv form written in Jupyter Notebook. Tweet Browser **does not** contain any features to scrape or generate data, so you must supply your own dataset. However, a sample dataset is provided. For a full feature list, see our finalized design [here](https://docs.google.com/presentation/d/1sLSYjo5U0F0wkxhPnb0jBVAOje8rez0vum3B_OEw8dc/edit?slide=id.p#slide=id.p)

## Installation
1. Clone the repo
`git clone https://github.com/Erikzhou2021/Tweet_browser.git`
or 
`git clone git@github.com:Erikzhou2021/Tweet_browser.git`
2. Change to the Frontend director
`cd Tweet_browser/Frontend/`

3. Install dependencies
`pip install -r requirements.txt`

4. Run the application
`voila tweet_browser.ipynb`

The application will be available at [localhost:8866]

Alternatively, pull the latest Docker build using
`docker pull erikzhou2021/tweet_browser:latest`
Then run with
`docker run -p 8866:8866 erikzhou2021/tweet_browser:latest`

## Documenation
For full documentation of the code base see [this doc](https://docs.google.com/document/d/1uauYSeUKuyeZZUwdbe6CVYJzr8YrTh0jxaB8ocpMj94/edit?tab=t.0)

## Issues
Our finalized design includes an interactive user interface to help streamline file uploads. However, this has not been implemented and the current version may not work in all cases. 