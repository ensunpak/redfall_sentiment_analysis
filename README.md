### <p>Project description</p>
This project aims to examine the trend of sentiment on Twitter related to the video game Redfall developed by
[Arkane Studios](https://bethesda.net/en/game/redfall) (A subsidiary of Bethesda).
The tweets used in this project were scraped and classified with a sentiment (positive, neutral, and negative) using unsupervised methods.
To classify new tweets, a select number of classification algorithms were developed to classify them with a sentiment, and results were cross-validated with K-Folds to determine the best model for use in this project.
___
![Redfall banner](/img/redfall_banner.png)

### <p>Background</p>
Redfall is a video game developed by Arkane Studios and was expected to be an A-list video game. However, the game was plagued by negative events throughout its development cycle such as high staff turnover and acquisition of the game studio by Microsoft. When the game was released, many features of the game were not delivered and it was full of bugs as well, thus attracting negative feedback from the gaming community. Based on gaming news articles written about Redfall, most of the sentiment is negative toward this game. This project examines whether the negative sentiment about the game is true by analyzing tweets about Redfall from Twitter.

### <p>Problem statement</p>
There will be two objectives in this project. Firstly, this project aims to examine if the trend of the sentiment remained the same before and after the game was released to the public by analyzing tweets from Twitter. Because the tweets will need to be scraped and subsequently labeled with a corresponding sentiment through unsupervised techniques, the project will also evaluate the performance of various classification algorithms used to classify the sentiment of the tweets.

### <p>Summary</p>
From the tweets analyzed, there is no change in the level of sentiment about the game before and after the game was launched. The negative sentiments seen before the game was launched were most probably attributed to the negative events that adversely affected the game’s publicity, while the negative sentiments observed after the game was launched were attributed to the poor quality of the finished product. Both of which are unrelated but when viewed in totality are unfortunate for the publicity of the game as well as the game studio that developed it.

Among all the algorithms implemented for this classification task, the models were evaluated based on classification accuracy. The word embedding method was also important in the performance of the models. In most cases, it was found that the count vectorization word embedding method gave better results compared to the term frequency-inverse document frequency (TFIDF) method. It was observed that the tree-based classifiers performed the best using count vectorizer word embedding. Using the count vectorizer word embedding method, the decision tree classifier was the best-performing model while the K-Nearest Neighbors (KNN) model was the worst performer.

### <p>Experiment</p>
#### <p>Tweet scraping</p>
We will need data for the project, as such tweets with the hashtag _#redfall_ or found through string search “_redfall_” in Twitter will be scraped using the **[snscrape](https://github.com/JustAnotherArchivist/snscrape)** Python package. 100 tweets daily were scrapped between Feb 2, 2023, and June 1, 2023, to observe the tweet sentiments about the game before and a month after the game was released on May 2, 2023. From this, 19,663 raw tweets were scraped. This was carried out in the "tweet_scraping.ipynb" notebook.
