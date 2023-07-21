### <p>Project description</p>
This project aims to examine the trend of sentiment on Twitter related to the video game Redfall developed by
[Arkane Studios](https://bethesda.net/en/game/redfall) (A subsidiary of Bethesda).
The tweets used in this project were scraped and classified with a sentiment (positive, neutral, and negative) using unsupervised methods.
To classify new tweets, a select number of classification algorithms were developed to classify them with a sentiment, and results were cross-validated with K-Folds to determine the best model for use in this project.
___
![Redfall banner](/img/redfall_banner.png)
<p align="center", font size=8>Screengrab from Arkane Studios.</p>

### <p>Background</p>
Redfall is a video game developed by Arkane Studios and was expected to be an A-list video game. However, the game was plagued by negative events throughout its development cycle such as high staff turnover and acquisition of the game studio by Microsoft. When the game was released, many features of the game were not delivered and it was full of bugs as well, thus attracting negative feedback from the gaming community. Based on gaming news articles written about Redfall, most of the sentiment is negative toward this game. This project examines whether the negative sentiment about the game is true by analyzing tweets about Redfall from Twitter.

### <p>Problem statement</p>
There will be two objectives in this project. Firstly, this project aims to examine if the trend of the sentiment remained the same before and after the game was released to the public by analyzing tweets from Twitter. Because the tweets will need to be scraped and subsequently labeled with a corresponding sentiment through unsupervised techniques, the project will also evaluate the performance of various classification algorithms used to classify the sentiment of the tweets.

### <p>Summary</p>
From the tweets analyzed, there is no change in the level of sentiment about the game before and after the game was launched. The negative sentiments seen before the game was launched were most probably attributed to the negative events that adversely affected the game’s publicity, while the negative sentiments observed after the game was launched were attributed to the poor quality of the finished product. Both of which are unrelated but when viewed in totality are unfortunate for the publicity of the game as well as the game studio that developed it.

Among all the algorithms implemented for this classification task, the models were evaluated based on classification accuracy. The word embedding method was also important in the performance of the models. In most cases, it was found that the count vectorization word embedding method gave better results compared to the term frequency-inverse document frequency (TFIDF) method. It was observed that the tree-based classifiers performed the best using count vectorizer word embedding. Using the count vectorizer word embedding method, the decision tree classifier was the best-performing model while the K-Nearest Neighbors (KNN) model was the worst performer.

### <p>Experiment</p>
#### <p>Tweet scraping</p>
We will need data for the project, as such tweets with the hashtag _#redfall_ or found through string search “_redfall_” in Twitter will be scraped using the **[snscrape](https://github.com/JustAnotherArchivist/snscrape)** Python package. 100 tweets daily were scrapped between Feb 2, 2023, and June 1, 2023, to observe the tweet sentiments about the game before and a month after the game was released on May 2, 2023. From this, 19,663 raw tweets were scraped. This was carried out in the "[tweet_scraping.ipynb](https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/tweets_scraping.ipynb)" notebook.

#### <p>Initial EDA</p>
Examining the distribution of tweets by language, the following plot is produced.<br>
<p></p>
<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/tweets_distribution_language.png" width="800">
<p></p>
The top five languages of tweets were English, Spanish, Portuguese, French, and Japanese.
<p></p>
These tweets were then filtered where only tweets in English were kept, and duplicated tweets were excluded. This resulted in 12,467 tweets to work with for the rest of the project.
<p></p>
<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/en_tweet_over_time.png" width="800">
<p></p>
Across our time window of analysis, it is observed that the volume of tweets scraped daily is not consistent daily. This will be revisited after the sentiment labels are assigned to the tweets, where the percentage of composition of sentiment daily is examined.

#### <p>Tweet pre-processing</p>
The tweets were subjected to text pre-processing techniques: word tokenization, removal of emoticons, URI links, the hashtag symbol (#), numbers, carriage return, Twitter handles (@user), punctuations, and words with less than a length of two. These were further processed to lower the case of each word, perform word lemmatization, and final removal of product and entity nouns. The NLTK Python package was used to perform the tweet tokenization and word lemmatization.

#### <p>Sentiment labelling</p>
The **[TextBlob](https://textblob.readthedocs.io/en/dev/)** Python package was used to evaluate and return a polarity score for each tweet. A polarity score ranges between [-1, 1], where a score towards -1 suggests strong negative sentiment, and conversely, a score towards 1 suggests strong positive sentiment. A score close to 0 suggests neutral sentiment.

After the sentiment labels are applied to the tweets, a distribution plot was produced to review the results of the labeling.
<p></p>
<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/tweet_polarity_histogram.png" width=550>
<p></p>
It is observed that a large majority of the tweets were classified as neutral, and the negative (< 0) and positive (> 0) labels were assigned equally.
<p></p>
<p></p>
Next, a discrete sentiment label was applied based on the polarity of each tweet. A sentiment label of 1 denotes positive sentiment, 0 denotes neutral sentiment and -1 denotes negative sentiment. A composition of sentiments as even as possible was desired, and this was achieved by assigning the following cut-off on the polarity score.<p></p>

| Polarity score cutoff | Sentiment |
| --------------------- | --------- |
| Polarity > 0.09       | Positive  |
| 0 <= Polarity <= 0.09 | Neutral   |
| Polarity < 0          | Negative  |


