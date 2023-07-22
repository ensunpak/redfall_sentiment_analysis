## Project description
This NLP project aims to examine the trend of sentiment on Twitter related to the video game Redfall developed by
[Arkane Studios](https://bethesda.net/en/game/redfall) (A subsidiary of Bethesda).
The tweets used in this project were scraped and classified with a sentiment (positive, neutral, and negative) using unsupervised methods.
To classify new tweets, a select number of classification algorithms were developed to classify them with a sentiment, and results were cross-validated with K-Folds to determine the best model for use in this project.

![Redfall banner](/img/redfall_banner.png)
<div align="center">Screengrab from Arkane Studios.</div>

### File organization
_Notebooks_<br>
* tweets_scraping.ipynb - Scrape the tweets
* tweets_analysis_eda.ipynb - Initial EDA of the tweets
* tweets_analysis_models.ipynb - Sentiment labeling, analysis and classification model training

_Data files_<br>
* redfall_tweets.pkl - Raw Redfall tweets with hashtag #redfall
* redfall_tweets_search.pkl - Raw Redfall tweets with search string "redfall"
* tweets_english.pkl - All raw Redfall tweets (English language only)

## 1. Background
Redfall is a video game developed by Arkane Studios and was expected to be an A-list video game. However, the game was plagued by negative events throughout its development cycle such as high staff turnover and acquisition of the game studio by Microsoft. When the game was released, many features of the game were not delivered and it was full of bugs as well, thus attracting negative feedback from the gaming community. Based on gaming news articles written about Redfall, most of the sentiment is negative toward this game. This project examines whether the negative sentiment about the game is true by analyzing tweets about Redfall from Twitter.

## 2. Problem statement
There will be two objectives in this project. Firstly, this project aims to examine if the trend of the sentiment remained the same before and after the game was released to the public by analyzing tweets from Twitter. Because the tweets will need to be scraped and subsequently labeled with a corresponding sentiment through unsupervised techniques, the project will also evaluate the performance of various classification algorithms used to classify the sentiment of the tweets.

## 3. Summary
From the tweets analyzed, there is no change in the level of sentiment about the game before and after the game was launched. The negative sentiments seen before the game was launched were most probably attributed to the negative events that adversely affected the game’s publicity, while the negative sentiments observed after the game was launched were attributed to the poor quality of the finished product. Both of which are unrelated but when viewed in totality are unfortunate for the publicity of the game as well as the game studio that developed it.

Among all the algorithms implemented for this classification task, the models were evaluated based on classification accuracy. The word embedding method was also important in the performance of the models. In most cases, it was found that the count vectorization word embedding method gave better results compared to the term frequency-inverse document frequency (TFIDF) method. It was observed that the tree-based classifiers performed the best using count vectorizer word embedding. Using the count vectorizer word embedding method, the decision tree classifier was the best-performing model while the K-Nearest Neighbors (KNN) model was the worst performer.

## 4. Sentiment Analysis

### 4.1 Tweet scraping
We will need data for the project, as such tweets with the hashtag _#redfall_ or found through string search “_redfall_” in Twitter will be scraped using the **[snscrape](https://github.com/JustAnotherArchivist/snscrape)** Python package. 100 tweets daily were scrapped between Feb 2, 2023, and June 1, 2023, to observe the tweet sentiments about the game before and a month after the game was released on May 2, 2023. From this, 19,663 raw tweets were scraped. This was carried out in the "[tweet_scraping.ipynb](https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/tweets_scraping.ipynb)" notebook.

### 4.2 Initial EDA
Examining the distribution of tweets by language, the following plot is produced.<br>

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/tweets_distribution_language.png" width="800">

The top five languages of tweets were English, Spanish, Portuguese, French, and Japanese.

These tweets were then filtered where only tweets in English were kept, and duplicated tweets were excluded. This resulted in 12,467 tweets to work with for the rest of the project.

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/en_tweet_over_time.png" width="800">

Across our time window of analysis, it is observed that the volume of tweets scraped daily is not consistent daily. This will be revisited after the sentiment labels are assigned to the tweets, where the percentage of composition of sentiment daily is examined.

### 4.3 Tweet pre-processing
The tweets were subjected to text pre-processing techniques: word tokenization, removal of emoticons, URI links, the hashtag symbol (#), numbers, carriage return, Twitter handles (@user), punctuations, and words with less than a length of two. These were further processed to lower the case of each word, perform word lemmatization, and final removal of product and entity nouns. The [NLTK](https://www.nltk.org/) Python package was used to perform the tweet tokenization and word lemmatization.

### 4.4 Sentiment labeling
The **[TextBlob](https://textblob.readthedocs.io/en/dev/)** Python package was used to evaluate and return a polarity score for each tweet. A polarity score ranges between [-1, 1], where a score towards -1 suggests strong negative sentiment, and conversely, a score towards 1 suggests strong positive sentiment. A score close to 0 suggests neutral sentiment.

After the sentiment labels are applied to the tweets, a distribution plot was produced to review the results of the labeling.

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/tweet_polarity_histogram.png" width=550>

It is observed that a large majority of the tweets were classified as neutral, and the negative (< 0) and positive (> 0) labels were assigned equally.

Next, a discrete sentiment label was applied based on the polarity of each tweet. A sentiment label of 1 denotes positive sentiment, 0 denotes neutral sentiment and -1 denotes negative sentiment. A composition of sentiments as even as possible was desired, and this was achieved by assigning the following cut-off on the polarity score.<p></p>

| Polarity score cutoff | Sentiment |
| --------------------- | --------- |
| Polarity > 0.09       | Positive  |
| 0 <= Polarity <= 0.09 | Neutral   |
| Polarity < 0          | Negative  |

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/pie_chart_sentiment_dist.png" width=350>

The pie chart (taboo visualization type 🤭) shows the distribution of the discrete sentiment labels of the tweets after applying the polarity score cutoffs according to the table above.

The ten most used words in tweets that are both positive and negative can be seen in the plot produced below. The count was transformed with natural log to make the points easier to read. Interestingly, most of the top words are found in positive tweets.

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/top_10_occuring_words.png" width=450>

The trend of the relative composition of the daily tweets was examined. This would be an appropriate approach to determine the sentiment trend of the tweets over the observation window, given that the daily sample size of tweets is not consistent.

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/sentiment_dist_ts.png" width=900>

Taking a different view of the sentiment trends by plotting each sentiment trend separately, we have the following plot.

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/sentiment_breakdown.png" width=960>

It is quite hard to tell the direction of the sentiments around the launch date. To make it clearer, linear models were fitted on one month of data before and after the launch date for both positive and negative sentiment plots. The following plot is produced.

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/sentiment_breakdown_trendline.png" width=960>

Now it can be seen clearly that the negative sentiment decreased towards the release date but increased again after. This confirms our expectation of the public sentiment of the game after its launch, however, it is interesting to note that negative sentiment actually decreased closer to when the game was about to be released.

## 5. Clustering Model Experiment

### 5.1 Word embedding (Feature extraction)
Two approaches to embedding the word tokens in each tweet were implemented: **Count vectorizer** and **Term Frequency Inverted Document Frequency (TFIDF)**.

The count vectorizer method collects the unique tokens found in the corpus of tweets and creates an index for each token and assigns a numerical value to it. Next, a count of the occurrences of each token in each tweet is performed and a matrix called document-term matrix (DTM) is formed. Each tweet is represented as a high-dimensional vector, each element in the vector corresponds to the count of a specific token in that tweet.

The TFIDF method measures how frequently a word token appears in a document. First, the term frequency in a tweet is obtained and then it is divided against the total number of terms in the entire tweet. The higher this value, the more important the term is in the context of the tweet. The inverse document term component takes the log of the ratio between the number of documents and the number of documents that contain the term. The TFIDF score is then calculated by multiplying the term frequency with the inverse document frequency. A low score suggests that the term is infrequent in the corpus and is less impactful, while a high score suggests that the term is frequent and more valuable.

## 6. Classification models
To classify new unseen tweets about Redfall, a classification model will be developed to do that. The model will be trained on the word-embedded corpus, and the following five algorithms were selected to classify a sentiment for a given tweet in this project.

1.	K-Nearest Neighbors (KNN)
2.	Decision tree classifier
3.	Random forest classifier
4.	Linear SVC (SVM)
5.	Naïve-Bayes

The model will be trained on both the word embedding performed by count vectorizer and TFIDF methods respectively. The contribution to performance from the size of the train/test dataset split will also be examined through cross-validation with different numbers of folds. Finally, the best performance from the combination of the model and word embedding approach will be recommended for use. The cross-validated classification accuracy of the model by K-Folds will be used to determine the model that performs the best.

### 6.1 Model performance
After cross-validating the models on the tweets with K-Folds, the accuracy of each model for each word embedding set is visualized in the charts below.

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/cv_model_result.png" width=700>

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/tfidf_model_result.png" width=700>

The performance of each type of algorithm is consistent among both word embedding methods where the decision tree, random forest, and linear SVC are the top performers while KNN is the worst. It is also observed that there are diminishing returns on model performance when the K-Fold parameter is greater than five.

<img src="https://github.com/ensunpak/redfall_sentiment_analysis/blob/main/img/comparison_performance.png" width=900>

Comparing each algorithm between the word embedding methods, it can clearly be seen that word embedding with count vectorizer tends to give better results in this task.

## 7. Conclusion
In this project, the sentiment trends about the Redfall video game before and after it was launched were analyzed. Tweets related to Redfall were scraped from Twitter. The tweets were then classified with a sentiment based on their polarity score. Subsequently, clustering models were fit with K-Fold cross-validation on the corpus of tweets to classify new tweets not in the corpus. Two word embedding methods such as count vectorizer and TFIDF were also used to observe the effect on the model performance.

From the analysis, it is concluded that despite the bad publicity surrounding Redfall's development, the negative sentiment trend was surprisingly on a downtrend close to the game's release date. However, this trend went back up, most probably caused by the poor quality of the final game. Word embedding with count vectorizer gave the best performance in the cross-validated models with K-Fold. There is not much model performance gain after 5 K-Folds.
