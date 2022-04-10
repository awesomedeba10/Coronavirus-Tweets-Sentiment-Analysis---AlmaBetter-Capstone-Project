## Coronavirus Tweets Sentiment Analysis

![image](https://user-images.githubusercontent.com/39692126/162616790-2161d8a9-b452-42cc-af68-6ffd8abe14bb.png)

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)]()
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/powered-by-responsibility.svg)](https://forthebadge.com)


NLP or Natural Language Processing is a new emerging hot topic in field of Data Science & Machine Learning. NLP is used to interpret human language and behavior. NLP combines the power of linguistics and computer science to study the rules and structure of language, and create intelligent systems (run on machine learning and NLP algorithms) capable of understanding, analyzing, and extracting meaning from text and speech. 

Sentiment Analysis is one of the convenient applications of NLP. It does the task of classifying the polarity of a given text. For instance, a text-based tweet can be categorized into either "positive", "negative", or even "neutral" also. Given the text and accompanying labels, a model can be trained to predict the correct sentiment. Vader, TextBlob, Google Cloud Natural Language API are examples of some of the pretrained model for sentiment analysis. 

This analysis focuses on the supervised ML-based approach, which is computationally fast and exhibits promising classification results. Aspect-based analysis has been performed using a text classifier model built from scratch.  Statistical models like Naïve Bayes, SVM, DecisionTree has been used to train and evaluate performance of our model, all these algorithms are available in python scikit-learn library. The model has been evaluated using standard metrics like balanced accuracy, f1 score, roc-auc score etc. The best model among those has been selected to predict the final result.

## 🔧 Libraries used:
| Base Library 		    | Visualization		    | ML  	|
|---			      		|---		    		|---		      	|
| - Numpy	    | - Matplotlib		    | - scikit-learn		|
| - Pandas				    | - seaborn			    | - imblearn			|
| - nltk		    | - plotly			    | - 			|
| - regex			    | - WordCloud			    | - 			|
| - json	    | - 			    | - 			|


## 🔧 Steps used:
* Text Preprocessing
* Text Normalization : Stemming
* Pipeline - Text Vectorization : CountVectorizer
* Pipeline - Model Training : MultinomialNB, LinearSVC, DecisionTreeClassifier, RandomForestClassifier, KNeighborClassifier
* Classification
* Prediction

## Metrics Chart

| model | class_counts	| accuracy |	precision |	recall |	f1_score |	best_params |	group |
|---	|---            |---       |---           |---     |---          |---	        |---	  |
| LinearSVC |	3 |	0.782677 |	0.758887 |	0.774365 |	0.764632 |	{'count__max_df': 0.1, 'count__max_features': ... |	No oversampling 3 Classes |
| LinearSVC | 	3 | 	0.773324 | 	0.750917 | 	0.772585 | 	0.756545 | 	{'count__max_df': 0.07, 'count__max_features':... | 	Over sampled 3 Classes |
| LinearSVC | 	3 | 	0.758503 | 	0.740619 | 	0.765132 | 	0.744295 	 | {'count__max_df': 0.1, 'count__max_features': ...	 | Down sampled 3 Classes |
| DecisionTreeClassifier | 	3 | 	0.716594 |	0.702761 |	0.728597	 | 0.711387	 | {'count__max_df': 0.1, 'count__max_features': ... | 	Over sampled 3 Classes |
| DecisionTreeClassifier | 	3 | 	0.686589 |	0.673124 |	0.704074	 | 0.680350	 | {'count__max_df': 0.1, 'count__max_features': ... | 	No oversampling 3 Classes |
| DecisionTreeClassifier | 	3 | 	0.680394 |	0.681406 |	0.705867	 | 0.678373	 | {'count__max_df': 0.07, 'count__max_features':... | 	Down sampled 3 Classes |
| MultinomialNB | 	3	 | 0.680637	 | 0.660151	 | 0.677974	 | 0.664728	 | {'count__max_df': 1.0, 'count__max_features': ...	 | No oversampling 3 Classes |
| MultinomialNB | 	3	 | 0.677235	 | 0.657858	 | 0.677271	 | 0.661878	 | {'count__max_df': 1.0, 'count__max_features': ...	 | Over sampled 3 Classes |
| MultinomialNB | 	3	 | 0.671283	 | 0.651277	 | 0.668895	 | 0.653743	 | {'count__max_df': 0.1, 'count__max_features': ...	 | Down sampled 3 Classes |
| LinearSVC | 	5	 | 0.554908	 | 0.555648	 | 0.593317	 | 0.565862	 | {'count__max_df': 0.4, 'count__max_features': ...	 | No oversampling 5 Classes |
| RandomForestClassifier | 	5	 | 0.557580	 | 0.604879	 | 0.534083	 | 0.547559	 | {'count__max_df': 0.1, 'count__max_features': ...	 | No oversampling 5 Classes |
| MultinomialNB | 	5	 | 0.484451	 | 0.494538	 | 0.497850	 | 0.496112	 | {'count__max_df': 0.55, 'count__max_features':...	 | No oversampling 5 Classes |
| DecisionTreeClassifier | 	5	 | 0.500364	 | 0.492508	 | 0.506935	 | 0.493443	 | {'count__max_df': 0.1, 'count__max_features': ...	 | No oversampling 5 Classes |
| KNeighborsClassifier | 	5	 | 0.296161	 | 0.394659	 | 0.297510	 | 0.264569	 | {'count__max_df': 1.0, 'count__max_features': ...	 | No oversampling 5 Classes |

## 🤝 Connect with me on
* Debanjan:
<br> [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/awesomedeba10/)
