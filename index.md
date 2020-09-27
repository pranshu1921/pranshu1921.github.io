## Data Science Portfolio

Hi, I am Pranshu Kumar.
Welcome to my Data Science portfolio page. This contains all my Data Science projects completed for academic, self learning, and professional consideration.
Most of the projects are presented in the form of Jupyter notebooks.


**Table of Contents**

 - [Machine Learning](#machine-learning)
   - *[Personalized Cancer Diagnosis](#Personalized-Cancer-Diagnosis)*
   - *[Facebook Friend Recommendation using Graph Mining](#dcgan-deep-convolutional-generative-adversarial-networks)*
   - *[NYC Taxi Demand Prediction](#face-generator)*
   - *[Microsoft Malware Detection](#image-classification)*
   - *[Quora Question Pair Similarity Problem](#train-a-quadcopter-how-to-fly)*
   - *[Stackoverflow Tag Predictor](#deep-qlearning-for-cart-pole)*
   - *[Amazon Fashion Discovery Engine](#train-a-self-driving-car)*
   - *[Netflix Movie Recommendation System](#topic-modeling)*
   - *[Amazon Fine Food Reviews](#word-embeddings)*
   - *[Battle of Neighborhoods](#finding-donors-for-charityml)*
   - *[Twitter Sentiments Analysis](#creating-customer-segments)*
  
 - [Data Analysis and Visualizations](#artificial-intelligence)
   - *[FIFA 2018 World's Best XI](#adversial-search)*
   - *[Nobel Prize Winners Visual History](#clasical-search)*
   - *[The Android App Market on Google Play](#coding-a-neural-network)*
   - *[Risk and Returns : The Sharpe Ratio](#predicting-boston-housing-prices)*
   - *[Reducing Traffic Mortality in USA](#titanic-survival-exploration)*
   - *[Generating Keywords for Google Ads](#coding-a-neural-network)*
   - *[Exploring the history of 67 years of LEGO](#predicting-boston-housing-prices)*
   
 - [SQL projects](#machine-learning)
   - *[Analyze International Debt Statistics](#price-prediction-for-ecommerce-products)*
   
- [Micro ML projects](#machine-learning)
   - *[Cricket Chirps Linear Regression](#price-prediction-for-ecommerce-products)*
   - *[Bigmart Sales Prediction](#dcgan-deep-convolutional-generative-adversarial-networks)*
   - *[Iris Data Logistic Regression](#face-generator)*
   - *[Random Password Generator](#image-classification)*
   - *[Rock Paper Scissors](#train-a-quadcopter-how-to-fly)*
   - *[KNN CLassifier on Titanic Dataset](#deep-qlearning-for-cart-pole)*
   
- [R Projects](#machine-learning)
   - *[NYC Births Time Series](#price-prediction-for-ecommerce-products)*
   - *[Clustering, Decision Tree on Iris data](#dcgan-deep-convolutional-generative-adversarial-networks)*   
   
   
## Machine Learning Projects

### Personalised Cancer Diagnosis  
> **Keywords**: MSKCC | Genetic-Variations | Classification | Genetic-Mutations | Neutral-Mutations. *[View Source](https://github.com/pranshu1921/Personalised-Cancer-Diagnosis)*.

This machine learning case study involves analyzing _**Memorial Sloan Kettering Cancer Center (MSKCC)**_ data for predicting the effect of **genetic variations in the cancer tumors** for enabling personalised medicine. By given data, we classify the given **genetic variations/mutations** based on evidence from text-based clinical literature.

### Facebook Friend Recommendation using Graph Mining
> **Keywords**: Facebook | Graph-Mining | Supervised Learning | XGBoost | Kaggle. *[View Source](https://github.com/pranshu1921/Facebook-Friend-Recommenation-Graph-Mining)*

The project involves using data from the FacebookRecruiting challenge on Kaggle to predict missing links from a given directed social graph to recommend users. This is a supervised machine learning problem.
Generated training samples of good and bad links from given directed graph and for each link got some features like no of followers, is he followed back, page rank, katz score, adar index, some svd fetures of adj matrix, some weight features etc. and trained ml model based on these features to predict link.

### NYC Taxi Demand Prediction
> **Keywords**: Time Series | Regression | XGBoost | CloudPickle | Folium. *[View Source](https://github.com/pranshu1921/Taxi-Demand-Prediction-NYC)*

This is a **time-series forecasting and regression** problem to find number of pickups, given location corrdinates(latitude and longitude) and time, in the surrounding regions, using data collected in Jan - Mar 2015 to predict the pickups in Jan - Mar 2016, provided by the **[NYC Taxi and Limousine Commission(TLC)]( http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml).**

### Microsoft Malware Detection
> **Keywords**: Multiclass-classification | Malware-analysis | K-Nearest-Neighbors | Decision-Tree | Logistic-Regression | Random-Forest-Classifier. *[View Source](https://github.com/pranshu1921/Microsoft-Malware-Detection)*

In the past few years, the malware industry has grown very rapidly that, the syndicates invest heavily in technologies to evade traditional protection, forcing the anti-malware groups/communities to build more robust softwares to detect and terminate these attacks. The major part of protecting a computer system from a malware attack is to identify whether a given piece of file/software is a malware.

There are nine different classes of malware that we need to classify a given a data point => Multi class classification problem.

### Quora Question Pair Similarity
> **Keywords**: Classification | NLP | Fuzzy-Matching | LinearSVM | XGBoost | Fuzzy-wuzzy. *[View Source](https://github.com/pranshu1921/Quora-Question-Pair-Similarity)*

This project involved a Kaggle competition hosted by Quora.com for finding which questions on Quora are duplicates of questions that have already been asked. Predictions were also made whether a pair of questions are duplicates or not.

- It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.
- Used Natural Language Processing and Fuzzy Features for Advanced feature extraction.
- Compared Logistic Regression, Linear SVM, and XGBoost for finding the best model for classification.


### Stackoverflow Tag Predictor
> **Keywords**: Multilabel-classification | keyword-extraction | Tag-Predictor | wordcloud | sqlalchemy. *[View Source](https://github.com/pranshu1921/Stackoverflow-Tag-Predictor)*

This is a **multi-label classification problem** to Identify keywords and tags from millions of text questions and suggest the tags based on the content that there is in the question posted on Stackoverflow.
It uses dataset provided in the **['Facebook Recruiting III - Keyword Extraction'](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/)** posted on Kaggle.

### Amazon Fashion Discovery Engine
> **Keywords**: Recommendation-system | content-based | Word2Vec | Tf-Idf | BagofWords | CNN | Keras | Gensim. *[View Source](https://github.com/pranshu1921/Amazon-Fashion-Discovery-Engine)*

This is a content based recommendation engine for recommending apparel items or products at Amazon, using text and image data retreived from website. Suggested text based recommendations using **Bag of Words (BoW)**, **Word2Vec** and **TF-IDF** techniques. Made image based recommendations using **Convolutional Neural Network(CNN)**.


### Netflix Movie Recommendation System
> **Keywords**: Recommendation-system | Regression | Cinematch | XGBoost | XGBoost | KNN. *[View Source](https://github.com/pranshu1921/Netflix-Movie-Recommendation-System)*

A classic recommendation system machine learning problem to predict movie rating to be given by a user that he/she has not yet rated on [Netflix](https://www.netflix.com).
It is also seen as a regression problem.
Data is picked up from [this](https://www.kaggle.com/netflix-inc/netflix-prize-data) Kaggle challenge.

### Amazon Fine Food Reviews
> **Keywords**: NLP | Porter-Stemmer | Bag-of-Words | Beautiful-Soup | n-grams | Tf-Idf | Word2Vec. *[View Source](https://github.com/pranshu1921/Amazon-Fine-Food-Reviews)*

This is a _**Natural Language Processing(NLP)**_ based project that uses data provided on the [**'Amazon Fine Food reviews'**](https://www.kaggle.com/snap/amazon-fine-food-reviews/tasks?taskId=797) challenge posted on Kaggle, to determine the polarity of a given user review, following a score/rating of 4 or 5 considered positive, 1 or 2 negative and 3 neutral and ignored.

### The Battle of Neighborhoods
> **Keywords**: Capstone | Unsupervised-Learning | K-Means | Clustering | Londom-crimes | Beautiful-Soup | Google-Maps-API. *[View Source](https://github.com/pranshu1921/Coursera_Capstone)*

This is the the Capstone Project - The Battle of the Neighborhoods for the  Applied Data Science Capstone by IBM/Coursera.
This project aims to select the safest borough in London based on the total crimes, explore the neighborhoods of that borough to find the 10 most common venues in each neighborhood and finally cluster the neighborhoods using k-mean clustering.

### Twitter Sentiments Analysis
> **Keywords**: Non-negative-matrix-factorization | Tf-Idf | Stopwords | wordcloud. *[View Source](https://github.com/pranshu1921/Twitter_Sentiments_Analysis)*

This repository contains a comprehensive Jupyter notebook to discover and visualize topics from a corpus of Twitter tweets.
Calculated tf-idf matrix for non-negative matrix factorization, filtering stop words and getting words' frequency in the corpus. Visualised top words using 'wordcloud' package in Python.



## Data Analysis and Visualizations Projects 

### FIFA 2018 World's Best XI
> **Keywords**: FIFA | Pandas | Numpy | Matplotlib. *[View Source](https://github.com/pranshu1921/FIFA2018_World-s_Best_XI)*.

Once in every 4 years, we celebrate FIFA World Cup. All priorities change to footbaall, and all predications change to the teams and players that perform in the tournament. So, it was quite exciting to "Predict the World's Best XI players" in FIFA 2018 using Python for Data Analysis.


### Nobel Prize Winners Visual History
> **Keywords**: Nobel-Prize | Python | Pandas | Numpy | Matplotlib. *[View Source](https://github.com/pranshu1921/Nobel-Prize-Winners-Visual-History)*

The Nobel Foundation has made a dataset available of all prize winners from the start of the prize, in 1901, to 2016.
This data analytics project is all about analyzing data of all Nobel Prize winners from its beginning to 2016, all compiled by **the Nobel Foundation**.

### The Android App Market on Google Play
> **Keywords**: Sentiment-Analysis| Seaborn | Pandas | Numpy | Matplotlib. *[View Source](https://github.com/pranshu1921/Android-App-Market-Google-Play)*

This data analytics project analyzes the Google Play Store data by comparison of apps across categories to look for data insights to devise strategies to **drive growth and retention**.

### Risk and Returns : The Sharpe Ratio
> **Keywords**: Sharpe-Ratio| Facebook | Amazon | Seaborn | Pandas | Numpy | Matplotlib. *[View Source](https://github.com/pranshu1921/Risk-and-Returns-Sharpe-Ratio)*

The Sharpe ratio is usually calculated for a portfolio and uses the risk-free interest rate as benchmark. We will simplify our example and use stocks instead of a portfolio. We explore Facebook and Amazon stocks and calculate the Sharpe ratio, for analysis using Python.


### Reducing Traffic Mortality in USA
> **Keywords**: Data-Wrangling | Clustering | Principal-Component-Analysis | Linear-Regression. *[View Source](https://github.com/pranshu1921/Reducing-Traffic-Mortality-USA)*

We analyze data collected by the _**National Highway Traffic Safety Administration**_ and _**the National Association of Insurance Commissioners**_ to wrangle, plot, dimensionally reduce and cluster data to make an attempt to find patterns and help _**reduce Traffic Mortality in USA**_.


### Generating Keywords for Google Ads
> **Keywords**: Google-AdWords| Python | Pandas | exact-match. *[View Source](https://github.com/pranshu1921/Generating-Keywords-Google-Ads)*

This analysis project focuses on generating keywords for ad campaigns for triggering the right ad using Google AdWords using Python.


### Exploring the history of 67 years of LEGO
> **Keywords**: LEGO| Python | Pandas. *[View Source](https://github.com/pranshu1921/Exploring-67-years-of-Lego)*

This data analysis micro project explores the glorious 67 years of LEGO database provided by [Rebrickable](https://rebrickable.com/downloads/).



## SQL Projects

### Analyzing International Debt Statistics
> **Keywords**: Sqlite | mySQL | aggregate-functions | ddl-commands | dml-commands. *[View Source](https://github.com/pranshu1921/Analyzing-International-Debt-Statistics)*.

This is a SQL project which analyzes the international debt data collected by the [World Bank](https://www.worldbank.org/).
We seek to find total debt owned by conuntries, the country with the highest debt amount and more relevant info.



## Micro ML/Python Projects

### Cricket Chirps Linear Regression  
> **Keywords**: Linear-Regression | Harvard-College-Press. *[View Source](https://github.com/pranshu1921/Cricket_Chirps_Linear_Regression)*.

THis repo contains files including Jupyter notebook that containing a walkthrough about using Linear Regression for predicting temperature from the number of chirps from the 'The Song of Insects', by Dr. G. W. Pierce, Harvard College Press, using Linear Regression.

### Bigmart Sales Prediction
> **Keywords**: Logistic-Regression | Regression-problem. *[View Source](https://github.com/pranshu1921/Bigmart_Sales_Prediction)*.

his repo contains the project of predicting the Big mart sales using Logistic Regression.
As the name suggests, the dataset comprises of transaction records of a sales store. This is a regression problem.

### Iris Data Logistic Regression
> **Keywords**: Classification | Logistic-Regression. *[View Source](https://github.com/pranshu1921/IrisDataLogisticRegression)*.

The Iris Flower data set is probably the most versatile, easy and resourceful dataset in pattern recognition literature.
We created a classification (Logistic Regression) model to predict the class of the flower based on available attributes.

### Random Password Generator
> **Keywords**: ASCII | Python. *[View Source](https://github.com/pranshu1921/RandomPasswordGenerator)*.

A Python script to generate a random password of 8 characters in accordance with ASCII code.



### Rock Paper Scissors
> **Keywords**: Python. *[View Source](https://github.com/pranshu1921/RockPaperScissors)*.

This is an interactive Python game of Rock, Paper and Scissors. The user competes with computer for the win.

### KNN CLassifier on Titanic Dataset
> **Keywords**: KNN-Classifier | Titanic | Python. *[View Source](https://github.com/pranshu1921/Machine_Learning_Internshala/tree/master/7%20k-NN)*.

Implemented KNN-Classifier on the cleaned version of the very known 'Titanic Survival' dataset, easily found on kaggle.com, as part of a machine learning course on [Internshala.com](https://www.internshala.com).



## R Projects

### NYC Births Time Series 
> **Keywords**: Time-Series | ARIMA | Auto-Correlation | Exponential-Smoothing. *[View Source](https://github.com/pranshu1921/Time-Series-NYC-Births)*.

The project deals with applying time series data in R on a dataset for decomposing the **seasonal time series, forecasting** using **Exponential Smoothing** and using **ARIMA** to address issues of correlations between successive values of time series on data of **number of births per month in New York city, from January 1946 to December 1959**.

### Clustering, Decision Tree on Iris data
> **Keywords**: kmeans-clustering | decision-trees | density-based-clustering | R-programming. *[View Source](https://github.com/pranshu1921/Clustering-Decision-Tree-Iris-data)*.

This project involves implementing Decision trees, k-means clusters and density based clusters on Iris dataset in R.

--

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

## Contact

If you liked what you've seen, want to have a chat with me about the portfolio, work opportunities, or collaboration, feel free to reach out:

* Email: pranshu1921@gmail.com
* [LinkedIn](https://www.linkedin.com/in/pranshu-kumar/)
* Twitter: @Pranshu1921
