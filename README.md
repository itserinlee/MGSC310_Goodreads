
# MGSC 310 Final Project: Building Predictive Models
*Fall 2020 MGSC-310-01 Final Project using the **"Goodreads-books"** data set from Kaggle*


![mixkit-person-using-binoculars-peering-out-from-behind-a-bush-24-original (2)](https://user-images.githubusercontent.com/69367085/102598800-68787780-40d1-11eb-80e3-87a247d07ba0.png)

(Image source: https://mixkit.co/free-stock-art/person-using-binoculars-peering-out-from-behind-a-bush-24/ )

# Reference
https://www.kaggle.com/jealousleopard/goodreadsbooks

# Note
- Scraped from Goodreads API

# Programming Language
- R

# Data At-A-Glance
- variables used in project:
  - outcome: “average_rating”
  - predictors (9 total):
    - "num_pages"
    - “book_ratings_count”
    - “text_reviews_count”
    - “title_sentiment_avg”
    - “authorworkcount”
    - “author_fans”
    - “author_ratings_count
    - “author_review_count”
    - “gender”
- removed 11 variables from original data set

# Known Errors
N/A

# Instructions
- Open & run in RStudio.

# Models
*1. Linear Regression*
![coef_table](https://user-images.githubusercontent.com/69367085/102663560-f2f2c280-4135-11eb-9874-a155fc480e90.jpg)

![lr_coef](https://user-images.githubusercontent.com/69367085/102663748-4cf38800-4136-11eb-96c1-405ece9d064a.jpg)

*2. Elastic Net*

![plot_enet](https://user-images.githubusercontent.com/69367085/102663800-65fc3900-4136-11eb-8e45-0c6eff858a87.jpg)

![coef_path](https://user-images.githubusercontent.com/69367085/102663836-744a5500-4136-11eb-9541-e3ddbf89beda.jpg)

*3. Bootstrap Aggregated (Bagged) Decision Tree*

![bagged_tree](https://user-images.githubusercontent.com/69367085/102663882-875d2500-4136-11eb-8139-68e4e9a6e493.jpg)

*4. Random Forest*

![plot_rf_fit](https://user-images.githubusercontent.com/69367085/102663922-9b088b80-4136-11eb-89b9-f2f88739ae47.jpg)

![rf_var_imp](https://user-images.githubusercontent.com/69367085/102664000-be333b00-4136-11eb-9a5d-a20cecf0ca15.jpg)

![rf_explainer_pckg](https://user-images.githubusercontent.com/69367085/102664015-c68b7600-4136-11eb-8064-726edc9d7ad2.jpg)

# Model Evaluation: Comparison of Metrics

![model_metrics](https://user-images.githubusercontent.com/69367085/102664028-ce4b1a80-4136-11eb-8656-7db3be418ea2.jpg)
