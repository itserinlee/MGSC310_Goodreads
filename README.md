
# MGSC 310 Final Project: Building Predictive Models
***Fall 2020 MGSC-310-01***

*Corinne Smith, Erin Lee, Jon Le, Adam Gonzalez, Debbie Lu*

## Datasets
- [books.csv](https://www.kaggle.com/jealousleopard/goodreadsbooks)
- [final_dataset.csv](https://www.kaggle.com/choobani/goodread-authors?select=final_dataset.csv)
###### Note: Both Kaggle data sets were originally scraped from the Goodreads API.

## Programming Language
R

## Data At-A-Glance
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

###### Note: removed 11 variables from original data set

## Instructions

###### • Recommended: Run the program using the R Markdown format: "MGSC310FinalProject.Rmd" • 
- Download the "datasets" folder.
- Download the "MGSC310FinalProject.Rmd" file.
- Given that the "datasets" folder & "MGSC310FinalProject.Rmd" file are in the same directory, open & run the R Markdown file in [RStudio](https://rstudio.cloud/).

## Models
### 1. Linear Regression

*Table of Coefficients*

![coef_table](https://user-images.githubusercontent.com/69367085/102663560-f2f2c280-4135-11eb-9874-a155fc480e90.jpg)

*Plot of Coefficients*

![lr_coef](https://user-images.githubusercontent.com/69367085/102663748-4cf38800-4136-11eb-96c1-405ece9d064a.jpg)

### 2. Elastic Net

*Plot of the Error Versus the Penalty (Regression with Regularization)*

![plot_enet](https://user-images.githubusercontent.com/69367085/102663800-65fc3900-4136-11eb-8e45-0c6eff858a87.jpg)

*Plot of the Path of the Coefficients*

![coef_path](https://user-images.githubusercontent.com/69367085/102663836-744a5500-4136-11eb-9541-e3ddbf89beda.jpg)

### 3. Bootstrap Aggregated (Bagged) Decision Tree

*Examining an Individual Model from Bagging*

![bagged_tree](https://user-images.githubusercontent.com/69367085/102663882-875d2500-4136-11eb-8139-68e4e9a6e493.jpg)

### 4. Random Forest

*Plot of Error Vesus the Number of Trees*

![plot_rf_fit](https://user-images.githubusercontent.com/69367085/102663922-9b088b80-4136-11eb-89b9-f2f88739ae47.jpg)

*Variable Importance Plot*

![rf_var_imp](https://user-images.githubusercontent.com/69367085/102664000-be333b00-4136-11eb-9a5d-a20cecf0ca15.jpg)

*Plotting the Minimum Depth Distribution*

![rf_explainer_pckg](https://user-images.githubusercontent.com/69367085/102664015-c68b7600-4136-11eb-8064-726edc9d7ad2.jpg)

## Model Evaluation: Comparison of Metrics

![model_metrics](https://user-images.githubusercontent.com/69367085/102664028-ce4b1a80-4136-11eb-8656-7db3be418ea2.jpg)
