# Final Project R File

### Group: Adam Gonzalez, Jon Le, Erin Lee, Debbie Lu & Corinne Smith


# 1) CLEAR ENVIRONMENT


# code to remove objects in Environment before knitting
rm(list = ls())



# 2) LOAD THE DATA

### ---------------------------------------
### Data sets:
### "books" is from https://www.kaggle.com/jealousleopard/goodreadsbooks
### "books_two" is from https://www.kaggle.com/choobani/goodread-authors?select=final_dataset.csv
### ---------------------------------------


books <- read.csv(here::here("datasets", "books.csv"))
books_two <- read.csv(here::here("datasets", "final_dataset.csv"))



# 3) LOAD LIBRARIES

#### ---------------------------------------
library('yardstick')
### ---------------------------------------

### ---------------------------------------
### data visualization
### --------------------------------------
library('ggplot2')
library('plotly')
library('gganimate')
library('ggridges')

### ---------------------------------------
### data manipulation
### --------------------------------------
library('forcats')
library('tidyverse')
library('magrittr')
library('lubridate')
library('dplyr')
library('DT')
#install.packages("formattable")
#install.packages("tidyr")
library('formattable')
library('tidyr')
library('data.table')
library('kableExtra')

### ---------------------------------------
### sentiment analysis
### --------------------------------------
library('sentimentr')

### ---------------------------------------
### summary statistics
### --------------------------------------
#install.packages("qwraps2")
library("qwraps2")

### ---------------------------------------
### model validation library
### ---------------------------------------
library('rsample')

### ---------------------------------------
### generalized linear model libraries
### ---------------------------------------
library('glmnet')
library('glmnetUtils')

### ---------------------------------------
### regression output
### ---------------------------------------
### install.packages('sjPlot')
library('sjPlot')
### install.packages('sjPlot')
library('tidymodels')

### ---------------------------------------
### random forest libraries
### ---------------------------------------
library('partykit')
#library('tidyverse')
library('PerformanceAnalytics')
library('rpart')      
library('rpart.plot')  
library('randomForest')
#install.packages("randomForestExplainer")
library('randomForestExplainer')

### ---------------------------------------
# lasso libraries
### ---------------------------------------
library('broom')
library('coefplot')
### ---------------------------------------




books <- books %>% rename(avg_book_rating = average_rating,
                          book_ratings_count = ratings_count,
                          author = authors)
books_two <- books_two %>% rename(author = name,
                                  authorworkcount = workcount,
                                  author_fans = fan_count,
                                  avg_author_rating = average_rate,
                                  author_ratings_count = rating_count,
                                  author_review_count = review_count,
)



# 4) SENTIMENT ANALYSIS

### Need 'sentimentr' library.

sentiment_DF <- get_sentences(books$title) %>% sentiment_by(books$title)

head(sentiment_DF)



# 5) MERGING


books_s <- inner_join(x = books,
                      y = sentiment_DF,
                      by = "title")
head(books_s)


books_sa <-
  inner_join(x = books_s,
             y = books_two,
             by = "author")
head(books_sa)



# 6) DATA CLEANING


### mutate to correct column data types
books_1 <- books_sa %>% mutate(num_pages = as.numeric(num_pages),
                               avg_book_rating = as.numeric(avg_book_rating),
                               text_reviews_count = as.numeric(text_reviews_count),
                               publication_date = as.Date(publication_date, format="%m/%d/%Y"),
                               born = as.Date(born, format="%m/%d/%Y"),
                               died = as.Date(died, format="%m/%d/%Y"),
                               gender = as.factor(gender)
)


### remove NAs
books_total <- books_1 %>%
  filter(
    (!is.na(avg_book_rating)), (!is.na(book_ratings_count)), (!is.na(text_reviews_count)), (!is.na(publication_date)),
    (!duplicated(title)),
    (avg_book_rating != 0),
    (author != "NOT A BOOK"),
    (!is_greater_than(num_pages, 2000)),
    (num_pages != 0),
    (bookID != 9796),
    (!is_less_than(num_pages, 10))
  )


### remove irrelevant variables (11):
### sd(standard deviation of words in title), author ID, image_URL, about, influence, website, twitter, original hometown, country, latitude, longitude

books_corti <- books_total %>% select(-isbn13,
                                      -sd,
                                      -authorid,
                                      -image_url,
                                      -about,
                                      -influence,
                                      -website,
                                      -twitter,
                                      -original_hometown,
                                      -country,
                                      -latitude,
                                      -longitude) %>% rename(
                                        title_sentiment_avg = ave_sentiment,
                                        title_word_count = word_count
                                      )




### View(books_corti)



# 7) DATA EXPLORATION

### NA VISUALIZATION
### to see the number of missing values in each column

### STEPS:
### 1) We need to sum through every column using a FOR loop.
### 2) Then print the variable name using names(movies[i]).
### 3) Finally, we print the sum of is.na() for just that variable.

### FOR loop to see each column in books data set
for(i in 1:ncol(books_corti)){

  ### print the following
  print(

    # first print "Variable: "
    paste0("Variable: ",

           # then print the variable name, then "NAs: "
           names(books_corti)[i], " NAs: ",

           # then print the sum of the number of missing values
           # for that variable
           sum(is.na(books_corti %>% select(i)))
    )

  )
}




### starts_with() function for certain columns...
books_corti %>% select(starts_with("isbn")) %>% glimpse()


### exploring first 10 rows using slice() function

explore_data <- books_corti %>% arrange(desc(avg_book_rating)) %>% slice(1:10) %>% select(title, author, avg_book_rating)
print(explore_data)


datatable(books_corti)



### ONLY select "NOT A BOOK" under author variable (a.k.a. the column) and store this as a new data frame

not_a_book <- books_corti %>% filter(author == "NOT A BOOK") %>% nrow()
print(not_a_book)



### Expectation:
### Linear regression analysis is sensitive to outliers.
### Use histogram to see where this will occur.


ggplot(books_corti, aes(x = avg_book_rating)) +
  xlab("Average Book Rating") +
  ylab("Count") +
  geom_histogram(fill = "skyblue", color = "#879bcd") +
  theme_dark(base_size = 18) +
  ggtitle("               Histogram to View Outliers")



p <- books_corti %>%
  ggplot(aes(avg_book_rating, title_sentiment_avg)) +
  xlab("Average Book Rating") + ylab("Title Sentiment") +
  geom_point(color = "skyblue", alpha = 1/2, size = 0.5) +
  theme_bw(base_size = 18) +
  ggtitle("Exploring the Data: Visualization 1")

ggplotly(p)



p <- ggplot(books_corti %>%
              mutate(genderMutated = fct_lump(gender, n = 10)),
            aes(x = avg_book_rating, y = genderMutated, fill = genderMutated)) +
  theme_minimal(base_size = 18) +
  geom_density_ridges(color="black") +
  xlab("Average Book Rating") +
  ylab("Gender of Author") +  
  ggtitle("   Exploring the Data: Visualization 2")

p + theme(legend.position = "none")



# 8) EXAMINING DATA STRUCTURE

str(books_corti)



# 9) SUMMARY STATS

### mean, std dev, min, max
### need to install & load 'qwraps2' library

options(qwraps2_markup = "markdown")
View(books_corti)
our_summary1 <-
  list("Average Book Rating" =
         list("min"       = ~ min(avg_book_rating),
              "mean"      = ~ mean(avg_book_rating),
              "max"       = ~ max(avg_book_rating),
              "st. dev"   = ~ sd(avg_book_rating)),
       "Number of Pages" =
         list("min"       = ~ min(num_pages),
              "mean"    = ~ mean(num_pages),
              "max"       = ~ max(num_pages),
              "st.dev" = ~ sd(num_pages)),
       "Book Ratings Count" =
         list("min"       = ~ min(book_ratings_count),
              "mean"      = ~ mean(book_ratings_count),
              "max"       = ~ max(book_ratings_count),
              "st. dev"   = ~ sd(book_ratings_count)),
       "Text Reviews Count" =
         list("min"       = ~ min(text_reviews_count),
              "mean"      = ~ mean(text_reviews_count),
              "max"       = ~ max(text_reviews_count),
              "st. dev"   = ~ sd(text_reviews_count)),
       "Average Title Sentiment Score" =
         list("min"       = ~ min(title_sentiment_avg),
              "mean"      = ~ mean(title_sentiment_avg),
              "max"       = ~ max(title_sentiment_avg),
              "st. dev"   = ~ sd(title_sentiment_avg)),
       "Author's Work Count" =
         list("min"       = ~ min(authorworkcount),
              "mean"      = ~ mean(authorworkcount),
              "max"       = ~ max(authorworkcount),
              "st. dev"   = ~ sd(authorworkcount)),
       "Author's Fan Count" =
         list("min"       = ~ min(author_fans),
              "mean"      = ~ mean(author_fans),
              "max"       = ~ max(author_fans),
              "st. dev"   = ~ sd(author_fans)),
       "Author Ratings Count" =
         list("min"       = ~ min(author_ratings_count),
              "mean"      = ~ mean(author_ratings_count),
              "max"       = ~ max(author_ratings_count),
              "st. dev"   = ~ sd(author_ratings_count)),
       "Author Review Count" =
         list("min"       = ~ min(author_review_count),
              "mean"      = ~ mean(author_review_count),
              "max"       = ~ max(author_review_count),
              "st. dev"   = ~ sd(author_review_count))
  )
sum_stats <- summary_table(books_corti, our_summary1) %>% round(1)

print(sum_stats)



# 10) LINEAR MODEL VALIDATION: TRAIN-TEST-SPLIT

### Need to load 'rsample' library here.


set.seed(1818)

train_prop <- 0.8

books_split <- initial_split(books_corti, prop = train_prop)
books_train <- training(books_split)
books_test <- testing(books_split)


nrow(books_train)
nrow(books_test)
head(books_train)



# 11)  MODEL 1: LINEAR REGRESSION

### Need 'dplyr', 'glmnet', and 'glmnetUtils' libraries here.

options(scipen = 999)

mod1 <- lm(avg_book_rating ~ num_pages + book_ratings_count + text_reviews_count + title_sentiment_avg + authorworkcount + author_fans + author_ratings_count + author_review_count + gender, data = books_train)

summary(mod1)


### --------------------------------------------------------
### estimating "prettier" regression output
### --------------------------------------------------------

### Need 'sjPlot' and 'tidymodels' libraries.

### --------------------------------------------------------
### tab_model() outputs a table of results
### --------------------------------------------------------

tab_model(mod1, digits = 3)


### --------------------------------------------------------
### plot_model() outputs a plot of regression coefficients
### --------------------------------------------------------

plot_model(mod1)+ ylim(-0.1,0.1)  + ggtitle("            Average Book Rating Coefficients") + theme_minimal(base_size = 16)


### --------------------------------------------------------
### tidy() outputs a table of coefficients and their p-values, t-stats
### --------------------------------------------------------

tidy(mod1)



# 12) MODEL 2: ELASTIC NET

### Note: We used an alpha sequence from 0 to 1 in steps of 0.1.


enet_mod <- cva.glmnet(avg_book_rating ~ num_pages + book_ratings_count + text_reviews_count + title_sentiment_avg + authorworkcount + author_fans + author_ratings_count + author_review_count + gender,
                       data = books_train,
                       alpha = seq(0,1, by = 0.1))


print(enet_mod)

plot(enet_mod)

minlossplot(enet_mod,
            cv.type = "min")


# 13) EXTRACT BEST LINEAR MODEL


### Use this function to find the best alpha.
get_alpha <- function(fit) {
  alpha <- fit$alpha
  error <- sapply(fit$modlist,
                  function(mod) {min(mod$cvm)})
  alpha[which.min(error)]
}


### Get all parameters.
get_model_params <- function(fit) {
  alpha <- fit$alpha
  lambdaMin <- sapply(fit$modlist, `[[`, "lambda.min")
  lambdaSE <- sapply(fit$modlist, `[[`, "lambda.1se")
  error <- sapply(fit$modlist, function(mod) {min(mod$cvm)})
  best <- which.min(error)
  data.frame(alpha = alpha[best], lambdaMin = lambdaMin[best],
             lambdaSE = lambdaSE[best], error = error[best])
}


### Extract the best alpha value & model parameters.
best_alpha <- get_alpha(enet_mod)
print(best_alpha)
get_model_params(enet_mod)

### Extract the best model object.
best_mod <- enet_mod$modlist[[which(enet_mod$alpha == best_alpha)]]


# 14) MODEL 3: BEST ELASTIC NET MODEL

enet_best_mod <- cv.glmnet(avg_book_rating ~ num_pages + book_ratings_count + text_reviews_count + title_sentiment_avg + authorworkcount + author_fans + author_ratings_count + author_review_count + gender,
                           data = books_train,
                           alpha = 0.1)
summary(enet_best_mod)

print(enet_best_mod)



### Print the model's two suggested values of lambda.

print(enet_best_mod$lambda.min)
print(enet_best_mod$lambda.1se)


### Plot how the MSE varies as we vary lambda.
plot(enet_best_mod)

coefpath(enet_best_mod)




### Compare lambda min & lambda 1SE...

### put into coefficient vector
enet_coefs <- data.frame(
  `lasso_min` = coef(enet_best_mod, s = enet_best_mod$lambda.min) %>%
    as.matrix() %>% data.frame() %>% round(3),
  `lasso_1se` = coef(enet_best_mod, s = enet_best_mod$lambda.1se) %>%
    as.matrix() %>% data.frame() %>% round(3)
) %>%  rename(`lasso_min` = 1, `lasso_1se` = 2)

print(enet_coefs)

enet_coefs %>% kable() %>% kable_styling()


# 15) MODEL 3: BOOTSTRAP AGGREGATING (BAGGING)

### Need 'partykit', 'PerformanceAnalytics', 'rpart', 'rpart.plot', and 'randomForest' libraries.


options(scipen = 10)
#set.seed(1818)



### store row names as columns
books_boot_preds <- books_corti %>% rownames_to_column() %>%
  mutate(rowname = as.numeric(rowname))

B <- 100      # number of bootstrap samples
num_b <- 500  # sample size of each bootstrap
boot_mods <- list() # store our bagging models
for(i in 1:B){
  boot_idx <- sample(1:nrow(books_corti),
                     size = num_b,
                     replace = FALSE)
  ### fit a tree on each bootstrap sample
  boot_tree <- ctree(avg_book_rating ~ num_pages + book_ratings_count + text_reviews_count + title_sentiment_avg + authorworkcount + author_fans + author_ratings_count + author_review_count+ gender,
                     data = books_corti %>%
                       slice(boot_idx))
  ### store bootstraped model
  boot_mods[[i]] <- boot_tree
  # generate predictions for that bootstrap model
  preds_boot <- data.frame(
    preds_boot = predict(boot_tree),
    rowname = boot_idx
  )  
  ### rename prediction to indicate which boot iteration it came from
  names(preds_boot)[1] <- paste("preds_boot",i,sep = "")
  ### merge predictions to dataset
  books_boot_preds <- left_join(x = books_boot_preds, y = preds_boot,
                                by = "rowname")
}



### --------------------------------------------------------
### plot() examines an individual model from bagging
### --------------------------------------------------------

plot(boot_mods[[1]], gp = gpar(fontsize = 8))

books_boot_preds <- books_boot_preds %>%
  mutate(preds_bag =
           select(., preds_boot1:preds_boot100) %>%
           rowMeans(na.rm = TRUE))

### NOTE: At this point in the code, the model has been bootstrapped.


# 16) MODEL 4: RANDOM FOREST

rf_fit <- randomForest(avg_book_rating ~ num_pages + book_ratings_count + text_reviews_count + title_sentiment_avg + authorworkcount + author_fans + author_ratings_count + author_review_count + gender,
                       data = books_corti,
                       type = regression,
                       mtry = 11/3,
                       ntree = 200,
                       importance = TRUE)

print(rf_fit)

plot(rf_fit)


varImpPlot(rf_fit, type = 1)

plot_min_depth_distribution(rf_fit)

plot_predict_interaction(rf_fit, books_corti, "author_ratings_count", "num_pages")
plot_predict_interaction(rf_fit, books_corti, "authorworkcount", "num_pages")
plot_predict_interaction(rf_fit, books_corti, "num_pages", "title_sentiment_avg")


### Storing predictions data frames for Linear and ElasticNet models...

lm_preds_train <- predict(mod1, newdata = books_train)
lm_preds_test <- predict(mod1,
                         newdata = books_test)

enet_preds_train <- predict(enet_best_mod,
                            newdata = books_train,  s = "lambda.min")
enet_preds_test <- predict(enet_best_mod,
                           newdata = books_test,  s = "lambda.min")

head(lm_preds_train)
head(lm_preds_test)

head(enet_preds_train)
head(enet_preds_test)


### Storing results data frames for Linear and ElasticNet models...

training_predictions <- data.frame(lm_preds_train, enet_preds_train)

results_train <- data.frame(books_train, training_predictions) %>% rename(enet_training = X1)

head(results_train)



testing_predictions <- data.frame(
  "lm_testing" = lm_preds_test,
  "enet_testing" = enet_preds_test)

results_test <- data.frame(books_test, testing_predictions) %>% rename(enet_testing = X1)

head(results_test)



# 17) GGPLOT OF LINEAR REGRESSION: TRAINING RESULTS


ggplot(results_train, aes(x = avg_book_rating, y = lm_preds_train)) +
  geom_point(alpha = 1/10, size = 4) +
  theme_minimal(base_size = 16)+
  geom_abline(color = "turquoise")+
  xlab("True Average Ratings")+
  ylab("Predicted Average Ratings")+
  xlim(0, 5) + ylim(0, 5)+
  ggtitle("              Linear Regression: Training True vs Predicted")



# 18) GGPLOT OF ELASTIC NET: TRAINING RESULTS


ggplot(results_train, aes(x = avg_book_rating, y = enet_preds_train)) +
  geom_point(alpha = 1/10, size = 4) +
  theme_minimal(base_size = 16)+
  geom_abline(color = "turquoise")+
  xlab("True Average Ratings")+
  ylab("Predicted Average Ratings")+
  xlim(0, 5) + ylim(0, 5)+
  ggtitle("        Best ElasticNet: Training True vs Predicted")



# 19) GGPLOT OF LINEAR REGRESSION: TESTING RESULTS

ggplot(results_test, aes(x = avg_book_rating, y = lm_preds_test)) +
  geom_point(alpha = 1/10, size = 4) +
  geom_abline(color = "coral")+
  theme_minimal(base_size = 16)+
  xlab("True Average Ratings")+
  ylab("Predicted Average Ratings")+
  xlim(0, 5) + ylim(0, 5)+
  ggtitle("              Linear Regression: Testing True vs Predicted")



# 20) GGPLOT OF ELASTIC NET: TESTING RESULTS

ggplot(results_test, aes(x = avg_book_rating, y = enet_preds_test)) +
  geom_point(alpha = 1/10, size = 4) +
  geom_abline(color = "coral")+
  theme_minimal(base_size = 16)+
  xlab("True Average Ratings")+
  ylab("Predicted Average Ratings")+
  xlim(0, 5) + ylim(0, 5)+
  ggtitle("         Best ElasticNet: Testing True vs Predicted")



# 21) MODEL EVALUATION

## LINEAR REGRESSION TRAINING METRICS

rmse(books_train, truth = avg_book_rating, estimate = lm_preds_train)
mae(books_train, truth = avg_book_rating, estimate = lm_preds_train)
rsq(books_train, truth = avg_book_rating, estimate = lm_preds_train)



## LINEAR REGRESSION TEST METRICS

lm_rmse <- rmse(books_test, truth = avg_book_rating, estimate = lm_preds_test)
lm_mae <- mae(books_test, truth = avg_book_rating, estimate = lm_preds_test)
lm_rsq <- rsq(books_test, truth = avg_book_rating, estimate = lm_preds_test)



## ELASTIC NET TRAINING METRICS

rmse(books_train, truth = avg_book_rating, estimate = as.vector(enet_preds_train))
mae(books_train, truth = avg_book_rating, estimate = as.vector(enet_preds_train))
rsq(books_train, truth = avg_book_rating, estimate = as.vector(enet_preds_train))



## ELASTIC NET TEST METRICS

enet_rmse <- rmse(books_test, truth = avg_book_rating, estimate = as.vector(enet_preds_test))
enet_mae <- mae(books_test, truth = avg_book_rating, estimate = as.vector(enet_preds_test))
enet_rsq <- rsq(books_test, truth = avg_book_rating, estimate = as.vector(enet_preds_test))



## Tree OUT-OF-BAG Predictions...

books_right_join <- right_join(books_corti, books_boot_preds)
books_right_join <- books_right_join %>% ungroup()

tree_rmse <- rmse(books_right_join, truth = avg_book_rating, estimate = preds_bag)
tree_mae <- mae(books_right_join, truth = avg_book_rating, estimate = preds_bag)
tree_rsq <- rsq(books_right_join, truth = avg_book_rating, estimate = preds_bag)



## Random Forest OUT-OF-BAG Predictions...

preds_OOB <- predict(rf_fit)

rf_rsq <- rsq(books_corti, truth = avg_book_rating, estimate = preds_OOB)
rf_rmse <- rmse(books_corti, truth = avg_book_rating, estimate = preds_OOB)
rf_mae <- mae(books_corti, truth = avg_book_rating, estimate = preds_OOB)



# 22) MERGING -- RSQ, RMSE & MAE COMBINED DATA FRAME

## All testing data predictions...


rsq_DF <- merge(rf_rsq, enet_rsq, by=c(".metric", ".estimator"))

rsq_DF1 <- merge(rsq_DF, lm_rsq, by=c(".metric", ".estimator")) %>% rename("Random Forest" = .estimate.x, "ElasticNet" = .estimate.y, "Linear" = .estimate)

rsq_DF2 <- merge(rsq_DF1, tree_rsq, by=c(".metric", ".estimator")) %>% select(-.estimator)

print(rsq_DF2)

rmse_DF <- merge(rf_rmse, enet_rmse, by=c(".metric", ".estimator"))

rmse_DF1 <- merge(rmse_DF, lm_rmse, by=c(".metric", ".estimator")) %>% rename("Random Forest" = .estimate.x, "ElasticNet" = .estimate.y, "Linear" = .estimate)

rmse_DF2 <- merge(rmse_DF1, tree_rmse, by=c(".metric", ".estimator")) %>% select(-.estimator)

print(rmse_DF2)

mae_DF <- merge(rf_mae, enet_mae, by=c(".metric", ".estimator"))

mae_DF1 <- merge(mae_DF, lm_mae, by=c(".metric", ".estimator")) %>% rename("Random Forest" = .estimate.x, "ElasticNet" = .estimate.y, "Linear" = .estimate)

mae_DF2 <- merge(mae_DF1, tree_mae, by=c(".metric", ".estimator")) %>% select(-.estimator)

print(mae_DF2)

total <- rbind(rsq_DF2, rmse_DF2)

final <-rbind(total, mae_DF2) %>% rename("Tree" = .estimate, "Metrics" = .metric)

print(final)



# 23) METRICS DATA TABLE

### Credit for the code below:
###  https://rfortherestofus.com/2019/11/how-to-make-beautiful-tables-in-r/


### Need to load 'kableExtra' library.


final %>% kable() %>% kable_styling()


### Credit for code below:
### https://www.littlemissdata.com/blog/prettytables

### Need to load 'formattable', 'tidyr', and 'data.table' libraries.

custom_one = "#CCCCFF"
custom_two = "skyblue"
custom_three = "#4ec5a5"
custom_coral = "#FA7268"

formattable(final,
            align =c("l","c","c","c","c", "c", "c", "c", "r"),
            list(`Metrics` = formatter(
              "span", style = ~ style(color = "grey",font.weight = "bold")),
              `Random Forest`= color_tile(custom_one, custom_one),
              `ElasticNet`= color_tile(custom_two, custom_two),
              `Linear`= color_tile(custom_three, custom_three),
              `Tree`= color_tile(custom_coral, custom_coral)
            ))
