library(rpart)
library(rpart.plot)
library(tidyverse)
library(rsample)
library(caret)
library(partykit)

dataset <- read.csv("song_data cleaned.csv")
dataset <- select(dataset, song_popularity:audio_valence)

dataset <- dataset %>%
  mutate(
    song_duration_ms = song_duration_ms/1000
  )

colnames(dataset)[2] <- "song_duration_seconds" 
table(dataset$song_popularity)
view(dataset)

dataset %>%
  ggplot(aes(song_popularity)) + geom_histogram()       

dataset <- dataset %>% 
  mutate(song_popularity = ifelse(song_popularity >= 0 & song_popularity <= 43, "unpopular",
                           ifelse(song_popularity >= 44 & song_popularity <= 60, "neutral",
                           ifelse(song_popularity >= 61 & song_popularity <= 100, "popular", NA))))


dataset <- dataset %>% mutate(song_popularity = factor(song_popularity))
table(dataset$song_popularity)

dataset <- dataset %>%
  mutate(
    key = factor(key),
    audio_mode = factor(audio_mode),
    time_signature = factor(time_signature)
  )

set.seed(123)
library(rsample)

dataset_split <- initial_split(dataset, prop = 0.6, strata = "song_popularity")
#antrenare
dataset_train <- training(dataset_split)
dataset_testval <- testing(dataset_split)

dataset_testval_split <- initial_split(dataset_testval, prop = 0.5, strata = "song_popularity")
#validare
dataset_val <- training(dataset_testval_split)
#testare
dataset_test <- testing(dataset_testval_split)

table(dataset_train$song_popularity)
table(dataset_test$song_popularity)
table(dataset_val$song_popularity)


tuneGrid <- data.frame(cp = seq(0.0001, 0.1, by = 0.001))       

set.seed(123)
m1 <- train(
  song_popularity ~ .,
  data = dataset_train,
  method = "rpart",
  tuneGrid = tuneGrid,
  metric = "Accuracy"
)

m1$bestTune     #cel mai bun cp -> 0.0031
m1$results

m1 = rpart(
  formula = song_popularity ~. ,
  data = dataset_train,
  method = "class",
  control = list(cp=0.0031)     
)

rpart.plot(m1)

pred_m1 <- predict(m1, newdata = dataset_val, target = "class")

pred_m1 <- as_tibble(pred_m1) %>%
  mutate(class = ifelse(popular > unpopular & popular > neutral, "popular", 
                        ifelse(neutral > unpopular & neutral > popular, "neutral", 
                               ifelse(unpopular > popular & unpopular > neutral, "unpopular", NA_character_))))

confusionMatrix(factor(pred_m1$class), factor(dataset_val$song_popularity))

#ENTROPIA

m1_entropy <- rpart(
  formula = song_popularity ~. ,
  data = dataset_train,
  method = "class",
  control = rpart.control(cp = 0.0031, minsplit = 20, minbucket = 10, parms = list(split = "entropy"))
)
printcp(m1_entropy)

pred_m1_entropy <- predict(m1_entropy, newdata = dataset_val, type = "class")

confusionMatrix(factor(pred_m1_entropy), factor(dataset_val$song_popularity))

#BAGGING

library(ipred)
set.seed(123)
bagged_m1 <- bagging(song_popularity ~ .,
                     data = dataset_train, coob = TRUE)
bagged_m1

pred_bagged_m1 <- predict(bagged_m1, newdata = dataset_val, target = "class")
confusionMatrix(pred_bagged_m1, factor(dataset_val$song_popularity))

ntree <- 10:500
err <- vector(mode = "numeric", length = length(ntree))

for (i in seq_along(ntree)) {
  set.seed(123)
  model <- bagging(
    formula = song_popularity ~ .,
    data = dataset_train,
    coob = TRUE,
    nbagg = ntree[i] )
  err[i] = model$err
  print(i)
}

plot(ntree, err, type = "l", lwd = 2, xlab = "ntree", ylab = "Error", xlim = c(1, length(ntree)))


bagged_m1_500 <- bagging(song_popularity ~ .,
                        data = dataset_train, coob = TRUE, nbag = 500)
                  
bagged_m1_500
pred_bagged_m1_500 <- predict(bagged_m1_500, newdata = dataset_val, target = "class")
confusionMatrix(pred_bagged_m1_500, factor(dataset_val$song_popularity))

#Random Forest
library(randomForest)
set.seed(123)
m1_rf <- randomForest(
  formula = song_popularity ~ .,
  data = dataset_train
)
plot(m1_rf)

# Tuning
set.seed(123)
library(ranger)
hyper_grid <- expand.grid(
  mtry = seq(2, 13, by = 1),
  node_size = seq(3, 9, by = 2),
  sample_size = c(0.55, 0.632, 0.7, 0.8, 0.3),
  OOB_ERR = 0
)

for (i in 1:nrow(hyper_grid)) {
  model <- ranger(
    formula = song_popularity ~ .,
    data = dataset_train,
    num.trees = 500,
    mtry = hyper_grid$mtry[i],
    min.node.size = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sample_size[i],
    seed = 123
  )
  hyper_grid$OOB_ERR[i] <- model$prediction.error
}

hyper_grid %>%
  arrange(desc(OOB_ERR)) %>%
  top_n(-10)

best_combinations <- hyper_grid[which.min(hyper_grid$OOB_ERR), ]
best_combinations
order(m1_rf$importance[, 1], decreasing = TRUE)
selected_predictor_indices <- order(m1_rf$importance[, 1], decreasing = TRUE)[1:best_combinations$mtry]
selected_predictor_indices
selected_predictor_names <- names(dataset_train)[-ncol(dataset_train)][selected_predictor_indices]
selected_predictor_names  #instrumentalness, liveness


OOB_ERR <- vector(mode = "numeric", length = 100)
for(i in seq_along(OOB_ERR)) {
  optimal_ranger <- ranger(
    formula         = song_popularity ~ ., 
    data            = dataset_train, 
    num.trees       = 500,
    mtry            = 2,
    min.node.size   = 9,
    sample.fraction = 0.7,
    importance      = 'none'
  )
  
  OOB_ERR[i] <- optimal_ranger$prediction.error
  
}


mean(OOB_ERR)

library(ggplot2)

df <- data.frame(OOB_ERR)

ggplot(df, aes(x = OOB_ERR)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 20) +
  labs(x = "OOB Error", y = "Frequency", title = "OOB Error Distribution")

pred_ranger <- predict(optimal_ranger, dataset_val, target ="class")  
confusionMatrix(pred_ranger$predictions, factor(dataset_val$song_popularity))   #42.48% acuratetea

#Testare
pred_ranger <- predict(optimal_ranger, dataset_test, target ="class")  
confusionMatrix(pred_ranger$predictions, factor(dataset_test$song_popularity))   #42.16% acuratetea



