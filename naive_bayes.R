library(rsample)
library(tidyverse)
library(caret)
library(corrplot)

dataset <- read.csv("song_data cleaned.csv")
dataset <- select(dataset, song_popularity:audio_valence)
dataset <- dataset %>%
  mutate(
    song_duration_ms = song_duration_ms/1000
  )
colnames(dataset)[2] <- "song_duration_seconds"
table(dataset$song_popularity)
view(dataset)
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

dataset %>%
  ggplot(aes(key)) +
  geom_density(show.legend = TRUE)


dataset %>%
  filter(song_popularity == "popular") %>%
  select_if(is.numeric) %>%    
  cor() %>%
  corrplot::corrplot()

dataset %>%
  filter(song_popularity == "neutral") %>%
  select_if(is.numeric) %>%    
  cor() %>%
  corrplot::corrplot()

dataset %>%
  filter(song_popularity == "unpopular") %>%
  select_if(is.numeric) %>%    
  cor() %>%     
  corrplot::corrplot()

set.seed(123) 

split <- initial_split(dataset, prop = 0.6, strata = "song_popularity")
train <- training(split)
testval <- testing(split)

testval_split <- initial_split(testval, prop = 0.5, strata = "song_popularity")
val <- training(testval_split)
test <- testing(testval_split)

table(train$song_popularity)
table(test$song_popularity)
table(val$song_popularity)

features <- setdiff(names(train), "song_popularity")

x <- train[,features] 
y <- train$song_popularity

set.seed(123)
modNbSimple <- train(
  x = x,
  y = y,
  method = "nb",
)
pred <- predict(modNbSimple, val)         
pred
confusionMatrix(pred, val$song_popularity) #37.15%

searchGrid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0.5,       
  adjust = seq(0, 10, by = 1) 
)

modNbCVSearch <- train(
  x = x,
  y = y,
  method = "nb",
  tuneGrid = searchGrid
)

modNbCVSearch$results %>%
  top_n(5, wt = Accuracy) %>%
  arrange(desc(Accuracy))

pred <- predict(modNbCVSearch, val)
pred
confusionMatrix(pred, val$song_popularity)    #37.19%

library(pROC)

fitControlROC <- trainControl(
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = TRUE
)

modNbCVSearchROC <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = fitControlROC,
  tuneGrid = searchGrid,
  metric = "ROC"
)

predProb <- predict(modNbCVSearch, val, type = "prob")

roc.unpopular <- roc(val$song_popularity == "unpopular", predProb[, "unpopular"])
roc.popular <- roc(val$song_popularity == "popular", predProb[, "popular"])
roc.neutral <- roc(val$song_popularity == "neutral", predProb[, "neutral"])

plot(roc.unpopular, col = "red", print.auc = TRUE, main = "ROC Curve for Song Popularity")
lines(roc.popular, col = "green", print.auc = TRUE)
lines(roc.neutral, col = "blue", print.auc = TRUE)

legend("bottomright", legend = c("Unpopular", "Popular", "Neutral"), col = c("red", "green", "blue"), lty = 1)



