# load libraries
require(readr)

# set seed & working directory
setwd("~/Kaggle/Prudential")
set.seed(2)

# load data and validation fold ids
train <- read_csv("./Data/svm_train.csv")
test <- read_csv("./Data/svm_test.csv")
fold_id <- read_csv("./ensemble2/validation_id.csv")

# train and test meta containers
lm_val <- data.frame(Fold = numeric(), Id = numeric(), ground_truth = numeric(), lm_preds = numeric())
lm_test <- data.frame(Id = test$Id)

# features to use in the model
feature.names <- names(train)[!names(train) %in% c("Id", "Response", "trainFlag")]

# linear regression formula
frm <- as.formula(paste0("Response ~ ", paste(feature.names, collapse = "+")))

# generate train meta features with
# 10-fold stratified cross validation
for ( i in 1:10) {
  
  cat("\n-----------------------------------")
  cat("\n.......Fold ", i, "...............")
  cat("\n------------------------------------")
  
  idx <- fold_id[[i]]
  idx <- idx[!is.na(idx)]
  
  validationSet <- train[which(train$Id %in% idx),]
  trainingSet <- train[which(!train$Id %in% idx),]
  
  model <- lm(frm, data = trainingSet)
  preds <- predict(model, newdata = validationSet[, feature.names])
  
  df <- data.frame(Fold = rep(i, length(idx)), Id = validationSet$Id, ground_truth = validationSet$Response, lm_preds = preds)
  lm_val <- rbind(lm_val, df)
  
  tpreds <- predict(model, test[, feature.names])
  cname <- paste0("Fold_", i)
  lm_test[[cname]] <- tpreds
}

# generate test meta features
# train on all training points
m <- lm(frm, data = train)
tp <- predict(m, newdata = test[, feature.names])
ts <- data.frame(Id = test$Id, lm_preds = tp)

# save meta features to disk
write_csv(lm_val, "./ensemble2/lm_val.csv")
write_csv(ts, "./ensemble2/lm_test.csv")
