require(readr)
require(xgboost)
require(Metrics)

setwd("F:/Kaggle/Prudential2")
set.seed(133)

train <- read_csv("RawData/train.csv")
test <- read_csv("RawData/test.csv")
kf <- read_csv("Ensemble2/validation_id_ensemble_2.csv")

train$trainFlag <- 1
test$trainFlag <- 0
test$Response <- NA
alldata <- rbind(train, test)

alldata$na_count <- apply(alldata[,!names(alldata) %in% c("Id", "Response", "trainFlag", "Product_Info_2")], 1, function(x) sum(is.na(x)))
alldata$zero_count <- apply(alldata[,!names(alldata) %in% c("Id", "Response", "trainFlag", "Product_Info_2")], 1, function(x) sum(x == 0, na.rm = TRUE))
alldata$Product_Info_2 <- as.integer(as.factor(alldata$Product_Info_2))
alldata$int_1 <- alldata$BMI * alldata$Ins_Age

ptr <- alldata[alldata$trainFlag == 1,]
pte <- alldata[alldata$trainFlag == 0,]

feature.names <- names(ptr)[!names(ptr) %in% c("Id", "Response", "trainFlag")]

evalMatrix <- data.frame(Fold = numeric(), Id = numeric(), ground_truth = numeric(), xgb5_preds = numeric())
testMatrix <- data.frame(Id = pte$Id)

for(i in 1:10) {
  
  cat("\n---------------------------")
  cat("\n------- Fold: ", i, "----------")
  cat("\n---------------------------\n")
  cname <- paste0("Fold_", i)
  idx <- kf[[i]]
  idx <- idx[!is.na(idx)]
  
  trainingSet <- ptr[!ptr$Id %in% idx,]
  validationSet <- ptr[ptr$Id %in% idx,]
  
  
  cat("\nnrow train: ", nrow(trainingSet))
  cat("\nnrow eval: ", nrow(validationSet), "\n")
  
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainingSet[, feature.names]), label = trainingSet$Response)
  dval <- xgb.DMatrix(data = data.matrix(validationSet[, feature.names]), label = validationSet$Response)
  
  param <- list(objective = "count:poisson",
                max_depth = 6,
                eta = 0.01,
                subsample = 0.78,
                colsample_bytree = 0.3,
                min_child_weight = 30,
                eval_metric = "rmse")
  
  watchlist <- list(eval = dval, train = dtrain)
  
  clf <- xgb.train(params = param,
                   data = dtrain,
                   nround = 10000,
                   early.stop.round = 250,
                   print.every.n = 50,
                   watchlist = watchlist)
  
  
  preds <- predict(clf, data.matrix(validationSet[, feature.names]))
  valid <- data.frame(Fold = rep(i, nrow(validationSet)), Id = validationSet$Id, ground_truth = validationSet$Response, xgb5_preds = preds)
  
  evalMatrix <- rbind(evalMatrix, valid)
  
  tpreds <- predict(clf, data.matrix(pte[, feature.names]))
  testMatrix[[cname]] <- tpreds
  
}

write_csv(evalMatrix, "./Ensemble4Folds/XGB_1/xgb1_eval.csv")
write_csv(testMatrix, "./Ensemble4Folds/XGB_1/xgb1_test.csv")