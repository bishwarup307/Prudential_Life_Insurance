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
alldata$int_2 <- alldata$BMI/ alldata$Medical_History_23
alldata$int_3 <- alldata$BMI /(alldata$Medical_History_15 + 1)
alldata$int_4 <- alldata$Ins_Age - alldata$Product_Info_4
alldata$int_5 <- alldata$Ins_Age/alldata$InsuredInfo_6
alldata$int_6 <- alldata$Ins_Age - alldata$Medical_History_23
alldata$int_7 <- alldata$InsuredInfo_6 - alldata$Insurance_History_2
alldata$int_8 <- alldata$Medical_History_23 * alldata$Medical_History_15
alldata$int_9 <- alldata$Medical_History_23 * alldata$Medical_History_4
alldata$int_10 <- alldata$Medical_History_23 * alldata$Medical_History_39
alldata$int_11 <- alldata$Medical_History_15 * alldata$Medical_History_39
alldata$int_12 <- alldata$Medical_History_4 * alldata$Medical_History_39
alldata$int_13 <- alldata$Medical_History_23 * alldata$Medical_History_6
alldata$int_14 <- alldata$Medical_History_6 - alldata$Medical_History_16
alldata$int_15 <- alldata$Medical_History_6 + alldata$Medical_History_13
alldata$int_16 <- alldata$Medical_History_32 + alldata$Medical_History_40
alldata$int_17 <- alldata$Medical_Keyword_15 + alldata$Medical_Keyword_3
alldata$int_18 <- alldata$Medical_Keyword_3 + alldata$Medical_Keyword_48
alldata$int_19 <- alldata$Medical_Keyword_15 + alldata$Medical_Keyword_48

ptr <- alldata[alldata$trainFlag == 1,]
pte <- alldata[alldata$trainFlag == 0,]

feature.names <- names(ptr)[!names(ptr) %in% c("Id", "Response", "trainFlag")]

evalMatrix <- data.frame(Fold = numeric(), Id = numeric(), ground_truth = numeric(), xgb7_preds = numeric())
testMatrix <- data.frame(Id = pte$Id)

for(i in 1:10) {
  
  cat("\n---------------------------")
  cat("\n------- Fold: ", i, "--------")
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
                eta = 0.018,
                subsample = 0.78,
                colsample_bytree = 0.4,
                min_child_weight = 30,
                eval_metric = "rmse")
  
  watchlist <- list(eval = dval, train = dtrain)
  
  clf <- xgb.train(params = param,
                   data = dtrain,
                   nround = 2840,
                   print.every.n = 50,
                   watchlist = watchlist)
  
  
  preds <- predict(clf, data.matrix(validationSet[, feature.names]))
  valid <- data.frame(Fold = rep(i, nrow(validationSet)), Id = validationSet$Id, ground_truth = validationSet$Response, xgb7_preds = preds)
  evalMatrix <- rbind(evalMatrix, valid)
  
}

dtrain <- xgb.DMatrix(data = data.matrix(ptr[, feature.names]), label = ptr$Response)
watchlist <- list(train = dtrain)

clf_full <- xgb.train(params = param,
                      data = dtrain,
                      nround =2840,
                      print.every.n = 50,
                      watchlist = watchlist)

test_preds <- predict(clf_full, data.matrix(pte[, feature.names]))
testMatrix$xgb7_preds <- test_preds

write_csv(evalMatrix, "./Ensemble2/xgb7_eval.csv")
write_csv(testMatrix, "./Ensemble2/xgb7_test.csv")

