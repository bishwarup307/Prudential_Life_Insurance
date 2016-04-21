require(readr)
require(Hmisc)
require(caret)
require(MASS)
require(xgboost)
require(Metrics)

setwd("F:/Kaggle/Prudential2")
set.seed(13)

train <- read_csv("./RawData/train.csv")
test <- read_csv("./RawData/test.csv")

manage_na <- function(datafra)
{
  for(i in 1:ncol(datafra))
  {
    if(is.numeric(datafra[,i]))
    {
      datafra[is.na(datafra[,i]),i] <- median(datafra[!is.na(datafra[,i]),i])
    }
  }
  datafra
}

tra_clean <- manage_na(train[,-c(1,3,128)])
ogg <- lm.ridge(train$Response ~ ., data=tra_clean, lambda=0.5)
impo <- tra_clean[,(abs(ogg$coef) > quantile(abs(ogg$coef), 0.382))] #only "important" variables left
var_names <- names(impo)

train <- train[, c("Id", "Response", "Product_Info_2", var_names)]
test <- test[, c("Id",  "Product_Info_2",var_names)]

train$trainFlag <- 1
test$trainFlag <- 0
test$Response <- NA
alldata <- rbind(train, test)

alldata$na_count <- apply(alldata[,!names(alldata) %in% c("Id", "Response", "trainFlag", "Product_Info_2")], 1, function(x) sum(is.na(x)))
alldata$zero_count <- apply(alldata[,!names(alldata) %in% c("Id", "Response", "trainFlag", "Product_Info_2")], 1, function(x) sum(x == 0, na.rm = TRUE))
alldata$gender_iden <- ifelse(is.na(alldata$Family_Hist_2) & !is.na(alldata$Family_Hist_3), 0,
                             ifelse(!is.na(alldata$Family_Hist_2) & is.na(alldata$Family_Hist_3), 1, -1))
alldata$int_1 <- alldata$BMI * alldata$Ins_Age
alldata$Product_Info_2 <- as.integer(as.factor(alldata$Product_Info_2))

qbmic <- 0.8
qbmic2 <- 0.9

alldata$custom_var_1 <- as.numeric(alldata$Medical_History_15 < 10.0)
alldata$custom_var_1[is.na(alldata$custom_var_1)] <- 0.0 #impute these NAs with 0s, note that they were not median-imputed
alldata$custom_var_3 <- as.numeric(alldata$Product_Info_4 < 0.075)
alldata$custom_var_4 <- as.numeric(alldata$Product_Info_4 == 1)
alldata$custom_var_6 <- (alldata$BMI + 1.0)**2.0
alldata$custom_var_7 <- (alldata$BMI)**0.8
alldata$custom_var_8 <- alldata$Ins_Age**8.5
alldata$custom_var_9 <- (alldata$BMI*alldata$Ins_Age)**2.5
BMI_cutoff <- quantile(train$BMI, qbmic)
alldata$custom_var_10 <- as.numeric(alldata$BMI > BMI_cutoff)
alldata$custom_var_11 <- (alldata$BMI*alldata$Product_Info_4)**0.9
ageBMI_cutoff <- quantile(train$Ins_Age*train$BMI, qbmic2)
alldata$custom_var_12 <- as.numeric(alldata$Ins_Age*alldata$BMI > ageBMI_cutoff)
alldata$custom_var_13 <- (alldata$BMI*alldata$Medical_Keyword_3 + 0.5)**3.0


ptr <- alldata[alldata$trainFlag == 1,]
pte <- alldata[alldata$trainFlag == 0,]

kf <- read_csv("./Ensemble2/validation_id_ensemble_2.csv")
feature.names <- names(ptr)[!names(ptr) %in% c("Id", "Response", "trainFlag")]

cp <- data.frame(c1 = numeric(),
                 c2 = numeric(),
                 c3 = numeric(),
                 c4 = numeric(),
                 c5 = numeric(),
                 c6 = numeric(),
                 c7 = numeric())

SQWK <- function(init, df = valid) {
  
  actual <- df$ground_truth
  predicted <- df$predicted
  
  resp <- ifelse(predicted <= init[1], 1,
                 ifelse(predicted > init[1] & predicted <= init[2], 2,
                        ifelse(predicted > init[2] & predicted <= init[3], 3,
                               ifelse(predicted > init[3] & predicted <= init[4], 4,
                                      ifelse(predicted > init[4] & predicted <= init[5], 5,
                                             ifelse(predicted > init[5] & predicted <= init[6], 6,
                                                    ifelse(predicted > init[6] & predicted <= init[7], 7, 8)))))))
  
  err <- ScoreQuadraticWeightedKappa(actual, resp)
  return(-err)
}

ac <- data.frame(a1 = c(-1, 1, rep(0, 5)),
                 a2 = c(0, -1, 1, rep(0, 4)),
                 a3 = c(0, 0, -1, 1, rep(0, 3)),
                 a4 = c(rep(0, 3), -1, 1, 0, 0),
                 a5 = c(rep(0, 4), -1, 1, 0),
                 a6 = c(rep(0, 5), -1, 1))

ac <- t(as.matrix(ac))


evalMatrix <- data.frame(Fold = numeric(), Id = numeric(), ground_truth = numeric(), predicted = numeric(), hard_preds = numeric())
testMatrix <- data.frame(Id = pte$Id)
test_hard <- data.frame(Id = pte$Id)
CV <- c()
biter <- c()

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
  
  param <- list(objective = "reg:linear",
                max_depth = 6,
                eta = 0.012,
                subsample = 0.7,
                colsample_bytree = 0.6,
                min_child_weight = 40,
                eval_metric = "rmse",
                gamma = 1,
                max_delta_step = 0.5)
  
  watchlist <- list(eval = dval, train = dtrain)
  
  clf <- xgb.train(params = param,
                   data = dtrain,
                   nround = 10000,
                   early.stop.round = 200,
                   print.every.n = 50,
                   watchlist = watchlist)
  
  preds <- predict(clf, data.matrix(validationSet[, feature.names]))
  
  valid <- data.frame(Fold = rep(i, nrow(validationSet)), Id = validationSet$Id, ground_truth = validationSet$Response, predicted = preds)
  
  
  
  initx <- seq(1.5, 7.5, by = 1)  
  optCuts <- constrOptim(initx, SQWK, method = "Nelder-Mead", control = list(trace = T), ui = ac, ci  = rep(0, 6))  
  pct1 <- optCuts$par
  optCuts <- constrOptim(pct1, SQWK, method = "Nelder-Mead", control = list(trace = T), ui = ac, ci  = rep(0, 6))
  pct2 <- optCuts$par
  optCuts <- constrOptim(pct2, SQWK, method = "Nelder-Mead", control = list(trace = T), ui = ac, ci  = rep(0, 6))
  pct3 <- optCuts$par                   
  optCuts <- constrOptim(pct3, SQWK, method = "Nelder-Mead", control = list(trace = T), ui = ac, ci  = rep(0, 6))
  pct4 <- optCuts$par      
  optCuts <- constrOptim(pct4, SQWK, method = "Nelder-Mead", control = list(trace = T), ui = ac, ci  = rep(0, 6))
  finalpct <- optCuts$par
  
  preds_hardcoded <- as.integer(Hmisc::cut2(valid$predicted, cuts = finalpct))
  valid$hard_preds <- preds_hardcoded
  evalMatrix <- rbind(evalMatrix, valid)
  
  tpreds <- predict(clf, data.matrix(pte[, feature.names]))
  
  testMatrix[[cname]] <- tpreds
  
  #   fpreds <- predict(clf, data.matrix(holdoutSet[, feature.names]))
  #   f10[[cname]] <- fpreds
  
  tpreds_hard <- as.integer(Hmisc::cut2(tpreds, cuts = finalpct))
  test_hard[[cname]] <- tpreds_hard
  
  fpct <- data.frame(c1 = finalpct[1],
                     c2 = finalpct[2],
                     c3 = finalpct[3],
                     c4 = finalpct[4],
                     c5 = finalpct[5],
                     c6 = finalpct[6],
                     c7 = finalpct[7])
  cp <- rbind(cp, fpct)
  
  CV <- c(CV, ScoreQuadraticWeightedKappa(valid$ground_truth, valid$hard_preds, 1, 8))
  biter <- c(biter, clf$bestInd)
  
  cat("\n best kappa: ", ScoreQuadraticWeightedKappa(valid$ground_truth, valid$hard_preds, 1, 8))
  cat("\n best iter: ", clf$bestInd)
}

names(evalMatrix)[4] <- "xgb6_preds"
testMatrix$xgb6_preds <- apply(testMatrix[, -1], 1, median)
write_csv(evalMatrix, "./Ensemble2/xgb6_eval.csv")
write_csv(testMatrix, "./Ensemble2/xgb6_test.csv")
