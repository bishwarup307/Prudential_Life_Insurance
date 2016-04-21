require(readr)
require(MASS)
require(Metrics)
require(caret)
require(xgboost)

setwd("F:/Kaggle/Prudential2")
set.seed(25)

train <- read_csv("./RawData/train.csv")
test <- read_csv("./RawData/test.csv")
kf <- read_csv("./Ensemble2/validation_id_ensemble_2.csv")

a <- rbind(train[, c("Id", "Family_Hist_2", "Family_Hist_3")], test[, c("Id", "Family_Hist_2", "Family_Hist_3")])
a$gender_indicator <- ifelse(is.na(a$Family_Hist_2) & !is.na(a$Family_Hist_3), 0,
                             ifelse(!is.na(a$Family_Hist_2) & is.na(a$Family_Hist_3), 1, -1))

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


minMaxScale <- function(x) {
  
  minima <- min(x, na.rm= TRUE)
  maxima <- max(x, na.rm = TRUE)
  
  p <- (x - minima)/(maxima - minima)
  return(p)
}


tra_clean <- manage_na(train[,-c(1,3,128)])
ogg <- lm.ridge(train$Response ~ ., data=tra_clean, lambda=0.5)
impo <- tra_clean[,(abs(ogg$coef) > quantile(abs(ogg$coef), 0.382))]              #only "important" variables left
var_names <- names(impo)


test_clean <- manage_na(test[,- c(1, 3)])
train_clean <- cbind(impo, Product_Info_2 = train[, 3])
test_clean <- cbind(test_clean[, var_names], Product_Info_2 = test[, 3])


qbmic <- 0.8
qbmic2 <- 0.9

train <- cbind(train[, c("Id", "Response")], train_clean)
test <- cbind(Id = test$Id, test_clean)

train$trainFlag <- 1
test$trainFlag <- 0
test$Response <- NA

alldata <- rbind(train, test)
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
alldata <- merge(alldata, a[, c("Id", "gender_indicator")], all.x = TRUE)
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
alldata$int_15 <- alldata$Medical_History_6 + alldata$Medical_History_13
alldata$int_16 <- alldata$Medical_History_32 + alldata$Medical_History_40
alldata$int_17 <- alldata$Medical_Keyword_15 + alldata$Medical_Keyword_3

scaleDF <- alldata[, !names(alldata) %in% c("Id", "Response", "trainFlag", "Product_Info_2")]
s <- data.frame(sapply(scaleDF, minMaxScale))
s <- log(s + 1)

alldata <- alldata[!names(alldata) %in% names(scaleDF)]
alldata <- cbind(alldata, s)

dmy <- dummyVars("~.", data = alldata)
trsf <- data.frame(predict(dmy, newdata = alldata))

ptr <- trsf[trsf$trainFlag == 1,]
pte <- trsf[trsf$trainFlag == 0,]

evalMatrix <- data.frame(Fold = numeric(), Id = numeric(), ground_truth = numeric(), xgb9_preds = numeric())
testMatrix <- data.frame(Id = pte$Id)

feature.names <- names(ptr)[!names(ptr) %in% c("Id", "Response", "trainFlag")]

for(i in 1:10) {
  
  cat("\n---------------------------")
  cat("\n------- Fold: ", i, "----------")
  cat("\n---------------------------\n")
  cname <- paste("Fold_", i)
  idx <- kf[[i]]
  idx <- idx[!is.na(idx)]
  
  trainingSet <- ptr[!ptr$Id %in% idx,]
  validationSet <- ptr[ptr$Id %in% idx,]
  
  
  cat("\nnrow train: ", nrow(trainingSet))
  cat("\nnrow eval: ", nrow(validationSet), "\n")
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainingSet[, feature.names]), label = trainingSet$Response)
  dval <- xgb.DMatrix(data = data.matrix(validationSet[, feature.names]), label = validationSet$Response)
  
  
  param <- list(objective = "reg:linear",
                max_depth = 5,
                eta = 0.01,
                subsample = 0.8,
                colsample_bytree = 0.67,
                min_child_weight = 30,
                lambda = 1,
                eval_metric = "rmse")
  
  watchlist <- list(eval = dval, train = dtrain)
  
  clf <- xgb.train(params = param,
                   data = dtrain,
                   nround =3200,
                   print.every.n = 50,
                   watchlist = watchlist)
  
  preds <- predict(clf, data.matrix(validationSet[, feature.names]))
  valid <- data.frame(Fold = rep(i, nrow(validationSet)), Id = validationSet$Id, ground_truth = validationSet$Response, xgb9_preds = preds)
  evalMatrix <- rbind(evalMatrix, valid)
}


dtrain <- xgb.DMatrix(data = data.matrix(ptr[, feature.names]), label = ptr$Response)
watchlist <- list(train = dtrain)

clf_full <- xgb.train(params = param,
                      data = dtrain,
                      nround =3200,
                      print.every.n = 50,
                      watchlist = watchlist)

test_preds <- predict(clf_full, data.matrix(pte[, feature.names]))
testMatrix$xgb9_preds <- test_preds

write_csv(evalMatrix, "./Ensemble2/xgb9_eval.csv")
write_csv(testMatrix, "./Ensemble2/xgb9_test.csv")
