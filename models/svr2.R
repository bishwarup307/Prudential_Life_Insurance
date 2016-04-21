# load libraries
require(readr)
require(MASS)
require(Metrics)
require(caret)
require(LiblineaR)

#set seed & wd
setwd("F:/Kaggle/Prudential2")
set.seed(25)

# load data
train <- read_csv("./RawData/train.csv")
test <- read_csv("./RawData/test.csv")
kf <- read_csv("./Ensemble2/validation_id_ensemble_2.csv")

a <- rbind(train[, c("Id", "Family_Hist_2", "Family_Hist_3")], test[, c("Id", "Family_Hist_2", "Family_Hist_3")])
a$gender_indicator <- ifelse(is.na(a$Family_Hist_2) & !is.na(a$Family_Hist_3), 0,
                  ifelse(!is.na(a$Family_Hist_2) & is.na(a$Family_Hist_3), 1, -1))

# impute NA with median values
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

# variable selection with RIDGE regression
tra_clean <- manage_na(train[,-c(1,3,128)])
ogg <- lm.ridge(train$Response ~ ., data=tra_clean, lambda=0.5)
impo <- tra_clean[,(abs(ogg$coef) > quantile(abs(ogg$coef), 0.382))]              #only "important" variables left
var_names <- names(impo)

# remove useless features
test_clean <- manage_na(test[,- c(1, 3)])
train_clean <- cbind(impo, Product_Info_2 = train[, 3])
test_clean <- cbind(test_clean[, var_names], Product_Info_2 = test[, 3])

# best inetraction cut-offs found by cross-validation
qbmic <- 0.8
qbmic2 <- 0.9

# merge train & test
train <- cbind(train[, c("Id", "Response")], train_clean)
test <- cbind(Id = test$Id, test_clean)
train$trainFlag <- 1
test$trainFlag <- 0
test$Response <- NA
alldata <- rbind(train, test)

# basic feature engineering
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

# scale the features - aka standardize
scaleDF <- alldata[, !names(alldata) %in% c("Id", "Response", "trainFlag", "Product_Info_2")]
s <- data.frame(scale(scaleDF))
names(s) <- names(scaleDF)
alldata <- alldata[!names(alldata) %in% names(scaleDF)]
alldata <- cbind(alldata, s)

# dummy code categorical features
dmy <- dummyVars("~.", data = alldata)
trsf <- data.frame(predict(dmy, newdata = alldata))
ptr <- trsf[trsf$trainFlag == 1,]
pte <- trsf[trsf$trainFlag == 0,]

# Squared Quadratic Weighted Kappa
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

# optimum cut-point contrainer for oof samples
cp <- data.frame(c1 = numeric(),
                 c2 = numeric(),
                 c3 = numeric(),
                 c4 = numeric(),
                 c5 = numeric(),
                 c6 = numeric(),
                 c7 = numeric())

# constraint matrix for constrained maximimation of SQWK
ac <- data.frame(a1 = c(-1, 1, rep(0, 5)),
                 a2 = c(0, -1, 1, rep(0, 4)),
                 a3 = c(0, 0, -1, 1, rep(0, 3)),
                 a4 = c(rep(0, 3), -1, 1, 0, 0),
                 a5 = c(rep(0, 4), -1, 1, 0),
                 a6 = c(rep(0, 5), -1, 1))
ac <- t(as.matrix(ac))

# train & test prediction containers
evalMatrix <- data.frame(Fold = numeric(), Id = numeric(), ground_truth = numeric(), predicted = numeric(), hard_preds = numeric())
testMatrix <- data.frame(Id = pte$Id)
test_hard <- data.frame(Id = pte$Id)
CV <- c()

# features to consider in the model
feature.names <- names(ptr)[!names(ptr) %in% c("Id", "Response", "trainFlag")]

# generate train meta-features with 10-fold 
# stratified cross validation
for(i in 1:10) {
  
  cat("\n---------------------------")
  cat("\n------- Fold: ", i, "--------")
  cat("\n---------------------------\n")
  cname <- paste("Fold_", i)
  idx <- kf[[i]]
  idx <- idx[!is.na(idx)]
  trainingSet <- ptr[!ptr$Id %in% idx,]
  validationSet <- ptr[ptr$Id %in% idx,]
  

  cat("\nnrow train: ", nrow(trainingSet))
  cat("\nnrow eval: ", nrow(validationSet), "\n")
  
  svp <- LiblineaR(as.matrix(trainingSet[, feature.names]), trainingSet$Response, type =  11, cost = 10000, svr_eps = 1e-7, epsilon = 1e-7, verbose = TRUE)
  preds <- as.numeric(unlist(predict(svp, newx = as.matrix(validationSet[, feature.names]))))
  
  valid <- data.frame(Fold = rep(i, nrow(validationSet)), Id = validationSet$Id, ground_truth = validationSet$Response, predicted = preds)
  
  # optimize cut point for the validation frame
  initx <- seq(1.5, 7.5, by = 1)  
  optCuts <- constrOptim(initx, SQWK, method = "Nelder-Mead", control = list(trace = T), ui = ac, ci  = rep(0, 6))  
  pct1 <- optCuts$par
  
  preds_hardcoded <- as.integer(Hmisc::cut2(valid$predicted, cuts = pct1))
  valid$hard_preds <- preds_hardcoded
  evalMatrix <- rbind(evalMatrix, valid)
  
  tpreds <- as.numeric(unlist(predict(svp, newx = as.matrix(pte[, feature.names]))))
  testMatrix[[cname]] <- tpreds
  
  tpreds_hard <- as.integer(Hmisc::cut2(tpreds, cuts = pct1))
  test_hard[[cname]] <- tpreds_hard
  
  fpct <- data.frame(c1 = pct1[1],
                     c2 = pct1[2],
                     c3 = pct1[3],
                     c4 = pct1[4],
                     c5 = pct1[5],
                     c6 = pct1[6],
                     c7 = pct1[7])
  
  cp <- rbind(cp, fpct)
  CV <- c(CV, ScoreQuadraticWeightedKappa(valid$ground_truth, valid$hard_preds, 1, 8))
  cat("\n best kappa: ", ScoreQuadraticWeightedKappa(valid$ground_truth, valid$hard_preds, 1, 8))
  
}

# mean & median of optimized threshold values
# for all the folds
cp_mean <- as.numeric(sapply(cp, mean))
cp_med <- as.numeric(sapply(cp, median))

# hard-coded predictions cut at optimized mean and median
# cut-offs respectively
hc_mean <- as.integer(Hmisc::cut2(evalMatrix$predicted, cuts = cp_mean))
hc_med <- as.integer(Hmisc::cut2(evalMatrix$predicted, cuts = cp_med))

# SQWK
ScoreQuadraticWeightedKappa(evalMatrix$ground_truth, hc_mean)
ScoreQuadraticWeightedKappa(evalMatrix$ground_truth, hc_med)

# Nelder-Mead method for SQWK maximization 
# feeding the best oof cutoffs
o <- optim(cp_med, SQWK, method = "Nelder-Mead", control = list(trace = T))
opc1 <- o$par
o <- optim(opc1, SQWK, method = "Nelder-Mead", control = list(trace = T))
opc2 <- o$par
o <- optim(opc2, SQWK, method = "Nelder-Mead", control = list(trace = T))
opc3 <- o$par

testMatrix$svr2_preds <- apply(testMatrix[, -1], 1, median)
t_hc_preds <- as.integer(Hmisc::cut2(testMatrix$med, cuts = opc3))
sub <- data.frame(Id = testMatrix$Id, Response = t_hc_preds)
write_csv(sub, "./linear_model_custom_vars.csv")

# save meta to disk
names(evalMatrix)[4] <- "svr2_preds"
write_csv(evalMatrix, "./Ensemble2/svr2_eval.csv")
write_csv(testMatrix, "./Ensemble2/svr2_test.csv")
