# load libraries
require(readr)
require(MASS)

# set seed & working directory
setwd("F:/Kaggle/Prudential2")
set.seed(25)

# load data and validation fold ids
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
impo <- tra_clean[,(abs(ogg$coef) > quantile(abs(ogg$coef), 0.382))] #only "important" variables left
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

# split train & test
ptr <- alldata[alldata$trainFlag == 1,]
pte <- alldata[alldata$trainFlag == 0,]

# train & test prediction containers
evalMatrix <- data.frame(Fold = numeric(), Id = numeric(), ground_truth = numeric(), lm2_preds = numeric())
testMatrix <- data.frame(Id = pte$Id)

# features to use in the model
feature.names <- names(ptr)[!names(ptr) %in% c("Id", "Response", "trainFlag")]
# linear regression formula
frm <- as.formula(paste0("Response ~ ", paste(feature.names, collapse = "+")))

# generate train meta features with
# 10-fold stratified cross validation
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
  
  linear_model <- lm(frm, data=trainingSet) 
  # predict on oof samples
  preds <- predict(linear_model, newdata = validationSet[,feature.names])
  valid <- data.frame(Fold = rep(i, nrow(validationSet)), Id = validationSet$Id, ground_truth = validationSet$Response, lm2_preds = preds)
  evalMatrix <- rbind(evalMatrix, valid)
  
  # predict on test samples
  tpreds <- predict(linear_model, newdata = pte[, feature.names])
  testMatrix[[cname]] <- tpreds
  
}
# take median over all folds to 
# generate test meta feature
testMatrix$lm2_preds <- apply(testMatrix[, -1], 1, median)
#save to disk
write_csv(evalMatrix, "./Ensemble2/lm2_eval.csv")
write_csv(testMatrix, "./Ensemble2/lm2_test.csv")
