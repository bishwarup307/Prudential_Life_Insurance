require(readr)
require(LiblineaR)

setwd("~/Kaggle/Prudential")
set.seed(1)

fold_id <- read_csv("./ensemble2/validation_id.csv")
load("./Data/svc_processed.RData")

alldata$int_1 <- alldata$BMI * alldata$Ins_Age
alldata$int_2 <- alldata$Wt * alldata$Wt
alldata$int_3 <- alldata$BMI * alldata$Ht
alldata$int_4 <- alldata$Employment_Info_1 * alldata$Employment_Info_4
alldata$int_5 <- alldata$Employment_Info_1 * alldata$Employment_Info_6

ptr <- alldata[alldata$trainFlag == 1,]
pte <- alldata[alldata$trainFlag == 0,]

ptr[is.na(ptr)] <- -1
pte[is.na(pte)] <- -1

svr_val2 <- data.frame(Id = numeric(), ground_truth = numeric(), svrpreds2 = numeric())
svr_test2 <- data.frame(Id = pte$Id)

for (i in 1:10) {
  
  cat("\n-----------------------------------")
  cat("\n.......Fold ", i, "...............")
  cat("\n------------------------------------")
  
  idx <- fold_id[[i]]
  idx <- idx[!is.na(idx)]
  
  validationSet <- ptr[which(ptr$Id %in% idx),]
  trainingSet <- ptr[which(!ptr$Id %in% idx),]
  
  feature.names <- names(ptr)[!names(ptr) %in% c("Id", "Response", "trainFlag")]
  
  svp <- LiblineaR(as.matrix(trainingSet[, feature.names]), trainingSet$Response, type =  11, cost = 10000, svr_eps = 1e-7, epsilon = 1e-7, verbose = FALSE)
  p <- as.numeric(unlist(predict(svp, newx = as.matrix(validationSet[, feature.names]))))
  
  df <- data.frame(Id = validationSet$Id, ground_truth = validationSet$Response, svrpreds2 = p)
  svr_val2 <- rbind(svr_val2, df)
  
  p <- as.numeric(unlist(predict(svp, newx = as.matrix(pte[, feature.names]))))
  cname <- paste0("Fold_", i)
  svr_test2[[cname]] <- p
  
}


write_csv(svr_val2, "./ensemble2/liblinear_val2.csv")
write_csv(svr_test2, "./ensemble2/liblinear_test2.csv")