require(readr)
setwd("~/Kaggle/Prudential")

cv <- read_csv("fold_indices.csv")
train <- read_csv("./Data/train.csv")

valIds <- data.frame(dummy = 1:5939)

for (i in 1:10) {
  
  idx <- cv[[i]]
  LenIdx <- length(idx)
  idx <- idx[!is.na(idx)]
  
  trIdx <- train$Id[idx]
  LenTr <- length(trIdx)
  
  if(LenIdx > LenTr){
    trIdx <- c(trIdx, rep(NA, LenIdx - LenTr))
  }
  
  cname <- paste0("Fold_", i)
  valIds[[cname]] <- trIdx
  
}

valIds$dummy <- NULL

write_csv(valIds, "./ensemble2/validation_id.csv")