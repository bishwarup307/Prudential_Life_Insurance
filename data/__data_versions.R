# load require libraries
require(readr)
require(caret)
require(Metrics)
require(Rtsne)

# set working directory
setwd("~/Kaggle/Prudential")

# load raw data files
train <- read_csv("./Data/train.csv")
test <- read_csv("./Data/test.csv")
test$Response <- (-1)
train$split1 <- 0
test$split1 <- 2
md <- rbind(train, test)
md[is.na(md)] <- -1
##########
## V1 ####
##########
save(md, file = "./Data/Pru_v1.RData")

names(train)
sapply(train[,-1], function(x) length(unique(x)))
sapply(train[,-1], function(x) sum(is.na(x)))
which(sapply(train[,-1], class) == "character")
which(sapply(train[,-1], class) == "factor")
which(sapply(md, function(x) length(unique(x))) == 2)

nm <- names(md)[grepl("Medical_Keyword", names(md))]

numeric_cols <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5",
                  "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5")

discrete_cols <- c("Medical_History_1", "Medical_History_15", "Medical_History_24", "Medical_History_32", nm)
cat_cols <- setdiff(names(md), c(numeric_cols, discrete_cols, "Id", "Response"))

num.data <- md[, numeric_cols]
disc.data <- md[, discrete_cols]
cat.data <- md[, cat_cols]

##########
### V2 ###
##########
# row wise missing count
md$NA_count <- apply(md, 1, function(x) sum(is.na(x)))

md$Product_Info_2 <- as.integer(as.factor(md$Product_Info_2))

employmentCols <- names(md)[grepl("Employment_Info", names(md))]
insuranceHistoryCols <- names(md)[grepl("Insurance_History", names(md))]
familyHistoryCols <- names(md)[grepl("Family_Hist", names(md))]
medicalHistoryCols <- names(md)[grepl("Medical_History", names(md))]

employment.data <- md[, employmentCols]
insurance.data <- md[, insuranceHistoryCols]
family.data <- md[, familyHistoryCols]
medical.data <- md[, medicalHistoryCols]

md$employment_NA_count <- apply(employment.data, 1, function(x) sum(is.na(x)))
md$insurance_NA_count <- apply(insurance.data, 1, function(x) sum(is.na(x)))
md$family_NA_count <- apply(family.data, 1, function(x) sum(is.na(x)))
md$medical_NA_count <- apply(medical.data, 1, function(x) sum(is.na(x)))
md$is_medical_history1_NA <- ifelse(is.na(md$Medical_History_1), 1, 0)

c1 <- setdiff(medicalHistoryCols, c("Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32"))
c2 <- names(md)[grepl("Medical_Keyword", names(md))]
c3 <- setdiff(insuranceHistoryCols, "Insurance_History_5")
c4 <- names(md)[grepl("InsuredInfo", names(md))]
selectedCols <- c(c1, c2, c3, c4, "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", "Family_Hist_1")
selectedData <- md[, selectedCols]

md$KM_4 <- kmeans(selectedData, centers = 4, iter.max = 50)$cluster
md$KM_8 <- kmeans(selectedData, centers = 8, iter.max = 50)$cluster
md$KM_12 <- kmeans(selectedData, centers = 12, iter.max = 50)$cluster

# tsne <- Rtsne(selectedData, dims = 3, theta = 0.4, perplexity = 30, pca = FALSE, check_duplicates = FALSE, verbose = TRUE)
# tsne_components <- data.frame(tsne$Y)
# names(tsne_components) <- c("TSNE_1", "TSNE_2", "TSNE_3")
# save(tsne_components, file = "./Data/tsnetsne_components.RData")
#########
## V3 ###
#########

load("./Data/tsnetsne_components.RData")
md <- cbind(md, tsne_components)

md$int_1 <- md$Wt/md$Ht
md$int_11 <- md$Wt/md$Ins_Age
md$int_12 <- md$BMI/md$Ht
md$int_13 <- md$BMI/md$Ins_Age
md$int_14 <- md$BMI/md$Wt

md$int_2 <- md$Family_Hist_2 - md$Family_Hist_3                  ###########################################
md$int_3 <- md$Family_Hist_2 - md$Family_Hist_4                  ######## Family history interactions ######
md$int_4 <- md$Family_Hist_2 - md$Family_Hist_5                  ###########################################
md$int_5 <- md$Family_Hist_3 - md$Family_Hist_4
md$int_6 <- md$Family_Hist_3 - md$Family_Hist_5
md$int_7 <- md$Family_Hist_4 - md$Family_Hist_5

md$int_8 <- md$Employment_Info_1 - md$Employment_Info_4
md$int_9 <- md$Employment_Info_1 - md$Employment_Info_6
md$int_10 <- md$Employment_Info_4 - md$Employment_Info_6

md[is.na(md)] <- -1

save(md, file = "./Data/pru_v2.RData")
save(md, file = "./Data/pru_v3.RData")          # Same version as V2 just with NA's as is

###########
#### V4 ###
###########
load("./Data/pru_v3.RData")

medical.key <- md[, names(md)[grepl("Medical_Keyword", names(md))]]
md$int_11 <- apply(medical.key, 1, sum)

medical.history <-  md[, names(md)[grepl("Medical_History", names(md))]]
k <- sapply(medical.history, function(x) length(unique(x)))
selected.medical.history <- medical.history[, names(which(k == 3))]
md$int_12 <- apply(selected.medical.history, 1, function(x) sum(x == 1))
md$int_13 <- apply(selected.medical.history, 1, function(x) sum(x == 2))
md$int_14 <- apply(selected.medical.history, 1, function(x) sum(x == 3))
md$int_15 <- apply(selected.medical.history, 1, sum)

train <- read_csv("./Data/train.csv")
test <- read_csv("./Data/test.csv")
test$Response <- (-1)
cmb <- rbind(train, test)
md$int_16 <- as.integer(as.factor(substring(cmb$Product_Info_2, 1, 1)))

md$age_categorized <- as.integer(cut(md$Ins_Age, breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1)))
md$Product_Info_4_categorized <- as.integer(cut(md$Product_Info_4, breaks = seq(0, 1, 0.1)))
md <- md[, !names(md) %in% c("Product_Info_2_categorized")]

md$int_17 <- md$Employment_Info_1 + md$Employment_Info_6
md$int_18 <- md$Employment_Info_6 - md$Employment_Info_1

md$int_19 <- ifelse(is.na(md$Insurance_History_5), 1, 0)

insurance.history <-  md[, names(md)[grepl("Insurance_History", names(md))]]
k <- sapply(insurance.history, function(x) length(unique(x)))
selected.insurance.history <- insurance.history[, names(which(k == 3))]
md$int_20 <- apply(selected.insurance.history, 1, function(x) sum(x == 1))
md$int_21 <- apply(selected.insurance.history, 1, function(x) sum(x == 2))
md$int_22 <- apply(selected.insurance.history, 1, function(x) sum(x == 3))

family.history <- md[, c("Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5")]
md$int_23 <- family.history$Family_Hist_2 + family.history$Family_Hist_4
md$int_24 <- family.history$Family_Hist_4 - family.history$Family_Hist_2
md$int_25 <- ifelse(is.na(md$Family_Hist_2), 1, 0)
md$int_26 <- ifelse(is.na(md$Family_Hist_3), 1, 0)
md$int_27 <- ifelse(is.na(md$Family_Hist_4), 1, 0)
md$int_28 <- ifelse(is.na(md$Family_Hist_5), 1, 0)
md$int_29 <- ifelse(!is.na(md$Family_Hist_3) & !is.na(md$Family_Hist_4), 1, 0)
md$int_30 <- md$Family_Hist_4 - md$Family_Hist_3
md$int_31 <- md$Family_Hist_3 - md$Family_Hist_5
md$int_32 <- md$Family_Hist_3 + md$Family_Hist_5

md$int_33 <- ifelse(!is.na(md$Medical_History_15), 1, 0)
md$int_34 <- ifelse(!is.na(md$Medical_History_24), 1, 0)
md$int_35 <- ifelse(!is.na(md$Medical_History_15) & !is.na(md$Medical_History_24), 1, 0)
md$int_36 <- ifelse(!is.na(md$Medical_History_32), 1, 0)

save(md, file = "./Data/pru_v4.RData")

##########
### V5 ###
##########

load("./Data/pru_v4.RData")

md$Ht_categorized <- as.integer(cut(md$Ht, breaks = seq(0, 1, 0.1)))
md$Wt_categorized <- as.integer(cut(md$Wt, breaks = seq(0, 1, 0.1)))
md$BMI_categorized <- as.integer(cut(md$BMI, breaks = seq(0, 1, 0.1)))
md$Insurance_History_5_categorized <- as.integer(cut(md$Insurance_History_5, breaks = seq(0, 1, 0.1)))

md$int_37 <- ifelse(is.na(md$Insurance_History_5), 0, 1)
md$int_38 <- ifelse(!is.na(md$Medical_History_10), 1, 0)
md$int39 <- md$Ins_Age*md$Wt
md$int_40 <- md$Ins_Age*md$BMI
md$int_41 <- md$Wt*md$BMI
md$int_42 <- md$Ins_Age*md$Wt*md$BMI
md$int_43 <- md$Ins_Age*md$Wt*md$BMI*md$Ht

save(md, file = "./Data/pru_v5.RData")

###########
### V6 ####
###########

load("./Data/pru_v5.RData")
md$int_44 <- md$BMI^3 + md$Wt^3 + md$Employment_Info_1^2
save(md, file = "./Data/pru_v6.RData")