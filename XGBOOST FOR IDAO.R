#Load the libraries
library(data.table)
library(odbc)
library(DBI)
library(ROSE)
library(caret)
library(dplyr)   
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(caret) # rich package -- for exmple in this code confusionMatrix -- for gettind directly the confusion matrix values.
library(DMwR) # smote
library(xgboost) # for xgboost
library(Matrix) # you need bcos , you need to create matrix as inputs for xgboost
library(reshape) #for the melt function
library(pROC) # AUC

setwd("~/Desktop/Hackathons/Yandex DAO/IDAO-MuID")


con <- dbConnect(odbc::odbc(), .connection_string = "Driver={Microsoft Access Text Driver (*.txt, *.csv)};train_part_1_v2.csv
                 ")


data <- dbReadTable(con, "train_part_1_v2.csv")
data_test=dbReadTable(con, "test_public_v2.csv")
data3=dbReadTable(con, "train_part_2_v2.csv")


mycol=c("particle_type","Mextra_DY2(2)","PT","Lextra_X(1)","Lextra_Y(3)","MatchedHit_DY(2)","P","MatchedHit_Z(2)","MatchedHit_TYPE(3)","avg_cs(0)","MatchedHit_TYPE(0)","NShared","label")

testcol=c("Mextra_DY2.2.","PT","Lextra_X.1.","Lextra_Y.3.","MatchedHit_DY.2.","P","MatchedHit_Z.2.","MatchedHit_TYPE.3.","avg_cs.0.","MatchedHit_TYPE.0.","NShared")

data[data==-9999] <- NA
data_test[data_test==-9999] <- NA
tr1_new=data[mycol]
test=data_test[testcol]


test[c("Mextra_DY2.2.","PT","Lextra_X.1.","Lextra_Y.3.","MatchedHit_DY.2.","P","MatchedHit_Z.2.","MatchedHit_TYPE.3.","avg_cs.0.","MatchedHit_TYPE.0.","NShared")] <- test[c("Mextra_DY2.2.","PT","Lextra_X.1.","Lextra_Y.3.","MatchedHit_DY.2.","P","MatchedHit_Z.2.","MatchedHit_TYPE.3.","avg_cs.0.","MatchedHit_TYPE.0.","NShared")]

mystats_num = function(x){
  class=class(x)
  n = length(x)
  nmiss = sum(is.na(x),na.rm = T)
  nmiss_pct = mean(is.na(x))
  sum = sum(x, na.rm=T)
  mean = mean(x, na.rm=T)
  median = quantile(x, p=0.5, na.rm=T)
  std = sd(x, na.rm=T)
  cv = sd(x, na.rm=T)/mean(x, na.rm=T)
  var = var(x, na.rm=T)
  pctl = quantile(x, p=c(0, 0.01, 0.05,0.1,0.25,0.5, 0.75,0.9,0.95,0.99,1), na.rm=T)
  return(c(N=n, Nmiss =nmiss, Nmiss_pct = nmiss_pct, sum=sum, avg=mean, meidan=median, std=std, cv=cv, var=var, pctl=pctl))
}
numeric_vars=names(data)[sapply(data, FUN=is.numeric)]

summary_stats = as.data.frame(sapply(tr1,  FUN=mystats_num))

saveRDS(tr1_new,"train1.rds")
saveRDS(test,"test.rds")

tr1=readRDS("train1.RDS")
test <- readRDS("test.RDS")

#data2=data2[complete.cases(data2),]
#data2=data2[,-69]
corrm<- as.data.frame(cor(data2))                                ### CORRELATION MATRIX
#data2=data2[,-9]

FA<-fa(r=corrm, 10, rotate="varimax", fm="ml")               ### CONDUCTING FACTOR ANALYSIS
print(FA)                                                    ### PRINT THE RESULTS
FA_SORT<-fa.sort(FA)                                         ### SORTING THE LOADINGS
ls(FA_SORT)                                                  ### LISTING OUT THE OBJECTS
FA_SORT$loadings
#FA_SORT$e.values                                            ### FINDING EIGEN VALUES FROM THE RESULTS
Loadings<-data.frame(FA_SORT$loadings[1:ncol(data2),]) ### CAPTURING ONLY LOADINGS INTO DATA FRAME

View(Loadings)
getwd()
write.csv(Loadings, "loadings.csv") ### SAVING THE FILE

saveRDS(tr1_new,"train1.rds")
saveRDS(test,"test.rds")

tr1=readRDS("train1.RDS")
test <- readRDS("test.RDS")


tr1 <- head(tr1,50000)
ccdata=tr1
ccdata$particle_type <- NULL
str(ccdata)
# Check if the data is balanced or not ?
table(ccdata$label)
# TO see in percentage
prop.table(table(ccdata$label))
#ccdata$Time <- NULL# We do not need time.
#Split data into train, cv and test.
set.seed(1900)
# For dividing the data in train and cross validation and test
inTrain <- createDataPartition(y = ccdata$label, p = .6, list = F)
#p: the percentage of data that goes to training
#list: logical - should the results be in a list (TRUE) or a matrix with the number of rows equal to floor(p * length(y)) and times columns.
train <- ccdata[inTrain,]
testcv <- ccdata[-inTrain,]
inTest <- createDataPartition(y = testcv$label, p = .5, list = F)
#test
test_new <- testcv[inTest,]
cv <- testcv[-inTest,]
train$label <- as.factor(train$label)
rm(inTrain, inTest, testcv) # Removing unwanted objects from memory. Good practice
#SMOTE
#
colnames(test)
colnames(train)
#Very imbalanced dataset, so let's see if using smote can improve this model.
train_smote <- SMOTE(label ~ ., as.data.frame(train), perc.over = 20000, perc.under=100)
# perc.over : percentage of over sampling -- 200 % increase of the minority label.
# perc.under : percentage of under sampling -- equivalent to total oversampled data : 100 % means equal proportion
# Now 2 label are almost equal.
table(train_smote$label)
prop.table(table(train_smote$label))
# Something to be used for xgboost
i <- grep("label", colnames(train)) # Get index label column
# you can simply assign to i <- ncol(train)
#Prepare data for XGBoost and set parameters. Use AUC as evaluation metric, as accuracy does not make sense for such a imbalanced dataset.
# Back to numeric
train$label <- as.numeric(levels(train$label))[train$label]
train_smote$label <- as.numeric(levels(train_smote$label))[train_smote$label]
# As Matrix
train <- Matrix(as.matrix(train), sparse = TRUE)
train_smote <- Matrix(as.matrix(train_smote), sparse = TRUE)
test_new <- Matrix(as.matrix(test_new), sparse = TRUE)
cv <- Matrix(as.matrix(cv), sparse = TRUE)
# Create XGB Matrices

train_xgb <- xgb.DMatrix(data = train[,-i], label = train[,i])
train_smote_xgb <- xgb.DMatrix(data = train_smote[,-i], label = train_smote[,i])
test_xgb <- xgb.DMatrix(data = test_new[,-i], label = test_new[,i])
cv_xgb <- xgb.DMatrix(data = cv[,-i], label = cv[,i])

test <- setNames(test, c("Mextra_DY2(2)","PT","Lextra_X(1)","Lextra_Y(3)","MatchedHit_DY(2)","P","MatchedHit_Z(2)","MatchedHit_TYPE(3)","avg_cs(0)","MatchedHit_TYPE(0)","NShared"))
test_fin <- xgb.DMatrix(as.matrix(test))

# Watchlist
watchlist <- list(train = train_xgb, cv = cv_xgb)
# set parameters:
parameters <- list(
  # General Parameters
  booster = "gbtree",
  silent = 0,
  # Booster Parameters
  eta = 0.3,
  gamma = 0,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1,
  colsample_bylevel = 1,
  lambda = 1,
  alpha = 0,
  # Task Parameters
  objective = "binary:logistic",
  eval_metric = "auc",
  seed = 1900
)
# Some explaination
#eta= low -- model is more robust to overfitting
#gamma (default is 0) and with larger values we would like to have a more conservative algorithms and thus, we would like to avoid overfitting.
# subsample = lower values helps in overfitting.
# colsample_bytree
# missing = NA;
# Original
xgb.model <- xgb.train(parameters, train_xgb, nrounds = 25, watchlist)
#Plot:
melted <- melt(xgb.model$evaluation_log, id.vars="iter")
# For changing into long format.
ggplot(data=melted, aes(x=iter, y=value, group=variable, color = variable)) + geom_line()
#Try without group = variable ? What happens ?
# Smote
xgb_smote.model <- xgb.train(parameters, train_smote_xgb, nrounds = 25, watchlist)
#Plot:
melted <- melt(xgb_smote.model$evaluation_log, id.vars="iter")
ggplot(data=melted, aes(x=iter, y=value, group=variable, color = variable)) + geom_line()
# Feature importance
imp <- xgb.importance(colnames(train_xgb), model = xgb.model)
print(imp)
# Gain is the improvement in accuracy by a feature to the branches it is on.
xgb.plot.importance(imp)

# Original
predicted = predict(xgb.model, X_test,class = "raw")

predicted = predict(xgb_smote.model, test_fin,outputmargin=F,class = "prob")

# SMOTE
predicted = predict(xgb_smote.model, X_test,class = "raw")

predicted = predict(xgb_smote.model, test_fin,outputmargin=F,class = "prob")


View(predicted)
id=as.matrix(c(1:NROW(test)))
id=id-1
head(xgboost_pred,10)
xgboost_pred <- as.data.table(cbind(id,predicted))
table(xgboost_pred$predicted)
warnings(xgb_model)
xgboost_pred <- setNames(xgboost_pred,c("id","prediction"))       
write.csv(xgboost_pred,file = "Xgboost_pred.csv",row.names = F)