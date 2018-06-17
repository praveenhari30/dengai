test <-  read.csv('dengue_features_test.csv')
train <- read.csv('dengue_features_train.csv')
label <- read.csv('dengue_labels_train.csv')
format <- read.csv('submission_format.csv')

head(test)
head(train)
head(format)
head(label)

sapply(X = train, FUN = function(x) sum(is.na(x)))

cor(train[-c(1,4)])

str(train)
str(label)

tstest <- ts(test)
tstrain <- ts(train)

str(tstrain)

library(xgboost)
library(caret)

columnset = c("reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k","station_avg_temp_c", 
              "station_min_temp_c")

dmattrain = xgb.DMatrix(as.matrix(train[columnset]),label=label$total_cases)
dmattest = xgb.DMatrix(as.matrix(test[columnset]))

trainctrl = trainControl(method = "repeatedcv", repeats = 1,number = 4, allowParallel=T)

xgb.grid = expand.grid(nrounds = 750,
                       eta = c(0.01,0.005,0.001),
                       max_depth = c(4,6,8),
                       colsample_bytree=c(0,1,10),
                       min_child_weight = 2,
                       subsample=c(0,0.2,0.4,0.6),
                       gamma=0.01)
set.seed(45)
xgb_params = list(
  booster = 'gbtree',
  objective = 'reg:linear',
  colsample_bytree=1,
  eta=0.005,
  max_depth=4,
  min_child_weight=3,
  alpha=0.3,
  lambda=0.4,
  gamma=0.01, # less overfit
  subsample=0.6,
  seed=5,
  silent=TRUE)
boostrain = xgb.train(xgb_params,dmattrain, nrounds = 1000)
testy <- round(predict(boostrain,newdata = dmattest,type = "raw"))


output <- read.csv('submission_format.csv',stringsAsFactors = F)
output$total_cases <- testy
head(output)
write.csv(output,'submission.csv',row.names=F,quote = F)

plot(label$total_cases,type='l',ylab = 'Total Cases',xlab='Time')
plot(testy,type="l",ylim=c(0,450),xlim=c(100,300),main='predicted',ylab = 'Total Cases',xlab='Time')
