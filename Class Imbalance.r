install.packages("ROSE", repos = "http://cran.us.r-project.org")

install.packages('ROSE')

library(ROSE)

data(hacide)

str(hacide.train)
#The data contains 3 variables of 1000 observations. cls is the dependent variable. Let's check the class imbalance on cls.

table(hacide.train$cls)

prop.table(table(hacide.train$cls))

#Let's build a decision tree with this.
library(rpart)

dt <- rpart(formula = cls ~ ., data = hacide.train)

pred.dt <- predict(object = dt, newdata = hacide.test)

#Let's check the prediction accuracy of the default decision tree.
accuracy.meas(response = hacide.test$cls, predicted = pred.dt[,2])

roc.curve(response = hacide.test$cls, predicted = pred.dt[,2], plotit = T)
#AUC of 60% is a very poor performance

balanced_data_under = ovun.sample(formula = cls ~ ., data = hacide.train, method = "under", N = 40, seed = 1)$data
#N = total number of observations. Undersampled until majority class reaches minority class.
#seed = since random elimination is taking place, a seed is important.

table(balanced_data_under$cls)
#A lot of information is lost with this technique. Samples have reduced from 980 to 20!

balanced_data_over = ovun.sample(formula = cls ~ ., data = hacide.train, method = "over", N = 1960)
#over: tells the function to perform oversampling
#N: tells it oversample until the total number of observations reach 1960. We choose 1960 because 980+980=1960. The limit we can oversample with is 980.

table(balanced_data_over$data$cls)

balanced_data_both = ovun.sample(formula = cls~.,data = hacide.train, method = "both", p = 0.5,seed = 1)
#p=is the probability of the minority class. Default is 0.5
#seed is required due to undersampling

table(balanced_data_both$data$cls)

balanced_data_syn <- ROSE(formula = cls~., data = hacide.train, seed = 1)

table(balanced_data_syn$data$cls)

#devtools::install_github("ncordon/imbalance")

library(imbalance)

head(hacide.train)

dim(hacide.train)

table(hacide.train$cls)

prop.table(table(hacide.train$cls))

imbalanceRatio(hacide.train, classAttr = 'cls')

balanced_data_over_mwmote = oversample(dataset = hacide.train, ratio = 1, method = 'MWMOTE', classAttr = 'cls')
#classAttr = column identifying the class variable
#ratio = desired ratio of the minority class/majority class
#mwmote = an improved method of smote

prop.table(table(balanced_data_over_mwmote$cls))

dt.under = rpart(formula = cls~.,data = balanced_data_under)
dt.over = rpart(formula = cls~., data = balanced_data_over$data)
dt.both = rpart(formula = cls~., data = balanced_data_both$data)
dt.syn = rpart(formula = cls~., data = balanced_data_syn$data)

pred.under = predict(object = dt.under, newdata = hacide.test)
pred.over = predict(object = dt.over, newdata = hacide.test)
pred.both = predict(object = dt.both, newdata = hacide.test)
pred.syn = predict(object = dt.syn, newdata = hacide.test)

#Dataset for undersampling has only 40 observations
#keeping in mind that 1 was the minority class...
roc.curve(response = hacide.test$cls, predicted = pred.under[,2], plotit = F)
roc.curve(response = hacide.test$cls, predicted = pred.over[,2], plotit = F)
roc.curve(response = hacide.test$cls, predicted = pred.both[,2], plotit = F)
roc.curve(response = hacide.test$cls, predicted = pred.syn[,2], plotit = F)

rose.cv <- ROSE.eval(formula = cls~., data = hacide.train, learner = rpart, acc.measure = "auc", method.assess = "LKOCV", K = 10, seed = 1, extr.pred = function(obj)obj[,2])

rose.cv

rose.cv.under <- ROSE.eval(formula = cls~., data = balanced_data_under, learner = rpart, acc.measure = "auc", method.assess = "LKOCV", K = 10, seed = 1, extr.pred = function(obj)obj[,2])

rose.cv.under

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

iris = datasets.load_iris()

type(iris.data)

#Load 100 rows only
x = iris.data[:100,:]
y = iris.target[:100]

np.unique(y, return_counts=True)

#Make class highly imbalanced by removing the first 40 observations
x = x[40:,]
y = y[40:]

y

np.unique(y, return_counts=True)

#Standardize the features
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

#Defining the cost sensitive models
svc = SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=0)
decision_tree = DecisionTreeClassifier(random_state=0, class_weight='balanced')
logistic = LogisticRegression(random_state=0, class_weight='balanced')
ridge = RidgeClassifier(random_state=0, class_weight='balanced')

#Train models
svc_model = svc.fit(X=x_std, y=y)
dt_model = decision_tree.fit(X=x_std, y=y)
logistic_model = logistic.fit(X=x_std, y=y)
ridge_model = ridge.fit(X=x_std, y=y)

from sklearn.metrics import balanced_accuracy_score

# DETERMINE BALANCED ACCURACY SCORE OF A LOGISTIC REGRESSION
log_naive = LogisticRegression(random_state=0)

log_naive_model = log_naive.fit(X=x_std, y=y)

iris.target[:100]

np.unique(iris.target[:100],return_counts=True)

print(iris.target[30:60])
print(np.unique(iris.target[30:60], return_counts=True))

test_y = iris.target[30:60]
test_x = iris.data[30:60]

pred_y = log_naive_model.predict(X=test_x)

balanced_accuracy_score(y_true=test_y, y_pred=pred_y)

# DETERMINE BALANCED ACCURACY SCORE OF A DECISION TREE
dt_class_naive = DecisionTreeClassifier(random_state=0)

dt_class_naive_model = dt_class_naive.fit(X=x_std, y=y)

pred_y_dt = dt_class_naive_model.predict(test_x)

balanced_accuracy_score(y_pred=pred_y_dt, y_true=test_y)
# Results aren't apparent but if using the dataset/guide in the ref, performance was noted to improve

library(ROSE)

install.packages('rpart')

library(rpart)

data(hacide)

test_data <- hacide.test
train_data <- hacide.train

head(test_data)

cost.rpart <- rpart(data = test_data,
                    formula = cls ~.,
                    parms = list(loss=matrix(c(0,1,5,0),nrow = 2,byrow = TRUE)))

print(cost.rpart)

summary(cost.rpart)

plotcp(cost.rpart)

printcp(cost.rpart)
# The best complexity parameter is 0.13333

# Now we prune the model using cp
cost.rpart.pruned <- prune(cost.rpart,0.13333)

par(mfrow = c(1,2))
plot(cost.rpart);text(cost.rpart)
plot(cost.rpart.pruned);text(cost.rpart.pruned)

#install.packages('mlr3') - successor to mlr but poorer documentation
#install.packages('mlr') - this one is in maintenance mode since July 2019 but has better documentation

library(mlr3)
library(mlr)

data(GermanCredit, package = 'caret')

head(GermanCredit)

credit.task = makeClassifTask(data = GermanCredit, target = 'Class')

#We are preparing the data by specifying that this is a classification task

credit.task = removeConstantFeatures(credit.task)

credit.task
#One of the classes is arbitrarily selected as the positive class i.e. the class of interest
#and the other will be the negative class

#Defining the loss matrix
costs = matrix(c(0,0,1,-0.35),2)

colnames(costs) = rownames(costs) = getTaskClassLevels(credit.task)

costs

lnr = makeLearner(cl = 'classif.multinom', predict.type = 'prob')

#Select model

mod = train(learner = lnr, task = credit.task)

#Train model

pred = predict(object = mod, task = credit.task)
pred

th = 0.2593

#We now adjust our predictions using the new threshold
pred.th = setThreshold(pred = pred, threshold = th)
pred.th

# Calculate and compare costs between the new thresholds
credit.costs = makeCostMeasure(costs = costs, best = 0, worst = 100)
credit.costs
# This prepares the model for computing costs

performance(pred = pred, measures = list(credit.costs, mmce))
#with 0.5 threshold

performance(pred = pred.th, measures = list(credit.costs, mmce))
#with predictions adjusted at 0.25 threshold

#Define training/resampling strategy
rin = makeResampleInstance(desc = 'CV', task = credit.task, iters=3)

#Make learner
lrn.th = makeLearner(cl = 'classif.multinom', predict.type = 'prob', predict.threshold = th)

#Traing/resample
r = resample(learner = lrn.th, task = credit.task, resampling = rin, measures = list(credit.costs, mmce))
r

#As expected, costs are less but classification error are higher

#If we were to perform the same resampling/training at 0.5 threshold...
#We set the predictions from the resampling output at 0.5 threshold

performance(pred = setThreshold(r$pred,0.5), measures = list(credit.costs,mmce))

d = generateThreshVsPerfData(r, measures = list(credit.costs,mmce))
plotThreshVsPerf(d, mark.th = th)

d = generateThreshVsPerfData(r, measures = list(credit.costs))
plotThreshVsPerf(d, mark.th = th)
plotThreshVsPerf(d, mark.th = 0.5)
#Costs: -0.0953044 vs -0.0566209922497347
# You can see clearly from this graph that the threshold should be lowered

#Make a learner
lnr = makeLearner(cl = 'classif.multinom', predict.type = 'prob')

#Set resampling strategy
rin = makeResampleInstance(desc = 'CV', task = credit.task, iters=3)

#Resample
r = resample(learner = lnr, task = credit.task, resampling = rin, measures = list(credit.costs, mmce))
r

#Tune thresholds of the predictions according to the credit costs
tuneThreshold(r$pred, measure = credit.costs)

d = generateThreshVsPerfData(r, measures = list(credit.costs))
plotThreshVsPerf(d, mark.th = th)
plotThreshVsPerf(d, mark.th = 0.20112783300862)
# Costs: -0.0953044 vs -0.09275
# The issue is that the mean costs are not lower yet according to the graph they are lower

#Models that support 'class weights'
listLearners(obj = 'classif', properties = 'class.weights')[c('class','package')]

#Models that support 'observational weights'
listLearners(obj = 'classif', properties = 'weights')[c('class','package')]

#Calculating theoretical weight
w = (1-0.2593)/0.2593
w

#Make learner
lnr.w = makeLearner(cl = 'classif.multinom', predict.type = 'prob')

#Set weights to model/learner. This works for both observational and class weights
lnr.w = makeWeightedClassesWrapper(learner = lnr.w, wcw.weight = w)
lnr.w

#Resampling strategy
rin = makeResampleInstance(desc = 'CV', task = credit.task, iters=3)

#Perform resampling
r.w = resample(learner = lnr.w, task = credit.task, resampling = rin, measures = list(credit.costs, mmce))
r.w

#Make learner
lnr.ew = makeLearner(cl = 'classif.multinom', predict.type = 'prob')

#Set weights
lnr.ew = makeWeightedClassesWrapper(lnr.ew)

#Now we want to test a range of weights. Knowing the theoretical weight beforehand helps us set the range
#This is basically hyper-parameter tuning
ps = makeParamSet(makeDiscreteParam(id = 'wcw.weight', values = seq(from = 0.5, to = 5, by = 0.1)))

ctrl = makeTuneControlGrid()

tune.res = tuneParams(learner = lnr.ew,
                      task = credit.task,
                      resampling = rin,
                      measures = list(credit.costs, mmce),
                      par.set = ps,
                      control = ctrl)

tune.res
#empirical weight=2.7 vs 2.86
#cost=-0.0964307 vs -0.0945831
#Costs for empirical weights are lower

library(mlr)

#Use data from mlbench
df = mlbench::mlbench.waveform(500)

#Specify nature of task
wf.task = makeClassifTask(data = as.data.frame(df), target = 'classes')

#Define cost matrix
costs = matrix(c(0,5,10,30,0,8,80,4,0),3)
colnames(costs) = rownames(costs) = getTaskClassLevels(wf.task)

#Define cost measure
wf.costs = makeCostMeasure(costs = costs, best = 0, worst = 10)

#Init model
lnr.mc = makeLearner(cl = 'classif.rpart', predict.type = 'prob')

#Define resampling strategy
wf.rin = makeResampleInstance(desc = 'CV', task = wf.task, iters=3)

wf.r = resample(learner = lnr.mc, task = wf.task, resampling = wf.rin, measures = list(wf.costs, mmce))
wf.r

#Calculate theoretical threshold
wf.th = 2/rowSums(costs)
names(wf.th) = getTaskClassLevels(wf.task)
wf.th

#Set prediction threshold
pred.th = setThreshold(pred = wf.r$pred, threshold = wf.th)

#Measure performance at set threshold
performance(pred = pred.th, measures = list(wf.costs, mmce), task = wf.task)

tuneThreshold(pred = wf.r$pred, measure = wf.costs)

#For comparison, we show the standardised versions of the thoretical thresholds
wf.th/sum(wf.th)

lnr.mc = makeLearner(cl = 'classif.multinom', predict.type = 'prob')
lnr.mc = makeWeightedClassesWrapper(lnr.mc)

#Define training parameters
ps = makeParamSet(makeNumericVectorParam("wcw.weight", len = 3, lower = 0, upper = 1))

#Define grid search
ctrl = makeTuneControlRandom()

#Tune model/runner
tune.res = tuneParams(learner = lmr.mc,
           task = wf.task,
           resampling = wf.rin,
           measures = list(wf.costs,mmce),
           control = ctrl,
           par.set = ps)

tune.res

library(dplyr)

# To compute tp, fp, tn, fn we need 3 variables:
# observed classes (actuals), predicted classes, threshold at which these predictions are taking place
# Threshold is the point at which the probability of prediction > threshold is converted to a class 
# (an ROC curve is drawn over many thresholds)
# Therefore, these predictions need to be observed over a threshold
# In this case the threshold is the imbalance ratio i.e. minority class/majority class in the observations=0.2

getProfit <- function(obs, pred, threshold=0.2) {
    prob <- pred
    pred <- ifelse(pred >= threshold,1,0)
    #now define tp, fp, tn, fn....
    #tp is where both pred=1 & obs=1 [correct pred]
    #fp is where pred=1 but obs=0 [false alarm]
    #tn is where pred=0 & obs=0 [correct pred]
    #fn is where pred=0 & obs=1 [the bad one]
    
    tp_count <- sum(pred==1 & obs==1)
    fp_count <- sum(pred==1 & obs==0)
    tn_count <- sum(pred==0 & obs==0)
    fn_count <- sum(pred==0 & obs==1)
    
    #punish false negatives but reward true negatives
    profit <- tn_count*3 - fn_count*10
    return(profit)
}

getBenchmarkProfit <- function(obs) {
    n <- length(obs)
    getProfit(obs, rep(x = 0,times = n)) #predictions is a replication of 0, n times. Threshold is already set.
}

getLift <- function(probs, labels, thresh){
    pred_profit <- as.numeric(getProfit(obs=labels,pred=probs,threshold=thresh))
    naive_profit <- as.numeric(getBenchmarkProfit(labels))
    profit_lift <- pred_profit/naive_profit
    return(profit_lift)
}

library(xgboost)

#extract features and labels
head(hacide.train)

train_labels <- hacide.train$cls

length(train_labels)

train_features <- hacide.train

train_features$cls <- NULL

head(train_features)

test_labels <- hacide.test$cls

test_features <- hacide.test
test_features$cls <- NULL

head(test_features)

levels(test_labels)

levels(train_labels)

print(class(train_labels))
print(levels(train_labels))
print(length(train_labels))

class(test_labels)

train_labels <- as.numeric(as.character(train_labels))
# We insert as.character to preserve the levels as numerals. Otherwise we get 1,2

test_labels <- as.numeric(as.character(test_labels))

dtrain <- xgb.DMatrix(as.matrix(train_features), label = train_labels)
dtest <- xgb.DMatrix(as.matrix(test_features), label = test_labels)

# These params did not work
xgb_params <- list(objective='binary:logistic',
                  eta=0.03,
                  max_depth=4, #default booster is gbtree
                  colsample_bytree=1,
                  subsample=0.75,
                  min_child_weight=1)

xgb_params_2 <- list(objective='multi:softmax',num_class=2)

thresholds <- seq(from = 0.1, to = 0.69, by = 0.01)

performance <- vector(length = 60)

# Evaluation function
# Function for every iteration to use a new threshold
for(i in 1:9){
    xgb.getLift <- function(preds, dtrain)
        labels <- getinfo(dtrain, 'label')
    lift <- getLift(preds, labels, thresholds[i])
    return(list(metric='Lift', value=lift))
}

set.seed(512)

# Train model using current iteration threshold
xgb_fit <- xgb.cv(params = xgb_params, data = dtrain, nfold = 5, nrounds = 250, feval = xgb.getLift, maximize = TRUE, early_stopping_rounds = 50, verbose = TRUE, print_every_n = 100)

#Store results
