library(kopls) # for kopls
library(e1071) # for svm
library(microbenchmark)
library(randomForest) # for random forest
library(ada) # for adaboost
set.seed(0) # is this all I need to do?

## Utility functions.
generate.train.inxs <- function(n,holdout) {
  # hold out `holdout' for training
  k <- round((1-holdout)*n) # sample k numbers from 1:n for training
  train.inxs <- sample(n,k,replace=T)
  return(train.inxs)
}
load.dataset <- function(nam,holdout=0.3) {
  if(nam %in% c('tb','synthetic','fake')) {
    if(nam == 'tb') {
      files <- c('tb/proc_x.csv',
                 'tb/proc_y.csv',
                 'tb/phi.csv')
    } else if(nam == 'synthetic') {
      files <- c('synthetic/X.csv',
                 'synthetic/y.csv',
                 'synthetic/phi.csv')
    } else if(nam == 'fake') {
      files <- c('fake/x_f.csv',
                 'fake/y_f.csv',
                 'fake/phi.csv')
    } else {
      files <- NULL
    }

    if (!is.null(files)) {
      for(i in 1:length(files)) {
        files[i] <- paste('data_sets',files[i],sep='/')
      }
      
      # Load raw X, y, and phi.
      dataset <- list()
      dataset$X <- as.matrix(read.csv(files[1],header=F))
      a <- dim(dataset$X)
      dataset$n <- a[1]
      dataset$d0 <- a[2]
      dataset$y <- as.factor(as.matrix(read.csv(files[2],header=F)))
      dataset$phi <- as.matrix(read.csv(files[3],header=F))

      # Generate train indices and subset to create the train and test sets.
      dataset$train.inxs <- train.inxs <- generate.train.inxs(dataset$n,holdout=holdout)
      dataset$X.train <- dataset$X[train.inxs,]
      dataset$y.train <- dataset$y[train.inxs]
      dataset$phi.train <- dataset$phi[train.inxs,]
      dataset$X.test <- dataset$X[-train.inxs,]
      dataset$y.test <- dataset$y[-train.inxs]
      dataset$phi.test <- dataset$phi[-train.inxs,]
      
      print(paste(
        "Loaded dataset '",
        nam,
        "' of ",
        nrow(dataset$X),
        " ",
        ncol(dataset$X),
        "-dimensional samples (phi dim: ",
        ncol(dataset$phi),
        ") using ",
        holdout*100,
        "% holdout for testing.",sep=''))
      return(dataset)
    }
  }
  print(paste("Unable to load data set '",nam,"'"))
  return(NULL)
}
calc.mcr <- function(y,yhat) {
  return(sum(as.integer(y != yhat)) / length(y))
}

## Evaluation methods
### RF
evaluate.rf <- function(X.train,y.train,X.test,y.test,ntree=500) {
  # general method to evaluate an RF (either FF or std, based on inputs)
  results <- list()
  results$st <- list() # system time of various actions
  results$pred <- list() # results of prediction (various stats)
  
  results$st$train <- system.time( # Time the training of the model
    results$model <- model <- randomForest(X.train,
                                           y.train,
                                           ntree=ntree,
                                           prox=T))
  
  results$st$test <- system.time( # Time the testing of the model
    results$pred$yhat <- yhat <- predict(model, X.test))
  
  results$pred$mcr <- calc.mcr(y.test,yhat)
  
  return(results)
}
evaluate.ff.rf <- function(data,ntree=500) {
  results <- evaluate.rf(data$phi.train,
                         data$y.train,
                         data$phi.test,
                         data$y.test,
                         ntree=ntree)
  return(results)
}

### SVM
evaluate.svm <- function(X.train,y.train,X.test,y.test,kernel='radial') {
  # general method to evaluate an SVM (either FF or std, based on inputs
  results <- list()
  results$st <- list() # system time of various actions
  results$pred <- list() # results of prediction (various stats)
  
  results$st$train <- system.time( # Time the training of the model
    results$model <- model <- svm(X.train,
                                  y.train,
                                  kernel=kernel))
  
  results$st$test <- system.time( # Time the testing of the model
    results$pred$yhat <- yhat <- predict(model,X.test)
  )
  
  results$pred$mcr <- calc.mcr(y.test,yhat)
  
  return(results)
}
evaluate.ff.svm <- function(data) {
  results <- evaluate.svm(data$phi.train,
                          data$y.train,
                          data$phi.test,
                          data$y.test,
                          kernel='linear')
  return(results)
}
evaluate.std.svm <- function(data) {
  results <- evaluate.svm(data$X.train,
                          data$y.train,
                          data$X.test,
                          data$y.test,
                          kernel='radial')
  return(results)
}

### Adaboost
evaluate.adaboost <- function(X.train,y.train,X.test,y.test) {
  # general method to evaluate adaboost (either FF or std, based on inputs)
  results <- list()
  results$st <- list() # system time of various actions
  results$pred <- list() # results of prediction (various stats)
  
  results$st$train <- system.time( # Time the training of the model
    results$model <- model <- ada(X.train,
                                  y.train))
  
  results$st$test <- system.time( # Time the testing of the model
    results$pred$yhat <- yhat <- predict(model,X.test)
  )
  
  results$pred$mcr <- calc.mcr(y.test,yhat)
  
  return(results)
}
evaluate.ff.adaboost <- function(data) {
  results <- evaluate.adaboost(data$phi.train,
                               data$y.train,
                               data$phi.test,
                               data$y.test)
  return(results)
}
evaluate.std.adaboost <- function(data) {
  results <- evaluate.adaboost(data$X.train,
                               data$y.train,
                               data$X.test,
                               data$y.test)
  return(results)
}

# Load data set.
data <- load.dataset('tb')

# Calculate and gather results.
results <- list()
# Fastfood algorithms
results$ff <- list()
results$ff$rf <- evaluate.ff.rf(data)
#results$ff$svm <- evaluate.ff.svm(data)
#results$ff$opls <- evaluate.ff.opls(data)
results$ff$adaboost <- evaluate.ff.adaboost(data)
# Standard algorithms
results$std <- list()
results$std$rf <- evaluate.std.rf(data)
results$std$svm <- evaluate.std.svm(data)
results$std$adaboost <- evaluate.std.adaboost(data)
#results$std$kopls <- evaluate.ff.kopls(data)
