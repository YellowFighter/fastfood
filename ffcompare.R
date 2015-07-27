library(kopls)
library(e1071)
library(microbenchmark)
set.seed(0) # is this all I need to do?

generate.train.inxs <- function(n,holdout) {
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
      dataset$y <- as.matrix(read.csv(files[2],header=F)) # perhaps as.vector?
      dataset$phi <- as.matrix(read.csv(files[3],header=F))
      
      # Generate train indices and subset to create the train and test sets.
      dataset$train.inxs <- generate.train.inxs(dataset$n,holdout=holdout)
      dataset$X.train <- dataset$X[train.inxs,]
      dataset$y.train <- dataset$y[train.inxs]
      dataset$phi.train <- dataset$phi[train.inxs,]
      dataset$X.test <- dataset$X[-train.inxs,]
      dataset$y.test <- dataset$y[-train.inxs]
      dataset$phi.test <- dataset$phi[-train.inxs,]
      
      return(dataset)
    }
  }
  print(paste("Unable to load data set '",nam,"'"))
  return(NULL)
}

evaluate.ff.rf <- function(data,ntree=500) {
  results <- list()
  results$st <- list() # system time of various actions
  results$pred <- list() # results of prediction (various stats)
  
  results$st$train <- system.time( # Time the training of the model
    results$model <- randomForest(data$X.train,
                                  data$y.train,
                                  xtest=data$X.test,
                                  ytest=data$y.test,
                                  ntree=ntree,
                                  prox=T))

  results$st$test <- system.time( # Time the testing of the model
    results$pred$yhat <- predict(results$model, data$X.test))
  
  return(results)
}

evaluate.ff.svm <- function(data) {
  results <- list()
  results$st <- list() # system time of various actions
  results$pred <- list() # results of prediction (various stats)
  
  results$st$train <- system.time( # Time the training of the model
    results$model <- svm(data$X.train,
                         data$y.train,
                         kernel='rbf'))
  
  results$st$test <- system.time( # Time the testing of the model
    results$pred$yhat <- predict(results$model, data$X.test)
  )
  
  return(results)
}

# Load data set.
data <- load.dataset('tb')

# Calculate and gather results.
results <- list()
# Fastfood algorithms
results$ff <- list()
results$ff$rf <- evaluate.ff.rf(data)
results$ff$svm <- evaluate.ff.svm(data)
results$ff$opls <- evaluate.ff.opls(data)
results$ff$adaboost <- evaluate.ff.adaboost(data)
# Standard algorithms
results$std <- list()
results$std$rf <- evaluate.std.rf(data)
results$std$svm <- evaluate.std.svm(data)
results$std$adaboost <- evaluate.std.adaboost(std)
results$std$kopls <- evaluate.ff.kopls(data)