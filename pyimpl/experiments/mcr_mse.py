"""FF = F

if (FF) {
  print("Performing FF performance")
  X.train <- data$phi.train
  y.train <- data$y.train
  ntrain <- length(y.train)
  X.test <- data$phi.test
  y.test <- data$y.test
  ntest <- length(y.test)
} else {
  print("Performing std performance")
  X.train <- data$X.train
  y.train <- data$y.train
  ntrain <- length(y.train)
  X.test <- data$X.test
  y.test <- data$y.test
  ntest <- length(y.test)
}

y.train <- as.numeric(y.train)
y.test <- as.numeric(y.test)

model <- penalized.pls(X.train,y.train)
pred <- new.penalized.pls(model,X.test)
print(mean((y.test-pred)^2))"""
