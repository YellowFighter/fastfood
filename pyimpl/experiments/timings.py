"""# Produce a plot of number of training samples vs (training+predicting) time
FF = F

if (FF) {
  print("Performing FF timings")
  X.train <- data$phi.train
  y.train <- data$y.train
  ntrain <- length(y.train)
  X.test <- data$phi.test
  y.test <- data$y.test
  ntest <- length(y.test)
} else {
  print("Performing std timings")
  X.train <- data$X.train
  y.train <- data$y.train
  ntrain <- length(y.train)
  X.test <- data$X.test
  y.test <- data$y.test
  ntest <- length(y.test)
}

y.train <- as.numeric(y.train)
y.test <- as.numeric(y.test)

times <- matrix(0,nrow=ntrain,ncol=10)
print(paste("ntrain:",ntrain))
for(ns in 2:ntrain) {
  inxs <- sample.int(ntrain,ns)
  x <- X.train[inxs,]
  y <- y.train[inxs]
  if(length(unique(y)) < 2) {
    next
  }
  print(paste("On iter ns=",ns))
  st <- system.time(
    model <- penalized.pls(x,y))
  times[ns,1:5] <- st
  
  inxs <- sample.int(ntest,min(ntest,ns))
  x <- X.test[inxs,]
  st <- system.time(new.penalized.pls(model,x))
  times[ns,6:10] <- st
}

colnames(times) <- c("train.user.self","train.sys.self","train.elapsed","train.user.child","train.sys.child",
                     "test.user.self","test.sys.self","test.elapsed","test.user.child","test.sys.child")
write.table(times,'data_sets/tb/times_std_penls.csv',col.names=T,row.names=F,quote=F,sep='\t')
"""
