library(kopls)
library(AUC)
library(kernlab)
library(foreach)
library(doParallel)

setwd('~/dev/fastfood')

optimize.kopls <- function(K,y,noxRange,kfold=2,cluster.size=7) {
  cl <- makeCluster(cluster.size)
  registerDoParallel(cl)
  
  a <- generate.ytr(y)
  ytr <- a[1]
  n_ortho <- a[2]
  kcauc <- matrix(0,nrow=length(noxRange),ncol=kfold)
  test.inxs <- generate.test.inxs(nrow(X),kfold)
  
  print('optimizing nox...')
  if(length(noxRange) > 1) {
    kcauc <- foreach(i=1:length(noxRange),.packages=c('kernlab','AUC','kopls'),.combine=rbind) %dopar% {
      n <- noxRange[i]
      kcauc.values <- c()
      for(j in 1:kfold) {
        kcauc.values[j] <- 0
        test <- na.omit(test.inxs[[j]])
        model <- koplsModel(K[-test,-test],ytr[-test,],n_ortho,n,
                            preProcK='no',preProcY='mc')
        modelPred <- koplsPredict(K[test,-test],K[test,test],K[-test,-test],
                                  model,n,rescaleY=T)
        labels <- factor(ytr[test,2])
        kcauc.values[j] <- auc(roc(modelPred$Yhat[,2],labels))
        kcauc[i,] <- kcauc.values
      }
      return(kcauc.values)
    }
    b <- which.max(rowMeans(kcauc))
    nox <- noxRange[b[1]]
  } else {
    nox <- 0
  }
  print(paste('finished, nox=',nox))
  
  return(nox)
}

generate.test.inxs <- function(n,kfold) {
  t.inxs <- shuffle(n)
  size <- round(n/kfold)
  test.inxs <- list()
  for(i in 1:kfold) {
    start <- 1 + size*(i-1)
    end <- min(nrow(X),size+size*(i-1))
    test.inxs[[i]] <- t.inxs[start:end]
  }
  return(test.inxs)
}

generate.ytr <- function(y) {
  values <- sort(unique(y))
  ytr <- matrix(0,nrow=length(y),ncol=length(values))
  for(i in 1:length(values)) {
    ytr[y==values[i],i] <- 1
  }
  return(c(ytr,length(values)))
}

predict.kopls <- function(K,y,nox,test.inxs) {
  ytr <- generate.ytr(y)[1]
  
  model <- koplsModel(K[-test.inxs,-test.inxs],ytr[-test.inxs,],
                      length(values)-1,nox,preProcK='no',
                      preProcY='mc')
  modelPred <- koplsPredict(K[test.inxs,-test.inxs],K[test.inxs,test.inxs],
                            K[-test.inxs,-test.inxs],model,
                            rescaleY=T)
  roc.curve <- roc(modelPred$Yhat[,2],y[test.inxs])
  
  m <- modelPred$Yhat[,2]
  kcauc <- auc(roc.curve)
  labels <- y[test.inxs]
  r <- roc.curve
  return(list(roc.curve=roc.curve,
              labels=labels,
              predicted.labels=m,
              auc=kcauc))
}

# load X, Y
X <- read.csv('data_sets/tb/proc_x.csv',header=F)
X <- as.matrix(X)
Y <- read.csv('data_sets/tb/proc_y.csv',header=F)
Y <- as.matrix(Y)

# read in the FF phi matrix as computed in matlab
print('Make sure phi.csv exists properly, then press [enter]')
readline()
print('Continuing...')
phi <- read.csv('data_sets/tb/phi.csv',header=F)
phi <- as.matrix(phi)

# compute the FF kernel
print('Computing FF kernel matrix ...')
#K <- phi%*%t(phi)
st.ff.kcalc <- system.time(
  K <- crossprod(phi))
K <- as.kernelMatrix(K)

# hyperparams
kfold <- 5
sigma <- 10 # should match the value used in MATLAB to generate phi

# FF kopls
nox <- optimize.kopls(K,y,1:5)
test.inxs <- generate.test.inxs(nrow(X),kfold)
st.ff <- system.time(
  pred <- predict.kopls(K,y,nox,test.inxs))
st.ff <- st.ff[3]
print(paste("Time for ff kopls ",st.ff))

# std KOPLS
rbf <- rbfdot(sigma=sigma)
K <- kernelMatrix(rbf,X)
nox <- optimize.kopls(K,y,1:5)
test.inxs <- generate.test.inxs(nrow(X),kfold)
st.std <- system.time(
  pred <- predict.kopls(K,y,nox,test.inxs))
st.std <- st.std[3]

# display percent speed up for ff over std
pct.diff <- (st.std-st.ff)/mean(c(st.std,st.ff))
print(paste("Pct speed up for FF: ",pct.diff*100,"%"))