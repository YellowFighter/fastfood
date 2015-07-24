fastfood.params <- function(n, d) {
  d0 <- d
  n0 <- n
  l <- as.integer(ceiling(log2(d)))
  d <- 2^l
  k <- as.integer(ceiling(n/d))
  n <- d*k
  print(paste('d0=',d0,', d=',d))
  print(paste('n0=',n0,', n=',n))
  
  B <- matrix(nrow=k,ncol=d)
  G <- matrix(nrow=k,ncol=d)
  PI <- matrix(nrow=k,ncol=d)
  S <- matrix(nrow=k,ncol=d)
  for(ii in 1:k) {
    B[ii,] <- sample(c(-1,1),size=d,replace=T)
    G[ii,] <- rnorm(d)
    PI[ii,] <- sample.int(d,d)
    
    p1 <- runif(d)
    p2 <- d / 2
    Tmp <- sqrt(2*gammaincinv(p1, p2))
    s.i <- Tmp*(1/norm(G,'F'))
    S[ii,] <- s.i
  }
  
  params <- data.frame(B=B,G=G,PI=PI,S=S)
  return(params)
}

fastfood.forkernel <- function(X,para,sgm) {
  c(d0, m) <- dim(X)
  l <- as.integer(ceiling(log2(d0)))
  d <- 2^l
  if (d == d0) {
    XX <- X
  } else {
    XX <- matrix(nrow=d,ncol=m)
    XX[,] <- 0
    XX[1:d0,] <- X
  }
  k <- nrow(para$B)
  n <- d*k
  tht <- matrix(nrow=n,ncol=m)
  tht[,] <- 0
  for (ii in 1:k) {
    B <- diag(para$B[ii,],nrow=d,ncol=d)
    G <- diag(para$G[ii,],nrow=d,ncol=d)
    PI <- diag(para$PI[ii,],nrow=d,ncol=d)
    xx <- B%*%XX
    Tmp <- fwht(xx)
    Tmp <- Tmp[PI,]
    Tmp <- (d*G)%*%Tmp
    Tmp <- fwht(Tmp)
    idx1 <- (ii-1)*d
    idx2 <- ii*d
    tht[idx1:idx2,] <- Tmp
  }
  tht <- t((S*sqrt(d))*t(tht))
  Tmp <- tht/sgm
  #line 218
}