source('fastfood.R')

X <- read.csv('digits-2.csv',header=F)
n <- dim(X)[1]
d <- dim(X)[2]
Y <- read.csv('digits-2-y.csv',header=F)

params <- fastfood.params(n,d)
