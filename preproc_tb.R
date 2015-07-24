df.tb <- read.csv('data_sets/tb/TBdata_better_edited.csv')
df.tb <- na.omit(df.tb)
A.tb <- which(grepl('Active',df.tb[,1]))
a.tb <- which(grepl('active',df.tb[,1]))
c.tb <- which(grepl('control',df.tb[,1]))
df.tb <- df.tb[sort(c(A.tb,a.tb,c.tb)),]

y.tb <- df.tb[1]
y.tb <- as.matrix(y.tb)
#confounders
g.tb <- df.tb[2]
a.tb <- df.tb[3]
e.tb <- df.tb[4]
df.tb <- df.tb[,-c(1:4)]

X.tb <- as.matrix(df.tb)

y.tb[which(grepl('control',y.tb))] <- 0
y.tb[which(grepl('active',y.tb))] <- 1
y.tb[which(grepl('Active',y.tb))] <- 1
y.tb <- as.numeric(y.tb)
y.tb <- as.matrix(y.tb)
y.tb <- factor(y.tb[,1])
set.seed(0, kind=NULL, normal.kind=NULL)
samp <- sample(c(1:nrow(X.tb)))
y.tb <- y.tb[samp]
X.tb <- X.tb[samp,]

write.table(X.tb,'data_sets/tb/proc_x.csv',sep=',',quote=F,col.names=F,row.names=F)
write.table(y.tb,'data_sets/tb/proc_y.csv',sep=',',quote=F,col.names=F,row.names=F)