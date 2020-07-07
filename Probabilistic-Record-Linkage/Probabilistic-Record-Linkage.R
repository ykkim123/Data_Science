#############################
######### Probabilstic Record Linkage
#############################
# Load data
library(PPRL)
data = read.csv('testdata.csv', header=F, sep='\t')
write.table(data[1:5], file='testdata_new.csv', sep=',', row.names=F, col.names=c('id','gender','year','month','date'))

data = read.csv('testdata_new.csv', header=T, sep=',')
data1 = data[c(1,3,5,7),]
data2 = data[c(2,4,6,8),]
data.cp = merge(x=data1, y=data2, by=NULL)
data.cp
attach(data.cp)

# Estimate parameter by EM
## Calculate gamma
N = nrow(data.cp)
gamma = list()

for (i in 1:N){
  
  temp = c()
  
  if (gender.x[i]==gender.y[i]){
    g=1}
  else {
    g=0}
  
  if (year.x[i]==year.y[i]){
    y=1}
  else {
    y=0}
  
  if (month.x[i]==month.y[i]){
    m=1}
  else {
    m=0}
  
  if (date.x[i]==date.y[i]){
    d=1}
  else {
    d=0}
  
  gamma[[i]] = c(g,y,m,d)
}


## Initialization
K=4

m = rep(0.7,K)
u = rep(0.3,K)
p=0.5
eta = c(m,u,p)
eta.list = eta

## Implement EM
repeat{
  eta.prev = eta
  
  # E-step: update g
  g = c()
  for (i in 1:N){
    
    m.temp = 1
    u.temp = 1
    
    for (k in 1:K){
      m.temp = m.temp * (m[k]^(gamma[[i]][k])) *((1-m[k])^(1-gamma[[i]][k]))
      u.temp = u.temp * (u[k]^(gamma[[i]][k])) *((1-u[k])^(1-gamma[[i]][k]))}
    
    g.i = (p*m.temp)/(p*m.temp + (1-p)*u.temp)
    g = append(g, g.i)}
  
  # M-step
  ## Update m
  m = c()
  for (k in 1:K){
    num.temp = 0
    for (i in 1:N){
      num.i = g[i] * gamma[[i]][k]
      num.temp = num.temp + num.i}
    
    m.k = num.temp/sum(g)
    m = append(m, m.k)}
  
  ## Update u
  u = c()
  for (k in 1:K){
    num.temp = 0
    for (i in 1:N){
      num.i = (1-g[i]) * gamma[[i]][k]
      num.temp = num.temp + num.i}
    
    u.k = num.temp/sum(1-g)
    u = append(u, u.k)}
  
  ## Update p
  p = c()
  p.temp = sum(g)/N
  p = append(p, p.temp)
  
  
  # Convergence criteria
  eta = c(m,u,p)
  eta.list = rbind(eta.list,eta)
  diff = (eta.prev-eta)^2
  if (sum(diff)<1e-7)
    break}


# Calculate R
R = c()

for (i in 1:N){
  
  num = 1
  denom = 1
  
  for (k in 1:K){
    if (gamma[[i]][k]==1){
      p.num.temp = m[k]
      p.denom.temp = u[k]}
    else{
      p.num.temp = 1-m[k]
      p.denom.temp = 1-u[k]}
      
    num = num * p.num.temp
    denom = denom * p.denom.temp}
    
    R_i = num/denom
    R = append(R, R_i)}
R

matching = rep(0,N)
matching[c(12,16)] = c(1,1)

# Matching result
data.matching = cbind(data.cp, matching)
data.matching
