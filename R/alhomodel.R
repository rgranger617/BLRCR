##' A placeholder function using roxygen
##'
##' This function shows a standard text on the console. In a time-honoured
##' tradition, it defaults to displaying \emph{hello, world}.
##' @param txt An optional character variable, defaults to \sQuote{world}
##' @return Nothing is returned but as a side effect output is printed
##' @examples
##' hello2()
##' hello2("and goodbye")
##' @export

alhomodel <- function(Y1,Y2,X){
  sigmoid <- function(x){
    1/(1+exp(-x))
  }
  n1=sum(Y1)
  n2=sum(Y2)
  M=length(Y1)
  m=sum(rowSums(cbind(Y1,Y2))==2)
  
  u1 = as.integer(Y1==1 & Y2!=1)
  u2 = as.integer(Y1!=1 & Y2==1)
  m = as.integer(Y1==1 & Y2==1)
  
  X1=cbind(rep(1,M),X)
  X2=cbind(rep(1,M),X)
  Xmat=rbind(cbind(X1,matrix(rep(0,M*2),ncol=2)),cbind(matrix(rep(0,M*2),ncol=2),X2))
  Ymat=as.matrix(c(Y1,Y2),ncol=1)
  Tmat=t(Xmat)%*%Ymat
  
  #initialize
  a1=c(0,0)
  a2=c(0,0)
  theta=matrix(c(a1,a2),ncol=1)
  
  maxsim=100  
  for(sim in 1:maxsim){
    #Compute EYM
    Ktheta = exp(X1%*%a1)+exp(X2%*%a2)+exp(X1%*%a1+X2%*%a2)
    probu1 = exp(X1%*%a1)/Ktheta
    probu2 = exp(X2%*%a2)/Ktheta
    probm  = exp(X1%*%a1+X2%*%a2)/Ktheta
    EYM = matrix(c((probu1+probm),(probu2+probm)),ncol=1)
    
    #Compute Wmat
    p1=EYM[1:M]
    p2=EYM[(M+1):(2*M)]
    phi=p1+p2-p1*p2
    w1tilde=diag((p1/phi)-((p1^2)/(phi^2)))
    w2tilde=diag((p2/phi)-((p2^2)/(phi^2)))
    w3tilde=diag((p1*p2/phi)-((p1*p2)/(phi^2)))
    w4tilde=w3tilde
    Wmat=rbind(cbind(w1tilde,w3tilde),cbind(w4tilde,w2tilde))
    
    #Compute g
    g = Xmat%*%theta+qr.solve(Wmat)%*%(Ymat-EYM)
    
    #Compute update
    thetanew=qr.solve(t(Xmat)%*%Wmat%*%Xmat)%*%t(Xmat)%*%Wmat%*%g
    
    if(sqrt(sum((thetanew-theta)^2))<1e-6){
      break
    }
    theta=thetanew
    a1=theta[1:2]
    a2=theta[3:4]
    
  }
  finaltheta<-matrix(theta,ncol=2,byrow=TRUE)
  
  
  prob =1-apply(1-sigmoid(cbind(rep(1,length(X)),X) %*% t(finaltheta)),1,prod)
  N=sum(1/prob)
  
  mylist <- list("beta"=finaltheta,"N"=N)
  
  return(mylist)
}




