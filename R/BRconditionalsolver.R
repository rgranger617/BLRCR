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

condlogitbayes2 <- function(Y,X,priorb,priorB,prior="yes",gradparam=.01,maxiter=1000){
  sigmoid <- function(x){
    1/(1+exp(-x))
  }
  M=nrow(Y)
  J=ncol(Y)
  Xmat=as.matrix(cbind(rep(1,M),X))
  H=ncol(Xmat)-1
  B=priorB
  b=matrix(rep(priorb,J),nrow=J)
  Binv=solve(priorB)
  
  #initialize
  beta=matrix(rep(1,(H+1)*J),nrow=J,byrow=TRUE)
  
  for(sim in 1:maxiter){
    #Compute Gradient
    lambda=sigmoid(Xmat%*%t(beta))
    probmis=apply(1-lambda,1,prod)
    likelihoodpart = t(t(Xmat)%*%as.matrix((lambda-Y))+t(Xmat)%*%(lambda*probmis/(1-probmis)))
    
    #prior component
    if(prior=="yes"){
      priorpart=matrix(rep(NA,J*(H+1)),nrow=J)
      for(j in 1:J){
        priorpart[j,] = Binv%*%(beta[j,]-b[j,])
      }
      gradient=likelihoodpart +priorpart
    }else if(prior=="no"){
      gradient=likelihoodpart
    }
    
    betanew = beta - gradparam*gradient
    
    if(sqrt(sum((betanew-beta)^2))<1e-6){
      break
    }
    beta=betanew
  }

  prob=(1-apply(1-sigmoid(Xmat %*% t(beta)),1,prod))
  N=sum(1/prob)

  
  mylist <- list("beta"=beta,"N"=N)
  return(mylist)
}