
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
multdatasimulator <- function(N,beta,
                              normprobs,normmeans,normsigma,missing="yes"){
  #safety checks
  
  #parameters
  H=ncol(normmeans)
  K=nrow(normmeans)
  J=nrow(beta)
  
  #Covariates
  x=matrix(rep(NA,N*(H+1)),nrow=N)
  x[,1]=rep(1,N)
  NK=rmultinom(1,N,normprobs)
  start=cumsum(c(1,NK))
  end=cumsum(NK)
  for(k in 1:K){
    sigma=matrix(normsigma[k,],nrow=H,byrow=TRUE)
    x[start[k]:end[k],2:(H+1)]=mvtnorm::rmvnorm(NK[k],normmeans[k,],sigma)
  }
  
  #shuffle
  x=x[sample.int(N,N,replace=FALSE),]
  
  colnames(x)<-rep("",H+1)
  
  #generate psi
  lambda = x%*%t(beta)
  
  #sigmoid
  sigmoid <- function(x){
    1/(1+exp(-x))
  }
  
  #Generate the data
  myprob = sigmoid(lambda)
  y <- matrix(rep(NA,J*N),ncol=J)
  for(j in 1:J){
    y[,j] = rbinom(n=N,size=1,prob=myprob[,j])
  }
  
  mydata = data.frame(
    "y"=y,
    "x"=x[,-1]
  )
  colnames(mydata) <- gsub("\\.","",colnames(mydata))
  colnames(mydata) <- gsub("V","",colnames(mydata))
  
  #Remove unobserved
  if(missing=="yes"){
    myobserveddata = mydata[rowSums(mydata[,1:J])>0,]
  }else if (missing=="no"){
    return(mydata)
  }else{
    stop("Note a valid option for missing.")
  }
  
  return(myobserveddata)
}


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
datasimulator <- function(N,beta,covdists,missing="yes"){
  #safety checks
  if(length(covdists)<ncol(beta)-1){
    warning("The beta matrix has too few coefficients specified.")
  }else if(length(covdists)>ncol(beta)-1){
    warning("The beta matrix has too many coefficients specified.")
  }
  
  #parameters
  H=length(covdists)
  J=nrow(beta)
  
  #Covariates
  x=rep(1,N)
  for(h in 1:H){
    if(covdists[[h]][1]=="normal"){
      newx=rnorm(N,mean=covdists[[h]][[2]],sd=covdists[[h]][[3]])
    }else if(covdists[[h]][1]=="gamma"){
      newx=rgamma(N,as.numeric(covdists[[h]][[2]]),as.numeric(covdists[[h]][[3]]))
    }else if(covdists[[h]][1]=="chisq"){
      newx=rchisq(N,df=as.numeric(covdists[[h]][[2]]))
    }else if(covdists[[h]][1]=="tdist"){
      newx=rt(N,df=as.numeric(covdists[[h]][[3]]),ncp=as.numeric(covdists[[h]][[2]]))
    }else if(covdists[[h]][[1]]=="normalmixture"){
      nummix=nrow(covdists[[h]][[2]])
      probs=covdists[[h]][[2]][,1]
      newx=rep(NA,N)
      for(i in 1:N){
        mixc=sample(1:3,1,prob=probs)
        newx[i]=rnorm(1,mean=covdists[[h]][[2]][mixc,2],sd=covdists[[h]][[2]][mixc,3])
      }
    }
    x=cbind(x,newx)
  }
  colnames(x)<-rep("",H+1)
  
  #generate psi
  lambda = x%*%t(beta)
  
  #sigmoid
  sigmoid <- function(x){
    1/(1+exp(-x))
  }
  
  #Generate the data
  myprob = sigmoid(lambda)
  y <- matrix(rep(NA,J*N),ncol=J)
  for(j in 1:J){
    y[,j] = rbinom(n=N,size=1,prob=myprob[,j])
  }
  
  mydata = data.frame(
    "y"=y,
    "x"=x[,-1]
  )
  colnames(mydata) <- gsub("\\.","",colnames(mydata))
  colnames(mydata) <- gsub("V","",colnames(mydata))
  
  #Remove unobserved
  if(missing=="yes"){
    myobserveddata = mydata[rowSums(mydata[,1:J])>0,]
  }else if (missing=="no"){
    return(mydata)
  }else{
    stop("Note a valid option for missing.")
  }
  
  return(myobserveddata)
}
