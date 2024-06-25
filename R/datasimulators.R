
##' Capture-Recapture Data Simulator with Normal Mixture Distributions
##'
##' This function generates a capture-recapture dataset using normal
##' mixture distributions for the covariates.
##' 
##' @param N the true population size.
##' @param beta a matrix of beta coefficients.
##' @param normprobs a vector of probabilities. Should sum to 1.
##' @param normmeans a vector of means.
##' @param normsigma a covariance matrix.
##' @param missing logical. if FALSE, then all individuals will be returned, even those with
##' capture histories of all 0.
##' 
##' @return df of capture histories and covariates.
##' 
##' @examples
##' #Create the Data
##' mybeta = matrix(c(-2,-1,1,
##'                   -2,1,-1,
##'                   -2,-1,1,
##'                   -2,1,-1),nrow=4,byrow=TRUE)
##'                   
##' mynormprobs=c(0.3,0.4,0.3)
##' 
##' mynormmeans=matrix(c(2,2,
##'                      0,0,
##'                      -2,-2),ncol=2,byrow=TRUE)
##'                      
##' mynormsigma=matrix(c(.5, .45, .45,.5,
##'                       1,   0,   0, 1,
##'                      .5,-.35,-.35,.5),nrow=3,byrow=TRUE)
##' mydata = multdatasimulator(2000,mybeta,mynormprobs,mynormmeans,mynormsigma,missing=TRUE)
##' 
##' @export
multdatasimulator <- function(N,beta,
                              normprobs,normmeans,normsigma,missing=TRUE){
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
  if(missing==TRUE){
    myobserveddata = mydata[rowSums(mydata[,1:J])>0,]
  }else if (missing==FALSE){
    return(mydata)
  }else{
    stop("Not a valid option for missing.")
  }
  
  return(myobserveddata)
}


##' Capture-Recapture Data Simulator
##'
##' This function generates a capture-recapture dataset.
##' 
##' @param N the true population size.
##' @param beta a matrix of beta coefficients.
##' @param covdists a list of covariate distributions and their parameters,
##' options include normal, gamma, chisq, and tdist.
##' @param missing logical. if FALSE, then all individuals will be returned, even those with
##' capture histories of all 0.
##' @return df of capture histories and covariates.
##' @examples
##' N = 500
##' mybeta = matrix(c(-1,-1,-1,
##'                   -1,-1,-2,
##'                   -1,0,1,
##'                   -1,-1,1),nrow=4,byrow=TRUE)
##' mycovariates = list(c("gamma",1,1),
##'                     c("gamma",1,1))
##'                   
##' CRsimdata=datasimulator(N,mybeta,mycovariates)
##' CRsimdata
##' 
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
      newx=rnorm(N,mean=as.numeric(covdists[[h]][[2]]),sd=as.numeric(covdists[[h]][[3]]))
    }else if(covdists[[h]][1]=="gamma"){
      newx=rgamma(N,as.numeric(covdists[[h]][[2]]),as.numeric(covdists[[h]][[3]]))
    }else if(covdists[[h]][1]=="chisq"){
      newx=rchisq(N,df=as.numeric(covdists[[h]][[2]]))
    }else if(covdists[[h]][1]=="tdist"){
      newx=rt(N,df=as.numeric(covdists[[h]][[3]]),ncp=as.numeric(covdists[[h]][[2]]))
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
  if(missing==TRUE){
    myobserveddata = mydata[rowSums(mydata[,1:J])>0,]
  }else if (missing==FALSE){
    return(mydata)
  }else{
    stop("Not a valid option for missing.")
  }
  
  return(myobserveddata)
}
