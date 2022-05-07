
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

BLRCRsolver2 <- function(y,x,priorb,priorB,
                   priornu0,priorLAMBDA0,priorkappa0,priorMU0,
                   aalpha,balpha,Kstar,
                   rhomethod=2,samples=100){
  #test if things make sense
  
  #put data in matrix
  yobs = as.matrix(y)
  xobsNO0 = as.matrix(x)
  
  #known parameters
  maxasam=2000 #sets the maximum number of tries to find a sample in missing
  n=nrow(yobs)
  H=ncol(xobsNO0)
  J=ncol(yobs)
  Binverse = solve(priorB)
  Bb =  Binverse%*%priorb
  priorLAMBDA0inverse = solve(priorLAMBDA0)
  
  
  #add column of 1s for intercept
  xobs = cbind(rep(1,n),xobsNO0)
  
  #sigmoid
  sigmoid <- function(x){
    1/(1+exp(-x))
  }
  
  #Storage
  savebeta = array(rep(NA,J*(H+1)*samples),dim=c(J,H+1,samples))
  saveN = rep(NA,samples)
  savePIK = matrix(rep(NA,Kstar*samples),nrow=samples)
  saveNK = matrix(rep(NA,Kstar*samples),nrow=samples)
  saveMUK = array(rep(NA,Kstar*H*samples),dim=c(Kstar,H,samples))
  saveSIGMA = array(rep(NA,H*H*Kstar*samples),dim=c(H,H,Kstar,samples))
  saveX = matrix(rep(NA,H*samples),nrow=samples)
  saveXmis = matrix(rep(NA,H*samples),nrow=samples)
  
  #initializations
  beta = matrix(rep(NA,(H+1)*J),ncol=H+1)
  for(j in 1:J){
    beta[j,] = coef(glm(yobs[,j]~xobs-1,family="binomial"))
  }
  savebeta[,,1]=beta
  n0 = 0
  N = n+n0
  saveN[1] = N
  x=xobs
  y=yobs
  
  ###################################
  #Initialize Covariate Distribution#
  ###################################
  
  #Create Space for Parameter Matrices
  Zlab = sample.int(Kstar,N,replace=TRUE,prob=cumsum(1:Kstar)[Kstar:1])
  NK = rep(NA,Kstar)
  XbarK = matrix(rep(NA,Kstar*H),nrow=Kstar)
  SK = array(rep(NA,H*H*Kstar),dim=c(H,H,Kstar))
  LAMBDANK=array(rep(NA,H*H*Kstar),dim=c(H,H,Kstar)) 
  LAMBDANKinv=array(rep(NA,H*H*Kstar),dim=c(H,H,Kstar))
  MUNK = matrix(rep(NA,Kstar*H),nrow=Kstar)
  SIGMAK = array(rep(NA,H*H*Kstar),dim=c(H,H,Kstar))
  MUK = matrix(rep(NA,Kstar*H),nrow=Kstar)
  
  #Initialize Alpha
  alpha=aalpha/balpha
  
  #Initialize PIK
  if(Kstar==1){
    Vk=1
    PIK=1
  }else{
    Vk = rep(NA,Kstar); Vk[Kstar]=1
    for(i in 1:(Kstar-1)){
      Vk[i] = rbeta(1,1,alpha)
    }
    PIK=Vk*c(1,cumprod((1-Vk))[1:(Kstar-1)])
  }
  savePIK[1,]=PIK
  
  #Initialize Z
  Zlab=sample.int(Kstar,size=N,replace=TRUE,prob=PIK)
  
  #initialize SIGMAK and MUK
  for(i in 1:Kstar){
    NK[i] <- sum(Zlab==i)
  }
  saveNK[1,]=NK
  NUNK = priornu0 + NK
  kappaNK = priorkappa0 + NK
  for(i in 1:Kstar){
    if(NK[i]==0){
      SIGMAK[,,i]=MCMCpack::riwish(max(H,NUNK[i]),priorLAMBDA0)
      MUK[i,]=mvtnorm::rmvnorm(1,priorMU0,SIGMAK[,,i]/kappaNK[i])
    }else if(NK[i]==1){
      if(sum(Zlab==i)==0){
        XbarK[i,1:H]=rep(1,H)
      }else{
        XbarK[i,1:H] <- x[Zlab==i,2:(H+1)]
      }
      SK[,,i] = matrix(rep(0,H^2),nrow=H)
      priorpart1=(priorkappa0/(priorkappa0+1))
      priorpart2=outer(XbarK[i,1:H]-priorMU0,XbarK[i,1:H]-priorMU0)
      LAMBDANK[,,i]=priorLAMBDA0+priorpart1*priorpart2
      LAMBDANKinv[,,i]=solve(LAMBDANK[,,i])
      MUNKpart1=priorkappa0/(priorkappa0+NK[i])*priorMU0
      MUNKpart2=NK[i]/(priorkappa0+NK[i])*XbarK[i,1:H]
      MUNK[i,]=MUNKpart1+MUNKpart2
      SIGMAK[,,i]=MCMCpack::riwish(max(H,NUNK[i]),LAMBDANK[,,i])
      MUK[i,]=mvtnorm::rmvnorm(1,MUNK[i,],SIGMAK[,,i]/kappaNK[i])
    }else{
      Xi = x[Zlab==i,2:(H+1)]
      XbarK[i,1:H] <- colMeans(Xi)
      SK[,,i] = t(Xi-XbarK[i,1:H])%*%(Xi-XbarK[i,1:H])
      priorpart1=(priorkappa0*NK[i]/(priorkappa0+NK[i]))
      priorpart2=outer(XbarK[i,1:H]-priorMU0,XbarK[i,1:H]-priorMU0)
      LAMBDANK[,,i]=priorLAMBDA0+SK[,,i]+priorpart1*priorpart2
      LAMBDANKinv[,,i]=solve(LAMBDANK[,,i])
      MUNKpart1=priorkappa0/(priorkappa0+NK[i])*priorMU0
      MUNKpart2=NK[i]/(priorkappa0+NK[i])*XbarK[i,1:H]
      MUNK[i,]=MUNKpart1+MUNKpart2
      SIGMAK[,,i]=MCMCpack::riwish(NUNK[i],LAMBDANK[,,i])
      MUK[i,]=mvtnorm::rmvnorm(1,MUNK[i,],SIGMAK[,,i]/kappaNK[i])
    }
  }
  saveMUK[,,1]=MUK
  saveSIGMA[,,,1]=SIGMAK 
  
  specialZ=sample.int(Kstar,1,prob=PIK)
  saveX[1,]=mvtnorm::rmvnorm(1,
                    mean=MUK[specialZ,],
                    sigma=SIGMAK[,,specialZ])
  saveXmis[1,]=mvtnorm::rmvnorm(1,
                       mean=MUK[specialZ,],
                       sigma=SIGMAK[,,specialZ])
  
  
  #############################
  ########Gibbs Sampling#######
  #############################
  for(sample in 2:samples){
    
    #Sample beta
    psi = x%*%t(beta)
    for(j in 1:J){
      w = BayesLogit::rpg(N,1,psi[,j])
      Vw = solve(t(x)%*%diag(w)%*%x + Binverse)
      kappa = y-0.5
      Mw = Vw%*%(t(x)%*%kappa[,j] + Bb)
      beta[j,] = mvtnorm::rmvnorm(1,mean=Mw,sigma = Vw)
    }
    savebeta[,,sample]=beta
    
    #sample n0 and N
    if(rhomethod==1){
      misprob <- apply(1-sigmoid(beta%*%t(x)),2,prod)
      rho=mean(misprob)
    }else if(rhomethod==2){
      simxsamplenum=200*H*Kstar #this is arbitrary, but bigger is better
      simx=matrix(c(rep(1,simxsamplenum),rep(NA,simxsamplenum*H)),ncol=H+1)
      simNK=rmultinom(1,simxsamplenum,PIK)
      simstart=c(1,cumsum(simNK)+1)
      simstop=cumsum(simNK)
      for(k in 1:Kstar){
        if(simNK[k]==0){
          next
        }
        simx[simstart[k]:simstop[k],-1]=mvtnorm::rmvnorm(simNK[k],MUK[k,],SIGMAK[,,k])
      }
      misprob <- apply(1-sigmoid(beta%*%t(simx)),2,prod)
      rho=mean(misprob)
    }
    n0 = rnbinom(1,n,1-rho)
    N = n+n0
    saveN[sample] = N
    
    #sample xmis
    xmis=matrix(rep(NA,(H+1)*n0),ncol=H+1)
    if(n0>0){
      for(i in 1:n0){
        for(asam in 1:maxasam){
          propZ=sample.int(Kstar,1,prob=PIK)
          propx=c(1,mvtnorm::rmvnorm(1,MUK[propZ,],SIGMAK[,,propZ]))
          lambda=sigmoid(propx%*%t(beta))
          acceptanceprob = prod(1-lambda)
          accept = rbinom(1,1,acceptanceprob)
          if(accept==1){
            xmis[i,]=propx
            break
          }
          if(asam==maxasam){
            stop('Error Max iterations hit without accepting a missing value')
          }
        }
      }
    }
    #update x and y
    x = rbind(xobs,xmis)
    y = rbind(yobs,matrix(rep(0,J*n0),ncol=J))
    
    #############################
    ## Sample Covariate Dist ####
    #############################
    #Sample Z
    Zlab=rep(NA,N)
    Zmatrix=matrix(rep(NA,Kstar*N),nrow=N)
    for(k in 1:Kstar){
      Zmatrix[,k]=PIK[k]*dmvnorm(x[,2:(H+1)],MUK[k,],SIGMAK[,,k])
    }
    Zprobs=Zmatrix/rowSums(Zmatrix)
    for(i in 1:N){
      Zlab[i]=sample.int(Kstar,1,prob=Zprobs[i,])
    }
    
    #Sample SIGMAK and MUK
    for(i in 1:Kstar){
      NK[i] <- sum(Zlab==i)
    }
    saveNK[sample,]=NK
    NUNK = priornu0 + NK
    kappaNK = priorkappa0 + NK
    for(i in 1:Kstar){
      if(NK[i]==0){
        SIGMAK[,,i]=MCMCpack::riwish(max(H,NUNK[i]),priorLAMBDA0)
        MUK[i,]=mvtnorm::rmvnorm(1,priorMU0,SIGMAK[,,i]/kappaNK[i])
      }else if(NK[i]==1){
        if(sum(Zlab==i)==0){
          XbarK[i,1:H]=rep(1,H)
        }else{
          XbarK[i,1:H] <- x[Zlab==i,2:(H+1)]
        }
        SK[,,i] = matrix(rep(0,H^2),nrow=H)
        priorpart1=(priorkappa0/(priorkappa0+1))
        priorpart2=outer(XbarK[i,1:H]-priorMU0,XbarK[i,1:H]-priorMU0)
        LAMBDANK[,,i]=priorLAMBDA0+priorpart1*priorpart2
        LAMBDANKinv[,,i]=solve(LAMBDANK[,,i])
        MUNKpart1=priorkappa0/(priorkappa0+NK[i])*priorMU0
        MUNKpart2=NK[i]/(priorkappa0+NK[i])*XbarK[i,1:H]
        MUNK[i,]=MUNKpart1+MUNKpart2
        SIGMAK[,,i]=MCMCpack::riwish(max(H,NUNK[i]),LAMBDANK[,,i])
        MUK[i,]=mvtnorm::rmvnorm(1,MUNK[i,],SIGMAK[,,i]/kappaNK[i])
      }else{
        Xi = x[Zlab==i,2:(H+1)]
        XbarK[i,1:H] <- colMeans(Xi)
        SK[,,i] = t(Xi-XbarK[i,1:H])%*%(Xi-XbarK[i,1:H])
        priorpart1=(priorkappa0*NK[i]/(priorkappa0+NK[i]))
        priorpart2=outer(XbarK[i,1:H]-priorMU0,XbarK[i,1:H]-priorMU0)
        LAMBDANK[,,i]=priorLAMBDA0+SK[,,i]+priorpart1*priorpart2
        LAMBDANKinv[,,i]=solve(LAMBDANK[,,i])
        MUNKpart1=priorkappa0/(priorkappa0+NK[i])*priorMU0
        MUNKpart2=NK[i]/(priorkappa0+NK[i])*XbarK[i,1:H]
        MUNK[i,]=MUNKpart1+MUNKpart2
        SIGMAK[,,i]=MCMCpack::riwish(NUNK[i],LAMBDANK[,,i])
        MUK[i,]=mvtnorm::rmvnorm(1,MUNK[i,],SIGMAK[,,i]/kappaNK[i])
      }
    }
    saveMUK[,,sample]=MUK
    saveSIGMA[,,,sample]=SIGMAK
    
    ############
    #Sample PIK#
    ############
    #This uses gammas to sample betas for computational purposes
    # l_acc_prod=0.0
    # for(i in 1:(Kstar-1)){
    #   a=1+NK[i]
    #   b=alpha+sum(NK[(i+1):Kstar])
    #   if(a<0.5){
    #     lgamma1=(log(runif(1))/a) + log(rgamma(1,1+a,1))
    #   }else{
    #     lgamma1=log(rgamma(1,a,1))
    #   }
    #   if(b<0.5){
    #     lgamma2=(log(runif(1))/b) + log(rgamma(1,1+b,1))
    #   }else{
    #     lgamma2=log(rgamma(1,b,1))
    #   }
    #   if(lgamma1>lgamma2){
    #     lsumgamma=log(exp(lgamma2-lgamma1)+1)-exp(lgamma2-lgamma1)+lgamma1
    #   }else{
    #     lsumgamma=log(exp(lgamma1-lgamma2)+1)-exp(lgamma1-lgamma2)+lgamma2
    #   }
    #   if(lgamma1<10|lgamma2<10){
    #     lsumgamma=log(exp(lgamma2)+exp(lgamma1))
    #   }
    #   logPIK=lgamma1-lsumgamma+l_acc_prod
    #   PIK[i]=exp(logPIK)
    #   l_acc_prod = l_acc_prod + lgamma2-lsumgamma
    # }
    # PIK[Kstar]=1-sum(PIK[1:(Kstar-1)])
    
    #This samples in the standard way, but may have computational issues
    Vk[Kstar]=1
    
    if(Kstar==1){
      PIK=1
    }else{
      for(i in 1:(Kstar-1)){
        Vk[i] = rbeta(1,1+NK[i],alpha+sum(NK[(i+1):Kstar]))
      }
      PIK=Vk*c(1,cumprod((1-Vk))[1:(Kstar-1)])
    }
    savePIK[sample,]=PIK
    
    
    #Sample Alpha
    alpha=rgamma(1,aalpha+Kstar-1,balpha-log(PIK[Kstar]))
    
    #Take a Sample from mixture distribution
    specialZ = sample.int(Kstar,1,prob=PIK)
    saveX[sample,]=mvtnorm::rmvnorm(1,
                           mean=MUK[specialZ,],
                           sigma=SIGMAK[,,specialZ])
    saveXmis[sample,]=xmis[sample.int(n0,1),2:(H+1)]
    
  }
  mylist <- list("beta"=savebeta,"N"=saveN,"PIK"=savePIK,"NK"=saveNK,
                 "MUK"=saveMUK,"SIGMA"=saveSIGMA,"X"=saveX,"Xmis"=saveXmis)
  return(mylist)
}