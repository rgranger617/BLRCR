% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{BLRCRsolver}
\alias{BLRCRsolver}
\title{Semi-Parametric Bayesian Logistic Regression Capture-Recapture}
\usage{
BLRCRsolver(
  formula,
  df,
  covmethod = "bootstrap",
  coefprior = "normal",
  bprior = 0L,
  Bprior = 1,
  Homega = 1L,
  Kstar = 1L,
  samples = 1000L,
  burnin = 0L
)
}
\arguments{
\item{formula}{an object of class "formula".}

\item{df}{Data containing capture histories and individual covariates.}

\item{covmethod}{specify a method for simulating the covariates.
Options include "bootstrap", "mixture", and "empirical".}

\item{coefprior}{specify a prior for the logistic regression coefficients.
Options include "normal" and "horseshoe".}

\item{bprior}{vector of hyperparameters for the mean of the coefficient prior. If a single 
value is used, it will apply this value for all coefficients. If "normalhier" is selected, it will use
these values as the coefficient prior on b.}

\item{Bprior}{value to assign the diagonals of the covariance matrix B for all variables.}

\item{Homega}{the number of latent intercepts}

\item{Kstar}{an integer of maximum number of mixture classes. This only needs to be specified
if the covmethod is "mixture".}

\item{samples}{the number of MCMC samples to draw from the posterior.}

\item{burnin}{the number of MCMC samples to "burn-in."}
}
\value{
List with CR results.
}
\description{
This function computes MCMC samples of N using the the Bayesian Logistic
Regression Capture-Recapture model.
}
\examples{
#Create the Data
mybeta = matrix(c(-2,-1,1,
                  -2,1,-1,
                  -2,-1,1,
                  -2,1,-1),nrow=4,byrow=TRUE)
                  
mynormprobs=c(0.3,0.4,0.3)

mynormmeans=matrix(c(2,2,
                     0,0,
                     -2,-2),ncol=2,byrow=TRUE)
                     
mynormsigma=matrix(c(.5,.45,.45,.5,
                      1,0,0,1,
                      .5,-.35,-.35,.5),nrow=3,byrow=TRUE)

myCRdata = multdatasimulator(2000,mybeta,mynormprobs,mynormmeans,mynormsigma,
                           missing=TRUE)
                           
#Run the Algorithm
myformula = cbind(y1,y2,y3,y4)~x1+x2

CRresults = BLRCRsolver(myformula,myCRdata,covmethod="mixture",coefprior="normal",
                        Homega=10,Kstar=10, samples=1000)                     

}
