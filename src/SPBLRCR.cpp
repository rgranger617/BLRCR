// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]


//' Semi-Parametric Bayesian Logistic Regression Capture-Recapture
//' 
//' This function computes MCMC samples of N using the the Bayesian Logistic
//' Regression Capture-Recapture model.
//' 
//' @param y a nxJ matrix of capture patterns.
//' @param x a nxH matrix of covariates.
//' @param priorb a (H+1) vector of prior means for the coefficients
//' @param priorB a (H+1)x(H+1) covariance matrix for the coefficients
//' @param priorMU0 a Hx1 vector of prior covariate means. Default=(0,...,0).
//' @param priorlambda0 a HxH prior covariance matrix for the covariates
//' @param priorkappa0 a positve real value
//' @param priornu0 a positive real value
//' @param aalpha a positive real value
//' @param balpha a positive real value
//' @param Kstar an integer of maximum number of mixture classes
//' @param samples the number of MCMC samples to draw from the posterior
//' 
//' @return List with components "N" and "beta". 
//' \describe{
//' \item{"N"}{is a single estimate of the population size.}
//' \item{"beta"}{is a matrix of beta estimates.}
//' }
//' 
//' @examples
//' #Generate the Data
//' mybeta = matrix(c(-2,-1,1,
//'                   -2,1,-1,
//'                   -2,-1,1,
//'                   -2,1,-1),nrow=4,byrow=TRUE)
//'                   
//' mynormprobs=c(0.3,0.4,0.3)
//' 
//' mynormmeans=matrix(c(2,2,
//'                      0,0,
//'                      -2,-2),ncol=2,byrow=TRUE)
//'                      
//' mynormsigma=matrix(c(.5,.45,.45,.5,
//'                       1,0,0,1,
//'                       .5,-.35,-.35,.5),nrow=3,byrow=TRUE)
//' 
//' mydata = multdatasimulator(2000,mybeta,mynormprobs,mynormmeans,mynormsigma,
//'                            missing="yes")
//'                            
//' #Run the Algorithm
//' mypriorb=rep(0,3) #prior on intercept and both beta means set to 0
//' mypriorB=diag(3) #identity matrix, size 3     
//' 
//' condBLRCRsolver(as.matrix(mydata[,1:4]),
//'                 as.matrix(mydata[,5:6]),mypriorb,mypriorB)                      
//' 
//' @export 
// [[Rcpp::export]]
Rcpp::List SPBLRCRsolver(arma::mat y, arma::mat x, 
                           arma::vec priorb, arma::mat priorB,
                           arma::vec priorMU0, arma::mat priorlambda0,
                           double priorkappa0=1, double priornu0=1,
                           double aalpha=0.25, double balpha=0.25,
                           int Kstar=10, int samples=1000){
  
  //Compute Parameters of the data
  int n = y.n_rows;
  int J = y.n_cols;
  int H = x.n_cols;
  
  //Compute Useful constants
  arma::mat Binv = inv_sympd(priorB);
  arma::mat Bb = Binv*priorb;
  arma::mat priorlambda0inv = inv_sympd(priorlambda0);
  
  //Add column of 1s for intercept
  arma::mat allOne(n, 1, arma::fill::ones);
  x.insert_cols(0, allOne);
  
  //Initializations
  arma::mat beta(J,H+1);
  beta.fill(1.0);  //initialize beta to all 1s
  int n0 = 0;
  int N = n+n0;
  arma::mat xobs=x;
  arma::mat yobs=y;
  
  arma::vec zinit=arma::normalise(sort(arma::cumsum(arma::regspace<arma::vec>(1,Kstar)),"descend"),1);
  arma::uvec Zlab(N);
  
  
  
  
  //Return Variables
  return Rcpp::List::create(Rcpp::Named("n") = n,
                            Rcpp::Named("zinit") = zinit,
                            Rcpp::Named("Zlab") = Zlab);
}



