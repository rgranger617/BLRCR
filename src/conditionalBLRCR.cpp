// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]



//' Bayesian Logistic Regression Capture-Recapture using Conditional Likelihood
//' 
//' This function computes the Bayesian Logistic Regression Capture-Recapture estimate for N
//' using the conditional maximum likelihood (instead of the full likelihood).
//' 
//' @param y a nxj matrix of capture patterns.
//' @param x a nxh matrix of covariates.
//' @param priorb a (h+1) vector of prior means for the coefficients
//' @param priorB a (h+1)x(h+1) covariance matrix for the coefficients
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
Rcpp::List condBLRCRsolver(arma::mat y, arma::mat x, 
                           arma::vec priorb, arma::mat priorB,
                           double gradparam=0.01, int prior=1,
                           double tol=1e-6,int maxiter=1000){
  //Compute Parameters of the data
  int M = y.n_rows;
  int J = y.n_cols;
  int H = x.n_cols;
  
  //Add a column of 1s to X
  arma::mat allOne(M, 1, arma::fill::ones);
  x.insert_cols(0, allOne);
  
  //Compute a few necessary matrices
  arma::mat gradparammat(J,H+1);
  gradparammat.fill(gradparam);
  arma::mat B = priorB;
  arma::mat b = priorb;
  arma::mat Binv = inv(B);
  
  //Initialize beta
  arma::mat beta(J,H+1);
  beta.fill(1.0);
  
  //Perform Gradient Ascent
  int converge = 0;
  for(int iter=0; iter<maxiter; iter++){
    
    //Compute lambda
    arma::mat lambda = 1.0/(1.0+exp(-x*beta.t()));
    
    //Compute Probability of missing
    arma::mat probmis(M,1,arma::fill::none);
    for(int i=0; i<M; i++){
      probmis(i) = prod(1.0-lambda.row(i));
    }
    
    //Compute Likelihood Part
    arma::mat probmismat = probmis / (1.0-probmis); //elementwise mult
    arma::mat likpart = (x.t()*(lambda-y)+x.t()*(lambda.each_col() % probmismat)).t();
    
    //Compute Prior Part
    arma::mat priorpart(J,H+1,arma::fill::none);
    for(int j=0; j<J; j++){
      priorpart.row(j)=(beta.row(j)-b.t())*Binv.t();
    }
    
    //Compute gradient
    arma::mat gradient = likpart;
    if(prior==1){
      gradient = likpart+priorpart;
    }
    arma::mat betanew = beta - gradient % gradparammat;
    
    //Check Convergence
    //Rcpp::Rcout << "The value is " << sqrt(accu(pow(betanew-beta,2))) << std::endl;
    if(sqrt(accu(pow(betanew-beta,2)))<tol){
      converge = 1;
      break;
    }
    
    //Update Beta
    beta = betanew;
  }
  
  //Compute N
  arma::mat lambda = 1.0/(1.0+exp(-x*beta.t()));
  arma::mat probmis(M,1,arma::fill::randu);
  for(int i=0; i<M; i++){
    probmis(i) = prod(1.0-lambda.row(i));
  }
  double N = accu(1.0/(1-probmis));


  return Rcpp::List::create(Rcpp::Named("N") = N,
                            Rcpp::Named("betas") = beta,
                            Rcpp::Named("converge") = converge);
}


