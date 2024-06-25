// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-


#include "RcppArmadillo.h"
#include <string> 
using namespace Rcpp;




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
//' @param beta a (H+1) vector of prior means for the coefficients
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
//' 
//' @export 
// [[Rcpp::export]]
SEXP bootstrapRCR(arma::mat y, arma::mat x, arma::mat beta){
  
  //Compute Parameters of the data
  unsigned int n = y.n_rows;
  unsigned int J = y.n_cols;
  unsigned int H = x.n_cols;
  
  //Add column of 1s for intercept
  arma::mat allOne(n, 1, arma::fill::ones);
  x.insert_cols(0, allOne);

  //Compute Bootstrap Probs
  arma::mat bootstrapprobs = 1/(1+exp(-x*beta.t()));
  
  //Compute NiHAT
  arma::vec NiHATreal = 1/(1-prod(1-bootstrapprobs,1));
  
  //Compute di
  arma::vec di = NiHATreal - floor(NiHATreal);
  arma::uvec addone(n);
  for(unsigned int i=0; i<n; i++){
    addone(i) = Rcpp::rbinom(1,1,di(i))(0);
  }
  
  //Update NiHat
  arma::vec NiHAT=floor(NiHATreal) + addone;
  int N = sum(NiHAT);
  
  //Create Bootstrapped Data
  arma::mat Yboot(N,J);
  arma::mat Xboot(N,H);
  int myindex = 0;
  for(unsigned int i=0; i<n; i++){
    for(int nh=0; nh<NiHAT(i); nh++){
      for(unsigned int j=0; j<J; j++){
        Yboot(myindex,j) = Rcpp::rbinom(1,1,bootstrapprobs(i,j))(0);
        Xboot.row(myindex) = x.submat(i,1,i,H);
      }
      myindex++;
    }
  }
  
  //Remove unobserved entries
  arma::uvec obsindex = find(sum(Yboot,1));
  arma::mat Ybootobs = Yboot.rows(obsindex);
  arma::mat Xbootobs = Xboot.rows(obsindex);
  
  //Combine into one matrix
  NumericMatrix mybootstrapdata = as<NumericMatrix>(wrap(join_rows(Ybootobs,Xbootobs)));
  
  //Name the Y and X variables
  CharacterVector mynames(J+H);
  for(unsigned int j=0; j<J; j++){
    std::string mynumber = std::to_string(j+1);
    mynames(j) = "y"+mynumber;
  }
  for(unsigned int h=0; h<H; h++){
    std::string mynumber = std::to_string(h+1);
    mynames(J+h) = "x"+mynumber;
  }
  
  Rcpp::colnames(mybootstrapdata) = mynames;

  ////////////////////
  //Return Variables//
  ///////////////////
  return Rcpp::wrap(mybootstrapdata);
}


