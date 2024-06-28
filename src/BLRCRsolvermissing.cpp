// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "RcppArmadillo.h"
#include <Rcpp/Benchmark/Timer.h>

using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]] 


//' Bayesian Logistic Regression Capture-Recapture with Missing Covariate Values
//' 
//' This function computes MCMC samples of N using the the Bayesian Logistic
//' Regression Capture-Recapture model but can also handle missing covariate values.
//' This method only allows for "bootstrap" and "empirical" covariate methods.
//' 
//' @param formula an object of class "formula".
//' @param df data containing capture histories and individual covariates.
//' 
//' @param covsupport specify a method for generating the support of the bootstrap distribution.  
//' Options include "all" and "unique".
//' @param covmethod specify a method for the distribution of the covariates.  
//' Options include "bootstrap" and "empirical".
//' @param coefprior specify a prior for the logistic regression coefficients.
//' Options include "normal", "normalhier", "normalhiervaronly", and "lasso".
//' 
//' @param bprior vector of hyperparameters for the mean of the coefficient prior. If a single 
//' value is used, it will apply this value for all coefficients. If "normalhier" is selected, it will use
//' these values as the coefficient prior on b.
//' @param Bprior value to assign the diagonals of the covariance matrix B for the observed variables.
//' @param LBprior value to assign the diagonals of the covariance matrix B for the latent variables.
//' 
//' @param alphaomega a value for the concentration parameter. Leave blank to apply a prior distribution
//' to the concentration parameter of Gamma(aomega,bomega).
//' @param aomega a value for the hyperparameter on the concentration parameter.
//' @param bomega a value for the hyperparameter on the concentration parameter.
//' 
//' @param Homega the number of latent intercepts
//' 
//' @param samples the number of MCMC samples to draw and return from the posterior.
//' @param burnin the number of MCMC samples to "burn-in."
//' @param thinning the number of MCMC samples to "thin".
//' 
//' @return 
//' \describe{
//' \item{"samples"}{is the number of samples.} 
//' \item{"H"}{is the number of covariate values after transformations} 
//' \item{"n"}{is the number of observed individuals in the data.}
//' \item{"N"}{is a sample of population sizes from the posterior distribution.}
//' \item{"Xmis"}{is a sample of missing covariate values. One of each covariate is sampled and saved
//' at each sampling occasion.}
//' \item{"alphaomega"}{is a sample of alphaomega values from the posterior.}
//' \item{"Beta"}{is a sample of JxH+Homega matrices of beta coefficients.}
//' \item{"Betacolnames"}{The names of the covariates for each column of the beta coefficients.}
//' \item{"PSIMATRIX"}{}
//' \item{"PSICOUNTS"}{}
//' \item{"PHIomega"}{}
//' \item{"omegalabels"}{}
//' \item{"priorBdiag"}{}
//' \item{"probcaptured"}{}
//' \item{"loglikelihood"}{}
//' \item{"BIC"}{}
//' }
//' 
//' 
//' @examples
//' #Create the Data
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
//' myCRdata = multdatasimulator(2000,mybeta,mynormprobs,mynormmeans,mynormsigma,
//'                            missing=TRUE)
//'                            
//' #Run the Algorithm
//' myformula = cbind(y1,y2,y3,y4)~x1+x2
//' 
//' CRresults=BLRCRsolvermissing(myformula,df=myCRdata,Homega=1,
//'                              Bprior=1,LBprior=1,
//'                              covsupport="unique",
//'                              samples = 1000,burnin=10,thinning = 10)
//' 
//' 
//' @export 
// [[Rcpp::export]]
Rcpp::List BLRCRsolvermissing(Formula formula, DataFrame df, 
                      String covmethod="bootstrap", 
                      String covsupport="unique",
                      String coefprior="normal",
                      Rcpp::Nullable<Rcpp::NumericMatrix> bprior = R_NilValue,
                      double Bprior=1, double LBprior=1,
                      double alphabayesbootstrap=0,
                      Rcpp::Nullable<double> alphaomega = R_NilValue,
                      double aomega = 0.25, double bomega = 0.25, 
                      unsigned int Homega=1,
                      int samples=1000, int burnin=0, int thinning=1){
  
 //Get functions from Rpackages
 Rcpp::Environment pkg1 = Rcpp::Environment::namespace_env("BayesLogit");
 Rcpp::Function rpg = pkg1["rpg"];
 
 Rcpp::Environment stats_env("package:stats");
 Rcpp::Function model_frame = stats_env["model.frame"];
 Rcpp::Function model_matrix = stats_env["model.matrix"];
 Rcpp::Function model_matrix_lm = stats_env["model.matrix.lm"];
 
 Rcpp::Environment base_env("package:base");
 Rcpp::Function all_vars = base_env["all.vars"];
 
 
 if(coefprior=="horseshoe"){
   Rcpp::Rcout << "Error: Horseshoe not programmed for Bootstrap, exact."<< std::endl;
   return(0);
 }else if(coefprior=="lasso"){
   //this is fine
 }else if(coefprior=="normalhier"){
   //this is fine
 }else if(coefprior=="normalhiervaronly"){
   //this is fine
 }else if(coefprior!="normal"){
   Rcpp::Rcout << "Error: Coefficient Prior Method does not exist."<< std::endl;
   return(0);
 }
 
 /////////////////
 //Safety Checks//
 /////////////////
 if(Homega<1){
   Rcpp::Rcout << "Error: The number of latent classes (Homega) cannot be less than 1." << std::endl;
   return(0);
 }
 if(samples<1){
   Rcpp::Rcout << "Error: The number of samples cannot be less than 1" << std::endl;
   return(0);
 }
 if(thinning<1){
   Rcpp::Rcout << "Error: The level of thinning cannot be less than 1" << std::endl;
   return(0);
 }
 if(covmethod=="empirical"){
 }else if(covmethod=="bootstrap"){
 }else{
   Rcpp::Rcout << "Error: Must select empirical or bootstrap for covmethod" << std::endl;
   return(0);
 }
 

 /// Create a copy and simple dataframe, does not include interactions in formula 
 Rcpp::DataFrame df_clone = clone(df); //do this so the original dataset is not overwritten


 //Create New Dataframe from only variables that are used
 CharacterVector yxnamesALL = all_vars(Rcpp::_["expr"] = formula);
 Rcpp::List mylist_new(yxnamesALL.length());
 for(int jh=0; jh<yxnamesALL.length(); jh++){
   std::string mycovname = as<std::string>(yxnamesALL[jh]);
   int mydfcolnumber = df_clone.findName(mycovname);
   mylist_new[jh] = df_clone[mydfcolnumber];
 }
 mylist_new.attr("names") = yxnamesALL;
 Rcpp::DataFrame df_new(mylist_new);
 
 
 //Create ymatrix
 Rcpp::DataFrame df_gety = model_frame(Rcpp::_["formula"] = formula, Rcpp::_["data"] = df_new, Rcpp::_["na.action"] = R_NilValue);
 arma::mat yobs = df_gety[0]; 
 int n = df_gety.nrow(); //get observed sample size
 unsigned int J = yobs.n_cols; //get number of lists
 int NMAXIMUM = 2*n; //preallocate the size of Y and X to this value;
 arma::mat ymis(NMAXIMUM-n,J,arma::fill::zeros);
 arma::mat ymatrix = join_cols(yobs,ymis);
 
 
 //Locate missing values
 int Hsimple = yxnamesALL.length()-J;
 arma::mat TrackMissingCovariates(n,Hsimple,arma::fill::zeros);
 for(int h=0; h<Hsimple; h++){
   Rcpp::NumericVector mycovariatecol = df_new[J+h]; 
   Rcpp::LogicalVector mymissingcovs = Rcpp::is_na(mycovariatecol);
   for(int i=0; i<n; i++){
     bool mycovval = mymissingcovs[i];
     if(mycovval==TRUE){
       TrackMissingCovariates.submat(i,h,i,h) = 1;
     }
   }
 }
 
 
 //impute df_new with random observed values
 for(int h=0; h<Hsimple; h++){
   if(Rf_isFactor(df_new[J+h])){
     IntegerVector mycovariatecol = df_new[J+h];
     IntegerVector mydfcolumnNONA = Rcpp::na_omit(mycovariatecol);
     int nNONA = mydfcolumnNONA.size();
     for(int i=0; i<n; i++){
       if(TrackMissingCovariates(i,h)==1){
         int propxindex = Rcpp::sample(nNONA, 1, TRUE)(0) - 1;
         int propx = mydfcolumnNONA[propxindex];
         mycovariatecol[i] = propx;
       }
     }
   }else{
     NumericVector mycovariatecol = df_new[h+J];
     NumericVector mydfcolumnNONA = Rcpp::na_omit(mycovariatecol);
     int nNONA = mydfcolumnNONA.size();
     for(int i=0; i<n; i++){
       if(TrackMissingCovariates(i,h)==1){
         int propxindex = Rcpp::sample(nNONA, 1, TRUE)(0) - 1;
         double propx = mydfcolumnNONA[propxindex];
         mycovariatecol[i] = propx;
       }
     }
   }
  }
 
 //Create xcovariatematrix
 //Note: This is a simpler matrix that only contains the x values before transformations, dummies, etc..
 //This is the set of variables that the covariate distribution will be calculated on
 arma::mat xcovariatematrix(NMAXIMUM,Hsimple,arma::fill::zeros);
 for(int h=0; h<Hsimple; h++){
   NumericVector mycovariatecol = df_new[h+J];
   xcovariatematrix.submat(0,h,n-1,h)  = as<arma::colvec>(wrap(mycovariatecol));
 }
 

 //Create xmatrix
 //Note: this creates the full X matrix, including an intercept
 //and latent group membership columns
 Rcpp::NumericMatrix createdXmatrix = model_matrix_lm(Rcpp::_["object"] = formula, Rcpp::_["data"] = df_new, Rcpp::_["na.action"] = R_NilValue);
 arma::mat createdXmatrixARMA = as<arma::mat>(wrap(createdXmatrix));
 unsigned int H = createdXmatrix.ncol()-1; //get number of covariates, use formula, subtract 1 to not count intercept
 arma::mat xmatrix(NMAXIMUM,H+Homega,arma::fill::zeros);
 if(H>0){
   xmatrix.submat(0,0,n-1,H-1) = createdXmatrixARMA.submat(0,1,n-1,H);
 }
 
 
 //get the xmatrix col names (save them to beta at the end)
 //need to remove intercept name and add in latent group names
 CharacterVector betacolnames(H+Homega);
 if(H>0){
   List xmatrixdimnames = createdXmatrix.attr("dimnames");
   CharacterVector xmatrixcolnamesALL = xmatrixdimnames[1];
   CharacterVector xmatrixcolnamesnointercept = xmatrixcolnamesALL[Range(1,H)];
   betacolnames[Range(0,H-1)] = xmatrixcolnamesnointercept;
   for(unsigned int h=1; h<Homega+1; h++){
     betacolnames[H+h-1] = "LGroup"+std::to_string(h);
   }
 }
 
 //////////////////////////////////////////////////////////////////////////////
 ////Create Support for Covariate Distribution and initialize probabilities////
 //////////////////////////////////////////////////////////////////////////////
 //Create PSIMATRIX, PSICOUNTS, PSICOUNTSindex
 int Kunique=1;
 arma::colvec myuniquecovariatecounts(Hsimple); //count of unique values in each column
 arma::field<arma::colvec> uniquecovariates(Hsimple);//field of all unique covariates, columns per covariate
 arma::mat PSIMATRIX; 
 arma::mat PSICOUNTS;
 arma::rowvec PSIPROBS;
 arma::uvec PSICOUNTSindex;
 arma::colvec covcolindextracker(Hsimple,arma::fill::zeros);
 if(H>0){
   //All combinations of covariates values
   if(covsupport=="all"){
     for(int h=0; h<Hsimple; h++){
       arma::colvec myuniquecovariatescol = arma::unique(xcovariatematrix.submat(0,h,n-1,h));
       myuniquecovariatecounts(h) = myuniquecovariatescol.n_rows;
       uniquecovariates(h) = myuniquecovariatescol;
       Kunique=Kunique*myuniquecovariatescol.n_rows;
     }
     PSIMATRIX.set_size(Kunique,Hsimple);
     PSICOUNTS.set_size(Kunique,1);
     for(int k=0; k<Kunique; k++){
       for(int h=0; h<Hsimple; h++){
         arma::colvec myuniquecovariatescol = uniquecovariates(h);
         PSIMATRIX(k,h) = myuniquecovariatescol(covcolindextracker(h));
       }
       for(int h=0; h<Hsimple; h++){
         covcolindextracker(h)++;
         if(covcolindextracker(h)<myuniquecovariatecounts(h)){
           break;
         }else{
           covcolindextracker(h)=0;
         }
       }
     }
   //Only unique combinations that show up in the data
   }else if(covsupport=="unique"){
     arma::uvec TrackMissingCovariatesrow(n,arma::fill::zeros);
     for(int i=0; i<n; i++){
       int tracker = accu(TrackMissingCovariates.row(i));
       if(tracker>0){
         TrackMissingCovariatesrow(i) = 1;
       }
     }
     int numNOTmissingvals = n-accu(TrackMissingCovariatesrow);
     arma::uvec TrackNotMissingCovariatesindices(numNOTmissingvals);
     int kmis=0;
     for(int i=0; i<n; i++){
       if(TrackMissingCovariatesrow(i)==0){
         TrackNotMissingCovariatesindices(kmis) = i;
         kmis++;
       }
     }
     arma::mat unique_rows(const arma::mat& x);
     PSIMATRIX = unique_rows(xcovariatematrix.rows(TrackNotMissingCovariatesindices));
     Kunique = PSIMATRIX.n_rows;
     PSICOUNTS.set_size(Kunique,1);
   }
   //Fill in PSICOUNTS and PSIPROBS
   PSICOUNTS.fill(0);
   PSICOUNTSindex.set_size(NMAXIMUM); //set size to be number of unique values
   for(int i=0; i<n; i++){
     unsigned int myindex = index_min(sum(abs(PSIMATRIX.each_row()-xcovariatematrix.row(i)),1));
     PSICOUNTSindex(i)=myindex;
     PSICOUNTS(myindex)++;
   }
   arma::rowvec PSICOUNTSRV = PSICOUNTS.t();
   for(int k=0; k<Kunique; k++){ //add the prior to each count
     PSICOUNTSRV[k] = PSICOUNTSRV[k] + alphabayesbootstrap;
   }
   arma::rowvec mydirichlet(arma::rowvec alphas);
   PSIPROBS = mydirichlet(PSICOUNTSRV);
 }
 
 //Create the PSIMATRIXLIST
 //but with the formula transformations/dummies
 Rcpp::List PSIMATRIXLIST(J+Hsimple);
 arma::mat PSIMATRIX_Y(Kunique,J,arma::fill::zeros);
 for(unsigned int j=0; j<J; j++){
   PSIMATRIXLIST[j] = PSIMATRIX_Y.col(j);
 }
 for(int h=0; h<Hsimple; h++){
   if(Rf_isFactor(df_new[J+h])){
     IntegerVector myfactorvar = as<IntegerVector>(wrap(PSIMATRIX.col(h)));
     IntegerVector mydfvar = df_new[J+h];
     myfactorvar.attr("levels") = mydfvar.attr("levels");
     myfactorvar.attr("class") = "factor";
     PSIMATRIXLIST[J+h] = myfactorvar;
   }else{
     PSIMATRIXLIST[J+h] = PSIMATRIX.col(h);
   }
 }
 PSIMATRIXLIST.attr("names") = yxnamesALL;
 Rcpp::DataFrame PSIMATRIX_DF(PSIMATRIXLIST);
 Rcpp::NumericMatrix PSIMATRIX_trans_NM = model_matrix_lm(Rcpp::_["object"] = formula, Rcpp::_["data"] = PSIMATRIX_DF, Rcpp::_["na.action"] = R_NilValue);
 arma::mat PSIMATRIX_trans = as<arma::mat>(wrap(PSIMATRIX_trans_NM));
 if(H>0){
   PSIMATRIX_trans.shed_col(0);
 }
 
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 /// Insert Priors Here /////////////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                                                                                           //////
 //Set prior for coefficients                                                                              //////
 //Normal//////                                                                                            //////
 arma::mat priorb(J,H+Homega);                                                                             //////
 if(bprior==R_NilValue){                                                                                   //////
   priorb.fill(0);                                                                                         //////
 }else{                                                                                                    //////
   priorb = as<arma::mat>(wrap(bprior));                                                                   //////
   if(priorb.n_rows==J){                                                                                   //////
    //safety check, correct number of rows                                                                 //////
   }else{                                                                                                  //////
     Rcpp::Rcout << "Error: bprior has the wrong number of rows" << std::endl;                             //////
     return(0);                                                                                            //////
   }                                                                                                       //////
   if(priorb.n_cols==H+Homega){                                                                            //////
     //safety check, correct number of cols                                                                //////
   }else{                                                                                                  //////
     Rcpp::Rcout << "Error: bprior has the wrong number of columns." << std::endl;                         //////
     return(0);
   }                                                                                                       //////
                                                         
 }                                                                                                         //////
                                                                                                           //////
                                                                                                           //////
 //Variance                                                                                                //////
 arma::mat priorB = arma::eye(H+Homega,H+Homega);                                                          //////
 for(unsigned int h=0; h<H; h++){                                                                                   //////
   priorB(h,h)=Bprior;                                                                                     //////
 }                                                                                                         //////
 for(unsigned int h=H; h<H+Homega; h++){                                                                            //////
   priorB(h,h)=LBprior;                                                                                    //////
 }                                                                                                         ////// 
                                                                                                           //////
 //Set priors for alpha_omega                                                                              //////
 //double aomega = 0.25;   //This is now declared by user                                                  //////
 //double bomega = 0.25;   //This is now declared by user                                                  //////
 //Bootstrap prior                                                                                         //////
 //double alphabayesbootstrap = 0; //This is now declared by user                                          //////                                                  
                                                                                                           //////
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
 //////////////////////////////////
 //Compute some useful constants //
 //////////////////////////////////
 arma::mat Binv = inv_sympd(priorB);
 arma::mat Bb(J,H+Homega); //not constant under "normalhier"
 for(unsigned int j=0; j<J; j++){
   Bb.row(j) = priorb.row(j)*Binv;
 }
 
 
 ///////////////////
 //Initializations//
 ///////////////////
 
 //Initialize beta and N//
 arma::mat beta(J,H+Homega,arma::fill::value(0.0));//initialize beta to all 0s
 int n0 = 0; 
 int N = n+n0;
 //Try to initialize beta and N with conditional BLRCR value
 if(H>0){
   arma::vec condpriorb(H+1,arma::fill::zeros);
   arma::mat condpriorB = arma::eye(H+1,H+1)*Bprior;
   Rcpp::List condBLRCRsolver(arma::mat y, arma::mat x, 
                              arma::vec priorb, arma::mat priorB,
                              double gradparam, int prior,
                              double tol,int maxiter,
                              Rcpp::Nullable<Rcpp::NumericMatrix> initbeta = R_NilValue);
   double mygradparam = .01;
   Rcpp::Rcout << "Attempting to initialize beta matrix with condBLRCR."  << std::endl;
   Rcpp::List betainit = condBLRCRsolver(yobs,xmatrix.submat(0,0,n-1,H-1),condpriorb,condpriorB,mygradparam,1,1e-6,1e4);
   int convergence = betainit[2];//checks convergence
   if(convergence==1){
     Rcpp::Rcout << "Success!"  << std::endl;
     arma::mat condbeta = betainit[1];
     int condN = betainit[0];
     beta.cols(0,H-1) = condbeta.cols(1,H);
     beta.col(H) = condbeta.col(0);
     N = condN;
     Rcpp::Rcout << "Initializing N to: " << N<< std::endl;
     if(N>=NMAXIMUM){
       int NMAXIMUMprev = NMAXIMUM;
       NMAXIMUM = N*1.2;
       arma::mat yaddrows(NMAXIMUM-NMAXIMUMprev,J,arma::fill::zeros);
       arma::mat xaddrows(NMAXIMUM-NMAXIMUMprev,H+Homega,arma::fill::zeros);
       arma::mat xcovaddrows(NMAXIMUM-NMAXIMUMprev,Hsimple,arma::fill::zeros);
       arma::mat ymatrix = join_cols(ymatrix,yaddrows);
       arma::mat xmatrix = join_cols(xmatrix,xaddrows);
       arma::mat xcovmatrix = join_cols(xcovariatematrix,xcovaddrows);
     }
     n0 = N-n;
   }
   
 }
 
 
 //Fill in Missing Y and X values (and xcovariatematrix)
 //For this, we are essentially using the empirical approach (i.e. equal probability assigned to each observation) 
 //but conditioning on the capture history
 arma::mat lambda(NMAXIMUM,J);
 lambda.rows(0,n-1) = 1.0/(1.0+exp(-xmatrix.rows(0,n-1)*beta.t()));
 NumericVector probmis(n);  //this is set to n for the initialization because this is the same as previous N at the start
 for(int i=0; i<n; i++){
   probmis(i) = prod(1.0-lambda.row(i));
 }
 Vector<INTSXP> indexlocations = Rcpp::sample(n, n0, TRUE, probmis);
 for(int i=0; i<n0; i++){
   int mymisindex = indexlocations(i)-1; //need to -1 as our index seq is 1:N, but C indexing is 0:N-1
   xmatrix.row(n+i) = xmatrix.row(mymisindex);
   xcovariatematrix.row(n+i) = xcovariatematrix.row(mymisindex);
 }
 
 
 //Latent grouping initialization
 bool fixalphaomega = true;
 double alphaomegaVAL;
 if(alphaomega==R_NilValue){
   fixalphaomega = false;
   alphaomegaVAL = 1.0;
 }else{
   alphaomegaVAL = as<double>(wrap(alphaomega));
 }

 arma::rowvec PHIomega(Homega,arma::fill::value(1.0/(Homega)));
 arma::rowvec logPHIomega = log(PHIomega);
 arma::mat logprobomega(N,Homega,arma::fill::ones);
 arma::vec latentclassmembership(n);
 if(Homega==1){
   xmatrix.col(H).fill(1);
 }else if(Homega>1){
   arma::mat X_times_beta_nolatent(N,J,arma::fill::zeros);
   if(H>0){
     X_times_beta_nolatent = xmatrix.submat(0,0,N-1,H-1)*beta.cols(0,H-1).t();
   }
   arma::mat X_times_beta = X_times_beta_nolatent;
   for(unsigned int w=0; w<Homega; w++){
     arma::rowvec betalatentw = beta.col(H+w).t();
     for(int i=0; i<N; i++){
       X_times_beta.row(i) = X_times_beta_nolatent.row(i) + betalatentw;
     }
     arma::mat MYLAMBDA = (1/(1+exp(-X_times_beta)));
     arma::mat logprobomegavec = arma::sum(arma::log(MYLAMBDA)%ymatrix.rows(0,N-1) + arma::log(1.0-MYLAMBDA)%(1.0-ymatrix.rows(0,N-1)),1)+ log(PHIomega(w));
     logprobomega.col(w) = logprobomegavec;
   }
   arma::rowvec logomegarow(H+Homega);
   arma::rowvec probomegarow(H+Homega);
   double mysampledval=0;
   for(int i=0; i<N; i++){
     logomegarow = logprobomega.row(i);
     probomegarow = exp(logomegarow - logomegarow.max());
     mysampledval = R::runif(0,sum(probomegarow));
     double probomegarowcum=0;
     for(unsigned int w=0; w<Homega; w++){
       probomegarowcum += probomegarow(w);
       if(probomegarowcum>mysampledval){
         if(i<n){
           latentclassmembership(i) = w;
         }
         xmatrix(i,H+w) = 1;
         break;
       }
     }
   }
   
 }

 ///////////////////////////////
 ////Variables to be saved//////
 ///////////////////////////////
 arma::vec saveN(samples);
 arma::mat saveXmis(samples,Hsimple);
 arma::mat savePHIomega(samples,Homega);
 arma::cube saveBETA(J,H+Homega,samples);
 arma::mat saveOMEGAlab(n,Homega,arma::fill::ones);
 arma::vec savealphaomega(samples);
 arma::mat savelikelihood(samples,1,arma::fill::zeros);
 arma::mat savePSICOUNTS(samples,Kunique); 
 arma::mat probcaptured(n,samples);
 
 arma::cube saveinvlambda2_betasampling(J,H+Homega,samples); //only for lasso
 
 arma::cube savepriorBdiag(samples,J,H+Homega); //only for normalhiervaronly
 
 //////////////////////////////////////////////////////////////
 //Allocate Constants to be used during Gibbs sampling steps //
 //////////////////////////////////////////////////////////////
 
 //Beta Sampling
 arma::mat kappa_betasampling(NMAXIMUM,J);
 kappa_betasampling = ymatrix-0.5;
 arma::mat Vw_betasampling(H+Homega,H+Homega);
 arma::colvec Mw_betasampling(H+Homega);
 
 //n0 and N Sampling
 int simxsamplenum=1000*(Homega+H);
 arma::mat simX(simxsamplenum,H+Homega,arma::fill::ones);
 arma::mat simlambda(simxsamplenum,J,arma::fill::ones);
 NumericVector simprobmis(simxsamplenum);
 double rho=0;
 
 
 //sample covariates of unobserved individuals
 arma::uvec PSICOUNTSindex_beforeexpanding(Kunique*Homega);
 arma::mat PSIMATRIX_trans_expanded(Kunique*Homega,H+Homega,arma::fill::ones);
 arma::mat PSIMATRIX_expanded(Kunique*Homega,Hsimple,arma::fill::ones);
 arma::colvec PSIPROBSexpanded(Kunique*Homega);
 int mypsiindex = 0;
 for(int k=0; k<Kunique; k++){
   for(unsigned int h=0; h<Homega; h++){
     PSICOUNTSindex_beforeexpanding(mypsiindex)=k;
     arma::rowvec homegarow(Homega,arma::fill::zeros);
     homegarow[h]=1;
     if(Hsimple>0){
       PSIMATRIX_expanded.submat(mypsiindex,0,mypsiindex,Hsimple-1) = PSIMATRIX.row(k);
     }
     if(H>0){
       PSIMATRIX_trans_expanded.submat(mypsiindex,0,mypsiindex,H-1) = PSIMATRIX_trans.row(k);
     }
     if(Homega>1){
       PSIMATRIX_trans_expanded.submat(mypsiindex,H,mypsiindex,H+Homega-1) = homegarow;
     }
     if(H>0){
       PSIPROBSexpanded[mypsiindex] = PSIPROBS[k]*PHIomega[h];
     }else if(H==0){
       PSIPROBSexpanded[mypsiindex] = PHIomega[h];
     }
     
     mypsiindex++;
   }
 }
 NumericVector PSIPROBSexpandedNV = as<NumericVector>(wrap(PSIPROBSexpanded));
 arma::mat lambda_PSIMATRIXEXPANDED(Kunique*Homega,J);
 NumericVector probmis_PSIMATRIXEXPANDED(Kunique*Homega);
 NumericVector myprobs_misunobs(Kunique*Homega);
 
 
 //Sample Missing Covariates of Observed Individuals
 arma::mat PSIPROBS_individualindex(Kunique,n,arma::fill::ones);
 arma::vec TrackMissingCovariateRow(n);
 //this is a column matrix of 0s and 1s that will be multiplied by PSIPROBS.
 //hence 0 weight will be assigned to the probability of rows that are not possible for that index.
 for(int i=0; i<n; i++){
   TrackMissingCovariateRow(i)=accu(TrackMissingCovariates.row(i));
   if(TrackMissingCovariateRow(i)>0){
     for(int k=0; k<Kunique; k++){
       for(int h=0; h<Hsimple; h++){
         if(TrackMissingCovariates(i,h)==0){
           if(xcovariatematrix(i,h)!=PSIMATRIX(k,h))
             PSIPROBS_individualindex(k,i) = 0;
         }
       }
     }
   }
 }
 arma::vec PSIPROBS_missingrowcounts(n,arma::fill::zeros);
 for(int i=0; i<n; i++){
   if(TrackMissingCovariateRow(i)>0){
     PSIPROBS_missingrowcounts(i) = accu(PSIPROBS_individualindex.col(i));
   }
 }
 arma::umat PSIPROBS_missingindices(n,max(PSIPROBS_missingrowcounts),arma::fill::zeros);
 for(int i=0; i<n; i++){
   if(TrackMissingCovariateRow(i)>0){
     int kind=0;
     for(int k=0; k<Kunique; k++){
       if(PSIPROBS_individualindex(k,i)==1){
         PSIPROBS_missingindices(i,kind) = k;
         kind++;
       }
     }
   }
 }
 arma::mat PSIMATRIX_times_beta_nolatent(Kunique,J);
 arma::cube lambda_PSIMATRIX_eachlatentclass(Kunique,J,Homega);
 arma::vec PSIPROBS_ind(Kunique);
 arma::rowvec xlatentvars(Homega);
 arma::mat PSIMATRIX_trans_indwithlatent(Kunique,H+Homega);
 if(H>0){
   PSIMATRIX_trans_indwithlatent.cols(0,H-1) = PSIMATRIX_trans; //this is the PSIMATRIX transformed but we can add a latent variable to the end
 }
 arma::mat lambda_PSIMATRIX(Kunique,J);
 
 
 //sample latent class membership
 arma::mat OMEGALABELS(n,Homega,arma::fill::ones);
 arma::rowvec Nomega(Homega);
 arma::vec logphiomega(Homega);
 
  
  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
  // Gibbs Sampling Steps ////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
  NumericMatrix MYTIMER(samples,10);
  CharacterVector timernames = {"beta","N","reallocatesize","missingunobservedX","missingobservedX",
                                "PHIomega","OMEGALABELS","PSIPROBS","loglike","saving"};
  colnames(MYTIMER) = timernames;
  for(int sample=-burnin; sample<samples; sample++){
    for(int thin=0; thin<thinning; thin++){
      Timer timer;
      ///////////////
      //Sample Beta//
      ///////////////
      
      timer.step("start");
      
      if(coefprior=="normal"){
        arma::mat psi_betasampling=xmatrix.rows(0,N-1)*beta.t();
        for(unsigned int j=0; j<J; j++){
          arma::vec w_betasampling=as<arma::vec>(wrap(rpg(N, 1.0, psi_betasampling.col(j))));
          Vw_betasampling = inv_sympd(xmatrix.rows(0,N-1).t()*diagmat(w_betasampling)*xmatrix.rows(0,N-1) + Binv);
          Mw_betasampling = Vw_betasampling*(xmatrix.rows(0,N-1).t()*kappa_betasampling.submat(0,j,N-1,j)+trans(Bb.row(j)));
          beta.row(j) = arma::mvnrnd(Mw_betasampling,Vw_betasampling).t();
        }
      }else if(coefprior=="lasso"){
        arma::mat psi_betasampling=xmatrix.rows(0,N-1)*beta.t();
        for(unsigned int j=0; j<J; j++){
          arma::vec w_betasampling=as<arma::vec>(wrap(rpg(N, 1.0, psi_betasampling.col(j))));
          arma::vec invlambda2_betasampling(H+Homega);//sample from IGauss, see wiki, note lambda=2
          for(unsigned int h=0; h<H+Homega; h++){
            double invtau_bs = Binv(h,h); 
            double mu2j_bs = 2.0/(invtau_bs*(pow(beta(j,h),2)+(1e-6)));
            double muj_bs = pow(mu2j_bs,.5);
            double v_bs = R::rnorm(0,1);
            double y_bs = pow(v_bs,2);
            double x_bs = muj_bs + (mu2j_bs*y_bs)/4 - (muj_bs/4)*pow((8*muj_bs*y_bs)+mu2j_bs*pow(y_bs,2),.5);
            double z_bs = R::unif_rand();  
            if((muj_bs/(muj_bs+x_bs))>z_bs){ //note: 1/lambda2 is IGauss, so these are inverted to get lambda2
              invlambda2_betasampling[h] = mu2j_bs/x_bs; 
            }else{
              invlambda2_betasampling[h] = x_bs;
            }
          }
          
          if(sample>=0){
            if(thin==thinning-1){
              saveinvlambda2_betasampling(arma::span(j,j),arma::span(0,H+Homega-1),arma::span(sample,sample)) = invlambda2_betasampling.t();
            }
          }
          arma::mat invLAMBDAp = diagmat(invlambda2_betasampling%Binv.diag());
          Vw_betasampling = inv_sympd(xmatrix.rows(0,N-1).t()*diagmat(w_betasampling)*xmatrix.rows(0,N-1) + invLAMBDAp);
          Mw_betasampling = Vw_betasampling*(xmatrix.rows(0,N-1).t()*kappa_betasampling.submat(0,j,N-1,j)+trans(priorb.row(j)*invLAMBDAp));
          beta.row(j) = arma::mvnrnd(Mw_betasampling,Vw_betasampling).t();
        }
      }else if(coefprior=="normalhier"){
        arma::cube Bmatrix_bs(H+Homega,H+Homega,J);
        arma::cube invBmatrix_bs(H+Homega,H+Homega,J);
        arma::mat bvector_bs(J,H+Homega);
        arma::mat psi_betasampling=xmatrix.rows(0,N-1)*beta.t();
        for(unsigned int j=0; j<J; j++){
          //page 73 in BDA 3rd edition
          //note: n=1, k0=1, v0=H+Homega, 
          Bmatrix_bs.slice(j) = arma::iwishrnd(inv_sympd(Binv+0.5*trans(beta.row(j)-priorb.row(j))*(beta.row(j)-priorb.row(j))),H+Homega+1);
          invBmatrix_bs.slice(j) = inv_sympd(Bmatrix_bs.slice(j));
          bvector_bs.row(j) = arma::mvnrnd(trans(0.5*(beta.row(j)+priorb.row(j))),Bmatrix_bs.slice(j)/2).t();
          arma::vec w_betasampling=as<arma::vec>(wrap(rpg(N, 1.0, psi_betasampling.col(j))));
          Vw_betasampling = inv_sympd(xmatrix.rows(0,N-1).t()*diagmat(w_betasampling)*xmatrix.rows(0,N-1) + invBmatrix_bs.slice(j));
          Mw_betasampling = Vw_betasampling*(xmatrix.rows(0,N-1).t()*kappa_betasampling.submat(0,j,N-1,j)+trans(bvector_bs.row(j)*invBmatrix_bs.slice(j)));
          beta.row(j) = arma::mvnrnd(Mw_betasampling,Vw_betasampling).t();
        }
      }else if(coefprior=="normalhiervaronly"){
        arma::cube Bmatrix_bs(H+Homega,H+Homega,J);
        arma::cube invBmatrix_bs(H+Homega,H+Homega,J);
        arma::mat bvector_bs(J,H+Homega);
        arma::mat psi_betasampling=xmatrix.rows(0,N-1)*beta.t();
        for(unsigned int j=0; j<J; j++){
          //Derived with conjugate invWishart(Binv,H+Homega)
          Bmatrix_bs.slice(j) = arma::iwishrnd(inv_sympd(Binv+trans(beta.row(j)-priorb.row(j))*(beta.row(j)-priorb.row(j))),H+Homega+1);
          if(sample>=0){
            if(thin==thinning-1){
              savepriorBdiag(arma::span(sample,sample),arma::span(j,j),arma::span(0,H+Homega-1)) = trans(Bmatrix_bs.slice(j).diag());
            }
          }
          invBmatrix_bs.slice(j) = inv_sympd(Bmatrix_bs.slice(j));
          arma::vec w_betasampling=as<arma::vec>(wrap(rpg(N, 1.0, psi_betasampling.col(j))));
          Vw_betasampling = inv_sympd(xmatrix.rows(0,N-1).t()*diagmat(w_betasampling)*xmatrix.rows(0,N-1) + invBmatrix_bs.slice(j));
          Mw_betasampling = Vw_betasampling*(xmatrix.rows(0,N-1).t()*kappa_betasampling.submat(0,j,N-1,j)+trans(priorb.row(j)*invBmatrix_bs.slice(j)));
          beta.row(j) = arma::mvnrnd(Mw_betasampling,Vw_betasampling).t();
        }
      }
      

      timer.step("beta");
      
      
      ////////////////////
      //Sample n0 and N //
      ////////////////////
      //Compute rho (marginal probability an individual is missing)
      simX.fill(1);
      Vector<INTSXP> indexlocations = Rcpp::sample(Kunique*Homega, simxsamplenum, TRUE, PSIPROBSexpandedNV);
      for(int i=0; i<simxsamplenum; i++){
        simX.row(i) = PSIMATRIX_trans_expanded.row(indexlocations(i)-1);
      }
      simlambda = 1.0/(1.0+exp(-simX*beta.t()));
      for(int i=0; i<simxsamplenum; i++){
        simprobmis(i) = prod(1.0-simlambda.row(i));
      }
      rho=mean(simprobmis);
      
      //Sample N
      n0 = Rcpp::rnbinom( 1, n, 1-rho )(0);
      N = n+n0;
      timer.step("N");
      
      if(thin==thinning-1){
        Rcpp::Rcout << "Iteration: " << sample+1 << std::endl;
        Rcpp::Rcout << "N: " << N<< std::endl;
      }
      
      //If N is greater than the maximum allocated memory, add extra rows
      if(N>=NMAXIMUM){
        Rcpp::Rcout << "Gotta boost that allocation!" << std::endl;
        int NMAXIMUMprev = NMAXIMUM;
        NMAXIMUM = N*1.2;
        Rcpp::Rcout << "new NMAXIMUM: " << NMAXIMUM << std::endl;
        arma::mat yaddrows(NMAXIMUM-NMAXIMUMprev,J,arma::fill::zeros);
        arma::mat xaddrows(NMAXIMUM-NMAXIMUMprev,H+Homega,arma::fill::zeros);
        arma::mat xcovaddrows(NMAXIMUM-NMAXIMUMprev,Hsimple,arma::fill::zeros);
        arma::mat lambdaaddrows(NMAXIMUM-NMAXIMUMprev,J,arma::fill::zeros);
        ymatrix = join_cols(ymatrix,yaddrows);
        xmatrix = join_cols(xmatrix,xaddrows);
        xcovariatematrix = join_cols(xcovariatematrix,xcovaddrows);
        lambda = join_cols(lambda,lambdaaddrows);
        PSICOUNTSindex.resize(NMAXIMUM);
        kappa_betasampling.set_size(NMAXIMUM,J);
        kappa_betasampling = ymatrix-0.5;
      }
      timer.step("reallocatesize");
      
      //////////////////////////////////////////
      // Sample Unobserved Covariate X Values //
      // Note: Does not include missing covs  //
      //       of the observed individuals    //
      //////////////////////////////////////////
      lambda_PSIMATRIXEXPANDED = 1.0/(1.0+exp(-PSIMATRIX_trans_expanded*beta.t()));
      for(unsigned int i=0; i<Kunique*Homega; i++){
        probmis_PSIMATRIXEXPANDED(i) = prod(1.0-lambda_PSIMATRIXEXPANDED.row(i));
      }
      myprobs_misunobs = PSIPROBSexpandedNV*probmis_PSIMATRIXEXPANDED; //note this is not standardized
      Vector<INTSXP> indexlocations_misunobs = Rcpp::sample(Kunique*Homega, n0, TRUE, myprobs_misunobs);
      for(int i=0; i<n0; i++){
        int mymisindex = indexlocations_misunobs(i)-1; //need to -1 as our index seq is 1:N, but C indexing is 0:N-1
        if(H>0){
          PSICOUNTSindex(n+i) = PSICOUNTSindex_beforeexpanding(mymisindex);
        }
        xmatrix.row(n+i) = PSIMATRIX_trans_expanded.row(mymisindex);
        xcovariatematrix.row(n+i) = PSIMATRIX_expanded.row(mymisindex);
      }    
      timer.step("missingunobservedX");
      
      
      ///////////////////////////////////////////////////////
      // Sample Missing Covariates of Observed Individuals //
      ///////////////////////////////////////////////////////
      
      if(H>0){
        PSIMATRIX_times_beta_nolatent = PSIMATRIX_trans*beta.cols(0,H-1).t();
        for(unsigned int w=0; w<Homega; w++){
          arma::rowvec betalatentw = beta.col(H+w).t();
          for(int k=0; k<Kunique; k++){
            lambda_PSIMATRIX_eachlatentclass.subcube(k,0,w,k,J-1,w) = 1/(1+exp(-(PSIMATRIX_times_beta_nolatent.row(k) + betalatentw)));
          }
        }
        for(int i=0; i<n; i++){
          if(TrackMissingCovariateRow(i)>0){
            arma::urowvec PSIPROBS_missingindices_ind = PSIPROBS_missingindices.submat(i,0,i,PSIPROBS_missingrowcounts(i)-1);
            arma::vec PSIPROBS_ind = PSIPROBS.elem(PSIPROBS_missingindices_ind);
            int w = latentclassmembership(i);
            lambda_PSIMATRIX = lambda_PSIMATRIX_eachlatentclass.slice(w);
            arma::mat lambda_PSIMATRIX_ind = lambda_PSIMATRIX.rows(PSIPROBS_missingindices_ind);
            arma::vec probcapturehistory(PSIPROBS_missingrowcounts(i),arma::fill::value(1));
            for(unsigned int k=0; k<PSIPROBS_missingrowcounts(i); k++){
              for(unsigned int j=0; j<J; j++){
                if(ymatrix(i,j)==1){
                  probcapturehistory[k] = probcapturehistory[k]*lambda_PSIMATRIX_ind(k,j);
                }else{
                  probcapturehistory[k] = probcapturehistory[k]*(1-lambda_PSIMATRIX_ind(k,j));
                }
              }
            }
            arma::vec mysamplingprobs = probcapturehistory%PSIPROBS_ind;
            double mysampledval = R::runif(0,sum(mysamplingprobs));
            double probcum=0;
            for(int k=0; k<PSIPROBS_missingrowcounts(i); k++){
              probcum += mysamplingprobs(k);
              if(probcum>mysampledval){
                int myindex = PSIPROBS_missingindices_ind(k);
                PSICOUNTSindex(i) = myindex;
                xcovariatematrix.row(i) = PSIMATRIX.row(myindex);
                xmatrix.submat(i,0,i,H-1) = PSIMATRIX_trans.row(myindex) ;
                break;
              }
            }
          }
        }
      }

      timer.step("missingobservedX");
      
      
      
      ////////////////////////////////////
      // Sample Latent Group Membership //
      ////////////////////////////////////
      if(Homega==0){
        xmatrix.col(H).fill(1);
      }
      if(Homega>1){
        //Sample Latent Class Membership Probabilities
        Nomega = sum(xmatrix.submat(0,H,N-1,H+Homega-1)); //rowsums
        int NHcum = 0;
        double accu_log_prod = 0.0;
        for(unsigned int h=0; h<Homega-1; h++){
          NHcum += Nomega(h);
          double aalphaomega = 1 + Nomega(h);
          double balphaomega = alphaomegaVAL + N - NHcum;
          //sampling from beta will result in computational 0s if b is too small, so do Dan's suggestion:
          // for small values of shape uses the method from p.45 of Robert & Casella (2002)
          double lgammaa = log(R::runif(0,1))/aalphaomega + log(R::rgamma(1+aalphaomega,1));
          double lgammab = log(R::runif(0,1))/balphaomega + log(R::rgamma(1+balphaomega,1));
          double sumlogab = log(exp(lgammaa) + exp(lgammab)); //this could be improved
          logphiomega(h) = lgammaa - sumlogab + accu_log_prod;
          accu_log_prod += lgammab - sumlogab; //this is log(1-Vh)
          PHIomega(h) = exp(logphiomega(h));
        }
        logphiomega(Homega-1)=accu_log_prod;
        PHIomega(Homega-1) = exp(logphiomega(Homega-1));
        
        //Sample alphaomega
        if(fixalphaomega == false){
          alphaomegaVAL = R::rgamma(aomega+Homega-1,1/(bomega-logphiomega(Homega-1)));
        }
        
        timer.step("PHIomega");
        
        //Sample Latent Class Memberships
        arma::mat logprobomega(N,Homega,arma::fill::ones);
        arma::mat X_times_beta_nolatent(N,J,arma::fill::zeros);
        if(H>0){
          X_times_beta_nolatent = xmatrix.submat(0,0,N-1,H-1)*beta.cols(0,H-1).t();
        }
        arma::mat X_times_beta = X_times_beta_nolatent; 
        for(unsigned int w=0; w<Homega; w++){
          arma::rowvec betalatentw = beta.col(H+w).t();
          for(int i=0; i<N; i++){
            X_times_beta.row(i) = X_times_beta_nolatent.row(i) + betalatentw;
          }
          arma::mat MYLAMBDA = (1/(1+exp(-X_times_beta)));
          arma::mat logprobomegavec = arma::sum(arma::log(MYLAMBDA)%ymatrix.rows(0,N-1) + arma::log(1.0-MYLAMBDA)%(1.0-ymatrix.rows(0,N-1)),1)+ log(PHIomega(w));
          logprobomega.col(w) = logprobomegavec;
        }  
        arma::rowvec logomegarow(H+Homega);
        arma::rowvec probomegarow(H+Homega);
        double mysampledval=0;  
        for(int i=0; i<n; i++){
          xmatrix.submat(i,H,i,H+Homega-1).fill(0); //reset latent classes all to 0
          logomegarow = logprobomega.row(i);
          probomegarow = exp(logomegarow - logomegarow.max());
          mysampledval = R::runif(0,sum(probomegarow));
          double probomegarowcum=0;
          for(unsigned int w=0; w<Homega; w++){
            probomegarowcum += probomegarow(w);
            if(probomegarowcum>mysampledval){
              latentclassmembership(i) = w;
              xmatrix(i,H+w) = 1;
              break;
            }
          }
          OMEGALABELS.row(i) = xmatrix.submat(i,H,i,H+Homega-1);
        }  
        
        //Update individual's average
        if(sample>=0){
          if(thin==thinning-1){
            saveOMEGAlab=(saveOMEGAlab*sample+OMEGALABELS)/(sample+1);
          }
        }
        
      }
      
      timer.step("OMEGALABELS");
      
      
      //////////////////////////////////
      //Sampling Covariate Distribution//
      ///////////////////////////////////
      if(H>0){
        PSICOUNTS.fill(0); 
        for(int i=0; i<N; i++){
          PSICOUNTS(PSICOUNTSindex(i))++;
        }
        arma::rowvec PSICOUNTSRV = PSICOUNTS.t();
        for(int k=0; k<Kunique; k++){
          PSICOUNTSRV[k] = PSICOUNTSRV[k] + alphabayesbootstrap;
        }
        arma::rowvec mydirichlet(arma::rowvec alphas);
        PSIPROBS = mydirichlet(PSICOUNTSRV); 
        if(sample>=0){
          if(thin==thinning-1){
            savePSICOUNTS.row(sample) = PSICOUNTS.t();
          }
        }
      }  
      //update PSIPROBSexpanded   
      int mypsiindex = 0;
      for(int k=0; k<Kunique; k++){
        for(unsigned int h=0; h<Homega; h++){
          if(H>0){
            PSIPROBSexpanded[mypsiindex] = PSIPROBS[k]*PHIomega[h];
          }else if(H==0){
            PSIPROBSexpanded[mypsiindex] = PHIomega[h];
          }
          
          mypsiindex++;
        }
      }
      PSIPROBSexpandedNV = as<NumericVector>(wrap(PSIPROBSexpanded));
      timer.step("PSIPROBS");
      
      
      
      //////////////////////////////////////////////
      //// Compute Log Likelihood //////////////////
      //// Needed for Model Selection //////////////
      //// Also save capture probs /////////////////
      //////////////////////////////////////////////
      if(sample>=0){
        if(thin==thinning-1){
          double firstterm = R::lchoose(N,n);
          lambda.rows(0,N-1) = (1/(1+exp(-xmatrix.rows(0,N-1)*beta.t())));
          for(int i=0; i<n; i++){
            probmis(i) = prod(1.0-lambda.row(i));
          }
          probcaptured.submat(0,sample,n-1,sample) = 1-as<arma::colvec>(wrap(probmis));
          arma::mat probcaptured(n,samples);
          double secondterm = arma::accu(log(lambda.rows(0,N-1))%ymatrix.rows(0,N-1) + log(1.0-lambda.rows(0,N-1))%(1.0-ymatrix.rows(0,N-1)));
          savelikelihood(sample,0) = firstterm+secondterm; 
        }
      }
      timer.step("loglik");
      
      ////////////////////////
      //// Save Values ///////
      ////////////////////////
      if(sample>=0){
        if(thin==thinning-1){
          saveN(sample)=N;
          saveBETA.subcube(0,0,sample,J-1,H+Homega-1,sample) = beta;
          savePHIomega.row(sample) = PHIomega;
          savealphaomega(sample) = alphaomegaVAL;
          if(H>0){
            if(n0==1){
              saveXmis.row(sample)=xcovariatematrix(N,arma::span(0,Hsimple-1));
            }else if(n0>1){
              int savedmiss = Rcpp::sample(n0, 1, TRUE)(0)-1;
              saveXmis.row(sample)=xcovariatematrix(n+savedmiss,arma::span(0,Hsimple-1));
            }
          }
        }
      }
      
      timer.step("saving");
      NumericVector mytime(timer);
      if(sample>=0){
        if(thin==thinning-1){
          for(int i=0; i<10;i++){
            if(i==0){
              MYTIMER(sample,i) = mytime[i+1];
            }else{
              MYTIMER(sample,i) = mytime[i+1]-mytime[i];
            }
          }
        }
      }
    }
  }
  
  
  /// Compute BIC
  arma::vec logliky = savelikelihood.col(0);
  double loglikymax = logliky.max();
  //number of parameters: J*H+Homega beta coefficients and N
  double BIC = (J*(H)+1)*log(n) - 2*loglikymax; 
  
  
  //Transpose savePSICOUNTS and savePHIomega
  arma::mat savePSICOUNTStransposed = savePSICOUNTS.t();
  arma::mat savePHIomegatransposed = savePHIomega.t();
  
  
  //Add names PSIMATRIX
  CharacterVector xnames(Hsimple);
  for(int h=0; h<xnames.length(); h++){
    xnames(h) = as<std::string>(yxnamesALL[J+h]);
  }
  Rcpp::List PSIMATRIX_LISTX(Hsimple);
  for(int h=0; h<Hsimple; h++){
    PSIMATRIX_LISTX[h] = PSIMATRIX_DF[J+h];
  }
  PSIMATRIX_LISTX.attr("names")=xnames;
  Rcpp::DataFrame PSIMATRIX_DFX(PSIMATRIX_LISTX);
  
  if(coefprior=="normal"){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("Betacolnames") = betacolnames,
                              Rcpp::Named("PSIMATRIX") = PSIMATRIX_DFX,
                              Rcpp::Named("PSICOUNTS") = savePSICOUNTStransposed,
                              Rcpp::Named("PHIomega") = savePHIomegatransposed,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("probcaptured") = probcaptured,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC,
                              Rcpp::Named("MYTIMER")=MYTIMER);
    
  }else if(coefprior=="lasso"){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("Betacolnames") = betacolnames,
                              Rcpp::Named("PSIMATRIX") = PSIMATRIX_DFX,
                              Rcpp::Named("PSICOUNTS") = savePSICOUNTStransposed,
                              Rcpp::Named("PHIomega") = savePHIomegatransposed,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("lambda2") = saveinvlambda2_betasampling,
                              Rcpp::Named("probcaptured") = probcaptured,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC,
                              Rcpp::Named("MYTIMER")=MYTIMER);
    
  }else if(coefprior=="normalhier"){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("Betacolnames") = betacolnames,
                              Rcpp::Named("PSIMATRIX") = PSIMATRIX_DFX,
                              Rcpp::Named("PSICOUNTS") = savePSICOUNTStransposed,
                              Rcpp::Named("PHIomega") = savePHIomegatransposed,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("probcaptured") = probcaptured,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC,
                              Rcpp::Named("MYTIMER")=MYTIMER);
    
  }else if(coefprior=="normalhiervaronly"){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("ymatrix") = ymatrix,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("Betacolnames") = betacolnames,
                              Rcpp::Named("PSIMATRIX") = PSIMATRIX_DFX,
                              Rcpp::Named("PSICOUNTS") = savePSICOUNTStransposed,
                              Rcpp::Named("PHIomega") = savePHIomegatransposed,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("priorBdiag") = savepriorBdiag,
                              Rcpp::Named("probcaptured") = probcaptured,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC,
                              Rcpp::Named("MYTIMER")=MYTIMER);
    
  }
  

}


