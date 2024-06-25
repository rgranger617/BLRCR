// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "RcppArmadillo.h"
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]] 


//Create a function for sampling from Dirichlet
arma::rowvec mydirichlet(arma::rowvec alphas){
  int K = alphas.n_elem;
  arma::rowvec thetas(K);
  double sumthetas = 0;
  for(int k=0; k<K; ++k){
    thetas[k]=R::rgamma(alphas[k],1.0);
    sumthetas += thetas[k];
  }
  thetas = thetas/sumthetas;
  return(thetas);
}


//Create a function for computing the log density of a multivariate normal
arma::vec logdmvNORM(arma::mat X,arma::rowvec MU,arma::mat SIGMA){
  int n = X.n_rows;
  double firstpart=-0.5*arma::log_det_sympd(SIGMA*2*arma::datum::pi);
  arma::mat secondpart=X.each_row()-MU;
  arma::vec thirdpart(n);
  for(int i=0; i<n; i++){
    arma::mat likval = -0.5*secondpart.row(i)*inv_sympd(SIGMA,arma::inv_opts::allow_approx)*secondpart.row(i).t();
    thirdpart(i)=likval(0);
  }
  arma::vec result = thirdpart+firstpart;
  
  
  return result;
}

//Create a function for getting unique rows
//https://stackoverflow.com/questions/37143283/finding-unique-rows-in-armamat
template <typename T>
inline bool rows_equal(const T& lhs, const T& rhs, double tol = 0.00000001) {
  return arma::approx_equal(lhs, rhs, "absdiff", tol);
}

// [[Rcpp::export]]
arma::mat unique_rows(const arma::mat& x) {
  unsigned int count = 1, i = 1, j = 1, nr = x.n_rows, nc = x.n_cols;
  arma::mat result(nr, nc);
  result.row(0) = x.row(0);
  
  for ( ; i < nr; i++) {
    bool matched = false;
    if (rows_equal(x.row(i), result.row(0))) continue;
    
    for (j = i + 1; j < nr; j++) {
      if (rows_equal(x.row(i), x.row(j))) {
        matched = true;
        break;
      }
    }
    
    if (!matched) result.row(count++) = x.row(i);
  }
  
  return result.rows(0, count - 1);
}


//' Semi-Parametric Bayesian Logistic Regression Capture-Recapture
//' 
//' This function computes MCMC samples of N using the the Bayesian Logistic
//' Regression Capture-Recapture model.
//' 
//' @param formula an object of class "formula".
//' @param df Data containing capture histories and individual covariates.
//' 
//' @param covmethod specify a method for simulating the covariates.  
//' Options include "bootstrap", "mixture", and "empirical".
//' @param coefprior specify a prior for the logistic regression coefficients.
//' Options include "normal" and "horseshoe".
//' 
//' @param bprior vector of hyperparameters for the mean of the coefficient prior. If a single 
//' value is used, it will apply this value for all coefficients. If "normalhier" is selected, it will use
//' these values as the coefficient prior on b.
//' @param Bprior value to assign the diagonals of the covariance matrix B for all variables.
//' 
//' @param Homega the number of latent intercepts
//' @param Kstar an integer of maximum number of mixture classes. This only needs to be specified
//' if the covmethod is "mixture".
//' 
//' @param samples the number of MCMC samples to draw from the posterior.
//' @param burnin the number of MCMC samples to "burn-in."
//' 
//' @return List with CR results. 
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
//' CRresults = BLRCRsolver(myformula,myCRdata,covmethod="mixture",coefprior="normal",
//'                         Homega=10,Kstar=10, samples=1000)                     
//' 
//' @export 
// [[Rcpp::export]]
Rcpp::List BLRCRsolver(Formula formula, DataFrame df, 
                       String covmethod="bootstrap", 
                       String coefprior="normal",
                       arma::vec bprior=0,
                       double Bprior=1,
                       int Homega=1,int Kstar=1, 
                       int samples=1000, int burnin=0){
  
  //Get functions from Rpackages
  Rcpp::Environment pkg1 = Rcpp::Environment::namespace_env("BayesLogit");
  Rcpp::Function rpg = pkg1["rpg"];
  Rcpp::Environment stats_env("package:stats");
  Rcpp::Function model_frame = stats_env["model.frame"];
  Rcpp::Function model_matrix = stats_env["model.matrix"];
  
  /////////////////
  //Safety Checks//
  /////////////////
  if(Homega<1){
    Rcpp::Rcout << "Error: The number of latent classes (Homega) cannot be less than 1." << std::endl;
    return(0);
  }
  if(Kstar<1){
    Rcpp::Rcout << "Error: The number of latent classes (Kstar) cannot be less than 1." << std::endl;
    return(0);
  }
  if(samples<1){
    Rcpp::Rcout << "Error: The number of samples cannot be less than 1" << std::endl;
    return(0);
  }
  
  ////////////////////////////////////////////////////////////////////////////
  /// Create a simple dataframe, does not include interactions in formula ////
  ////////////////////////////////////////////////////////////////////////////
  
  Rcpp::DataFrame df_clone = clone(df); //do this so the original dataset is not overwritten
  Rcpp::DataFrame df_new = model_frame(Rcpp::_["formula"] = formula, Rcpp::_["data"] = df_clone, Rcpp::_["na.action"] = R_NilValue);
  Rcpp::List checkterms = df_new.attr("terms");
  int intercept = checkterms.attr("intercept"); //Include Intercept? 1=yes, 0=no
  if(intercept==0){
    Rcpp::Rcout << "Error: You must include an intercept." << std::endl;
    return(0);
  }
  
  //Create ymatrix
  int n = df_new.nrow(); //get observed sample size
  arma::mat ymatrix = df_new[0]; //this will ultimately have unobserved added to it
  arma::mat yobs = ymatrix; //this will always be just the observed
  int J = yobs.n_cols; //get number of lists
  if(J<2){
    Rcpp::Rcout << "Error: You must have at least two (J=2) lists" << std::endl;
    return(0);
  }
  if(yobs.has_nan()){ //Check for missing capture history, return error
    Rcpp::Rcout << "Cannot have NA for a capture occasion." << std::endl;
    return(0);
  }

  
  
  //////////////////////////////////////
  ///Check for missing Covariate data///
  //////////////////////////////////////
  //Note: this checks over the raw dataset for missing values//
  //Note: we do not have to worry about capture histories being missing
  //as they would have been detected in the ymatrix above and returned an error
  int JHsimple = df_clone.ncol();
  Rcpp::List missingindex(JHsimple); //list of missing indices
  bool MissingTrue=FALSE; //creates a flag for missing data
  if(JHsimple>J){
    arma::mat TrackMissingCovariates(n,JHsimple,arma::fill::zeros);
    for(int jh=0; jh<JHsimple; jh++){
      Rcpp::NumericVector mycovariate = df_clone[jh]; 
      Rcpp::LogicalVector mymissingcovs = Rcpp::is_na(mycovariate);
      for(int i=0; i<n; i++){
        bool mycovval = mymissingcovs[i];
        if(mycovval==TRUE){
          TrackMissingCovariates.submat(i,jh,i,jh) = 1;
        }
      }
      MissingTrue = accu(TrackMissingCovariates)>0;
      for(int jh=0; jh<JHsimple; jh++){
        missingindex[jh] = arma::find(TrackMissingCovariates.col(jh) > 0);
      }
    }
  }
  
  //Impute missing values with randomly selected available covariate values
  if(MissingTrue==TRUE){
    for(int jh=0; jh<JHsimple; jh++){
      Rcpp::NumericVector mycolumn = df_clone[jh];
      Rcpp::NumericVector mydfcolumnNONA = Rcpp::na_omit(mycolumn);
      int nNONA = mydfcolumnNONA.size();
      IntegerVector samframe=seq_len(nNONA);
      arma::ivec indexofmisvals = missingindex[jh];
      int nMIS = indexofmisvals.size();
      for(int i=0; i<nMIS; i++){
        int misvalindex = indexofmisvals[i]; //get location of missing value
        int propxindex = RcppArmadillo::sample(samframe, 1, TRUE)(0) - 1; //get location of proposed value
        double propx = mydfcolumnNONA[propxindex]; //get proposed value
        mycolumn[misvalindex] = propx;
      }
    }
  }  
  
  
  //Create xmatrix
  //Note: this creates the full X matrix, including an intercept
  //and latent group membership columns
  Rcpp::NumericMatrix createdXmatrix = model_matrix(Rcpp::_["object"] = formula, Rcpp::_["data"] = df_clone);
  arma::mat createdXmatrixARMA = as<arma::mat>(wrap(createdXmatrix));
  int H = createdXmatrix.ncol()-1; //get number of covariates, use formula, subtract 1 to not count intercept
  arma::mat xmatrix(n,H+Homega,arma::fill::zeros);
  xmatrix.submat(0,0,n-1,H) = createdXmatrixARMA;
  
  
  
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Insert Priors Here /////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                                                                                            //////
  //Set prior for coefficients                                                                              //////
  //Normal//////                                                                                            //////
  int bpriorlength = bprior.n_elem;                                                                         //////
  arma::vec priorb(H+Homega,arma::fill::ones);                                                              //////
  if(bpriorlength==1){                                                                                      //////
    priorb = priorb*bprior;                                                                                 //////
  }else if(bpriorlength==H+Homega){                                                                         //////
    priorb = bprior;                                                                                        ////// 
  }else{                                                                                                    //////
    Rcpp::Rcout << "Error: bprior is of improper length" << std::endl;                                      //////
    return(0);                                                                                              //////
  }                                                                                                         //////
  arma::mat priorB = arma::eye(H+Homega,H+Homega)*Bprior;                                                   //////
                                                                                                            //////
  //Horseshoe                                                                                               //////
  arma::cube LAMBDAVARhorse(H+Homega-1,H+Homega-1,J);                                                       //////
  arma::vec invxi2horse(J,arma::fill::ones);                                                                //////
  arma::vec invtau2horse(J,arma::fill::ones);                                                               //////
  arma::mat invlambda2horse(J,H+Homega-1,arma::fill::ones);                                                 //////
  arma::mat invnu2horse(J,H+Homega-1,arma::fill::ones);                                                     //////
  if(H>0){                                                                                                  //////
    for(int j=0; j<J; j++){                                                                                 //////
      LAMBDAVARhorse.subcube(0,0,j,H+Homega-2,H+Homega-2,j) = arma::eye(H+Homega-1,H+Homega-1);             //////
    }                                                                                                       //////
  }                                                                                                         //////
                                                                                                            //////
  //Set priors for alpha_omega                                                                              //////
  double aomega = 0.25;                                                                                     //////
  double bomega = 0.25;                                                                                     //////
                                                                                                            //////
  //Set prior for mixture distribution                                                                      //////
  arma:: vec priorMU0;                                                                                      //////
  arma::mat priorlambda0;                                                                                   //////
  double priorkappa0=1;                                                                                     //////
  double priornu0=H+1;//should be bigger than H                                                             //////
  double aalpha=0.25;                                                                                       //////
  double balpha=0.25;                                                                                       //////
  if(H>0){                                                                                                  //////
    priorMU0 = arma::vec(H,arma::fill::zeros);                                                              //////
    priorlambda0 = arma::eye(H,H);                                                                          //////
  }                                                                                                         //////
  //Set prior for Bayes Bootstrap                                                                           //////
  double alphabayesbootstrap = 0;                                                                           //////  
                                                                                                            //////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //////////////////////////////////
  //Compute some useful constants //
  //////////////////////////////////
  arma::mat Binv = inv_sympd(priorB);
  arma::mat Bb = Binv*priorb;
  arma::mat priorlambda0inv;
  if(H>0){
    priorlambda0inv = inv_sympd(priorlambda0); 
  }
  
  
  ///////////////////////////////
  ////Variables to be saved//////
  ///////////////////////////////
  arma::vec saveN(samples);
  arma::mat saveXmis(samples,H);
  arma::mat savePHIomega(samples,Homega);
  arma::cube saveBETA(J,H+Homega,samples);
  arma::mat saveOMEGAlab(n,Homega,arma::fill::zeros);
  arma::vec savealphaomega(samples);
  arma::vec savealphaPIK(samples);
  arma::mat savelikelihood(samples,2,arma::fill::zeros);
  
  
  
  ///////////////////
  //Initializations//
  ///////////////////
  
  //Initialize beta and N//
  arma::mat beta(J,H+Homega,arma::fill::value(0.0));//initialize beta to all 0s
  int n0 = 0; //initialize n0 to be 0 *DO NOT CHANGE*
  int N = n+n0;
  //Try to initialize beta and N with conditional BLRCR value
  if(H>0){
    arma::vec condpriorb(H+1,arma::fill::zeros);
    arma::mat condpriorB = arma::eye(H+1,H+1);
    Rcpp::List condBLRCRsolver(arma::mat y, arma::mat x, 
                               arma::vec priorb, arma::mat priorB,
                               double gradparam, int prior,
                               double tol,int maxiter,
                               Rcpp::Nullable<Rcpp::NumericMatrix> initbeta = R_NilValue);
    double mygradparam = .01;
    Rcpp::Rcout << "Attempting to initialize beta matrix with condBLRCR."  << std::endl;
    Rcpp::List betainit = condBLRCRsolver(yobs,xmatrix.submat(0,1,n-1,H),condpriorb,condpriorB,mygradparam,1,1e-6,1e4);
    int convergence = betainit[2];//checks convergence
    if(convergence==1){
      Rcpp::Rcout << "Success!"  << std::endl;
      arma::mat condbeta = betainit[1];
      int condN = betainit[0];
      beta.submat(0,0,J-1,H) = condbeta;
      N = condN;
      n0 = N-n;
    }

  }
  
  //Fill in Missing Y and X values (just use empirical method)
  if(H>0){
    arma::mat ymis(n0,J,arma::fill::zeros);
    arma::mat xmis(n0,H+Homega);
    int maxasam=10000;
    IntegerVector samframe=seq_len(n);
    for(int i=0; i<n0; i++){
      for(int asam=0; asam<maxasam; asam++){
        arma::rowvec propX(H+Homega,arma::fill::zeros); //all covariates initialized to 0
        propX(0) = 1.0; //set intercept equal to 1
        int propXindex = RcppArmadillo::sample(samframe, 1, TRUE)(0); //sample x from empirical dist
        propX(arma::span(1,H)) = xmatrix(propXindex-1,arma::span(1,H)); 
        double acceptanceprob = prod(1-(1.0/(1.0+exp(-propX*beta.t()))));
        int accept = Rcpp::rbinom(1,1,acceptanceprob)(0);
        if(accept==1){
          xmis.row(i)=propX;
          break;
        }
        if(asam==maxasam-1){
          return("Error: Failed to find Missing Values");
        }
      }
    }
    xmatrix = join_cols(xmatrix.submat(0,0,n-1,H+Homega-1),xmis); 
    ymatrix = join_cols(yobs,ymis);
  }
  
  
  //Latent grouping initialization
  double alphaomega = 1.0;
  arma::rowvec PHIomega(Homega,arma::fill::value(1.0/(Homega)));
  if(Homega==1){
    PHIomega=1;
  }
  if(Homega>1){
    arma::mat logprobomega(N,Homega,arma::fill::zeros);
    for(int w=0; w<Homega; w++){
      arma::mat omegamat(N,Homega-1,arma::fill::zeros);
      arma::mat allOne(N, 1, arma::fill::ones);
      omegamat.insert_cols(w, allOne);
      omegamat.shed_col(0); //removes first row (base class)
      arma::mat XOMEGA = join_rows(xmatrix.cols(0,H),omegamat);
      arma::mat lambda = (1/(1+exp(-XOMEGA*beta.t())));
      arma::mat logprobomegavec = arma::sum(arma::log(lambda)%ymatrix + arma::log(1.0-lambda)%(1.0-ymatrix),1)+ log(PHIomega(w));
      logprobomega.col(w) = logprobomegavec;
    } 
    for(int i=0; i<N; i++){
      arma::rowvec logomegarow = logprobomega.row(i);
      arma::rowvec omegaprobrowARMAnum = exp(logomegarow - logomegarow.max());
      double omegaprobrowARMAsum = sum(omegaprobrowARMAnum);
      arma::rowvec omegaprobrowARMA = omegaprobrowARMAnum/omegaprobrowARMAsum;
      NumericVector omegaprobrow = as<NumericVector>(wrap(omegaprobrowARMA));
      IntegerVector omegarowvec(Homega);
      R::rmultinom(1,omegaprobrow.begin(),Homega,omegarowvec.begin());
      arma::rowvec omegarow = as<arma::rowvec>(wrap(omegarowvec)); 
      xmatrix.submat(i,1+H,i,H+Homega-1) = omegarow.cols(1,Homega-1);
    }
  }
  
  //////////////////////////////
  //Covariate Initializations //
  //////////////////////////////
  arma::mat MUK;
  arma::cube SIGMAK;
  if(H>0){
    SIGMAK = arma::cube(H,H,Kstar);
  }
  arma::vec PIK(Kstar);
  arma::ivec NK(Kstar);
  double alpha;
  arma::mat PSIMATRIX;
  arma::mat PSICOUNTS;
  arma::mat PSICOUNTSobs;
  arma::rowvec PSIPROBS;
  arma::vec FIXEDMEAN;
  arma::mat FIXEDCOVAR;
  
  //Initialize if Mixture Covariate Method is selected//
  if(H>0){
    if(covmethod=="mixture"){
      
      //Initialize Alpha
      alpha=aalpha/balpha;
      
      //Initialize MUK with Kmeans clustering
      arma::mat MUKtranspose;
      arma::mat Xnoconst = xmatrix.submat(0,1,N-1,H);
      bool status = kmeans(MUKtranspose, 
                           Xnoconst.t(), 
                           Kstar, arma::random_subset, 1000, false); //do not use intercept or latent variables
      if(status == false){
        Rcpp::Rcout << "Initialization of MUs Failed " << std::endl;
        return(0);
      }
      MUK=MUKtranspose.t();
      
      //Initialize Z by Euclidean Distance
      arma::vec dist(Kstar);
      arma::ivec Zlab(N);
      for(int i=0; i<N; i++){
        for(int k=0; k<Kstar; k++){
          dist(k) = accu(square(Xnoconst.row(i)-MUK.row(k)));
        }
        Zlab(i)=dist.index_min()+1;
      }
      
      //Initialize NK and PIK
      for(int k=0; k<Kstar; k++){
        arma::uvec position = arma::find(Zlab==k+1);
        NK(k) = position.n_rows;
        PIK(k) = (float)NK(k)/(float)N;
        arma::mat subsetX = Xnoconst.rows(position);
        if(NK(k)<H+1){ //condition necessary or you end up with a nonsingular covariance matrix
          SIGMAK.slice(k)=priorlambda0; 
        }else{
          SIGMAK.slice(k)=arma::cov(Xnoconst.rows(position))+priorlambda0; 
        }
      }
      
      //Sort Classes in Ascending order by NK
      arma::uvec NKindices = sort_index(NK, "descend");
      NK = sort(NK,"descend");
      PIK = sort(PIK,"descend");
      MUK = MUK.rows(NKindices);
      SIGMAK = SIGMAK.slices(NKindices);
      
      //Initialize if Empirical Covariate Method is selected //
    }else if(covmethod=="empirical"){
      //Could track discretely, but easier to sample uniformly from previous iteration
      
      //Initialize if Bootstrap Covariate Method is selected //
    }else if(covmethod=="bootstrap"){
      PSIMATRIX = unique_rows(xmatrix.cols(1,H));
      unsigned int Kunique = PSIMATRIX.n_rows;
      PSICOUNTS.set_size(Kunique,1); //set size to be number of unique values
      PSICOUNTS.fill(alphabayesbootstrap); //initialize counts at the prior
      for(int i=0; i<n; i++){
        unsigned int myindex = index_min(sum(abs(PSIMATRIX.each_row()-xmatrix(arma::span(i,i),arma::span(1,H))),1));
        PSICOUNTS(myindex)++;
      }
      PSICOUNTSobs = PSICOUNTS; //this is the prior + observed value counts
      arma::rowvec PSICOUNTSRV = PSICOUNTS.t();
      PSIPROBS = mydirichlet(PSICOUNTSRV); //note PSICOUNTS includes the prior
    }else if(covmethod=="fixed"){
      FIXEDMEAN = arma::vec(H,arma::fill::zeros);; //INSERT MEAN HERE
      FIXEDCOVAR = arma::eye(H,H);; //INSERT COVARIANCE HERE
    }
  }
  
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  //////////////////////////
  // Gibbs Sampling Steps //
  //////////////////////////
  for(int sample=-burnin; sample<samples; sample++){
    //if(sample>=0){
    //if((sample+1) % 100 == 0){
      Rcpp::Rcout << "Iteration: " << sample+1 << std::endl;
      Rcpp::Rcout << "N: " << N<< std::endl;
    //}
    //}
    ///////////////
    //Sample Beta//
    ///////////////
    if(coefprior=="normal"){
      arma::mat psi=xmatrix*beta.t();
      arma::mat kappa = ymatrix-0.5;
      for(int j=0; j<J; j++){
        NumericVector wnumericvector=rpg(N, 1.0, psi.col(j));
        arma::vec w=as<arma::vec>(wrap(wnumericvector));
        arma::mat Vw = inv_sympd(xmatrix.t()*diagmat(w)*xmatrix + Binv);
        arma::mat Mw = Vw*(xmatrix.t()*kappa.col(j)+Bb);
        beta.row(j) = arma::mvnrnd(Mw,Vw).t();
      }
    }else if(coefprior=="horseshoe"){
      arma::mat psi=xmatrix*beta.t();
      arma::mat kappa = ymatrix-0.5;
      arma::mat xnoconst = xmatrix.cols(1,H+Homega-1);
      for(int j=0; j<J; j++){ 
        arma::mat lambdavarj = LAMBDAVARhorse.subcube(0,0,j,H+Homega-2,H+Homega-2,j);
        NumericVector invw2numericvector=rpg(N, 1.0, psi.col(j));
        arma::vec invw2=as<arma::vec>(wrap(invw2numericvector));
        arma::mat invOMEGA=diagmat(invw2);
        arma::vec zhorseshoe = (1/invw2)%kappa.col(j);
        double sigma20horse = 1/sum(invw2); //intercept
        double mubar0horse = sigma20horse*sum(invw2%(zhorseshoe-xnoconst*beta.submat(j,1,j,H+Homega-1).t())); //intercept
        double beta0horse = R::rnorm(mubar0horse,sigma20horse); //intercept
        arma::mat invAP = inv_sympd(xnoconst.t()*invOMEGA*xnoconst+inv_sympd(lambdavarj,arma::inv_opts::allow_approx),arma::inv_opts::allow_approx); //slope
        arma::mat MUbarhorse = invAP*xnoconst.t()*invOMEGA*(zhorseshoe-beta0horse); //slope
        arma::mat betahorse = mvnrnd(MUbarhorse,invAP).t(); //slope
        beta(j,0) = beta0horse; //update intercept
        beta.submat(j,1,j,H+Homega-1) = betahorse; //update slopes
        
        invxi2horse(j) = R::rgamma(1,1/(1+invtau2horse(j)));
        invtau2horse(j) = R::rgamma(0.5*(H+Homega),1/(invxi2horse(j)+0.5*sum(pow(betahorse%invlambda2horse.row(j),2))));
        for(int h=0; h<(H+Homega-1); h++){ 
          invnu2horse(j,h) = R::rexp(1+invlambda2horse(j,h));
          invlambda2horse(j,h) = R::rexp(invnu2horse(j,h)+pow(beta(j,h),2)*invtau2horse(j)*0.5);
        }
        LAMBDAVARhorse.subcube(0,0,j,H+Homega-2,H+Homega-2,j) = (1/invtau2horse(j))*diagmat(1/invlambda2horse.row(j)); //update LAMBDAVAR
        
      }
      
      
      
    }else{
      Rcpp::Rcout << "Error: Coefficient Prior Method does not exist."<< std::endl;
      return(0);
    }
    
    
    
    ////////////////////
    //Sample n0 and N //
    ////////////////////
    
    //Compute rho (marginal probability an individual is missing)
    double rho=0;
    
    //special case of no covariates
    if(H==0){
      int simxsamplenum=1000*(1+Homega);
      arma::mat simX(simxsamplenum,Homega,arma::fill::ones);
      NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      for(int i=0; i<simxsamplenum; i++){
        if(Homega>1){
          IntegerVector omegarowvec(Homega);
          R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
          omegarowvec.erase(0);
          simX(i,arma::span(H+1,Homega-1)) = as<arma::rowvec>(wrap(omegarowvec));
        }
      }
      arma::mat lambda = 1.0/(1.0+exp(-simX*beta.t()));
      arma::vec probmis(simxsamplenum);
      for(int i=0; i<simxsamplenum; i++){
        probmis(i) = prod(1.0-lambda.row(i));
      }
      rho=mean(probmis);
      
      //mixture distribution  
    }else if(covmethod=="mixture"){
      
      //Use Monte Carlo within MCMC to get Rho
      int simxsamplenum=1000*(1+Homega+H);
      IntegerVector Zsim(simxsamplenum);
      IntegerVector samframe=seq_len(Kstar);
      Zsim = RcppArmadillo::sample(samframe, simxsamplenum, TRUE, PIK);
      arma::mat simX(simxsamplenum,H+Homega,arma::fill::ones);
      NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      for(int i=0; i<simxsamplenum; i++){
        simX(i,arma::span(1,H)) = mvnrnd(MUK.row(Zsim(i)-1).t(),SIGMAK.slice(Zsim(i)-1)).t();
        if(Homega>1){
          IntegerVector omegarowvec(Homega);
          R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
          omegarowvec.erase(0);
          simX(i,arma::span(H+1,Homega+H-1)) = as<arma::rowvec>(wrap(omegarowvec));
        }
      }
      arma::mat lambda = 1.0/(1.0+exp(-simX*beta.t()));
      arma::vec probmis(simxsamplenum);
      for(int i=0; i<simxsamplenum; i++){
        probmis(i) = prod(1.0-lambda.row(i));
      }
      rho=mean(probmis);
      
      //empirical distribution
    }else if(covmethod=="empirical"){
      arma::mat lambda = 1.0/(1.0+exp(-xmatrix*beta.t()));
      arma::vec probmis(N);
      for(int i=0; i<N; i++){
        probmis(i) = prod(1.0-lambda.row(i));
      }
      rho=mean(probmis);
    }else if(covmethod=="bootstrap"){
      int simxsamplenum=1000*(1+Homega+H);
      IntegerVector indexlocations(simxsamplenum);
      IntegerVector samframe=seq_len(PSIMATRIX.n_rows);
      indexlocations = RcppArmadillo::sample(samframe, simxsamplenum, TRUE, PSIPROBS.t());
      arma::mat simX(simxsamplenum,H+Homega,arma::fill::ones);
      NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      for(int i=0; i<simxsamplenum; i++){
        simX(i,arma::span(1,H)) = PSIMATRIX.row(indexlocations(i)-1);
        if(Homega>1){
          IntegerVector omegarowvec(Homega);
          R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
          omegarowvec.erase(0);
          simX(i,arma::span(H+1,Homega+H-1)) = as<arma::rowvec>(wrap(omegarowvec));
        }
      }
      arma::mat lambda = 1.0/(1.0+exp(-simX*beta.t()));
      arma::vec probmis(simxsamplenum);
      for(int i=0; i<simxsamplenum; i++){
        probmis(i) = prod(1.0-lambda.row(i));
      }
      rho=mean(probmis);  
      
    }else if(covmethod=="fixed"){
      //Use Monte Carlo within MCMC to get Rho
      int simxsamplenum=1000*(1+Homega+H);
      arma::mat simX(simxsamplenum,H+Homega,arma::fill::ones);
      NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      for(int i=0; i<simxsamplenum; i++){
        simX(i,arma::span(1,H)) = mvnrnd(FIXEDMEAN,FIXEDCOVAR).t();
        if(Homega>1){
          IntegerVector omegarowvec(Homega);
          R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
          omegarowvec.erase(0);
          simX(i,arma::span(H+1,Homega+H-1)) = as<arma::rowvec>(wrap(omegarowvec));
        }
      }
      arma::mat lambda = 1.0/(1.0+exp(-simX*beta.t()));
      arma::vec probmis(simxsamplenum);
      for(int i=0; i<simxsamplenum; i++){
        probmis(i) = prod(1.0-lambda.row(i));
      }
      rho=mean(probmis);
    }
    
    //Sample N
    n0 = Rcpp::rnbinom( 1, n, 1-rho )(0);
    int prevN=N; //useful for empirical distribution covariates
    N = n+n0;
    
    
    //////////////////////////////////////////
    // Sample Unobserved Covariate X Values //
    // Note: Does not include missing covs  //
    //       of the observed individuals    //
    //////////////////////////////////////////
    arma::mat xmis(n0,H+Homega);
    if(H==0){
      int maxasam=10000;
      NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      IntegerVector samframe=seq_len(Kstar);
      for(int i=0; i<n0; i++){
        for(int asam=0; asam<maxasam; asam++){
          arma::rowvec propX(Homega,arma::fill::ones);
          if(Homega>1){
            IntegerVector omegarowvec(Homega);
            R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
            omegarowvec.erase(0);
            propX(arma::span(1,Homega-1)) = as<arma::rowvec>(wrap(omegarowvec));
          }
          double acceptanceprob = prod(1-(1.0/(1.0+exp(-propX*beta.t()))));
          int accept = Rcpp::rbinom(1,1,acceptanceprob)(0);
          if(accept==1){
            xmis.row(i)=propX;
            break;
          }
          if(asam==maxasam-1){
            return("Error: Failed to find Missing Values");
          }
        }
      }
    }else if(covmethod=="mixture"){
      int maxasam=10000;
      NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      IntegerVector samframe=seq_len(Kstar);
      for(int i=0; i<n0; i++){
        for(int asam=0; asam<maxasam; asam++){
          int propZ = RcppArmadillo::sample(samframe, 1, TRUE, PIK)(0);
          arma::rowvec propX(H+Homega,arma::fill::ones);
          propX(arma::span(1,H)) = mvnrnd(MUK.row(propZ-1).t(),SIGMAK.slice(propZ-1)).t();
          if(Homega>1){
            IntegerVector omegarowvec(Homega);
            R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
            omegarowvec.erase(0);
            propX(arma::span(H+1,Homega+H-1)) = as<arma::rowvec>(wrap(omegarowvec));
          }
          double acceptanceprob = prod(1-(1.0/(1.0+exp(-propX*beta.t()))));
          int accept = Rcpp::rbinom(1,1,acceptanceprob)(0);
          if(accept==1){
            xmis.row(i)=propX;
            break;
          }
          if(asam==maxasam-1){
            return("Error: Failed to find Missing Values");
          }
        }
      }
    }else if(covmethod=="empirical"){
      //discrete method
      arma::mat lambda = 1.0/(1.0+exp(-xmatrix*beta.t()));
      arma::vec probmis(prevN);
      for(int i=0; i<prevN; i++){
        probmis(i) = prod(1.0-lambda.row(i));
      }
      IntegerVector indexlocations(n0);
      IntegerVector samframe=seq_len(prevN);
      indexlocations = RcppArmadillo::sample(samframe, n0, TRUE, probmis);
      for(int i=0; i<n0; i++){
        int mymisindex = indexlocations(i)-1; //need to -1 as our index seq is 1:N, but C indexing is 0:N-1
        xmis.row(i) = xmatrix.row(mymisindex);
      }
      //rejection method
      //int maxasam=10000;
      //NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      //IntegerVector samframe=seq_len(prevN);
      //for(int i=0; i<n0; i++){
      //  for(int asam=0; asam<maxasam; asam++){
      //    int propXindex = RcppArmadillo::sample(samframe, 1, TRUE)(0);
      //    arma::rowvec propX(H+Homega,arma::fill::ones);
      //    propX(arma::span(1,H)) = xmatrix(propXindex-1,arma::span(1,H));
      //    if(Homega>1){
      //      IntegerVector omegarowvec(Homega);
      //      R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
      //      omegarowvec.erase(0);
      //      propX(arma::span(H+1,Homega+H-1)) = as<arma::rowvec>(wrap(omegarowvec));
      //    }
      //    double acceptanceprob = prod(1-(1.0/(1.0+exp(-propX*beta.t()))));
      //    int accept = Rcpp::rbinom(1,1,acceptanceprob)(0);
      //    if(accept==1){
      //      xmis.row(i)=propX;
      //      break;
      //    }
      //    if(asam==maxasam-1){
      //      return("Error: Failed to find Missing Values");
      //    }
      //  }
      //}
    }else if(covmethod=="bootstrap"){
      //rejection method
      int maxasam=10000;
      NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      IntegerVector samframe=seq_len(PSIMATRIX.n_rows);
      for(int i=0; i<n0; i++){
        for(int asam=0; asam<maxasam; asam++){
          int indexlocation = RcppArmadillo::sample(samframe, 1, TRUE, PSIPROBS.t())(0);
          arma::rowvec propX(H+Homega,arma::fill::ones);
          propX(arma::span(1,H)) = PSIMATRIX.row(indexlocation-1);;
          if(Homega>1){
            IntegerVector omegarowvec(Homega);
            R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
            omegarowvec.erase(0);
            propX(arma::span(H+1,Homega+H-1)) = as<arma::rowvec>(wrap(omegarowvec));
          }
          double acceptanceprob = prod(1-(1.0/(1.0+exp(-propX*beta.t()))));
          int accept = Rcpp::rbinom(1,1,acceptanceprob)(0);
          if(accept==1){
            xmis.row(i)=propX;
            break;
          }
          if(asam==maxasam-1){
            return("Error: Failed to find Missing Values");
          }
        }
      }
    }else if(covmethod=="fixed"){
      int maxasam=10000;
      NumericVector PHIomegaNV = as<NumericVector>(wrap(PHIomega));
      for(int i=0; i<n0; i++){
        for(int asam=0; asam<maxasam; asam++){
          arma::rowvec propX(H+Homega,arma::fill::ones);
          propX(arma::span(1,H)) = mvnrnd(FIXEDMEAN,FIXEDCOVAR).t();
          if(Homega>1){
            IntegerVector omegarowvec(Homega);
            R::rmultinom(1,PHIomegaNV.begin(),Homega,omegarowvec.begin());
            omegarowvec.erase(0);
            propX(arma::span(H+1,Homega+H-1)) = as<arma::rowvec>(wrap(omegarowvec));
          }
          double acceptanceprob = prod(1-(1.0/(1.0+exp(-propX*beta.t()))));
          int accept = Rcpp::rbinom(1,1,acceptanceprob)(0);
          if(accept==1){
            xmis.row(i)=propX;
            break;
          }
          if(asam==maxasam-1){
            return("Error: Failed to find Missing Values");
          }
        }
      }
    }
    
    ///////////////////////////////
    //Update X and Y Matrices /////
    ///////////////////////////////
    arma::mat ymis(n0,J,arma::fill::zeros);
    xmatrix = join_cols(xmatrix.submat(0,0,n-1,H+Homega-1),xmis); 
    ymatrix = join_cols(yobs,ymis);
    
    
    ///////////////////////////////////////////////////////
    // Sample Missing Covariates of Observed Individuals //
    ///////////////////////////////////////////////////////
    //For now, we only have the empirical programmed //
    //Assumes independence between covariates /////////
    ///////////////////////////////////////////////////
    
    if(MissingTrue==TRUE){
      for(int jh=0; jh<JHsimple; jh++){
        arma::ivec indexofmisvals = missingindex[jh];
        int nMIS = indexofmisvals.size();
        for(int i=0; i<nMIS; i++){
          int misvalindex = indexofmisvals[i]; //get location of missing value
          arma::rowvec trueY = ymatrix.row(misvalindex);
          arma::rowvec propX = xmatrix.row(misvalindex);
          int maxasam=1e7;
          for(int asam=0; asam<maxasam; asam++){
            if(covmethod=="bootstrap"){
              IntegerVector samframe=seq_len(PSIMATRIX.n_rows); //this was recently moved inside the loop, not sure why it was outside, I hope this is ok
              int propxindex = RcppArmadillo::sample(samframe, 1, TRUE, PSIPROBS.t())(0)-1;
              double propxreplacement = PSIMATRIX(propxindex,jh);
              propX(jh+1) = propxreplacement;
              arma::rowvec proplambda = (1.0/(1.0+exp(-propX*beta.t())));
              double acceptanceprob = 1.0;
              for(int j=0; j<J; j++){ //I think this is faster than matrix multiplying w/ exponents
                if(trueY(j)==1){
                 acceptanceprob = acceptanceprob*proplambda(j);
                }else{
                  acceptanceprob = acceptanceprob*(1-proplambda(j));
                }
              }
              int accept = Rcpp::rbinom(1,1,acceptanceprob)(0);
              if(accept==1){
                xmatrix(misvalindex,jh+1) = propxreplacement;
                break;
              }
            }else if(covmethod=="empirical"){
              IntegerVector samframe=seq_len(N);
              int propXindex = RcppArmadillo::sample(samframe, 1, TRUE)(0)-1;
              double propxreplacement = xmatrix(propXindex,jh+1);
              propX(jh+1) = propxreplacement;
              arma::rowvec proplambda = (1.0/(1.0+exp(-propX*beta.t())));
              double acceptanceprob = 1.0;
              for(int j=0; j<J; j++){ //I think this is faster than matrix multiplying w/ exponents
                if(trueY(j)==1){
                  acceptanceprob = acceptanceprob*proplambda(j);
                }else{
                  acceptanceprob = acceptanceprob*(1-proplambda(j));
                }
              }
              int accept = Rcpp::rbinom(1,1,acceptanceprob)(0);
              if(accept==1){
                xmatrix(misvalindex,jh+1) = propxreplacement;
                break;
              }
            }else{
              Rcpp::Rcout << "Error: Only bootstrap and empirical can be used if missing values" << std::endl;
              return(0);
            }
            
      
            if(asam==maxasam-1){
              Rcpp::Rcout << "Error: Failed to find Missing Values" << std::endl;
              return(0);
            }
          }
        }
      }
    }
    
    
    
    
    ////////////////////////////////////
    // Sample Latent Group Membership //
    ////////////////////////////////////
    
    arma::mat OMEGALABELS(n,Homega,arma::fill::ones);
    if(Homega>1){
      //Sample Latent Class Membership Probabilities
      arma::rowvec Nomega(Homega);
      Nomega.cols(1,Homega-1) = sum(xmatrix.cols(H+1,H+Homega-1)); //rowsums
      Nomega.col(0) = N - sum(Nomega.cols(1,Homega-1)); 
      int NHcum = 0;
      double accu_log_prod = 0.0;
      arma::vec logphiomega(Homega);
      for(int h=0; h<Homega-1; h++){
        NHcum += Nomega(h);
        double aalphaomega = 1 + Nomega(h);
        double balphaomega = alphaomega + N - NHcum;
        //sampling from beta will result in computational 0s if b is too small, so do Dan's suggestion:
        // for small values of shape uses the method from p.45 of RObert & Casella (2002)
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
      alphaomega = R::rgamma(aomega+Homega-1,1/(bomega-logphiomega(Homega-1)));
      
      
      //Sample Latent Class Memberships
      arma::mat logprobomega(N,Homega,arma::fill::zeros);
      for(int w=0; w<Homega; w++){
        arma::mat omegamat(N,Homega-1,arma::fill::zeros);
        arma::mat allOne(N, 1, arma::fill::ones);
        omegamat.insert_cols(w, allOne);
        omegamat.shed_col(0); //removes first row (base class)
        arma::mat XOMEGA = join_rows(xmatrix.cols(0,H),omegamat);
        arma::mat lambda = (1/(1+exp(-XOMEGA*beta.t())));
        arma::mat logprobomegavec = arma::sum(arma::log(lambda)%ymatrix + arma::log(1.0-lambda)%(1.0-ymatrix),1)+ log(PHIomega(w));
        logprobomega.col(w) = logprobomegavec;
      }
      for(int i=0; i<n; i++){ //note, we are sampling only the observed captures
        arma::rowvec logomegarow = logprobomega.row(i);
        arma::rowvec omegaprobrowARMAnum = exp(logomegarow - logomegarow.max());
        double omegaprobrowARMAsum = sum(omegaprobrowARMAnum);
        arma::rowvec omegaprobrowARMA = omegaprobrowARMAnum/omegaprobrowARMAsum;
        NumericVector omegaprobrow = as<NumericVector>(wrap(omegaprobrowARMA));
        IntegerVector omegarowvec(Homega);
        R::rmultinom(1,omegaprobrow.begin(),Homega,omegarowvec.begin());
        arma::rowvec omegarow = as<arma::rowvec>(wrap(omegarowvec)); 
        OMEGALABELS.row(i) = omegarow;
        xmatrix.submat(i,1+H,i,H+Homega-1) = omegarow.cols(1,Homega-1);
      }
      
    }
    
    //Update individual's average
    if(sample>=0){
      saveOMEGAlab=(saveOMEGAlab*sample+OMEGALABELS)/(sample+1);
    }
    
    
    ///////////////////////////////////
    //Sampling Covariate Distribution//
    ///////////////////////////////////
    arma::uvec Zlab(N);
    if(H==0){
      
    }else if(covmethod=="mixture"){
      
      
      //Create Useful submatrix
      arma::mat Xnoconst = xmatrix.cols(1,H);
      
      //Sample Z
      arma::mat Zlogprobs(N,Kstar);
      arma::vec logPIK=arma::log(PIK); //Unnecessary with Method 2???
      for(int k=0; k<Kstar; k++){
        arma::vec lognormpart=logdmvNORM(Xnoconst,MUK.row(k),SIGMAK.slice(k));
        Zlogprobs.col(k)= lognormpart + logPIK(k);
      }
      arma::mat Zprobs=arma::exp(Zlogprobs);
      IntegerVector samframe=seq_len(Kstar);
      for(int i=0; i<N; i++){
        Zlab(i) = RcppArmadillo::sample(samframe, 1, TRUE, Zprobs.row(i).t())(0);
      }
      
      
      //Sample SIGMAK and MUK
      for(int k=0; k<Kstar; k++){
        arma::uvec position = arma::find(Zlab==k+1);
        NK(k) = position.n_rows;
      }
      arma::vec NUNK=priornu0+as<arma::vec>(wrap(NK));
      arma::vec kappaNK=priorkappa0+as<arma::vec>(wrap(NK));
      
      ///////////////////////   
      
      for(int k=0; k<Kstar; k++){
        if(NK(k)==0){
          SIGMAK.slice(k)=arma::iwishrnd(priorlambda0,NUNK(k));
          MUK.row(k) = mvnrnd(priorMU0,SIGMAK.slice(k)/kappaNK(k)).t(); 
        } else if (NK(k)==1){
          arma::uvec position = arma::find(Zlab==k+1);
          arma::mat Xk = Xnoconst.rows(position);
          arma::rowvec Xbark = mean(Xk,0);
          double priorp1 = priorkappa0/(priorkappa0+1);
          arma::mat priorp2 = (Xbark.t()-priorMU0)*(Xbark.t()-priorMU0).t();
          arma::mat lambdaNK = priorlambda0+priorp1*priorp2;
          arma::rowvec MUNKp1 = priorkappa0/(priorkappa0+1)*priorMU0.t();
          arma::rowvec MUNKp2 = 1/(priorkappa0+1)*Xbark;
          arma::rowvec MUNK = MUNKp1+MUNKp2;
          SIGMAK.slice(k)=arma::iwishrnd(lambdaNK,NUNK(k)); 
          MUK.row(k) = mvnrnd(MUNK.t(),SIGMAK.slice(k)/kappaNK(k)).t();
        } else {
          arma::uvec position = arma::find(Zlab==k+1);
          arma::mat Xk = Xnoconst.rows(position);
          arma::rowvec Xbark = mean(Xk,0);
          arma::mat Sk = (Xk.each_row()-Xbark).t()*(Xk.each_row()-Xbark);
          double priorp1 = priorkappa0*NK(k)/(priorkappa0+NK(k));
          arma::mat priorp2 = (Xbark.t()-priorMU0)*(Xbark.t()-priorMU0).t();
          arma::mat lambdaNK = priorlambda0+Sk+priorp1*priorp2;
          arma::rowvec MUNKp1 = priorkappa0/(priorkappa0+NK(k))*priorMU0.t();
          arma::rowvec MUNKp2 = NK(k)/(priorkappa0+NK(k))*Xbark;
          arma::rowvec MUNK = MUNKp1+MUNKp2;
          SIGMAK.slice(k)=arma::iwishrnd(lambdaNK,NUNK(k));
          MUK.row(k) = mvnrnd(MUNK.t(),SIGMAK.slice(k)/kappaNK(k)).t();
        }
      }
      
      //Sample PIK
      int NKcum = 0;
      double accu_log_prod = 0.0;
      for(int k=0; k<Kstar-1; k++){
        NKcum += NK(k);
        double a = 1 + NK(k);
        double b = alpha + N - NKcum;
        //sampling from beta will result in computational 0s if b is too small, so do Dan's suggestion:
        // for small values of shape uses the method from p.45 of RObert & Casella (2002)
        double lgammaa = log(R::runif(0,1))/a + log(R::rgamma(1+a,1));
        double lgammab = log(R::runif(0,1))/b + log(R::rgamma(1+b,1));
        double sumlogab = log(exp(lgammaa) + exp(lgammab));
        logPIK(k) = lgammaa - sumlogab + accu_log_prod;
        accu_log_prod += lgammab - sumlogab; //this is log(1-Vh)
        PIK(k) = exp(logPIK(k));
      }
      logPIK(Kstar-1)=accu_log_prod;
      PIK(Kstar-1) = exp(logPIK(Kstar-1));
      //Sample alphaomega
      if(Kstar>1){
        alpha = R::rgamma(aalpha+Kstar-1,1/(balpha-logPIK(Kstar-1)));
      }
      
      
    }else if(covmethod=="empirical"){
      //Could track discretely, but easier to sample uniformly from previous iteration
      
    }else if(covmethod=="bootstrap"){
      PSICOUNTS = PSICOUNTSobs; //reset counts to just observed + prior
      for(int i=n; i<N; i++){
        unsigned int myindex = index_min(sum(abs(PSIMATRIX.each_row()-xmatrix(arma::span(i,i),arma::span(1,H))),1));
        PSICOUNTS(myindex)++;
      }
      arma::rowvec PSICOUNTSRV = PSICOUNTS.t();
      PSIPROBS = mydirichlet(PSICOUNTSRV); //note PSICOUNTS includes the prior
      
    }else if(covmethod=="fixed"){
      //nothing needs specified
    }else{
      Rcpp::Rcout << "Covariate method incorrectly specified." << std::endl;
      return(0);
    }
    
    
    //////////////////////////////////////////////
    //// Compute Log Likelihood //////////////////
    //// Needed for Model Selection //////////////
    //////////////////////////////////////////////
    if(sample>=0){
      double firstterm = R::lchoose(N,n);
      arma::mat lambda = (1/(1+exp(-xmatrix*beta.t())));
      double secondterm = arma::accu(log(lambda)%ymatrix + log(1.0-lambda)%(1.0-ymatrix));
      double thirdterm=0;
      if(H==0){
        thirdterm=0;
      }else if(covmethod=="bootstrap"){
        thirdterm = arma::accu(log(PSIPROBS)%(PSICOUNTSobs-alphabayesbootstrap).t()); //Recall PSICOUNTobs includes prior, so need to remove it
      }else if(covmethod=="mixture"){
        arma::vec lognormpart;
        arma::mat  xobs = xmatrix.submat(0,1,n-1,H);
        for(int i=0; i<n; i++){
          int k = Zlab(i)-1; //k is one less than the label, which corresponds to 0 starting point
          lognormpart=logdmvNORM(xobs.row(i),MUK.row(k),SIGMAK.slice(k));
        }
        thirdterm=arma::accu(lognormpart);
      }else if(covmethod=="empirical"){
        thirdterm=0;
      }else if(covmethod=="fixed"){
        arma::vec lognormpart;
        arma::mat  xobs = xmatrix.submat(0,1,n-1,H);
        for(int i=0; i<n; i++){
          lognormpart=logdmvNORM(xobs.row(i),FIXEDMEAN.t(),FIXEDCOVAR.t());
        }
        thirdterm=arma::accu(lognormpart);
      }
      savelikelihood(sample,0) = firstterm+secondterm; 
      savelikelihood(sample,1) = thirdterm; 
    }
    
    
    ////////////////////////
    //// Save Values ///////
    ////////////////////////
    if(sample>=0){
      saveN(sample)=N;
      saveBETA.subcube(0,0,sample,J-1,H+Homega-1,sample) = beta;
      savePHIomega.row(sample) = PHIomega;
      savealphaomega(sample) = alphaomega;
      savealphaPIK(sample) = alpha;
      if(H>0){
        if(n0==1){
          saveXmis.row(sample)=xmis(0,arma::span(1,H));
        }else if(n0>1){
          IntegerVector sammis=seq_len(n0-1); //note that the first generated missing value will have no chance of being selected
          int savedmiss = RcppArmadillo::sample(sammis, 1, TRUE)(0);
          saveXmis.row(sample)=xmis(savedmiss,arma::span(1,H));
        }
      }
    }
    
  }
  
  /// Compute BIC
  arma::vec logliky = savelikelihood.col(0);
  double loglikymax = logliky.max();
  //number of parameters: J*H+Homega beta coefficients and N
  double BIC = (J*(H+Homega)+1)*log(n) - 2*loglikymax; 
  
  ////////////////////
  //Return Variables//
  ///////////////////
  if(H==0){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("df_new") = df_new,
                              Rcpp::Named("intercept") = intercept,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("PHIomega") = savePHIomega,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("ymatrix")=ymatrix,
                              Rcpp::Named("xmatrix")=xmatrix,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC);
  }else if(covmethod=="mixture"){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("df_new") = df_new,
                              Rcpp::Named("intercept") = intercept,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("PHIomega") = savePHIomega,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("ymatrix")=ymatrix,
                              Rcpp::Named("xmatrix")=xmatrix,
                              Rcpp::Named("PIK")=PIK,
                              Rcpp::Named("alphaPIK")=savealphaPIK,
                              Rcpp::Named("SIGMAK")=SIGMAK,
                              Rcpp::Named("MUK")=MUK,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC);
  }else if(covmethod=="empirical"){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("df_new") = df_new,
                              Rcpp::Named("intercept") = intercept,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("PHIomega") = savePHIomega,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("ymatrix")=ymatrix,
                              Rcpp::Named("xmatrix")=xmatrix,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC);
  }else if(covmethod=="bootstrap"){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("df_new") = df_new,
                              Rcpp::Named("intercept") = intercept,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("PHIomega") = savePHIomega,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("PSIMATRIX") = PSIMATRIX,
                              Rcpp::Named("PSICOUNTS") = PSICOUNTS,
                              Rcpp::Named("PSIPROBS") = PSIPROBS,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC);
  }else if(covmethod=="fixed"){
    return Rcpp::List::create(Rcpp::Named("samples") = samples,
                              Rcpp::Named("df_new") = df_new,
                              Rcpp::Named("intercept") = intercept,
                              Rcpp::Named("H") = H,
                              Rcpp::Named("n") = n,
                              Rcpp::Named("J") = J,
                              Rcpp::Named("N") = saveN,
                              Rcpp::Named("Xmis") = saveXmis,
                              Rcpp::Named("PHIomega") = savePHIomega,
                              Rcpp::Named("omegalabels") = saveOMEGAlab,
                              Rcpp::Named("alphaomega") = savealphaomega,
                              Rcpp::Named("covariatemean") = FIXEDMEAN,
                              Rcpp::Named("covariatecovariance") = FIXEDCOVAR,
                              Rcpp::Named("Beta") = saveBETA,
                              Rcpp::Named("loglikelihood")=savelikelihood,
                              Rcpp::Named("BIC")=BIC);
  }else{
    Rcpp::Rcout << "Covariate method incorrectly specified." << std::endl;
    return(0);
  }
}



  