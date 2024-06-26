
<!-- README.md is generated from README.Rmd. Please edit that file -->

# BLRCR

<!-- badges: start -->
<!-- badges: end -->

A package for using the BLRCR model.

## Installation

### R Installation

Begin with installing [R and
RStudio](https://posit.co/download/rstudio-desktop/).

### Install Package “BLRCR”

You can install this beta version of BLRCR from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("rgranger617/BLRCR")
```

## Usage

Check out the BLRCRsolvermissing() function:

``` r
library(BLRCR)

#Create the Data
myN = 2000

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

myCRdata = multdatasimulator(myN,mybeta,mynormprobs,mynormmeans,mynormsigma,
                           missing=TRUE)
                           
#Run the Algorithm
myformula = cbind(y1,y2,y3,y4)~x1+x2

CRresults=BLRCRsolvermissing(myformula,df=myCRdata,Homega=1,
                             Bprior=1,LBprior=1,
                             covsupport="unique",
                             samples = 1000,burnin=10,thinning = 10)

quantile(CRresults$N,c(.025,.5,.975))
```
