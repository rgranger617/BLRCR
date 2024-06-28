
library(BLRCR)

kosovodata = readRDS("KosovoData-clean.RDS")
kosovodata$agestandard = (kosovodata$age - mean(kosovodata$age,na.rm=TRUE))/sd(kosovodata$age,na.rm=TRUE)
kosovodata$age0_14 = as.numeric(kosovodata$age<=14)
kosovodata$age15_17 = as.numeric((kosovodata$age>14)*(kosovodata$age<18))
kosovodata$age40_69 = as.numeric((kosovodata$age>=40)*(kosovodata$age<70))
kosovodata$age70plus = as.numeric(kosovodata$age>=70)


#myformula = cbind(exh,aba,osce,hrw)~1
#myformula = cbind(exh,aba,osce,hrw)~female+district
#myformula = cbind(exh,aba,osce,hrw)~female*district
#myformula = cbind(exh,aba,osce,hrw)~female+agestandard+I(agestandard^2)
myformula = cbind(exh,aba,osce,hrw)~female+agestandard+I(agestandard^2)+I(agestandard^3)+district
#myformula = cbind(exh,aba,osce,hrw)~female+I(age<=14)+I(age>14&age<18)+I(age>=40&age<70)+I(age>=70)+district
#myformula = cbind(exh,aba,osce,hrw)~female*I(age<=14)*I(age>14&age<18)*I(age>=40&age<70)*I(age>=70)*district
#myformula = cbind(exh,aba,osce,hrw)~female+age0_14+age15_17+age40_69+age70plus+district
#myformula = cbind(exh,aba,osce,hrw)~female*age0_14*age15_17*age40_69*age70plus*district

start.time=Sys.time()
#set.seed(20230323)
Homegaval=5
bprior=matrix(rep(0,4*(ncol(model.matrix(myformula,kosovodata))+Homegaval-1)),nrow=4)
modelresults=BLRCRsolvermissing(myformula,df=kosovodata,Homega=Homegaval,
                                covmethod="bootstrap",covsupport="all",
                                coefprior="normalhiervaronly",
                                Bprior=10,LBprior=10,
                                bprior=bprior,
                                samples = 200,burnin=10,thinning = 10,
                                alphabayesbootstrap = .01,
                                aomega=.25,bomega=.25)
end.time <- Sys.time()
end.time-start.time


modelresults$ymatrix

modelresults$priorBdiag[100,,]
modelresults$Beta

modelresults$alphaomega
lambda2=modelresults$lambda2[,,95]
round(modelresults$invlambda2_betasampling,3)
modelresults$invLAMBDAp
modelresults$Vw_betasampling
modelresults$Mw_betasampling

probcaptured=rowMeans(modelresults$probcaptured)

plot(kosovodata$age[-is.na(kosovodata$age)],probcaptured[-is.na(kosovodata$age)])

PSIPROBS_missingindices = modelresults$PSIPROBS_missingindices
PSIMATRIX=modelresults$PSIMATRIX
PSICOUNTS = modelresults$PSICOUNTS

test=kosovodata[kosovodata$age==0,]

plot(modelresults$N,type="l")

Beta = matrix(rep(NA,4*dim(modelresults$Beta)[2]),nrow=4)
for(j in 1:4){
  for(h in 1:dim(modelresults$Beta)[2]){
    Beta[j,h] = median(modelresults$Beta[j,h,])
  }
}
colnames(Beta) <- modelresults$Betacolnames
round(Beta,3)



plot(modelresults$alphaomega,type="l")

plot(modelresults$PHIomega[10,],type="l")


sum(modelresults$PSICOUNTS[,5])
modelresults$N[5]

sum(modelresults$PSICOUNTS[,1]==0)
sum(modelresults$PSICOUNTS[,2]==0)
sum(modelresults$PSICOUNTS[,3]==0)
sum(modelresults$PSICOUNTS[,4]==0)
sum(modelresults$PSICOUNTS[,5]==0)
sum(modelresults$PSICOUNTS[,6]==0)
sum(modelresults$PSICOUNTS[,7]==0)
sum(modelresults$PSICOUNTS[,8]==0)
sum(modelresults$PSICOUNTS[,9]==0)
sum(modelresults$PSICOUNTS[,10]==0)
nrow(modelresults$PSIMATRIX)


MYTIMER=modelresults$MYTIMER
MYTIMERPERC = MYTIMER/rowSums(MYTIMER)

#plotting age vs capturehistory
#works for formula=5
par(mfrow=c(5,2))
Homega=1
for(w in 1:Homega){
sexX = rep(0,100)
agestandardX = rbind(seq(-2.22,3.211,length.out=100),seq(-2.22,3.211,length.out=100)^2,seq(-2.22,3.211,length.out=100)^3)
ageX = rbind(seq(1,100,by=1),seq(1,100,by=1)^2,seq(1,100,by=1)^3)
districtX=rbind(rep(1,100),rep(0,100),rep(0,100),rep(0,100),rep(0,100),rep(0,100))
latentX=matrix(rep(0,100*Homega),nrow=Homega);latentX[w,]=1
myX = t(rbind(sexX,agestandardX,districtX,latentX))
probcaptured=1-apply(1-1/(1+exp(-myX%*%t(Beta))),1,prod)
plot(seq(1,100,by=1),probcaptured,type="l")
}

sum(mydata$age<18,na.rm=TRUE)
sum(mydata$age>65,na.rm=TRUE)
sum(mydata$age>17|mydata$age<66,na.rm=TRUE)

rowMeans(modelresults$PHIomega)
quantile(modelresults$N,c(.025,.5,.975))
plot(modelresults$N,type="l")


femaleN=colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$female==1),])
maleN=colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$female==0),])

Pristina = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$district=="Pristina"),])
Gnjilane = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$district=="Gnjilane"),])
Pecki = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$district=="Pećki"),])
Urosevac = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$district=="Uroševac"),])
Prizren = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$district=="Prizren"),])
Dakovica = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$district=="Đakovica"),])
Kosovska_Mitrovica = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$district=="Kosovska Mitrovica"),])


childN = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$age<18),])
adultN = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$age>=18&modelresults$PSIMATRIX$age<=64),])
elderlyN = colSums(modelresults$PSICOUNTS[which(modelresults$PSIMATRIX$age>64),])

par(mfrow=c(2,1))
hist(femaleN);hist(maleN)
quantile(femaleN,c(.025,.5,.975))
quantile(maleN,c(.025,.5,.975))

par(mfrow=c(3,3))
median(Pristina)+median(Gnjilane)+median(Pecki)+median(Urosevac)+median(Prizren)+median(Dakovica)+median(Kosovska_Mitrovica)
hist(Pristina);hist(Gnjilane);hist(Pecki);hist(Urosevac);hist(Prizren);hist(Dakovica);hist(Kosovska_Mitrovica)

par(mfrow=c(2,2))
hist(childN);hist(adultN);hist(elderlyN)
quantile(childN,c(.025,.5,.975))
quantile(adultN,c(.025,.5,.975))
quantile(elderlyN,c(.025,.5,.975))

####################################################
library(BLRCR)
casanaredata <- readRDS("mycleaneddataCasanare.RDS")

covhistorystr = "cbind(in_1,in_2,in_6,in_8,in_9,in_10,in_12,in_19,in_20,in_23,in_24,in_28,in_29,in_30,in_32,in_33,in_combine)" 
myformula = paste0(covhistorystr,"~age_cat+sex+ethnicity")

modelresults=BLRCRsolvermissing(myformula,df=casanaredata,Homega=10,
                                covmethod="bootstrap",bprior=0,Bprior=10,
                                samples = 1000,burnin=1,
                                missmethod="exact")

CRresults = BLRCRsolver(myformula,myCRdata,covmethod="mixture",coefprior="normal",
                        Homega=10,Kstar=10, samples=1000)     
