suppressMessages({library(survey); library(splines); library(jsonlite)})
options(survey.lonely.psu="adjust")
cfg<-fromJSON("epi_config.json"); d<-read.csv("epi_analytic.csv")
y<-cfg$outcome; cov<-cfg$cov; xterm<-cfg$exposure; xbin<-cfg$exposure_bin; xcont<-cfg$exposure_cont
des<-svydesign(ids=~psu,strata=~kstrata,weights=~wt_pool,data=d,nest=TRUE)
orci<-function(fit,term){s<-summary(fit)$coefficients; if(!(term%in%rownames(s)))return(c(NA,NA,NA))
  b<-s[term,1];se<-s[term,2];c(exp(b),exp(b-1.96*se),exp(b+1.96*se))}

## VIF
terms<-c(xterm,cov); vif<-c()
for(t in terms){ot<-setdiff(terms,t)
  r2<-tryCatch(summary(lm(as.formula(paste(t,"~",paste(ot,collapse="+"))),data=d))$r.squared,error=function(e)NA)
  vif<-c(vif, if(is.na(r2)) NA else round(1/(1-r2),2))}
write.csv(data.frame(term=terms,VIF=vif),"epi_vif.csv",row.names=FALSE)

## RCS 비선형성
pnl<-NA
if(!is.null(xcont)&&!is.na(xcont)&&nzchar(xcont)){
  ff<-tryCatch(svyglm(as.formula(paste(y,"~ ns(",xcont,",3)+",paste(cov,collapse="+"))),design=des,family=quasibinomial()),error=function(e)NULL)
  fl<-tryCatch(svyglm(as.formula(paste(y,"~",xcont,"+",paste(cov,collapse="+"))),design=des,family=quasibinomial()),error=function(e)NULL)
  if(!is.null(ff)&&!is.null(fl)) pnl<-tryCatch(as.numeric(anova(fl,ff)$p)[1],error=function(e)NA)}
write.csv(data.frame(test="RCS_nonlinearity_p",value=pnl),"epi_rcs.csv",row.names=FALSE)

## 인과추론 준비
dd<-d[!is.na(d[[xbin]]) & complete.cases(d[,cov,drop=FALSE]) & !is.na(d[[y]]),]
desb<-svydesign(ids=~psu,strata=~kstrata,weights=~wt_pool,data=dd,nest=TRUE)
A<-dd[[xbin]]; Y<-dd[[y]]; sw<-weights(desb,"sampling")
psf<-svyglm(as.formula(paste(xbin,"~",paste(cov,collapse="+"))),design=desb,family=quasibinomial())
ps<-pmin(pmax(as.numeric(predict(psf,type="response")),1e-4),1-1e-4)
ptx<-as.numeric(coef(svymean(as.formula(paste0("~",xbin)),desb))[1])
iptw<-ifelse(A==1,ptx/ps,(1-ptx)/(1-ps)); cw<-sw*iptw
gm<-svyglm(as.formula(paste(y,"~",xbin,"+",paste(cov,collapse="+"))),design=desb,family=quasibinomial())
d1<-dd;d1[[xbin]]<-1; d0<-dd;d0[[xbin]]<-0
m1<-as.numeric(predict(gm,newdata=d1,type="response")); m0<-as.numeric(predict(gm,newdata=d0,type="response")); mA<-as.numeric(predict(gm,type="response"))
wm<-function(x,w) sum(w*x)/sum(w); wv<-function(x,w){m<-wm(x,w);sum(w*(x-m)^2)/sum(w)}

## Love plot
smd1<-function(x,A,w){ if(length(unique(x[!is.na(x)]))<=2){p1<-wm(x[A==1],w[A==1]);p0<-wm(x[A==0],w[A==0]);pb<-(p1+p0)/2
    if(pb<=0||pb>=1)return(NA);(p1-p0)/sqrt(pb*(1-pb))} else {m1_<-wm(x[A==1],w[A==1]);m0_<-wm(x[A==0],w[A==0]);s<-sqrt((wv(x[A==1],w[A==1])+wv(x[A==0],w[A==0]))/2); if(s==0)return(NA);(m1_-m0_)/s}}
love<-data.frame()
for(c0 in cov){x<-dd[[c0]];ok<-!is.na(x); love<-rbind(love,data.frame(covariate=c0,before=round(abs(smd1(x[ok],A[ok],sw[ok])),3),after=round(abs(smd1(x[ok],A[ok],cw[ok])),3)))}
write.csv(love,"epi_love.csv",row.names=FALSE)

res<-data.frame(); addm<-function(n,e,l,h,s) res<<-rbind(res,data.frame(method=n,estimate=round(e,3),lo=round(l,3),hi=round(h,3),scale=s))
# Crude / Min-adj / Full-adj (OR)
r<-orci(svyglm(as.formula(paste(y,"~",xbin)),design=desb,family=quasibinomial()),xbin); addm("Crude",r[1],r[2],r[3],"OR")
mc<-intersect(c("age","men"),cov)
if(length(mc)>0){r<-orci(svyglm(as.formula(paste(y,"~",xbin,"+",paste(mc,collapse="+"))),design=desb,family=quasibinomial()),xbin);addm("Min-adj (age,sex)",r[1],r[2],r[3],"OR")}
r<-orci(gm,xbin); addm("Full-adj",r[1],r[2],r[3],"OR")
# IPTW (OR)
desi<-svydesign(ids=~psu,strata=~kstrata,weights=~cw,data=cbind(dd,cw=cw),nest=TRUE)
r<-orci(svyglm(as.formula(paste(y,"~",xbin)),design=desi,family=quasibinomial()),xbin); addm("IPTW",r[1],r[2],r[3],"OR")
# PSM (greedy 1:1 nearest neighbor on logit PS, caliper 0.2 SD) → matched OR
lps<-qlogis(ps); cal<-0.2*sd(lps); ti<-which(A==1); ci<-which(A==0); used<-rep(FALSE,length(A)); pt<-c(); pc<-c()
for(t in ti){cand<-ci[!used[ci]]; if(length(cand)==0)break; dgap<-abs(lps[cand]-lps[t]); j<-which.min(dgap)
  if(dgap[j]<=cal){pt<-c(pt,t);pc<-c(pc,cand[j]);used[cand[j]]<-TRUE}}
if(length(pt)>=10){midx<-c(pt,pc); dm_<-dd[midx,]
  desm<-svydesign(ids=~psu,strata=~kstrata,weights=~wt_pool,data=dm_,nest=TRUE)
  r<-tryCatch(orci(svyglm(as.formula(paste(y,"~",xbin)),design=desm,family=quasibinomial()),xbin),error=function(e)c(NA,NA,NA))
  addm(sprintf("PSM (n=%d pairs)",length(pt)),r[1],r[2],r[3],"OR")} else addm("PSM",NA,NA,NA,"OR")
# G-computation (RD)
rd_g<-wm(m1,sw)-wm(m0,sw); addm("G-computation",rd_g,NA,NA,"RD")
# AIPW (RD)
ai<-(A*(Y-m1)/ps+m1)-((1-A)*(Y-m0)/(1-ps)+m0); rd_a<-wm(ai,sw); se_a<-sqrt(sum((sw/sum(sw))^2*(ai-rd_a)^2))
addm("AIPW",rd_a,rd_a-1.96*se_a,rd_a+1.96*se_a,"RD")
# TMLE (RD) — clever covariate 변동
Hobs<-ifelse(A==1,1/ps,-1/(1-ps)); off<-qlogis(pmin(pmax(mA,1e-4),1-1e-4))
eps<-tryCatch(coef(glm(Y~ -1+Hobs+offset(off),family=binomial,weights=sw)),error=function(e)0); if(length(eps)==0||is.na(eps))eps<-0
q1<-plogis(qlogis(pmin(pmax(m1,1e-4),1-1e-4))+eps*(1/ps)); q0<-plogis(qlogis(pmin(pmax(m0,1e-4),1-1e-4))+eps*(-1/(1-ps)))
rd_t<-wm(q1,sw)-wm(q0,sw); ict<-(A/ps)*(Y-q1)-((1-A)/(1-ps))*(Y-q0)+(q1-q0)-rd_t; se_t<-sqrt(sum((sw/sum(sw))^2*ict^2))
addm("TMLE",rd_t,rd_t-1.96*se_t,rd_t+1.96*se_t,"RD")
write.csv(res,"epi_table6.csv",row.names=FALSE)
cat("epi_adv.R OK\n")
