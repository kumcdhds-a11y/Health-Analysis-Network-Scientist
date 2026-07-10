suppressMessages({library(survey); library(jsonlite)})
options(survey.lonely.psu="adjust")
cfg<-fromJSON("epi_config.json"); d<-read.csv("epi_analytic.csv")
y<-cfg$outcome; expo<-cfg$exposure; cov<-cfg$cov; subs<-cfg$subgroups
des<-svydesign(ids=~psu,strata=~kstrata,weights=~wt_pool,data=d,nest=TRUE)
orr<-function(fit,term){s<-summary(fit)$coefficients
  if(!(term%in%rownames(s))) return(c(NA,NA,NA,NA))
  b<-s[term,1];se<-s[term,2];c(exp(b),exp(b-1.96*se),exp(b+1.96*se),s[term,4])}
terms<-c(expo,cov)
# Table 2 crude / Table 3 adjusted OR
crude<-data.frame(); fitA<-svyglm(as.formula(paste(y,"~",paste(terms,collapse="+"))),design=des,family=quasibinomial())
adj<-data.frame()
for(t in terms){
  fc<-tryCatch(svyglm(as.formula(paste(y,"~",t)),design=des,family=quasibinomial()),error=function(e)NULL)
  rc<-if(is.null(fc)) c(NA,NA,NA,NA) else orr(fc,t); crude<-rbind(crude,data.frame(term=t,OR=rc[1],lo=rc[2],hi=rc[3],p=rc[4]))
  ra<-orr(fitA,t); adj<-rbind(adj,data.frame(term=t,OR=ra[1],lo=ra[2],hi=ra[3],p=ra[4]))
}
write.csv(crude,"epi_crude.csv",row.names=FALSE); write.csv(adj,"epi_adj.csv",row.names=FALSE)
# Table 4 overall risk (설계가중 사건율)
pr<-svymean(as.formula(paste0("~",y)),des,na.rm=TRUE)
write.csv(data.frame(N=nrow(d),weighted_risk_pct=100*coef(pr)[1]),"epi_risk.csv",row.names=FALSE)
# Table 5 하위군 + P-interaction (main exposure 효과)
sub<-data.frame()
for(sv in subs){
  for(lv in sort(unique(d[[sv]][!is.na(d[[sv]])]))){
    di<-des[which(d[[sv]]==lv),]
    cc<-setdiff(cov,sv); f<-paste(y,"~",paste(c(expo,cc),collapse="+"))
    fit<-tryCatch(svyglm(as.formula(f),design=di,family=quasibinomial()),error=function(e)NULL)
    r<-if(is.null(fit)) c(NA,NA,NA,NA) else orr(fit,expo)
    sub<-rbind(sub,data.frame(subgroup=sv,level=lv,OR=r[1],lo=r[2],hi=r[3],p=r[4],p_int=NA))
  }
  fi<-tryCatch(svyglm(as.formula(paste(y,"~",expo,"*",sv,"+",paste(setdiff(cov,sv),collapse="+"))),design=des,family=quasibinomial()),error=function(e)NULL)
  if(!is.null(fi)){s<-summary(fi)$coefficients; ix<-grep(":",rownames(s))
    if(length(ix)>0) sub$p_int[sub$subgroup==sv]<-s[ix[1],4]}
}
write.csv(sub,"epi_sub.csv",row.names=FALSE)
cat("epi.R OK\n")
