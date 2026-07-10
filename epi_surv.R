suppressMessages({library(survey); library(survival); library(jsonlite)})
options(survey.lonely.psu="adjust")
cfg<-fromJSON("surv_config.json"); d<-read.csv("surv_analytic.csv")
tcol<-cfg$time; ecol<-cfg$event; expo<-cfg$exposure; xbin<-cfg$exposure_bin; cov<-cfg$cov
d[[xbin]]<-factor(d[[xbin]])
des<-svydesign(ids=~psu,strata=~kstrata,weights=~wt_pool,data=d,nest=TRUE)
Sf<-function(rhs) as.formula(paste0("Surv(",tcol,",",ecol,")~",rhs))
hr<-function(s){se<-s[,"se(coef)"]; data.frame(term=rownames(s),HR=round(exp(s[,1]),3),
  lo=round(exp(s[,1]-1.96*se),3),hi=round(exp(s[,1]+1.96*se),3),p=signif(s[,ncol(s)],3))}
cr<-summary(svycoxph(Sf(expo),design=des))$coefficients
aj<-summary(svycoxph(Sf(paste(c(expo,cov),collapse="+")),design=des))$coefficients
write.csv(hr(cr),"surv_crude.csv",row.names=FALSE); write.csv(hr(aj),"surv_adj.csv",row.names=FALSE)
# KM (설계가중) — 노출군별 생존곡선
km<-svykm(Sf(paste0("factor(",xbin,")")),design=des)
out<-data.frame()
nm<-names(km); if(is.null(nm)) nm<-as.character(seq_along(km))
for(i in seq_along(km)){k<-km[[i]]; out<-rbind(out,data.frame(group=nm[i],time=k$time,surv=k$surv))}
write.csv(out,"surv_km.csv",row.names=FALSE)
# 발생률 (설계가중 events / person-time)
ev<-as.numeric(svytotal(as.formula(paste0("~",ecol)),des)[1]); pt<-as.numeric(svytotal(as.formula(paste0("~",tcol)),des)[1])
write.csv(data.frame(events=round(ev),person_time=round(pt),rate_per_1000py=round(1000*ev/pt,2)),"surv_inc.csv",row.names=FALSE)
cat("epi_surv.R OK\n")
