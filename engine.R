suppressMessages({library(survey); library(jsonlite)})
options(survey.lonely.psu="adjust")
cfg <- fromJSON("engine_config.json")
d <- read.csv("engine_analytic.csv")
grp<-cfg$group; cont<-cfg$cont; bin<-cfg$bin; cov<-cfg$cov; labs<-cfg$labels
fp<-function(p) ifelse(is.na(p),"",ifelse(p<0.001,"<0.001",sprintf("%.3f",p)))
if(nzchar(grp)) d[[grp]]<-factor(d[[grp]])
for(b in bin) if(!is.null(d[[b]])) d[[paste0(b,"_f")]]<-factor(d[[b]])
des<-svydesign(ids=~psu,strata=~kstrata,weights=~wt_pool,data=d,nest=TRUE)
glevels<-if(nzchar(grp)) levels(d[[grp]]) else "Overall"; L<-length(glevels)

chars<-character(); pcol<-character(); valmat<-matrix(character(),ncol=L,nrow=0)
addrow<-function(lab,vals,p){chars[[length(chars)+1]]<<-lab; pcol[[length(pcol)+1]]<<-fp(p); valmat<<-rbind(valmat,matrix(vals,nrow=1))}
for(v in cont){
  if(nzchar(grp)){
    m<-svyby(as.formula(paste0("~",v)),as.formula(paste0("~",grp)),des,svymean,na.rm=TRUE)
    vals<-sapply(glevels,function(g){s<-m[m[[grp]]==g,]; if(nrow(s)==0)"" else sprintf("%.1f (%.2f)",s[1,2],s[1,3])})
    p<-tryCatch(svyttest(as.formula(paste0(v,"~",grp)),des)$p.value,error=function(e)NA)
  } else {mm<-svymean(as.formula(paste0("~",v)),des,na.rm=TRUE); vals<-sprintf("%.1f (%.2f)",coef(mm)[1],SE(mm)[1]); p<-NA}
  addrow(labs[[v]],vals,p)
}
for(b in bin){
  if(nzchar(grp)){
    m<-svyby(as.formula(paste0("~",b)),as.formula(paste0("~",grp)),des,svymean,na.rm=TRUE)
    vals<-sapply(glevels,function(g){s<-m[m[[grp]]==g,]; if(nrow(s)==0)"" else sprintf("%.1f",100*s[1,2])})
    p<-tryCatch(svychisq(as.formula(paste0("~",b,"_f+",grp)),des)$p.value,error=function(e)NA)
  } else {mm<-svymean(as.formula(paste0("~",b)),des,na.rm=TRUE); vals<-sprintf("%.1f",100*coef(mm)[1]); p<-NA}
  addrow(labs[[b]],vals,p)
}
t1<-data.frame(Characteristic=chars,valmat,P=pcol,stringsAsFactors=FALSE,check.names=FALSE)
colnames(t1)<-c("Characteristic",glevels,"P"); write.csv(t1,"engine_table1.csv",row.names=FALSE)

# ── 회귀: 결과 유형별 자동 전환 (이분형=로지스틱 OR, 연속형=선형 beta) ──
res<-data.frame(); covstr<-if(length(cov)>0) paste("+",paste(cov,collapse=" + ")) else ""
pr<-cfg$pairs
for(i in seq_len(nrow(pr))){
  y<-pr$y[i]; x<-pr$x[i]; otype<-pr$otype[i]
  fam<-if(otype=="b") quasibinomial() else gaussian()
  fit<-tryCatch(svyglm(as.formula(sprintf("%s ~ %s %s",y,x,covstr)),design=des,family=fam),error=function(e)NULL)
  if(is.null(fit)) next
  s<-summary(fit)$coefficients
  if(!(x %in% rownames(s))) next
  b<-s[x,1]; se<-s[x,2]; p<-s[x,4]
  if(otype=="b"){est<-exp(b); lo<-exp(b-1.96*se); hi<-exp(b+1.96*se); meas<-"OR"}
  else {est<-b; lo<-b-1.96*se; hi<-b+1.96*se; meas<-"beta"}
  res<-rbind(res,data.frame(exposure=x,outcome=y,measure=meas,est=est,ci_low=lo,ci_high=hi,p=p))
}
write.csv(res,"engine_results.csv",row.names=FALSE)
cat(sprintf("engine OK: table1 %dx%d, results %d\n",nrow(t1),L,nrow(res)))
