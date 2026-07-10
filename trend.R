suppressMessages({library(MASS); library(jsonlite)})
cfg<-fromJSON("trend_config.json"); d<-read.csv("trend_series.csv")   # year,count,N,rate
yr<-d$year; cnt<-d$count; off<-log(d$N); fut<-cfg$future
## 음이항 예측
nb<-tryCatch(glm.nb(cnt~yr+offset(off)),error=function(e)NULL); if(is.null(nb)) nb<-glm(cnt~yr+offset(off),family=poisson)
nd<-data.frame(yr=fut, off=log(rep(mean(d$N),length(fut))))
pr<-predict(nb,newdata=nd,type="link",se.fit=TRUE)
write.csv(data.frame(year=fut,NB_forecast=round(exp(pr$fit-nd$off)*100,2),
  lo95=round(exp(pr$fit-nd$off-1.96*pr$se.fit)*100,2),hi95=round(exp(pr$fit-nd$off+1.96*pr$se.fit)*100,2)),"trend_nb.csv",row.names=FALSE)
## Joinpoint — 다구간(관측 충분 시 1 조인포인트=2구간 탐색), APC + 부트스트랩 CI
lr<-log(d$rate); n<-length(yr)
apc_of<-function(idx){ff<-lm(lr[idx]~yr[idx]); as.numeric((exp(coef(ff)[2])-1)*100)}
if(n>=5){best<-Inf;bj<-NA
  for(k in 2:(n-2)){f1<-lm(lr[1:k]~yr[1:k]);f2<-lm(lr[k:n]~yr[k:n]);sse<-sum(residuals(f1)^2)+sum(residuals(f2)^2)
    if(is.finite(sse)&&sse<best){best<-sse;bj<-k}}
  segs<-list(1:bj, bj:n)} else segs<-list(1:n)
set.seed(1);B<-2000; seg<-data.frame()
for(si in seq_along(segs)){idx<-segs[[si]]; apc<-apc_of(idx)
  bs<-replicate(B,{s<-sample(idx,replace=TRUE); if(length(unique(yr[s]))<2) NA else {ff<-lm(lr[s]~yr[s]);as.numeric((exp(coef(ff)[2])-1)*100)}})
  ci<-quantile(bs,c(.025,.975),na.rm=TRUE)
  seg<-rbind(seg,data.frame(segment=si,period=paste0(min(yr[idx]),"-",max(yr[idx])),APC_pct=round(apc,2),lo95=round(ci[1],2),hi95=round(ci[2],2)))}
write.csv(seg,"trend_apc.csv",row.names=FALSE)
cat("trend.R OK\n")
