# -*- coding: utf-8 -*-
"""KNHANES + NHANES Research Automation Platform (final) — Streamlit
Interactive analysis · Trajectory · Epidemiologic/Trend/ML reports (auto Word generation)
Run: streamlit run factory_app.py   (same folder: factory_core.py, epi_report.py, epi.R, epi_adv.R,
      trend_report.py, trend.R, ml_report.py, engine.R + R survey/rms/MASS + ollama optional)"""
import streamlit as st
import factory_core as fc
import epi_report, trend_report, ml_report
import copy
MIME="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DERIVED=["steatosis","dm","htn","mets","dld"]
DATA_DIRS={"KNHANES":"data/KNHANES","NHANES":"data/NHANES"}
CYCLES={"KNHANES":("08","09"),"NHANES":("j",)}
TREND_YEARS={"KNHANES":["08","09","10"],"NHANES":["j"]}
st.set_page_config(page_title="Research Automation Platform", layout="wide")

with st.sidebar:
    st.header("⚙️ Global settings")
    ds=st.radio("Dataset", fc.DATASETS, horizontal=True)
    DATA_DIRS[ds]=st.text_input("Data folder", DATA_DIRS[ds], key=f"dir_{ds}")
    st.divider(); st.subheader("🤖 LLM (ollama)")
    MODEL=st.text_input("Model","llama3.3:70b"); URL=st.text_input("URL","http://localhost:11434")
    USE_LLM=st.checkbox("Use LLM",value=True)
    st.divider(); AGE=st.number_input("Minimum age",0,100,20)
    st.caption("Reports are generated as Word (.docx). Analyses use R survey; ML uses scikit-learn.")

def get_df():
    k=f"DF_{ds}"
    if k not in st.session_state: st.session_state[k]=fc.load_raw(ds,DATA_DIRS[ds],CYCLES[ds])
    return st.session_state[k]
def defs_(): return {**copy.deepcopy(fc.DEFAULT_DEFS),"pop":{"age_min":AGE}}
def out_avail(): return [o for o in DERIVED if o in fc.AVAIL[ds]]
def dlbtn(docx,fname,key): st.download_button("⬇️ Download Word (.docx)",docx,fname,MIME,key=key,use_container_width=True)

st.title("🏭 KNHANES + NHANES Research Automation Platform")
st.caption(f"Dataset: {ds} · select variables → analysis → Word report. Steatosis: {fc.STEATOSIS_METHOD[ds]}")
T1,T2,T3,T4,T5=st.tabs(["📊 Interactive analysis","📈 Trajectory","🧬 Epidemiologic report","📉 Trend report","🤖 ML report"])

# ── Interactive analysis ──
with T1:
    c1,c2=st.columns(2)
    def vbox(title,dft,kp):
        st.markdown(f"**{title}**"); sel=[]
        for grp,tp in [("Continuous","c"),("Binary","b")]:
            st.caption(grp)
            for cvar in [x for x in fc.AVAIL[ds] if fc.typ(x)==tp]:
                if st.checkbox(fc.lab(cvar).split(",")[0],value=cvar in dft,key=f"{kp}_{cvar}"): sel.append(cvar)
        return sel
    with c1: exp=vbox("Exposure",("bodyfat_pct","asm_pct","bmi"),f"{ds}_e")
    with c2: out=vbox("Outcome",("steatosis","dm"),f"{ds}_o")
    cov=fc.auto_covariates(ds,exp,out) if (exp and out) else []
    st.info(f"Auto covariates: {', '.join(fc.lab(c) for c in cov) or '(shown after selection)'}")
    if st.button("▶ Run analysis",type="primary",key="ia_run",use_container_width=True):
        if not exp or not out: st.error("Select at least one exposure and one outcome")
        else:
            with st.spinner("Survey-weighted analysis..."):
                d=fc.apply_definitions(get_df(),defs_()); ana=fc.build_analytic(d,exp,out,cov,AGE)
                t1,res,cfg,pairs=fc.run_engine(ana,ds,exp,out,cov); res=fc.add_fdr(res)
                if fc.typ(out[0])=="b":
                    lb=fc.lab(out[0]).split(",")[0]; t1=t1.rename(columns={"0":f"No {lb}","1":lb})
                st.session_state["ia"]=dict(t1=t1,res=res,n=len(ana),exp=exp,out=out,cov=cov)
    if "ia" in st.session_state:
        R=st.session_state["ia"]
        st.markdown(f"#### Table 1 (n={R['n']:,})"); st.dataframe(R["t1"],use_container_width=True,hide_index=True)
        rr=R["res"].copy(); rr["Exposure"]=rr.exposure.str[2:].map(fc.lab); rr["Outcome"]=rr.outcome.map(fc.lab)
        rr["Measure"]=rr.measure.map(fc.measure_name); rr["Estimate (95% CI)"]=rr.apply(lambda x:f"{x.est:.2f} ({x.ci_low:.2f}–{x.ci_high:.2f})",axis=1)
        rr["FDR q"]=rr.q.map(lambda q:"<0.001" if q<0.001 else f"{q:.3f}")
        st.markdown("#### Associations (binary outcome=OR, continuous=β)"); st.dataframe(rr[["Exposure","Outcome","Measure","Estimate (95% CI)","FDR q"]].sort_values("FDR q"),use_container_width=True,hide_index=True)
        if st.button("📝 Generate manuscript Word",key="ia_ms",use_container_width=True):
            with st.spinner("Writing..."):
                ms=fc.gen_manuscript(R["t1"],R["res"],ds,R["exp"],R["out"],R["cov"],defs_(),MODEL,URL,USE_LLM)
                docx=fc.build_docx(f"Association study ({ds})",ms,R["t1"],R["res"])
            dlbtn(docx,f"manuscript_{ds}.docx","ia_dl")

# ── Trajectory ──
with T2:
    st.markdown("Age-specific survey-weighted means of outcome determinants (+ prevalence)")
    o=st.selectbox("Outcome",out_avail(),format_func=fc.lab,key="tj_o"); sx=st.checkbox("Split by sex",True,key="tj_s")
    if st.button("📈 Plot trajectory",key="tj_b",use_container_width=True):
        with st.spinner("Computing..."):
            d=fc.apply_definitions(get_df(),defs_()); tdf,vs=fc.trajectory(d,ds,o,split_sex=sx)
        if not vs: st.warning("Determinant variables are not available in this dataset.")
        else: st.caption(f"Determinants: {vs}"); st.pyplot(fc.plot_trajectory(tdf,o,ds,sx))

# ── Epidemiologic report ──
with T3:
    st.markdown("Select outcome and main exposure → Table 1-6 (SMD, crude/adj OR, subgroup, IPTW/AIPW/G-comp/PSM/TMLE, Love, VIF, RCS, E-value) Word")
    c1,c2=st.columns(2)
    o=c1.selectbox("Outcome (binary)",out_avail(),format_func=fc.lab,key="epi_o")
    mx=c2.selectbox("Main exposure",[v for v in fc.AVAIL[ds] if v!=o],format_func=fc.lab,key="epi_x")
    cov=fc.auto_covariates(ds,[mx],[o]); st.info(f"Auto covariates: {', '.join(fc.lab(c) for c in cov)}")
    if st.button("🧬 Generate epidemiologic report (Word)",type="primary",key="epi_b",use_container_width=True):
        with st.spinner("Epidemiologic analysis (survey-weighted + causal inference)... tens of seconds"):
            try:
                docx=epi_report.build_epi_report(ds,get_df(),o,mx,cov,["men","age50"],defs_(),MODEL,URL,USE_LLM)
                st.success("Epidemiologic report generated"); dlbtn(docx,f"epi_analysis_{ds}_{o}.docx","epi_dl")
            except Exception as e: st.error(f"Error: {e}")

# ── Trend report ──
with T4:
    st.markdown("Yearly prevalence, age-sex standardization, projection, stratification, joinpoint APC, NB forecast Word")
    o=st.selectbox("Outcome",out_avail(),format_func=fc.lab,key="tr_o")
    yrs=st.multiselect("Years (cycles)",TREND_YEARS[ds],default=TREND_YEARS[ds],key="tr_y")
    if st.button("📉 Generate trend report (Word)",type="primary",key="tr_b",use_container_width=True):
        if len(yrs)<2: st.error("Select at least 2 years.")
        else:
            with st.spinner("Trend analysis (standardization, APC bootstrap, NB forecast)..."):
                try:
                    docx=trend_report.build_trend_report(ds,DATA_DIRS[ds],yrs,o,{**defs_(),"pop":{"age_min":max(AGE,19)}},MODEL,URL,USE_LLM)
                    st.success("Trend report generated"); dlbtn(docx,f"trend_{ds}_{o}.docx","tr_dl")
                except Exception as e: st.error(f"Error: {e}")

# ── ML report ──
with T5:
    st.markdown("Prediction model — tuning, search space, full metrics, ROC/PR, threshold, calibration, SHAP, PDP Word")
    o=st.selectbox("Outcome (binary)",out_avail(),format_func=fc.lab,key="ml_o")
    availp=[v for v in fc.AVAIL[ds] if v!=o and v not in fc.determinants(ds,o)]
    preds=st.multiselect("Predictors (features)",availp,
        default=[v for v in ["age","men","bmi","wc","sbp","tg","alt","bodyfat_pct","asm_pct","smoking","alcohol"] if v in availp],
        format_func=fc.lab,key=f"ml_p_{o}")
    st.caption(f"Leakage prevention: definitional components of {fc.lab(o)} are auto-excluded from candidate features" + (f" ({', '.join(fc.lab(x) for x in fc.determinants(ds,o))})" if fc.determinants(ds,o) else "") + ".")
    mdls=st.multiselect("Models to compare (10+ recommended, incl. modern)", ml_report.ALL_MODELS, default=ml_report.DEFAULT_MODELS, key="ml_m")
    st.caption("SVM-RBF and MLP are excluded by default (slow) but selectable. The other 14 are selected by default.")
    if st.button("🤖 Generate ML report (Word)",type="primary",key="ml_b",use_container_width=True):
        if len(preds)<2 or len(mdls)<2: st.error("Select at least 2 features and 2 models.")
        else:
            with st.spinner(f"Tuning/evaluating {len(mdls)} models + SHAP... a few minutes"):
                try:
                    docx=ml_report.build_ml_report(ds,get_df(),o,preds,defs_(),MODEL,URL,USE_LLM,models=mdls)
                    st.success("ML report generated"); dlbtn(docx,f"ML_{ds}_{o}.docx","ml_dl")
                except Exception as e: st.error(f"Error: {e}")
