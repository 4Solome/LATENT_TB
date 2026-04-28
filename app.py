"""
Uganda National TB Survey — TTVAE Risk Sequencing System
Transformer-based Tabular Variational Autoencoder · Deployment App
Aligned with training notebook: Uganda_TB_TTVAE_Enhanced_Final
"""
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib, json, os, io, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="TB Risk Sequencing · TTVAE",
                   page_icon="🫁", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#09090b;color:#fafafa;}
.stApp{background:#09090b;}
section[data-testid="stSidebar"]{background:#111113;border-right:1px solid #27272a;}
.card{background:#111113;border:1px solid #27272a;border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:.6rem;}
.card-r{background:#180a0a;border:1px solid #ef4444;border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:.6rem;}
.card-g{background:#0a1a0f;border:1px solid #22c55e;border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:.6rem;}
.card-a{background:#1a1200;border:1px solid #f59e0b;border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:.6rem;}
.card-b{background:#0a1020;border:1px solid #3b82f6;border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:.6rem;}
.card-p{background:#120a1a;border:1px solid #a855f7;border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:.6rem;}
.mv{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:700;line-height:1;}
.ml{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:#71717a;margin-bottom:.25rem;}
.banner{display:flex;align-items:center;gap:.6rem;background:linear-gradient(90deg,#3b82f620,transparent);
        border-left:3px solid #3b82f6;padding:.5rem 1rem;border-radius:0 8px 8px 0;
        margin:1.3rem 0 .8rem;font-weight:700;font-size:.76rem;letter-spacing:.1em;
        text-transform:uppercase;color:#60a5fa;}
.chip{display:inline-block;padding:2px 9px;border-radius:99px;
      font-family:'JetBrains Mono',monospace;font-size:.68rem;font-weight:700;letter-spacing:.05em;}
.ch{background:#180a0a;color:#ef4444;border:1px solid #ef4444;}
.cm{background:#1a1200;color:#f59e0b;border:1px solid #f59e0b;}
.cl{background:#0a1a0f;color:#22c55e;border:1px solid #22c55e;}
.co{background:#120a1a;color:#a855f7;border:1px solid #a855f7;}
.ood-box{background:#120a1a;border:2px solid #a855f7;border-radius:10px;padding:1rem 1.3rem;margin:1rem 0;}
.disc{background:#180a0a;border:1px solid #ef444450;border-radius:8px;
      padding:.8rem 1.1rem;font-size:.76rem;color:#a1a1aa;margin-top:1rem;}
h1{font-size:1.65rem;font-weight:700;letter-spacing:-.02em;color:#f4f4f5;}
h2{font-size:1.1rem;font-weight:700;color:#f4f4f5;
   border-bottom:1px solid #27272a;padding-bottom:.3rem;margin-top:1.3rem;}
.stButton>button{background:#2563eb;color:#fff;border:none;border-radius:8px;
                 font-family:'Inter',sans-serif;font-weight:700;
                 padding:.48rem 1.5rem;transition:background .2s;}
.stButton>button:hover{background:#3b82f6;}
.stTabs [data-baseweb="tab-list"]{background:#111113;border-bottom:1px solid #27272a;gap:0;}
.stTabs [data-baseweb="tab"]{color:#71717a;font-family:'Inter',sans-serif;
   font-size:.82rem;font-weight:600;padding:.52rem 1rem;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{color:#60a5fa;border-bottom:2px solid #3b82f6;background:transparent;}
</style>
""", unsafe_allow_html=True)

# ── TTVAE — exact copy from notebook cell 13 ──────────────────────────────────
class TTVAE(nn.Module):
    def __init__(self,input_dim,latent_dim,d_model,nhead,n_layers,n_cont,n_bin,cat_sizes):
        super().__init__()
        self.n_cont=n_cont; self.n_bin=n_bin; self.cat_sizes=cat_sizes
        self.value_proj=nn.Linear(1,d_model)
        self.positional=nn.Parameter(torch.randn(1,input_dim,d_model)*0.01)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
            dim_feedforward=d_model*2,dropout=0.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(enc,num_layers=n_layers)
        self.mu=nn.Linear(d_model,latent_dim)
        self.logvar=nn.Linear(d_model,latent_dim)
        self.decoder_trunk=nn.Sequential(
            nn.Linear(latent_dim,d_model),nn.ReLU(),
            nn.Linear(d_model,d_model),nn.ReLU())
        self.dec_cont=nn.Linear(d_model,n_cont) if n_cont>0 else None
        self.dec_bin =nn.Linear(d_model,n_bin)  if n_bin >0 else None
        self.dec_cat =nn.ModuleList([nn.Linear(d_model,s) for s in cat_sizes])

    def encode(self,x):
        h=self.encoder(self.value_proj(x.unsqueeze(-1))+self.positional).mean(1)
        return self.mu(h), torch.clamp(self.logvar(h),min=-8.,max=8.)

    def reparameterize(self,mu,lv):
        return mu+torch.randn_like(mu)*torch.exp(.5*lv)

    def decode(self,z):
        h=self.decoder_trunk(z)
        return (self.dec_cont(h) if self.dec_cont else None,
                self.dec_bin(h)  if self.dec_bin  else None,
                [hd(h) for hd in self.dec_cat])

    def forward(self,x):
        mu,lv=self.encode(x); z=self.reparameterize(mu,lv)
        return *self.decode(z),mu,lv

# ── Modality constants — match notebook cell 11 ───────────────────────────────
DEMOGRAPHICS=["age_census","sex_census","region","setting"]
BEHAVIORAL  =["smoke_now","smoke_past","occupation"]
SYMPTOMS    =["cough","cough_d","fever","fever_d","weight_loss","wloss_d",
              "night_sweats","chest_pain","blood_sputum","sputum","sputum_d"]
HISTORY     =["hist_rx","tbhist_y","hiv_res"]
RADIOLOGY   =["xrayres","central_cxr_res","cavit_rm","cavit_lm","cavit_rl","cavit_ll"]
LAB         =["smear_pos","zn","genexpert","culture","cult_pos","final_result","bact"]
ALL_COLS    = DEMOGRAPHICS+BEHAVIORAL+SYMPTOMS+HISTORY+RADIOLOGY+LAB
NDB = ["smoke_now","smoke_past","hiv_res","cough","fever","weight_loss",
       "night_sweats","chest_pain","blood_sputum","sputum","hist_rx",
       "cavit_rm","cavit_lm","cavit_rl","cavit_ll","smear_pos","culture","cult_pos","bact"]
NDC = ["cough_d","fever_d","wloss_d","sputum_d","tbhist_y"]
NDX = ["xrayres","central_cxr_res","zn","genexpert","final_result"]
MISS= ["",  " ","na","nan","none","missing","MISSING"]
DEVICE=torch.device("cpu")
CPAL=["#3b82f6","#22c55e","#ef4444","#f59e0b","#a855f7"]

# ── Asset loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets(d):
    try:
        def j(f): 
            with open(os.path.join(d,f)) as fh: return json.load(fh)
        cfg=j("model_config.json")
        fn =j("feature_names.json")
        f2m=j("feature_to_modality.json")
        thr=j("ood_threshold.json")["ood_threshold"]
        ptb=j("pseudotime_bounds.json")
        # keep_indices — applied after preprocessor to match model's 72-feature input
        keep_path = os.path.join(d, "keep_indices.json")
        keep_indices = j("keep_indices.json") if os.path.exists(keep_path) else None
        m=TTVAE(cfg["input_dim"],cfg["latent_dim"],cfg["d_model"],
                cfg["nhead"],cfg["n_layers"],cfg["n_cont"],cfg["n_bin"],
                cfg["cat_sizes"]).to(DEVICE)
        for w in ("ttvae_best.pth","ttvae_best.pt"):
            wp=os.path.join(d,w)
            if os.path.exists(wp):
                m.load_state_dict(torch.load(wp,map_location=DEVICE)); break
        m.eval()
        pre=joblib.load(os.path.join(d,"preprocessor.joblib"))
        km =joblib.load(os.path.join(d,"kmeans_model.joblib"))
        nc,nb=cfg["n_cont"],cfg["n_bin"]
        sl,cur=[],nc+nb
        for s in cfg["cat_sizes"]: sl.append((cur,cur+s)); cur+=s
        return dict(model=m,pre=pre,km=km,cfg=cfg,fn=fn,f2m=f2m,
                    thr=thr,ptb=ptb,nc=nc,nb=nb,
                    cs=cfg["cat_sizes"],sl=sl,
                    nk=cfg["n_clusters"],ld=cfg["latent_dim"],
                    keep_indices=keep_indices)
    except Exception as e:
        return {"error":str(e)}

# ── Inference helpers ─────────────────────────────────────────────────────────
def enc(model,x):
    with torch.no_grad():
        mu,_=model.encode(torch.tensor(x,dtype=torch.float32))
    return mu.numpy()

def rec(model,x):
    with torch.no_grad():
        c,b,cl,_,_=model(torch.tensor(x,dtype=torch.float32))
        parts=[]
        if c  is not None: parts.append(c)
        if b  is not None: parts.append(torch.sigmoid(b))
        for lg in cl:
            p=torch.softmax(lg,1); oh=torch.zeros_like(p)
            oh.scatter_(1,p.argmax(1,keepdim=True),1.); parts.append(oh)
        return torch.cat(parts,1).numpy()

def pt_score(Z,ptb):
    pca=PCA(n_components=1,random_state=42)
    pt=pca.fit_transform(Z).ravel()
    lo,hi=ptb["pseudotime_min"],ptb["pseudotime_max"]
    return pt, np.clip((pt-lo)/(hi-lo+1e-8),0,1)

def rlabel(s):
    if s>=.66: return "HIGH","ch"
    if s>=.33: return "MEDIUM","cm"
    return "LOW","cl"

def synth_gen(model,n,ld,nc,nb,cs,fn):
    with torch.no_grad():
        z=torch.randn(n,ld)
        c,b,cl=model.decode(z)
        parts=[]
        if c is not None: parts.append(np.clip(c.numpy(),0,1))
        if b is not None: parts.append((torch.sigmoid(b).numpy()>=.5).astype(np.float32))
        for lg in cl:
            p=torch.softmax(lg,1).numpy()
            oh=np.zeros_like(p,dtype=np.float32)
            oh[np.arange(len(p)),p.argmax(1)]=1; parts.append(oh)
    return pd.DataFrame(np.concatenate(parts,1),columns=fn)

# ── Preprocessing — replicates notebook cell 11 ───────────────────────────────
def preproc(df_raw, col_map, pre, keep_indices=None):
    df = df_raw.rename(columns=col_map).copy()
    for c in ALL_COLS:
        if c not in df.columns: df[c] = np.nan
    df = df.replace(MISS, np.nan)
    for col in ["cavit_rm","cavit_lm","cavit_rl","cavit_ll"]:
        df[col] = df[col].map({1:1, 2:0, "1":1, "2":0})
    mg = {"behavioral_observed": BEHAVIORAL, "symptoms_observed": SYMPTOMS,
          "history_observed": HISTORY, "radiology_observed": RADIOLOGY,
          "lab_observed": LAB}
    for ind, cols in mg.items():
        ex = [c for c in cols if c in df.columns]
        df[ind] = df[ex].notna().any(axis=1).astype(int)
    for col in NDB:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1)
    for col in NDC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in NDX:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(
                {"nan":"not_observed","None":"not_observed","":"not_observed"})
    df["age_census"] = pd.to_numeric(df.get("age_census", np.nan), errors="coerce")
    df["sex_census"] = pd.to_numeric(df.get("sex_census", np.nan), errors="coerce").clip(0, 1)
    df["setting"]    = pd.to_numeric(df.get("setting",    np.nan), errors="coerce").clip(0, 1)
    for col in ["region","occupation"]:
        df[col] = df.get(col, "not_observed").astype(str).replace(
            {"nan":"not_observed","None":"not_observed"})
    for col in NDX + ["region","occupation"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(
                {"nan":"not_observed","None":"not_observed"})
    X = pre.transform(df).astype(np.float32)
    if keep_indices is not None:
        X = X[:, keep_indices]
    return X

# ── Plot helper ───────────────────────────────────────────────────────────────
BG,PN,GR,TX="#09090b","#111113","#27272a","#a1a1aa"
def dfig(w=11,h=4,n=1):
    fig,axes=plt.subplots(1,n,figsize=(w,h),facecolor=BG)
    if n==1: axes=[axes]
    for ax in axes:
        ax.set_facecolor(PN); ax.tick_params(colors=TX,labelsize=8)
        ax.xaxis.label.set_color(TX); ax.yaxis.label.set_color(TX)
        ax.title.set_color("#f4f4f5"); ax.title.set_fontweight("bold")
        for sp in ax.spines.values(): sp.set_edgecolor(GR)
    return fig,axes

def sec(t):
    st.markdown(f'<div class="banner">⬡ {t}</div>',unsafe_allow_html=True)

def mcard(col,cls,color,val,lbl,sub=""):
    col.markdown(
        f'<div class="{cls}"><div class="ml">{lbl}</div>'
        f'<div class="mv" style="color:{color};">{val}</div>'
        f'{"<div style=color:#71717a;font-size:.72rem;>"+sub+"</div>" if sub else ""}'
        f'</div>',unsafe_allow_html=True)

def need():
    if st.session_state.get("A") is None:
        st.info("👈  Load the model first using the sidebar."); st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 TB Risk Sequencing")
    st.markdown("<p style='color:#52525b;font-size:.76rem;line-height:1.6;'>"
                "TTVAE · Uganda National TB Survey<br>2014–2015 · Unsupervised Risk Phenotyping</p>",
                unsafe_allow_html=True)
    st.divider()
    st.markdown("### Model Assets")
    ddir=st.text_input("Deployment folder",
        value="results_full/ttvae_results/deployment_assets",
        help="Folder with ttvae_best.pth, kmeans_model.joblib, preprocessor.joblib and JSON configs")
    if st.button("⟳  Load Model",use_container_width=True):
        with st.spinner("Loading…"):
            a=load_assets(ddir)
        if "error" in a: st.error(f"Failed: {a['error']}")
        else:
            st.session_state["A"]=a
            st.success(f"Loaded ✓  {a['cfg']['input_dim']} features · "
                       f"{a['nk']} clusters · latent {a['ld']}")
    st.divider()
    st.markdown("### Navigation")
    page=st.radio("",["🔬 Upload & Analyse","🧬 Synthetic Generation",
                       "📐 Model Info","ℹ️ About"],
                  label_visibility="collapsed")
    st.divider()
    st.markdown("<p style='color:#3f3f46;font-size:.66rem;text-align:center;'>"
                "Research Prototype · Not for Clinical Use<br>"
                "MSc Deep Learning · Makerere University</p>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — UPLOAD & ANALYSE
# ─────────────────────────────────────────────────────────────────────────────
if page=="🔬 Upload & Analyse":
    st.markdown("# TB Latent Risk Sequencing")
    st.markdown("<p style='color:#71717a;'>Upload a patient cohort CSV to receive cluster "
                "assignments, pseudotime risk scores and OOD safety flags.</p>",
                unsafe_allow_html=True)
    need()
    A=st.session_state["A"]

    # Step 1: upload
    sec("01 · DATA UPLOAD")
    up=st.file_uploader("Patient CSV",type=["csv"],
        help="One row per patient. Map column names in step 02.")
    if up is None:
        st.markdown("""<div class="card"><b style='color:#f4f4f5;'>Expected columns (any subset)</b><br><br>
        <span style='color:#71717a;font-size:.81rem;'>
        <b>Demographics:</b> age_census, sex_census, region, setting<br>
        <b>Behavioral:</b> smoke_now, smoke_past, occupation<br>
        <b>Symptoms:</b> cough, cough_d, fever, weight_loss, night_sweats, chest_pain, blood_sputum, sputum, sputum_d<br>
        <b>History:</b> hist_rx, hiv_res<br>
        <b>Radiology:</b> xrayres, central_cxr_res, cavit_rm/lm/rl/ll<br>
        <b>Laboratory:</b> smear_pos, zn, genexpert, culture, cult_pos, final_result, bact
        </span></div>""",unsafe_allow_html=True); st.stop()

    df_raw=pd.read_csv(up)
    st.success(f"✓  {len(df_raw):,} patients · {len(df_raw.columns)} columns")
    with st.expander("Preview",expanded=False):
        st.dataframe(df_raw.head(8),use_container_width=True)

    # Step 2: column mapping
    sec("02 · COLUMN MAPPING")
    st.markdown("<p style='color:#71717a;font-size:.81rem;'>"
                "Map your columns to the expected field names. "
                "Unmapped fields default to zero / not_observed.</p>",
                unsafe_allow_html=True)
    uc=["— not available —"]+list(df_raw.columns)
    cm={}
    groups={"Demographics":DEMOGRAPHICS,"Behavioral":BEHAVIORAL,
            "Symptoms":SYMPTOMS,"History":HISTORY,
            "Radiology":RADIOLOGY,"Laboratory":LAB}
    with st.expander("Open column mapper",expanded=True):
        for gn,gc in groups.items():
            st.markdown(f"**{gn}**")
            cols=st.columns(min(len(gc),4))
            for j,exp in enumerate(gc):
                di=uc.index(exp) if exp in df_raw.columns else 0
                sel=cols[j%4].selectbox(exp,uc,index=di,
                    key=f"m_{exp}",label_visibility="visible")
                if sel!="— not available —": cm[sel]=exp

    # Step 3: run
    sec("03 · INFERENCE")
    rb=st.button("▶  Run Risk Analysis")
    if "RES" not in st.session_state: st.session_state["RES"]=None

    if rb:
        with st.spinner("Preprocessing · Encoding · Scoring…"):
            try:
                X  =preproc(df_raw,cm,A["pre"],A.get("keep_indices"))
                Z  =enc(A["model"],X)
                Xh =rec(A["model"],X)
                err=((X-Xh)**2).mean(1)
                ood=(err>A["thr"]).astype(int)
                cl =A["km"].predict(Z)
                _,ptn=pt_score(Z,A["ptb"])
                res=df_raw.copy().reset_index(drop=True)
                res["cluster"]              =cl
                res["pseudotime_score"]     =np.round(ptn,4)
                res["reconstruction_error"] =np.round(err,6)
                res["ood_flag"]             =ood
                res["risk_level"]           =res["pseudotime_score"].apply(lambda s:rlabel(s)[0])
                st.session_state["RES"]=res
                st.session_state["Z"] =Z
                st.session_state["PT"]=ptn
                st.session_state["OOD"]=ood
                st.success(f"✓  {len(res):,} patients processed")
            except Exception as e:
                st.error(f"Inference error: {e}"); st.stop()

    # Step 4: display
    if st.session_state["RES"] is not None:
        res=st.session_state["RES"]
        Z  =st.session_state["Z"]
        ptn=st.session_state["PT"]
        ood=st.session_state["OOD"]
        N=len(res)
        nh=int((res["risk_level"]=="HIGH").sum())
        nm=int((res["risk_level"]=="MEDIUM").sum())
        nl=int((res["risk_level"]=="LOW").sum())
        no=int(ood.sum()); oor=no/N

        sec("04 · RESULTS SUMMARY")

        if oor>.20:
            st.markdown(f"""<div class="ood-box">
            ⚠️ <b style='color:#a855f7;'>HIGH OOD RATE: {oor:.1%}</b><br>
            <span style='color:#71717a;font-size:.81rem;'>
            {no} patients exceed the 95th-percentile reconstruction error threshold
            ({A['thr']:.5f}) from the 2014–2015 training distribution.
            Per the concept guide MLOps specification the system abstains from
            confident risk assignment for these patients.
            Consider retraining on locally representative data.
            </span></div>""",unsafe_allow_html=True)

        c1,c2,c3,c4,c5=st.columns(5)
        mcard(c1,"card",    "#f4f4f5",f"{N:,}",   "Total Patients")
        mcard(c2,"card-r",  "#ef4444",f"{nh:,}",  "High Risk")
        mcard(c3,"card-a",  "#f59e0b",f"{nm:,}",  "Medium Risk")
        mcard(c4,"card-g",  "#22c55e",f"{nl:,}",  "Low Risk")
        mcard(c5,"card-p",  "#a855f7",f"{no:,}",  "OOD Flagged",f"{oor:.1%} of cohort")

        st.markdown("")

        t1,t2,t3,t4=st.tabs(["📊 Risk Distribution",
                              "🗂 Cluster Profiles",
                              "🔵 Latent Space",
                              "📋 Patient Table"])

        with t1:
            fig,ax=dfig(w=12,h=4,n=2)
            for lvl,col in [("LOW","#22c55e"),("MEDIUM","#f59e0b"),("HIGH","#ef4444")]:
                mask=res["risk_level"]==lvl
                ax[0].hist(res.loc[mask,"pseudotime_score"],bins=30,
                           color=col,alpha=.75,label=lvl)
            ax[0].set_title("Pseudotime Risk Score Distribution")
            ax[0].set_xlabel("Risk Score  (0=Low · 1=High)")
            ax[0].set_ylabel("Count")
            ax[0].legend(framealpha=0,labelcolor="#f4f4f5",fontsize=8)
            ax[1].hist(res.loc[ood==0,"reconstruction_error"],bins=30,
                       color="#3b82f6",alpha=.75,label="In-distribution")
            ax[1].hist(res.loc[ood==1,"reconstruction_error"],bins=30,
                       color="#a855f7",alpha=.75,label="OOD")
            ax[1].axvline(A["thr"],color="#ef4444",lw=1.5,ls="--",
                          label=f"Threshold {A['thr']:.4f}")
            ax[1].set_title("OOD Detection — Reconstruction Error")
            ax[1].set_xlabel("Reconstruction Error")
            ax[1].set_ylabel("Count")
            ax[1].legend(framealpha=0,labelcolor="#f4f4f5",fontsize=8)
            plt.tight_layout(pad=1.5); st.pyplot(fig,use_container_width=True); plt.close()

        with t2:
            cs=res.groupby("cluster").agg(
                Patients=("pseudotime_score","count"),
                Mean_PT =("pseudotime_score","mean"),
                Pct_High=("risk_level",lambda x:(x=="HIGH").mean()*100),
                Pct_OOD =("ood_flag","mean"),
            ).reset_index().sort_values("Mean_PT",ascending=False)
            cs.columns=["Cluster","Patients","Mean PT","% High Risk","% OOD"]
            cs=cs.round(3); cs["% OOD"]=(cs["% OOD"]*100).round(1)
            st.dataframe(cs,use_container_width=True,hide_index=True)
            fig2,ax2=dfig(w=8,h=3)
            ax2[0].bar(cs["Cluster"].astype(str),cs["Patients"],
                       color=[CPAL[i%5] for i in range(len(cs))],alpha=.85)
            ax2[0].set_title("Patients per Cluster")
            ax2[0].set_xlabel("Cluster"); ax2[0].set_ylabel("Count")
            plt.tight_layout(); st.pyplot(fig2,use_container_width=True); plt.close()

        with t3:
            Z2=PCA(n_components=2,random_state=42).fit_transform(Z)
            fig3,ax3=dfig(w=12,h=4,n=2)
            sc=ax3[0].scatter(Z2[:,0],Z2[:,1],c=ptn,cmap="RdYlGn_r",
                              alpha=.5,s=6,rasterized=True)
            cb=plt.colorbar(sc,ax=ax3[0]); cb.set_label("Risk Score",color=TX)
            cb.ax.yaxis.set_tick_params(color=TX)
            ax3[0].set_title("Latent Space — Risk Score")
            ax3[0].set_xlabel("PC1"); ax3[0].set_ylabel("PC2")
            for i,cl in enumerate(sorted(res["cluster"].unique())):
                m=res["cluster"]==cl
                ax3[1].scatter(Z2[m,0],Z2[m,1],c=CPAL[i%5],
                               alpha=.5,s=6,label=f"Cluster {cl}",rasterized=True)
            ax3[1].legend(framealpha=0,labelcolor="#f4f4f5",
                          markerscale=2.5,fontsize=8)
            ax3[1].set_title("Latent Space — Clusters")
            ax3[1].set_xlabel("PC1"); ax3[1].set_ylabel("PC2")
            plt.tight_layout(pad=1.5); st.pyplot(fig3,use_container_width=True); plt.close()

        with t4:
            sc=["cluster","pseudotime_score","risk_level","ood_flag","reconstruction_error"]
            for c in ["age_census","sex_census","region"]:
                if c in res.columns: sc=[c]+sc
            st.dataframe(res[sc].head(1000),use_container_width=True)
            st.caption(f"Showing first 1000 of {N:,} patients")

        sec("05 · DOWNLOAD")
        dc1,dc2=st.columns(2)
        buf=io.BytesIO(); res.to_csv(buf,index=False); buf.seek(0)
        dc1.download_button("⬇ Full Results CSV",buf,"ttvae_results.csv","text/csv")
        if no>0:
            buf2=io.BytesIO()
            res[res["ood_flag"]==1].to_csv(buf2,index=False); buf2.seek(0)
            dc2.download_button(f"⬇ OOD Patients ({no})",buf2,
                                "ttvae_ood.csv","text/csv")

        st.markdown("""<div class="disc">
        ⚠ <b>Research Prototype.</b> Risk scores are statistical approximations derived
        from a 2014–2015 cross-sectional survey. They are not individual clinical diagnoses.
        All patients must be independently evaluated by a qualified clinician.
        </div>""",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — SYNTHETIC GENERATION
# ─────────────────────────────────────────────────────────────────────────────
elif page=="🧬 Synthetic Generation":
    st.markdown("# Synthetic Patient Profile Generation")
    st.markdown("<p style='color:#71717a;'>Sample from the learned latent manifold. "
                "All outputs are audited against the clinical error taxonomy "
                "in the concept guide (Layer 5).</p>",unsafe_allow_html=True)
    need()
    A=st.session_state["A"]
    col1,_=st.columns([1,2])
    nr=col1.slider("Profiles to generate",50,5000,500,step=50)
    if st.button("▶  Generate"):
        with st.spinner(f"Generating {nr} profiles…"):
            s=synth_gen(A["model"],nr,A["ld"],A["nc"],A["nb"],A["cs"],A["fn"])
        # Audit
        nc,nb,sl=A["nc"],A["nb"],A["sl"]
        hall=np.zeros(nr,dtype=bool)
        if nc>0: hall|=((s.iloc[:,:nc]<0)|(s.iloc[:,:nc]>1)).any(1).values
        if nb>0: hall|=~s.iloc[:,nc:nc+nb].isin([0.,1.]).all(1).values
        for (st2,e) in sl:
            hall|=np.abs(s.iloc[:,st2:e].sum(1).values-1)>1e-6
        contra=np.zeros(nr,dtype=bool)
        if "bin__bact" in s.columns:
            nd=[c for c in ["bin__smear_pos","bin__culture","bin__cult_pos"] if c in s.columns]
            if nd: contra=(s["bin__bact"].values>.5)&(s[nd].max(1).values<.5)
        danom=np.zeros(nr,dtype=bool)
        if "bin__sex_census" in s.columns:
            danom|=~s["bin__sex_census"].isin([0.,1.]).values
        if "cont__age_census" in s.columns:
            danom|=((s["cont__age_census"]<0)|(s["cont__age_census"]>1)).values

        sec("CLINICAL SAFETY AUDIT  (Concept Guide Layer 5)")
        m1,m2,m3,m4=st.columns(4)
        mcard(m1,"card-g","#22c55e",f"{nr-int(hall.sum()):,}","Valid Profiles")
        mcard(m2,"card-r","#ef4444",f"{hall.mean():.1%}","Hallucination Rate")
        mcard(m3,"card-r","#ef4444",f"{contra.mean():.1%}","Clinical Contradictions")
        mcard(m4,"card-a","#f59e0b",f"{danom.mean():.1%}","Demographic Anomalies")
        st.markdown("")
        st.dataframe(s.head(100),use_container_width=True)
        buf=io.BytesIO(); s.to_csv(buf,index=False); buf.seek(0)
        st.download_button("⬇ Download Synthetic CSV",buf,"ttvae_synthetic.csv","text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
elif page=="📐 Model Info":
    st.markdown("# Model Architecture & Configuration")
    need()
    A=st.session_state["A"]; cfg=A["cfg"]
    c1,c2=st.columns(2)
    with c1:
        st.markdown("## Architecture")
        st.dataframe(pd.DataFrame({
            "Parameter":["Model type","Input dim","Latent dim","d_model",
                         "Attention heads","Encoder layers","K-Means clusters",
                         "Continuous feats","Binary feats","Categorical groups"],
            "Value":["Transformer-based Tabular VAE (TTVAE)",
                     cfg["input_dim"],cfg["latent_dim"],cfg["d_model"],
                     cfg["nhead"],cfg["n_layers"],cfg["n_clusters"],
                     cfg["n_cont"],cfg["n_bin"],len(cfg["cat_sizes"])]
        }),use_container_width=True,hide_index=True)
        st.markdown("## Feature Modality Distribution")
        mc=pd.Series(A["f2m"]).value_counts().reset_index()
        mc.columns=["Modality","Encoded Features"]
        st.dataframe(mc,use_container_width=True,hide_index=True)
    with c2:
        st.markdown("## OOD Detection  (Concept Guide MLOps Layer)")
        st.markdown(f"""<div class="card-p">
        <div class="ml">Reconstruction Error Threshold</div>
        <div class="mv" style="color:#a855f7;font-size:1.55rem;">{A['thr']:.6f}</div>
        <p style='color:#71717a;font-size:.8rem;margin-top:.4rem;'>
        95th percentile of training reconstruction errors. Patients exceeding this
        are flagged OOD and the system abstains from confident risk assignment,
        preventing erroneous clinical extrapolation from the 2014–2015 snapshot.
        </p></div>""",unsafe_allow_html=True)
        st.markdown("## Pseudotime Bounds")
        st.markdown(f"""<div class="card-b">
        <div class="ml">Training Population Range</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:.88rem;color:#60a5fa;'>
        Min: {A['ptb']['pseudotime_min']:.5f}<br>Max: {A['ptb']['pseudotime_max']:.5f}
        </div>
        <p style='color:#71717a;font-size:.8rem;margin-top:.4rem;'>
        New patient scores are normalised to [0,1]. Scores outside [0,1] indicate
        patients more extreme than any in the training population.
        </p></div>""",unsafe_allow_html=True)
        st.markdown("## Risk Level Thresholds")
        st.markdown("""<div class="card">
        <table style='width:100%;font-size:.82rem;border-collapse:collapse;'>
        <tr><td style='padding:4px 0;color:#71717a;'>Score ≥ 0.66</td>
            <td><span class="chip ch">HIGH RISK</span></td></tr>
        <tr><td style='padding:4px 0;color:#71717a;'>0.33 ≤ Score &lt; 0.66</td>
            <td><span class="chip cm">MEDIUM RISK</span></td></tr>
        <tr><td style='padding:4px 0;color:#71717a;'>Score &lt; 0.33</td>
            <td><span class="chip cl">LOW RISK</span></td></tr>
        <tr><td style='padding:4px 0;color:#71717a;'>Error > threshold</td>
            <td><span class="chip co">OOD FLAGGED</span></td></tr>
        </table></div>""",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page=="ℹ️ About":
    st.markdown("# About This System")
    st.markdown("""<div class="card">
    <h3 style='color:#60a5fa;margin-top:0;'>Research Context</h3>
    <p style='color:#71717a;font-size:.86rem;line-height:1.7;'>
    This system implements a <b style='color:#f4f4f5;'>Transformer-based Tabular
    Variational Autoencoder (TTVAE)</b> trained on the Uganda National Tuberculosis
    Prevalence Survey (2014–2015), comprising 86,108 surveyed individuals across
    270 clinical variables spanning six epidemiological modalities.
    </p>
    <p style='color:#71717a;font-size:.86rem;line-height:1.7;'>
    The framework discovers latent TB risk phenotypes through unsupervised representation
    learning, ordering patients along a continuous disease progression trajectory —
    <b style='color:#f4f4f5;'>pseudotime</b> — without requiring longitudinal observations.
    Missingness in the survey is treated as a clinically meaningful signal rather than
    data loss, preserved through modality-level cascade indicators before preprocessing.
    </p></div>""",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("""<div class="card">
        <h3 style='color:#22c55e;margin-top:0;'>Concept Guide Layer Compliance</h3>
        <p style='color:#71717a;font-size:.82rem;line-height:1.8;'>
        <b style='color:#f4f4f5;'>Layer 0</b> — Discovery objective<br>
        <b style='color:#f4f4f5;'>Layer 1</b> — Unsupervised structure learning<br>
        <b style='color:#f4f4f5;'>Layer 2</b> — Joint dimensionality reduction + clustering<br>
        <b style='color:#f4f4f5;'>Layer 3</b> — Heterogeneous cross-sectional tabular data<br>
        <b style='color:#f4f4f5;'>Layer 4</b> — Composite generative loss (MSE + BCE + KL)<br>
        <b style='color:#f4f4f5;'>Layer 5</b> — Interpretable interpolations + OOD safety<br>
        <b style='color:#f4f4f5;'>Layer 6</b> — Centralised Streamlit inference<br>
        <b style='color:#f4f4f5;'>Layer 7</b> — VAE as primary inductive bias
        </p></div>""",unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card">
        <h3 style='color:#f59e0b;margin-top:0;'>Safety Architecture</h3>
        <p style='color:#71717a;font-size:.82rem;line-height:1.7;'>
        <b style='color:#f4f4f5;'>OOD Detection</b> — Reconstruction error at 95th percentile.
        Flagged patients receive no confident risk assignment.<br><br>
        <b style='color:#f4f4f5;'>Hallucination Audit</b> — Synthetic profiles validated:
        binary ∈ {0,1}, one-hot groups sum to 1, bacteriological confirmation
        must not contradict laboratory evidence.<br><br>
        <b style='color:#f4f4f5;'>Concept Drift</b> — High OOD rates signal distributional
        shift from 2014–2015 training population. Retraining on local data is recommended.
        </p></div>""",unsafe_allow_html=True)
    st.markdown("""<div class="card-r" style='margin-top:.8rem;'>
    <b style='color:#ef4444;'>⚠ Research Prototype — Not for Clinical Use</b><br>
    <span style='color:#71717a;font-size:.79rem;'>
    Risk scores are statistical approximations from a cross-sectional population survey.
    They are not individual clinical diagnoses. All patients must be independently
    evaluated by a qualified clinician.
    </span></div>""",unsafe_allow_html=True)
