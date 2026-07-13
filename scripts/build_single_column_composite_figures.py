"""Render the requested single-column empirical composites (no workflow diagram).

The figures are deliberately data-driven: every panel is derived from a named
released record.  They are standalone layout assets and are not inserted into
the five-page letter automatically, because two 88-mm x ~145-mm figures cannot
fit in that page budget together with the manuscript text and references.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
DATA = OUT / "figure_source_data"
OUT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

COL = {"visdrone": "#0072B2", "uavdt": "#D55E00", "aitod": "#009E73", "dota_v2_val": "#7F7F7F"}
LAB = {"visdrone": "VisDrone", "uavdt": "Local UAVDT", "aitod": "AI-TOD", "dota_v2_val": "DOTA-v2"}
MARK = {"visdrone": "o", "uavdt": "s", "aitod": "^", "dota_v2_val": "D"}
LS = {"visdrone": "-", "uavdt": "--", "aitod": "-.", "dota_v2_val": ":"}
POLICIES = ["include_all", "exclude_without_ignore_protection", "exclude_with_ignore_protection"]
PSHORT = {"include_all": "I", "exclude_without_ignore_protection": "O", "exclude_with_ignore_protection": "N"}


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.7, "axes.labelsize": 7.2, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0, "axes.linewidth": .6, "lines.linewidth": .85,
        "xtick.major.width": .55, "ytick.major.width": .55, "xtick.major.size": 2.3, "ytick.major.size": 2.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight", "savefig.pad_inches": .02,
    })


def tidy(ax, grid="both"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.2)
    if grid: ax.grid(axis=grid, color="#DADADA", lw=.38, zorder=0)


def panel(ax, text):
    ax.text(-.16, 1.035, text, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def save(fig, stem):
    # Override the global tight-crop style: the publication asset must retain
    # its declared 88-mm canvas rather than expand to off-axis panel labels.
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        for ext in ("pdf", "svg"):
            fig.savefig(OUT / f"{stem}.{ext}", bbox_inches=None, pad_inches=0)
        fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def ecdf(values):
    x = np.sort(np.asarray(values, float)); return x, np.arange(1, len(x)+1)/len(x)


def fig_a():
    records = pd.read_parquet(ROOT / "outputs/scale/box_scale_records.parquet")
    coverage = pd.read_csv(ROOT / "outputs/coverage/source_common_intersection.csv").set_index("source")
    cards = pd.read_csv(ROOT / "outputs/structural/source_cards.csv").set_index("dataset_name")
    selection = pd.read_csv(ROOT / "outputs/coverage/aitod_coverage_selection_audit.csv").set_index("group")
    order = ["visdrone", "uavdt", "aitod", "dota_v2_val"]
    audited = ["visdrone", "uavdt", "aitod"]
    rows=[]
    for s in order:
        q=records[records.source.eq(s)]
        for x in q.max_side_px: rows.append((s,"absolute",float(x)))
        for x in q.normalized_side: rows.append((s,"normalized",float(x)))
    pd.DataFrame(rows, columns=["source","domain","support"]).to_csv(DATA / "Fig_A_source_data.csv", index=False)

    # 88 mm x 145 mm, as specified. Workflow panel intentionally omitted.
    fig = plt.figure(figsize=(88/25.4, 145/25.4))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[.98, .82, 1.10], hspace=.66, wspace=.48)
    axa, axb, axc, axd, axe, axf = [fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]

    retain_abs={s:100*np.mean(records[records.source.eq(s)].max_side_px.to_numpy(float)>=24) for s in audited}
    retain_norm={s:100*np.mean(records[records.source.eq(s)].normalized_side.to_numpy(float)>=.015) for s in audited}
    for s in order:
        x,y=ecdf(records[records.source.eq(s)].max_side_px); axa.plot(x,y,color=COL[s],ls=LS[s],label=LAB[s])
        x,y=ecdf(records[records.source.eq(s)].normalized_side); axb.plot(x,y,color=COL[s],ls=LS[s])
    axa.axvline(24,color="#333",ls="--",lw=.7); axb.axvline(.015,color="#333",ls="--",lw=.7)
    axa.annotate("24 px",(24,.985),xycoords=("data","axes fraction"),xytext=(-2,-2),textcoords="offset points",ha="right",va="top",fontsize=6,bbox=dict(fc="white",ec="none",alpha=.88,pad=.25))
    axb.annotate(".015",(.015,.985),xycoords=("data","axes fraction"),xytext=(-2,-2),textcoords="offset points",ha="right",va="top",fontsize=6,bbox=dict(fc="white",ec="none",alpha=.88,pad=.25))
    for s,(x,y) in {"visdrone":(54,.26),"uavdt":(29,.67),"aitod":(12,.42)}.items(): axa.text(x,y,f"{retain_abs[s]:.1f}%",color=COL[s],fontsize=6.1,bbox=dict(fc="white",ec="none",alpha=.72,pad=.18))
    for s,(x,y) in {"visdrone":(.060,.25),"uavdt":(.028,.58),"aitod":(.006,.72)}.items(): axb.text(x,y,f"{retain_norm[s]:.1f}%",color=COL[s],fontsize=6.1,bbox=dict(fc="white",ec="none",alpha=.72,pad=.18))
    for ax,xlabel in [(axa,"Max side (px)"),(axb,"Normalized max side")]:
        ax.set(xscale="log",ylim=(0,1.01),xlabel=xlabel); tidy(ax); panel(ax,"(a)" if ax is axa else "(b)")
    axa.set(xlim=(3,600),ylabel="ECDF"); axb.set(xlim=(.001,.5),yticklabels=[])
    handles=[Line2D([0],[0],color=COL[s],ls=LS[s],label=LAB[s]) for s in order]
    axa.legend(handles=handles,frameon=False,ncol=2,loc="lower right",handlelength=1.7,columnspacing=.55,fontsize=5.5)

    # Headline retained support bars.
    x=np.arange(3); width=.32
    for i,s in enumerate(audited):
        axc.bar(i-width/2,retain_abs[s],width,color=COL[s],ec="#333",lw=.3)
        axc.bar(i+width/2,retain_norm[s],width,color=COL[s],ec="#333",lw=.3,hatch="///")
        axc.text(i-width/2,retain_abs[s]+2,f"{retain_abs[s]:.1f}",ha="center",va="bottom",fontsize=5.5,rotation=90)
        axc.text(i+width/2,retain_norm[s]+2,f"{retain_norm[s]:.1f}",ha="center",va="bottom",fontsize=5.5,rotation=90)
    axc.set(xticks=x,xticklabels=["VisDrone","UAVDT","AI-TOD"],ylim=(0,112),ylabel="Retained boxes (%)")
    tidy(axc,"y"); panel(axc,"(c)")
    axc.legend([mpl.patches.Patch(fc="#666",label="24 px"),mpl.patches.Patch(fc="white",ec="#333",hatch="///",label=".015")], ["24 px",".015"],frameon=False,loc="upper right",fontsize=5.4,handlelength=1.2)

    # Coverage dumbbell on a log image-count axis.
    y=np.arange(4)[::-1]
    totals=np.array([coverage.loc["visdrone","source_evaluation_images"],coverage.loc["uavdt","source_evaluation_images"],coverage.loc["aitod","source_evaluation_images"],cards.loc["dota_v2_val","image_count"]],float)
    cov=np.array([coverage.loc["visdrone","common_prediction_covered_images"],coverage.loc["uavdt","common_prediction_covered_images"],coverage.loc["aitod","common_prediction_covered_images"],np.nan])
    for i,s in enumerate(order):
        axd.scatter(totals[i],y[i],s=18,marker="o",fc="white",ec=COL[s],lw=.7,zorder=3)
        if np.isfinite(cov[i]):
            axd.plot([cov[i],totals[i]],[y[i],y[i]],color=COL[s],lw=1.2,zorder=2)
            axd.scatter(cov[i],y[i],s=18,marker="s",fc=COL[s],ec=COL[s],lw=.4,zorder=4)
            label=f"{int(cov[i]):,}/{int(totals[i]):,}"
        else: label=None
        if label:
            axd.annotate(label,(totals[i],y[i]),xytext=(2,2),textcoords="offset points",fontsize=4.7,ha="left")
    axd.set(xscale="log",xlim=(200,9000),yticks=y,yticklabels=["VisDrone","UAVDT","AI-TOD","DOTA-v2"],xlabel="Images")
    tidy(axd,"x"); panel(axd,"(d)")

    # AI-TOD coverage selection microbars.
    mini=GridSpecFromSubplotSpec(2,2,subplot_spec=gs[2,0],wspace=.85,hspace=1.0)
    axe.remove(); mini_axes=[fig.add_subplot(mini[i,j]) for i in range(2) for j in range(2)]
    metric_specs=[("Images","images",2000,""),("Boxes","vehicle_boxes",46000,""),("Boxes/image","boxes_per_image",50,""),("Median side","median_max_side_px",20," px")]
    for ax,(title,key,top,suffix) in zip(mini_axes,metric_specs):
        vals=[selection.loc["covered",key],selection.loc["not_covered",key]]
        ax.bar([0,1],vals,color=[COL["aitod"],"white"],ec=COL["aitod"],lw=.55,hatch=["","///"])
        for k,v in enumerate(vals):
            label=f"{v:,.1f}" if key=="boxes_per_image" else f"{v:,.0f}{suffix}"
            ax.text(k,v*.55,label,ha="center",va="center",fontsize=4.5,rotation=90,color="white" if k==0 else "#166B54")
        ax.set(xticks=[0,1],xticklabels=["Cov.","Non."],ylim=(0,top)); ax.set_title(title,fontsize=5.3,pad=1.0)
        ax.set_yticks([]); ax.tick_params(axis="x",labelsize=4.8)
        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);ax.grid(axis="y",color="#E0E0E0",lw=.3)
    mini_axes[0].text(-.42,1.10,"(e)",transform=mini_axes[0].transAxes,fontsize=8,fontweight="bold",va="bottom")

    # Claim-scope glyph matrix.
    matrix=[("VisDrone",[1,1,1]),("Local UAVDT",[1,1,1]),("AI-TOD*",[1,1,0]),("DOTA-v2",[1,0,0])]
    cols=["Structural","Policy","Rank"]
    for iy,(name,vals) in enumerate(matrix[::-1]):
        axf.text(-.62,iy,name,ha="right",va="center",fontsize=5.8)
        for ix,v in enumerate(vals):
            axf.scatter(ix,iy,s=32,marker="o",fc=COL["aitod"] if v else "white",ec="#444",lw=.55)
            axf.text(ix,iy,"✓" if v else "–",ha="center",va="center",fontsize=6,color="white" if v else "#555")
    # Asterisk retains the coverage-conditioned qualification without covering the glyph.
    axf.text(1,1,"*",ha="center",va="center",fontsize=7,color="white",fontweight="bold")
    axf.set(xlim=(-.85,2.55),ylim=(-.55,3.65),xticks=range(3),xticklabels=cols,yticks=[])
    axf.tick_params(axis="x",length=0,labelsize=5.2); [sp.set_visible(False) for sp in axf.spines.values()]
    panel(axf,"(f)")

    fig.subplots_adjust(left=.16,right=.96,bottom=.06,top=.975)
    save(fig,"Fig_A_source_structure_audit_protocol")


def surface(mode, source, candidate):
    d=pd.read_parquet(ROOT/f"outputs/coverage_corrected_grid/{mode}/metrics_long.parquet")
    threshold="scale_threshold_px" if mode=="absolute" else "scale_threshold_norm"
    d=d[d.source.eq(source)&d.candidate.eq(candidate)&np.isclose(d.iou,.25)&d.small_object_policy.isin(POLICIES)]
    q=d.groupby(["confidence",threshold]).f1.agg(lambda x:float(x.max()-x.min())).reset_index(name="band")
    x=np.sort(q[threshold].unique());y=np.sort(q.confidence.unique()); z=q.pivot(index="confidence",columns=threshold,values="band").reindex(index=y,columns=x).to_numpy(float)
    return x,y,z,q


def add_uav_ap_duplicates(d):
    d=d.copy()
    for mode in ["absolute","normalized"]:
        if not (d.source.eq("uavdt")&d["mode"].eq(mode)&d.policy.eq("exclude_with_ignore_protection")).any():
            q=d[d.source.eq("uavdt")&d["mode"].eq(mode)&d.policy.eq("exclude_without_ignore_protection")].copy();q["policy"]="exclude_with_ignore_protection";d=pd.concat([d,q],ignore_index=True)
    return d


def forest_rows(ap,f1):
    ap=add_uav_ap_duplicates(ap)
    ap=ap.assign(metric="AP50",point=ap.point_ap50_difference)
    f=f1[np.isclose(f1.iou,.25)].copy().assign(metric="F1",point=lambda x:x.point_f1_difference)
    d=pd.concat([ap,f],ignore_index=True)
    order={"absolute":0,"normalized":1}; po={p:i for i,p in enumerate(POLICIES)}
    d["_m"]=d["mode"].map(order);d["_p"]=d.policy.map(po)
    return d.sort_values(["metric","source","_m","_p"])


def fig_b():
    cols=[("visdrone","visdrone_sahi640","VisDrone"),("uavdt","uavdt_tiling","Local UAVDT"),("aitod","aitod_baseline640","AI-TOD")]
    modes=[("absolute","Absolute"),("normalized","Normalized")]
    allq={}; heat_rows=[]
    for mode,_ in modes:
        for source,candidate,_ in cols:
            x,y,z,q=surface(mode,source,candidate);allq[(mode,source)]=(x,y,z);q["source"]=source;q["candidate"]=candidate;q["domain"]=mode;heat_rows.append(q)
    heat=pd.concat(heat_rows,ignore_index=True);heat.to_csv(DATA / "Fig_B_heatmap_source_data.csv",index=False)
    vmax=max(z.max() for _,_,z in allq.values()); vmax=np.ceil(vmax*100)/100 # .52, prevents clipping actual .512 max
    ap=pd.read_csv(ROOT/"outputs/statistics/bootstrap_headline_ap50_common_coverage.csv")
    f1=pd.read_csv(ROOT/"outputs/statistics/bootstrap_headline_f1_common_coverage.csv")
    forest=forest_rows(ap,f1);forest.to_csv(DATA / "Fig_B_bootstrap_source_data.csv",index=False)
    robust=pd.read_csv(ROOT/"outputs/statistics/pairwise_policy_robustness.csv")

    fig=plt.figure(figsize=(88/25.4,145/25.4))
    gs=GridSpec(3,3,figure=fig,height_ratios=[.89,.89,1.52],hspace=.48,wspace=.24)
    axes=[]; im=None
    for row,(mode,rowname) in enumerate(modes):
        for col,(source,candidate,title) in enumerate(cols):
            ax=fig.add_subplot(gs[row,col]);axes.append(ax);x,y,z=allq[(mode,source)]
            im=ax.imshow(z,origin="lower",aspect="auto",vmin=0,vmax=vmax,cmap="cividis",interpolation="nearest")
            ax.set_xticks(range(len(x)),[f"{int(t)}" if mode=="absolute" else f"{t:.4f}".rstrip("0") for t in x],rotation=45,ha="right")
            ax.set_yticks(range(len(y)),[f"{v:.2f}" for v in y] if col==0 else [])
            if col==0: ax.set_ylabel(f"{rowname}\nconfidence")
            if row==1: ax.set_xlabel("Threshold (px)" if mode=="absolute" else "Norm. threshold")
            if row==0: ax.text(.5,1.07,title,transform=ax.transAxes,ha="center",va="bottom",fontsize=6.5)
            headline=(24,.25) if mode=="absolute" else (.015,.25)
            ix=np.where(np.isclose(x,headline[0]))[0][0];iy=np.where(np.isclose(y,headline[1]))[0][0]
            ax.add_patch(mpl.patches.Rectangle((ix-.5,iy-.5),1,1,fill=False,ec="black",lw=.9))
            maxid=np.unravel_index(np.argmax(z),z.shape)
            for jy,jx in {(iy,ix),maxid}:
                val=z[jy,jx]; right_edge=jx==len(x)-1
                ax.text(jx+(.37 if right_edge else 0),jy,f"{val:.3f}",ha="right" if right_edge else "center",va="center",fontsize=5.0,color="white" if val>.28 else "black")
            ax.tick_params(length=0,pad=1);[sp.set_linewidth(.45) for sp in ax.spines.values()];panel(ax,f"({chr(97+row*3+col)})")
    cax=fig.add_axes([.905,.56,.022,.32]);cb=fig.colorbar(im,cax=cax);cb.set_label(r"$F_1$ policy band",fontsize=6.5);cb.ax.tick_params(labelsize=5.8,length=2)

    bottom=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[2,:],width_ratios=[.76,1.24],wspace=.55)
    axg=fig.add_subplot(bottom[0,0]); axh=fig.add_subplot(bottom[0,1])
    for _,r in robust.iterrows():
        filled=r.metric=="AP50"
        axg.scatter(r.reference_margin,r.differential_radius,s=23,marker=MARK[r.source],fc=COL[r.source] if filled else "white",ec=COL[r.source],lw=.8,zorder=3)
        short=("VD" if r.source=="visdrone" else "UAV")+(" AP50" if r.metric=="AP50" else " F1")
        dx=-2 if r.source=="uavdt" and r.metric=="AP50" else 2; ha="right" if dx<0 else "left"
        axg.annotate(short,(r.reference_margin,r.differential_radius),xytext=(dx,2),textcoords="offset points",fontsize=5.1,ha=ha)
    low,hi=5e-5,.5
    axg.fill_between([low,hi],[low,hi],hi,color="#F2F2F2",zorder=-2);axg.fill_between([low,hi],low,[low,hi],color="#F5FAF7",zorder=-2)
    axg.plot([low,hi],[low,hi],"--",color="#333",lw=.65);axg.text(.00013,.00045,"Certified\nstable",fontsize=4.7,color="#4D7863");axg.text(.055,.18,"Not\ncertified",fontsize=4.7,color="#777")
    axg.set(xscale="log",yscale="log",xlim=(low,hi),ylim=(low,hi),xlabel=r"Reference margin $\Delta_0$",ylabel=r"Differential radius $\Gamma$")
    tidy(axg,"both");panel(axg,"(g)")

    groups=[("visdrone","AP50","VisDrone AP50"),("uavdt","AP50","Local UAVDT AP50"),("visdrone","F1","VisDrone F1"),("uavdt","F1","Local UAVDT F1")]
    ycur=0; yt=[]; yl=[]; boundaries=[]
    for source,metric,title in groups:
        q=forest[(forest.source.eq(source))&forest.metric.eq(metric)].sort_values(["_m","_p"])
        # AP rows include collapsed-but-disclosed N duplicates for UAVDT; six rows remain visible.
        for i,(_,r) in enumerate(q.iterrows()):
            point,lo,hi_=r.point,r.ci95_low,r.ci95_high; solid=lo>0
            axh.errorbar(point,ycur,xerr=[[point-lo],[hi_-point]],fmt=MARK[source],ms=3,mfc=COL[source] if solid else "white",mec=COL[source],ecolor=COL[source],elinewidth=.75,capsize=1.2,zorder=3)
            rule=("A" if r["mode"]=="absolute" else "N")+"-"+PSHORT[r.policy]
            yt.append(ycur); yl.append(rule); ycur+=1
        boundaries.append(ycur-.5);ycur+=.72
    axh.axvline(0,color="#333",ls="--",lw=.65);axh.set(yticks=yt,yticklabels=yl,xlabel="Leading candidate − runner-up")
    axh.set_xlim(-.36,.42); axh.invert_yaxis(); tidy(axh,"x"); axh.tick_params(axis="y",labelsize=5.1)
    for y in boundaries: axh.axhline(y,color="#D0D0D0",lw=.35)
    panel(axh,"(h)")
    fig.subplots_adjust(left=.12,right=.89,bottom=.06,top=.975)
    save(fig,"Fig_B_sensitivity_robustness")


def readme():
    (OUT / "README_figure_reproduction.md").write_text("""# Composite-figure reproduction

Run `python scripts/build_single_column_composite_figures.py` from the project root.

Outputs are 88-mm-wide vector PDF/SVG and 600-dpi PNG previews.  Figure A contains only empirical source/coverage panels (a)--(f); the requested workflow/architecture panel is intentionally omitted. Figure B uses a common 0--0.52 `F1 policy band` range because the released grid maximum is 0.512; a 0--0.35 range would clip material values.

`figure_source_data/Fig_A_source_data.csv`, `Fig_B_heatmap_source_data.csv`, and `Fig_B_bootstrap_source_data.csv` are the records used to draw the composite panels.  Bootstrap intervals are read from the setting-level released files, not reconstructed from manuscript prose.
""")


if __name__ == "__main__":
    style(); fig_a(); fig_b(); readme()
