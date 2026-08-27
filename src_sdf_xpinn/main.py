"""Training entry point for the independent five-domain SDF-XPINN model."""
import argparse, csv, json, os, random, time
from pathlib import Path
import numpy as np
import torch

import config as C
import geometry as G
import sampling as S
import rar as RAR
from networks import TemperatureField
from losses import pde_one, interface_one, boundary_one, energy_loss


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--layout",choices=C.LAYOUTS,default="center")
    p.add_argument("--epochs",type=int,default=C.TRAIN["epochs"])
    p.add_argument("--lr",type=float,default=C.TRAIN["lr"])
    p.add_argument("--device",default="auto")
    p.add_argument("--width",type=int,default=C.TRAIN["width"])
    p.add_argument("--depth",type=int,default=C.TRAIN["depth"])
    p.add_argument("--power-scale",type=float,default=1.)
    p.add_argument("--ckptdir",default="checkpoint_m1/sdf_xpinn_v1")
    p.add_argument("--outdir",default="results/sdf_xpinn_v1")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--eval-every",type=int,default=C.TRAIN["eval_every"])
    p.add_argument("--save-every",type=int,default=C.TRAIN["save_every"])
    p.add_argument("--no-rar",action="store_true",
                   help="disable residual-adaptive domain points")
    p.add_argument("--no-plot",action="store_true")
    return p


def choose_device(value):
    if value=="auto": value="cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def chunks(tensor,size):
    n=tensor.shape[0]
    for i in range(0,n,size): yield tensor[i:i+size], tensor[i:i+size].shape[0]/n


def append_csv(path,row):
    new=not path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if new: w.writeheader()
        w.writerow(row)


def scalar(value):
    """Convert a scalar tensor/number to a CSV-safe Python float."""
    return float(value.detach()) if torch.is_tensor(value) else float(value)


def save(path,epoch,field,opt,args):
    payload=dict(epoch=epoch,field=field.state_dict(),optimizer=opt.state_dict(),
                 args=vars(args),case=C.CASE_VERSION)
    torch.save(payload,path/f"epoch_{epoch:06d}.pt")
    torch.save(payload,path/"latest.pt")


@torch.no_grad()
def temperature_max(field,x1,x2,device,n=61):
    result={}
    boxes={"dev1":(x1-C.D1/2,x1+C.D1/2,*C.DEV1_Y),
           "dev2":(x2-C.D2/2,x2+C.D2/2,*C.DEV2_Y),
           "dev3":(C.C3[0]-C.R3,C.C3[0]+C.R3,C.C3[1]-C.R3,C.C3[1]+C.R3)}
    for dom,(xa,xb,ya,yb) in boxes.items():
        xx,yy=torch.meshgrid(torch.linspace(xa,xb,n,device=device),
                             torch.linspace(ya,yb,n,device=device),indexing="ij")
        pts=torch.stack((xx.ravel(),yy.ravel()),1)
        if dom=="dev3": pts=pts[G.mask_domain(dom,pts,x1,x2)]
        result[dom]=float(field.temperature_K(dom,pts).max()-273.15)
    return result


def plot_field(field,x1,x2,device,outpath,n=201):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axis=torch.linspace(0,1,n,device=device); xx,yy=torch.meshgrid(axis,axis,indexing="xy")
    pts=torch.stack((xx.ravel(),yy.ravel()),1); labels=G.label_points(pts,x1,x2)
    temp=torch.full((len(pts),),torch.nan,device=device)
    with torch.no_grad():
        for dom in C.DOMAINS:
            m=labels==G.LABEL_ORDER.index(dom)
            if m.any(): temp[m]=field.temperature_K(dom,pts[m]).squeeze()-273.15
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(temp.reshape(n,n).cpu(),origin="lower",extent=(0,1,0,1),cmap="inferno")
    fig.colorbar(im,ax=ax,label="Temperature [C]"); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(outpath,dpi=150); plt.close(fig)


def main():
    args=parser().parse_args(); device=choose_device(args.device)
    torch.manual_seed(C.TRAIN["seed"]); np.random.seed(C.TRAIN["seed"]); random.seed(C.TRAIN["seed"])
    ckpt=Path(args.ckptdir); out=Path(args.outdir); ckpt.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    x1,x2=C.LAYOUTS[args.layout]
    field=TemperatureField(args.width,args.depth,C.TRAIN["fourier_sigma"],C.TRAIN["fourier_dim"]).to(device)
    opt=torch.optim.Adam(field.parameters(),lr=args.lr); start=1
    if args.resume:
        data=torch.load(ckpt/"latest.pt",map_location=device)
        if data.get("case")!=C.CASE_VERSION: raise ValueError("checkpoint case mismatch")
        field.load_state_dict(data["field"]); opt.load_state_dict(data["optimizer"]); start=data["epoch"]+1
        print(f"[resume] epoch={data['epoch']}")
    print(f"device={device} domains={C.DOMAINS} start={start} target={args.epochs} lr={args.lr:g}")
    log=ckpt/"loss_log.csv"
    physics_log=ckpt/"physics_log.csv"
    energy_log=ckpt/"energy_log.csv"
    t0=time.time()
    for epoch in range(start,args.epochs+1):
        opt.zero_grad(set_to_none=True)
        pde_value=0.; pde_by_domain={}
        rar_info={}
        for dom,n in C.TRAIN["n_dom"].items():
            pts=S.sample_domain(dom,n,x1,x2,device)
            near_n=C.TRAIN["n_near"].get(dom,0)
            if near_n:
                near=S.sample_near_boundary(dom,near_n,
                    C.TRAIN["near_width"][dom],x1,x2,device)
                pts=torch.cat((pts,near),dim=0)
            rar_n=0 if args.no_rar else C.TRAIN["n_rar"].get(dom,0)
            if rar_n:
                adaptive,info=RAR.sample_rar(
                    field,dom,rar_n,x1,x2,device,args.power_scale,
                    C.TRAIN["rar_candidate_factor"],C.TRAIN["rar_power"],
                    C.TRAIN["rar_uniform_mix"],
                    C.TRAIN["rar_score_microbatch"],return_info=True)
                pts=torch.cat((pts,adaptive),dim=0)
                rar_info[dom]=info
            for part,fraction in chunks(pts,C.TRAIN["pde_microbatch"]):
                raw=pde_one(field,dom,part,args.power_scale,C.TRAIN["w_pde_dev"])
                (C.TRAIN["w_pde"]*fraction*raw).backward()
                contribution=fraction*float(raw.detach())
                pde_value+=contribution
                pde_by_domain[dom]=pde_by_domain.get(dom,0.)+contribution
        interfaces=S.sample_all_interfaces(C.TRAIN["n_iface"],x1,x2,device)
        ift=ifq=0.; interface_by_name={}
        for interface_name,sample in interfaces.items():
            pts=sample["pts"]; n=len(pts)
            name_t=name_q=0.
            for i in range(0,n,C.TRAIN["interface_microbatch"]):
                sub={**sample,"pts":pts[i:i+C.TRAIN["interface_microbatch"]],"dirs":sample["dirs"][i:i+C.TRAIN["interface_microbatch"]]}
                fraction=len(sub["pts"])/n; lt,lq=interface_one(field,sub)
                (fraction*(C.TRAIN["w_if_T"]*lt+C.TRAIN["w_if_q"]*lq)).backward()
                ct=fraction*float(lt.detach()); cq=fraction*float(lq.detach())
                ift+=ct; ifq+=cq; name_t+=ct; name_q+=cq
            interface_by_name[interface_name]=(name_t,name_q)
        boundaries=S.sample_boundaries(C.TRAIN["n_bnd"],device)
        bc=0.; boundary_by_name={}
        for name,sample in boundaries.items():
            raw=boundary_one(field,name,sample)
            factor=C.TRAIN["w_adiabatic"] if name in ("top","bottom") else 1.
            (C.TRAIN["w_bc"]*factor*raw).backward()
            boundary_by_name[name]=float(raw.detach())
            bc+=factor*boundary_by_name[name]
        eng_if=S.sample_all_interfaces(C.TRAIN["n_energy"],x1,x2,device,True)
        eng_bnd=S.sample_boundaries(C.TRAIN["n_energy"],device,True)
        eng,details=energy_loss(field,eng_if,eng_bnd,args.power_scale)
        (C.TRAIN["w_eng"]*eng).backward(); opt.step()
        total=(C.TRAIN["w_pde"]*pde_value+C.TRAIN["w_if_T"]*ift+C.TRAIN["w_if_q"]*ifq+C.TRAIN["w_bc"]*bc+C.TRAIN["w_eng"]*float(eng.detach()))
        if epoch%args.eval_every==0 or epoch in (start,args.epochs):
            tm=temperature_max(field,x1,x2,device)
            row=dict(epoch=epoch,loss=total,pde=pde_value,ifT=ift,ifq=ifq,bc=bc,eng=float(eng.detach()),lr=opt.param_groups[0]["lr"],
                     Q_left=float(details["Q_left"]),Q_right=float(details["Q_right"]),Q_top=float(details["Q_top"]),Q_bottom=float(details["Q_bottom"]),
                     T1_C=tm["dev1"],T2_C=tm["dev2"],T3_C=tm["dev3"])
            for dom in C.DOMAINS:
                info=rar_info.get(dom,{})
                row[f"rar_{dom}_candidate_mean"]=info.get("candidate_mean",float("nan"))
                row[f"rar_{dom}_selected_mean"]=info.get("selected_mean",float("nan"))
            physics_row={"epoch":epoch,"phase":"adam"}
            for dom in C.DOMAINS:
                physics_row[f"pde_{dom}"]=pde_by_domain[dom]
            for name,_,_ in G.INTERFACES:
                physics_row[f"ifT_{name}"]=interface_by_name[name][0]
                physics_row[f"ifq_{name}"]=interface_by_name[name][1]
            for name in ("left","right","top","bottom"):
                physics_row[f"bc_{name}"]=boundary_by_name[name]

            energy_row={"epoch":epoch,"phase":"adam"}
            energy_row.update({key:scalar(value) for key,value in details.items()})

            append_csv(log,row)
            append_csv(physics_log,physics_row)
            append_csv(energy_log,energy_row)
            print(" ".join(f"{k}={v:.5g}" if isinstance(v,float) else f"{k}={v}" for k,v in row.items()))
            worst_pde=max(pde_by_domain,key=pde_by_domain.get)
            worst_ifT=max(interface_by_name,key=lambda k:interface_by_name[k][0])
            worst_ifq=max(interface_by_name,key=lambda k:interface_by_name[k][1])
            print(f"  physics: maxPDE={worst_pde}:{pde_by_domain[worst_pde]:.4g} "
                  f"maxIfT={worst_ifT}:{interface_by_name[worst_ifT][0]:.4g} "
                  f"maxIfq={worst_ifq}:{interface_by_name[worst_ifq][1]:.4g} "
                  f"bc[L/R/T/B]={boundary_by_name['left']:.3g}/"
                  f"{boundary_by_name['right']:.3g}/"
                  f"{boundary_by_name['top']:.3g}/"
                  f"{boundary_by_name['bottom']:.3g}")
            print(f"  energy: lossDev={scalar(details['eng_loss_device']):.4g} "
                  f"lossFace={scalar(details['eng_loss_face']):.4g} "
                  f"lossLR={scalar(details['eng_loss_lr']):.4g} "
                  f"lossAdiabatic={scalar(details['eng_loss_adiabatic']):.4g} "
                  f"rDev1={scalar(details['eng_dev1_dev']):+.3g}/"
                  f"{scalar(details['eng_dev1_recv']):+.3g} "
                  f"rDev2={scalar(details['eng_dev2_dev']):+.3g}/"
                  f"{scalar(details['eng_dev2_recv']):+.3g} "
                  f"rDev3={scalar(details['eng_dev3_dev']):+.3g}/"
                  f"{scalar(details['eng_dev3_recv']):+.3g} "
                  f"rLR={scalar(details['eng_lr']):+.3g} "
                  f"rT={scalar(details['eng_adiabatic_top']):+.3g} "
                  f"rB={scalar(details['eng_adiabatic_bottom']):+.3g}")
            if not args.no_plot: plot_field(field,x1,x2,device,out/"temperature.png")
        if epoch%args.save_every==0 or epoch==args.epochs: save(ckpt,epoch,field,opt,args)
    report=dict(case=C.CASE_VERSION,epoch=args.epochs,x1=x1,x2=x2,minutes=(time.time()-t0)/60)
    (out/"run_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__": main()
