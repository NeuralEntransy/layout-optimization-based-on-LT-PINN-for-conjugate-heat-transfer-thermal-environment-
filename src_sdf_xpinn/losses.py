"""Strong-form PDE, interface, boundary and integral energy constraints."""
import torch
import config as C
import geometry as G


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True,
                               retain_graph=True)[0]


def pde_residual(field, dom, pts, power_scale=1.):
    """Return the pointwise strong-form residual before reduction."""
    pts = pts.requires_grad_(True)
    th = field(dom, pts); g = grad(th, pts)
    lap = grad(g[:,0:1],pts)[:,0:1] + grad(g[:,1:2],pts)[:,1:2]
    if dom in C.S_OF_DOMAIN: lap = lap + C.S_OF_DOMAIN[dom]*power_scale
    return lap


def pde_one(field, dom, pts, power_scale=1., w_dev=1.):
    lap = pde_residual(field, dom, pts, power_scale)
    weight = w_dev if dom in C.S_OF_DOMAIN else 1.
    return weight*torch.mean(lap.square())


def interface_one(field, sample):
    pts = sample["pts"].requires_grad_(True); dirs=sample["dirs"]
    ta=field(sample["a"],pts); tb=field(sample["b"],pts)
    ga=(grad(ta,pts)*dirs).sum(1,keepdim=True)
    gb=(grad(tb,pts)*dirs).sum(1,keepdim=True)
    rt=ta-tb
    rq=(C.K_OF_DOMAIN[sample["a"]]*ga-C.K_OF_DOMAIN[sample["b"]]*gb)*C.DT/C.L_REF/C.Q_IF
    return torch.mean(rt.square()), torch.mean(rq.square())


def boundary_one(field, name, sample):
    pts=sample["pts"].requires_grad_(True); th=field("wall",pts)
    g=grad(th,pts); qn=(-C.K_AL*C.DT/C.L_REF*(g*sample["dirs"]).sum(1,keepdim=True))/C.Q_REF
    if name == "left":
        target=(C.H_TBL*C.DT*(th-C.THETA_COLD))/C.Q_REF
        return torch.mean((qn-target).square())
    if name == "right": return torch.mean((th-C.THETA_INF).square())
    return torch.mean(qn.square())


def integral(field, dom, sample, normal, length):
    pts=sample["pts"].requires_grad_(True); th=field(dom,pts); g=grad(th,pts)
    n = sample["dirs"] if normal == "radial" else torch.tensor(normal,device=pts.device).expand_as(pts)
    q=-C.K_OF_DOMAIN[dom]*C.DT/C.L_REF*(g*n).sum(1,keepdim=True)
    return q.mean()*length*C.B


def energy_loss(field, interfaces, boundaries, power_scale=1.):
    details={}; loss_device=0.; loss_face=0.; total=C.P_TOT*power_scale
    for dev,names in G.DEV_IFACES.items():
        qd=0.; qr=0.; target=C.P_OF_DEVICE[dev]*power_scale
        for name in names:
            s=interfaces[name]; normal,length=G.IFACE_META[name]
            recv=s["b"] if s["a"]==dev else s["a"]
            a=integral(field,dev,s,normal,length); b=integral(field,recv,s,normal,length)
            qd+=a; qr+=b
            rface=(a-b)/target
            loss_face=loss_face+rface.square()
            details[f"eng_face_{name}"]=rface.detach()
            details[f"Q_{name}_dev"]=a.detach()
            details[f"Q_{name}_recv"]=b.detach()
            details[f"dQ_{name}"]=(a-b).detach()
        rdev=(qd-target)/target; rrecv=(qr-target)/target
        loss_device=loss_device+rdev.square()+rrecv.square()
        details[f"eng_{dev}_dev"]=rdev.detach()
        details[f"eng_{dev}_recv"]=rrecv.detach()
        details[f"Q_{dev}_target"]=torch.as_tensor(target,device=qd.device)
        details[f"Q_{dev}_dev"]=qd.detach(); details[f"Q_{dev}_recv"]=qr.detach()
    outer={}
    for name,normal in (("left",(-1.,0.)),("right",(1.,0.)),
                        ("top",(0.,1.)),("bottom",(0.,-1.))):
        q=integral(field,"wall",boundaries[name],normal,1.)
        details[f"Q_{name}"]=q.detach(); outer[name]=q

    # The physical outer-boundary conditions are not one interchangeable
    # four-sided budget: only left/right may reject the generated heat,
    # whereas top and bottom are independently adiabatic.  Keeping these
    # residuals separate prevents an unphysical top/bottom leak from being
    # cancelled by an error in Q_left + Q_right.
    q_lr=outer["left"]+outer["right"]
    r_lr=(q_lr-total)/total
    r_top=outer["top"]/total
    r_bottom=outer["bottom"]/total
    loss_lr=r_lr.square()
    loss_adiabatic=r_top.square()+r_bottom.square()
    loss_outer=loss_lr+loss_adiabatic
    loss=loss_device+loss_face+loss_outer
    details["eng_lr"]=r_lr.detach()
    details["eng_adiabatic_top"]=r_top.detach()
    details["eng_adiabatic_bottom"]=r_bottom.detach()
    details["Q_lr"]=q_lr.detach()
    details["Q_outer"]=(q_lr+outer["top"]+outer["bottom"]).detach()
    details["Q_target_total"]=torch.as_tensor(total,device=q_lr.device)
    details["eng_loss_device"]=loss_device.detach()
    details["eng_loss_face"]=loss_face.detach()
    details["eng_loss_lr"]=loss_lr.detach()
    details["eng_loss_adiabatic"]=loss_adiabatic.detach()
    details["eng_loss_outer"]=loss_outer.detach()
    details["eng_loss_total"]=loss.detach()
    return loss,details
