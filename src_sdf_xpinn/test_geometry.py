"""Fast geometry invariants; executable without pytest."""
import torch
import config as C
import geometry as G
import sampling as S


def run():
    x1,x2=C.LAYOUTS["center"]
    pts=torch.tensor([[0.005,.5],[.8,.5],[x1,.11],[x2,.815],[.5,.4]])
    expected=["wall","air","dev1","dev2","dev3"]
    got=[G.LABEL_ORDER[i] for i in G.label_points(pts,x1,x2).tolist()]
    assert got==expected,(got,expected)
    # SDF signs and differentiability with respect to a moving device center.
    center=torch.tensor(x1,requires_grad=True)
    value=G.sdf_dev1(torch.tensor([[x1+.05,.11]]),center).sum()
    value.backward(); assert center.grad is not None
    for name, a, b in G.INTERFACES:
        interface, _ = S.sample_interface(name, 32, x1, x2, "cpu", True)
        assert float(G.domain_sdf(a, interface, x1, x2).abs().max()) < 1e-5
        assert float(G.domain_sdf(b, interface, x1, x2).abs().max()) < 1e-5
    print("geometry tests passed",got,"dx1=",float(center.grad))


if __name__=="__main__": run()
