"""Check that RAR returns in-domain points biased toward larger residuals."""
import torch
import config as C
import geometry as G
from networks import TemperatureField
import rar


def run():
    torch.manual_seed(7)
    field=TemperatureField(width=12,depth=2)
    # Break the constant initialization to create a spatial residual profile.
    with torch.no_grad():
        for net in field.nets.values():
            torch.nn.init.normal_(net.net[-1].weight,std=.2)
    x1,x2=C.LAYOUTS["center"]
    points,info=rar.sample_rar(field,"air",256,x1,x2,"cpu",
        candidate_factor=6,power=1.,uniform_mix=.02,
        score_microbatch=512,return_info=True)
    assert points.shape==(256,2)
    assert G.mask_domain("air",points,x1,x2).all()
    assert info["selected_mean"] > info["candidate_mean"]
    assert all(parameter.grad is None for parameter in field.parameters())
    print("RAR test passed",info)


if __name__=="__main__": run()
