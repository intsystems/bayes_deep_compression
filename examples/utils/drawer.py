import torch
import matplotlib.pyplot as plt

def plot_grad_flow(named_parameters, axs = None):    
    if axs is None:
        fig, axs = plt.subplots(1,1)
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            try:
                tmp = p.grad.detach().cpu()
            except Exception:
                tmp = torch.tensor([0.])
            ave_grads.append(tmp.abs().mean())
    axs.plot(ave_grads, alpha=0.3, color="b")
    axs.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    axs.set_xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    axs.set_xlim(xmin=0, xmax=len(ave_grads))
    axs.set_xlabel("Layers")
    axs.set_ylabel("average gradient")
    axs.set_title("Gradient flow")
    axs.grid(True)
    return fig, axs