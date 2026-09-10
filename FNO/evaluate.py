import torch
from.data import TIMES,model_input
from.losses import relative_l2_per_sample


def evaluate(model,data,time_dependent=False,batch_size=32,device='cpu'):
    model.to(device)
    model.eval()
    metrics={}
    horizons=range(1,5)if time_dependent else[4]
    with torch.no_grad():
        for j in horizons:
            errors=[]
            for batch in data.split(batch_size):
                batch=batch.to(device)
                u=batch[:,0]
                x=torch.linspace(0,1,u.shape[-1],device=device).expand_as(u)
                time=torch.full((len(u),),TIMES[j],device=device)
                inputs=model_input(x,u,time,time_dependent)
                output=model(inputs,time)if time_dependent else model(inputs)
                errors.append(relative_l2_per_sample(output,batch[:,j]).cpu())
            metrics[str(TIMES[j])]=torch.cat(errors).mean().item()
    return metrics
