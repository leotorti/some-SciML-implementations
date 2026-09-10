from copy import deepcopy
import torch
from.data import model_input
from.losses import time_weighted_loss,relative_l2_per_sample


def fit(model,training_set,validation_set=None,n_epochs=50,lr=1e-3,time_dependent=False,time_weight=0.0,device='cpu'):
    if n_epochs<1:
        raise ValueError('n_epochs must be positive.')
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=n_epochs)
    history=[]
    best_error=float('inf')
    best_weights=None
    for epoch in range(n_epochs):
        model.train()
        running_loss=0
        count=0
        for batch in training_set:
            x,u,time,target=[tensor.to(device)for tensor in batch]
            inputs=model_input(x,u,time,time_dependent)
            optimizer.zero_grad()
            output=model(inputs,time)if time_dependent else model(inputs)
            loss=time_weighted_loss(output,target,time,time_weight)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*len(x)
            count+=len(x)
        row={'epoch':epoch+1,'loss':running_loss/count,'lr':optimizer.param_groups[0]['lr']}
        if validation_set is not None:
            model.eval()
            error=0
            count=0
            with torch.no_grad():
                for batch in validation_set:
                    x,u,time,target=[tensor.to(device)for tensor in batch]
                    inputs=model_input(x,u,time,time_dependent)
                    output=model(inputs,time)if time_dependent else model(inputs)
                    error+=relative_l2_per_sample(output,target).sum().item()
                    count+=len(x)
            row['validation_relative_l2']=error/count
            if error/count<best_error:
                best_error=error/count
                best_weights=deepcopy(model.state_dict())
        history.append(row)
        scheduler.step()
        if epoch==0 or(epoch+1)%10==0:
            print(f"epoch {epoch+1}: {row}",flush=True)
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return history
