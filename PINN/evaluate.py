import torch
from.losses import interior_loss,boundary_loss


def relative_l2(prediction,target):
    return(torch.linalg.vector_norm(prediction-target)/torch.linalg.vector_norm(target).clamp_min(1e-12)).item()


def evaluate(model,data,kind='pinn'):
    model.eval()
    with torch.no_grad():
        prediction=model(data['test_pos']).flatten()
        interior_prediction=model(data['test_int_pos']).flatten()
        if kind=='dd':
            prediction=prediction*data['u_std']+data['u_mean']
            interior_prediction=interior_prediction*data['u_std']+data['u_mean']
    metrics={'relative_l2':relative_l2(prediction,data['test_u']),
             'interior_relative_l2':relative_l2(interior_prediction,data['test_int_u'])}
    if kind=='pinn':
        metrics['residual_mse']=interior_loss(model,data['test_interior']).item()
        metrics['boundary_mse']=boundary_loss(model,data['test_boundary']).item()
    return metrics,prediction
