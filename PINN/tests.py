import unittest
import numpy as np
import torch
from torch import nn
from.data import generate_data,create_dataloaders
from.losses import interior_loss
from.landscape import pca_directions,loss_landscape


class TestPINN(unittest.TestCase):
    def test_poisson_pair(self):
        data=generate_data(9,1,0.5,0)
        coefficient=np.random.RandomState(0).randn()
        class ExactSolution(nn.Module):
            def forward(self,pos):
                return coefficient/(np.pi*np.sqrt(2))*torch.sin(np.pi*pos[:,0:1])*torch.sin(np.pi*pos[:,1:2])
        batch=tuple(torch.tensor(data[key],dtype=torch.float64)for key in['pos','f','u'])
        self.assertLess(interior_loss(ExactSolution(),batch).item(),1e-25)
        np.testing.assert_allclose(ExactSolution()(batch[0]).detach().numpy().ravel(),data['u'],atol=1e-14)

    def test_split(self):
        raw=generate_data(9,2)
        data=create_dataloaders(raw,seed=7)
        again=create_dataloaders(raw,seed=7)
        train=data['train_loader'].dataset.tensors[0]
        test=data['test_pos']
        self.assertEqual(len(train)+len(test),81)
        self.assertEqual(len(set(map(tuple,train.tolist()))&set(map(tuple,test.tolist()))),0)
        torch.testing.assert_close(test,again['test_pos'])
        combined=torch.cat([data['training_interior'].dataset.tensors[0],data['training_boundary'].dataset.tensors[0]])
        torch.testing.assert_close(train,combined)

    def test_pca_and_restore(self):
        model=nn.Linear(2,1,bias=False)
        base=model.weight.detach().flatten().clone()
        trajectory=base+torch.tensor([[0.,0.],[1.,0.],[0.,2.],[1.,1.]])
        directions,explained=pca_directions(trajectory)
        torch.testing.assert_close(directions.T@directions,torch.eye(2,dtype=torch.float64))
        self.assertAlmostEqual(explained.sum().item(),1)
        result=loss_landscape(model,trajectory,lambda:model.weight.square().sum(),n_grid=3)
        self.assertTrue(np.isfinite(result['loss']).all())
        torch.testing.assert_close(model.weight.flatten(),base)
        self.assertTrue(model.training)
        def fail():
            raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            loss_landscape(model,trajectory,fail,n_grid=3)
        torch.testing.assert_close(model.weight.flatten(),base)


if __name__=='__main__':
    unittest.main()
