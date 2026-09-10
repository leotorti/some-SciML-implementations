import unittest
import torch
from.model import FNO,tdFNO,FILMLayer
from.data import EvolutionDataset,model_input,create_loader
from.losses import time_weighted_loss
from.train import fit


class TestFNO(unittest.TestCase):
    def test_resolutions_and_gradients(self):
        torch.set_num_threads(2)
        for model in[FNO(4,40),tdFNO(4,40)]:
            for n in[31,32,64,96,128]:
                timed=isinstance(model,tdFNO)
                x=torch.linspace(0,1,n).expand(2,-1)
                u=torch.randn(2,n)
                time=torch.tensor([0.25,1.0])
                inputs=model_input(x,u,time,timed)
                output=model(inputs,time)if timed else model(inputs)
                self.assertEqual(output.shape,(2,n))
                model.zero_grad()
                output.square().mean().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_film_identity(self):
        layer=FILMLayer(4)
        x=torch.randn(2,4,9)
        torch.testing.assert_close(layer(x,torch.tensor([0.25,1.])),x)

    def test_pairs_and_weight(self):
        data=torch.arange(10.).reshape(2,5,1).expand(-1,-1,8)
        dataset=EvolutionDataset(data,all_pairs=True)
        self.assertEqual(len(dataset),20)
        horizons=[dataset[i][2].item()for i in range(10)]
        self.assertEqual([horizons.count(t)for t in[0.25,0.5,0.75,1.0]],[4,3,2,1])
        for i in range(len(dataset)):
            _,u,time,target=dataset[i]
            torch.testing.assert_close(target-u,torch.full_like(u,4*time))
        pred=torch.ones(2,8,requires_grad=True)
        target=torch.zeros_like(pred)
        time=torch.tensor([0.25,1.])
        loss=time_weighted_loss(pred,target,time)
        self.assertAlmostEqual(loss.item(),1.625)
        loss.backward()
        self.assertGreater(pred.grad[1,0],pred.grad[0,0])
        self.assertEqual(time_weighted_loss(pred,target,time,0).item(),1.)

    def test_training_and_validation(self):
        data=torch.randn(3,5,16)
        loader=create_loader(data,batch_size=2)
        model=FNO(4,3)
        history=fit(model,loader,loader,n_epochs=1)
        self.assertTrue(history[0]['validation_relative_l2']>=0)


if __name__=='__main__':
    unittest.main()
