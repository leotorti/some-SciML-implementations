import json
import sys
from copy import deepcopy
from pathlib import Path
import torch
from.data import DATA_DIR,load_data,create_loader
from.model import FNO,tdFNO
from.train import fit
from.evaluate import evaluate
from.plot import plot_results


def run(config):
    torch.set_num_threads(config.get('threads',4))
    seed=config['seed']
    device=config.get('device','cpu')
    output_dir=Path(config['output_dir'])
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    data_dir=config.get('data_dir',DATA_DIR)
    limit=config.get('sample_limit')
    data=load_data('data_train_128.npy',data_dir,limit)
    val=load_data('data_val_128.npy',data_dir,limit)
    tests={n:load_data(f'data_test_{n}.npy',data_dir,limit)for n in[32,64,96,128]}
    unknown=load_data('data_test_unknown_128.npy',data_dir,limit)
    finetune=load_data('data_finetune_train_unknown_128.npy',data_dir,limit)
    histories={}
    metrics={}
    models={}
    for name in['fixed','unweighted','weighted']:
        torch.manual_seed(seed)
        timed=name!='fixed'
        model=tdFNO(config['n_lifting'],config['modes'])if timed else FNO(config['n_lifting'],config['modes'])
        train_loader=create_loader(data,timed,config['batch_size'],True,seed)
        val_loader=create_loader(val,timed,config['batch_size'])
        print(f'Training {name}',flush=True)
        histories[name]=fit(model,train_loader,val_loader,
            n_epochs=config['time_epochs']if timed else config['fixed_epochs'],
            lr=1e-3 if timed else 1e-4,time_dependent=timed,
            time_weight=config['time_weight']if name=='weighted' else 0.0,device=device)
        metrics[name]={'original':evaluate(model,tests[128],timed,device=device),
                       'unknown':evaluate(model,unknown,timed,device=device)}
        if not timed:
            metrics[name]['resolution']={str(n):evaluate(model,test,device=device)['1.0']for n,test in tests.items()}
        models[name]=model
        torch.save({'model':model.state_dict(),'config':config,'metrics':metrics[name]},output_dir/f'{name}.pt')

    for name in['finetuned','scratch']:
        torch.manual_seed(seed)
        model=deepcopy(models['weighted'])if name=='finetuned' else tdFNO(config['n_lifting'],config['modes'])
        loader=create_loader(finetune,True,config['batch_size'],True,seed)
        print(f'Training {name}',flush=True)
        histories[name]=fit(model,loader,n_epochs=config['finetune_epochs'],lr=config['finetune_lr'],
                            time_dependent=True,time_weight=config['time_weight'],device=device)
        metrics[name]={'original':evaluate(model,tests[128],True,device=device),
                       'unknown':evaluate(model,unknown,True,device=device)}
        torch.save({'model':model.state_dict(),'config':config,'metrics':metrics[name]},output_dir/f'{name}.pt')
    (output_dir/'history.json').write_text(json.dumps(histories,indent=2)+'\n')
    (output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    plot_results(histories,metrics,output_dir)
    print(json.dumps(metrics,indent=2))
    return metrics


if __name__=='__main__':
    path=Path(sys.argv[1])if len(sys.argv)>1 else Path(__file__).with_name('config.json')
    run(json.loads(path.read_text()))
