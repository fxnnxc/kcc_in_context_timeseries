from .models.psb import PSBGPT2LMHeadModel
from .models.tsic import SAILGPT2LMHeadModel


def load_model(data_path, model_type, attention_type, n_layer, n_head, bin_name='model_best.bin'):
    import os 
    import torch 
    assert model_type    in ['psb', 'tsic']
    assert attention_type in ['also', 'only']
    assert n_layer       in [0,1,2,3]
    assert n_head        in [0, 12]

    state_dict = torch.load(os.path.join(data_path, 
                                         'hub/electricity-over-16-25-50', 
                                         f"{model_type}-{n_layer}-{n_head}-{attention_type}", 
                                         bin_name))
    if model_type == 'psb':
        model = PSBGPT2LMHeadModel(n_layer, n_head, 25, True if attention_type=='only' else False)
    else:
        model = SAILGPT2LMHeadModel(n_layer, n_head, 25, True if attention_type=='only' else False)
    model.load_state_dict(state_dict)
    return model 