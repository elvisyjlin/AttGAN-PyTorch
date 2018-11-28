# https://github.com/sksq96/pytorch-summary
import torch
import torch.nn as nn
from collections import OrderedDict


class Logger():
    def __init__(self, silence=False):
        self.buffer = ''
        self.silence = silence
    
    def __call__(self, *strings, end='\n'):
        if not self.silence:
            print(*strings, end=end)
        for string in strings:
            self.buffer += string + end
        
    def __str__(self):
        return self.buffer
    
    def get_logs(self):
        return str(self)

def summary(model, input_size, batch_size=1, dtype=torch.float, use_gpu=False, return_str=False, forward_fn=None):
    logger = Logger(return_str)
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list,tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and 
           not isinstance(module, nn.ModuleList) and 
           not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    def zero_tensor(*size, dtype=torch.float):
        tensor = torch.zeros(*size, dtype=dtype)
        if torch.cuda.is_available() and use_gpu:
            return tensor.cuda()
        return tensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        if not isinstance(dtype, (list, tuple)):
            dtype = [dtype] * len(input_size)
        x = [zero_tensor(batch_size, *in_size, dtype=dt) for in_size, dt in zip(input_size, dtype)]
    else:
        x = zero_tensor(batch_size, *input_size, dtype=dtype)

    # print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    # print(x.shape)
    if isinstance(input_size[0], (list, tuple)):
        if forward_fn is None:
            model(*x)
        else:
            forward_fn(*x)
    else:
        if forward_fn is None:
            model(x)
        else:
            forward_fn(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    logger('---------------------------------------------------------------------')
    line_new = '{:>20}  {:>30} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
    logger(line_new)
    logger('=====================================================================')
    total_params = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>20}  {:>30} {:>15}'.format(layer, str(summary[layer]['output_shape']), summary[layer]['nb_params'])
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        logger(line_new)
    logger('=====================================================================')
    logger('Total params: ' + str(total_params))
    logger('Trainable params: ' + str(trainable_params))
    logger('Non-trainable params: ' + str(total_params - trainable_params))
    logger('---------------------------------------------------------------------')
    # return summary
    if return_str:
        return logger.get_logs()