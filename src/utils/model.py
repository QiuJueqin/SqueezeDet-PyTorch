import torch
import torch.nn as nn


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded model {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    success_loaded = True
    for layer in state_dict:
        if layer in model_state_dict:
            if state_dict[layer].shape != model_state_dict[layer].shape:
                success_loaded = False
                print('Skip loading param {}, required shape{}, loaded shape{}.'.format(
                    layer, model_state_dict[layer].shape, state_dict[layer].shape))
                state_dict[layer] = model_state_dict[layer]
        else:
            success_loaded = False
            print('Drop param {} in pre-trained model.'.format(layer))

    for layer in model_state_dict:
        if layer not in state_dict:
            success_loaded = False
            print('Param {} not found in pre-trained model.'.format(layer))
            state_dict[layer] = model_state_dict[layer]

    model.load_state_dict(state_dict, strict=False)
    print('Model successfully loaded.' if success_loaded else
          'The model does not fully load the pre-trained weight.')

    return model


def load_official_model(model, model_path):
    """
    load official models from https://pytorch.org/docs/stable/torchvision/models.html
    :param model:
    :param model_path:
    :return:
    """
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    layers = list(state_dict.keys())
    for layer in layers:
        new_layer = 'base.' + layer
        state_dict[new_layer] = state_dict.pop(layer)

    checkpoint = {'epoch': 0,
                  'state_dict': state_dict}
    converted_model_path = model_path.replace('.pth', '_converted.pth')
    torch.save(checkpoint, converted_model_path)

    return load_model(model, converted_model_path)


def save_model(model, path, epoch):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    torch.save(data, path)
