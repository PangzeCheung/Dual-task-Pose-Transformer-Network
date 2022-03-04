import torch
import models

def create_model(opt):
    '''
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'basic':
        from .basic_model import BasicModel
        model = BasicModel(opt)
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    '''
    model = models.find_model_using_name(opt.model)(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))
    return model
