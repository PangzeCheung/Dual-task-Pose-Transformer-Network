from data.data_loader import CreateDataLoader
from options.test_options import TestOptions
from models.models import create_model
import numpy as np
import torch

if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse()
    # creat a dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()


    print(len(dataset))

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)

    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
