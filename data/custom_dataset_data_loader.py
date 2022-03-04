import torch.utils.data
from data.base_data_loader import BaseDataLoader
import data

def CreateDataset(opt):
    '''
    dataset = None
    if opt.dataset_mode == 'fashion':
        from data.fashion_dataset import FashionDataset
        dataset = FashionDataset()
    else:
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    '''
    dataset = data.find_dataset_using_name(opt.dataset_mode)()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=(not opt.serial_batches) and opt.isTrain,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
