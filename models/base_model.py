import os
import torch
import sys
from collections import OrderedDict
from util import util
from util import pose_utils
import numpy as np
import ntpath
import cv2

class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    # visual multi-scale ouputs
                    for i in range(len(value)):
                        visual_ret[name + str(i)] = self.convert2im(value[i], name)
                else:
                    visual_ret[name] =self.convert2im(value, name)
        return visual_ret

    def convert2im(self, value, name):
        if 'label' in name:
            convert = getattr(self, 'label2color')
            value = convert(value)

        if 'flow' in name: # flow_field
            convert = getattr(self, 'flow2color')
            value = convert(value)

        if value.size(1) == 18: # bone_map
            value = np.transpose(value[0].detach().cpu().numpy(),(1,2,0))
            value = pose_utils.draw_pose_from_map(value)[0]
            result = value

        elif value.size(1) == 21: # bone_map + color image
            value = np.transpose(value[0,-3:,...].detach().cpu().numpy(),(1,2,0))
            # value = pose_utils.draw_pose_from_map(value)[0]
            result = value.astype(np.uint8)

        else:
            result = util.tensor2im(value.data)
        return result

    def get_current_errors(self):
        """Return training loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def save(self, label):
        pass

    # save model
    def save_networks(self, which_epoch):
        """Save all the networks to the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

    # load models
    def load_networks(self, which_epoch):
        """Load all the networks from the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                filename = '%s_net_%s.pth' % (which_epoch, name)
                path = os.path.join(self.save_dir, filename)
                net = getattr(self, 'net_' + name)
                try:
                    '''
                    new_dict = {}
                    pretrained_dict = torch.load(path)
                    for k, v in pretrained_dict.items():
                        if 'transformer' in k:
                            new_dict[k.replace('transformer', 'PTM')] = v
                        else:
                            new_dict[k] = v

                    net.load_state_dict(new_dict)
                    '''
                    net.load_state_dict(torch.load(path))
                    print('load %s from %s' % (name, filename))
                except FileNotFoundError:
                    print('do not find checkpoint for network %s'%name)
                    continue
                except:
                    pretrained_dict = torch.load(path)
                    model_dict = net.state_dict()
                    try:
                        pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        if len(pretrained_dict_) == 0:
                            pretrained_dict_ = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if
                                                k.replace('module.', '') in model_dict}
                        if len(pretrained_dict_) == 0:
                            pretrained_dict_ = {('module.' + k): v for k, v in pretrained_dict.items() if
                                                'module.' + k in model_dict}

                        pretrained_dict = pretrained_dict_
                        net.load_state_dict(pretrained_dict)
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % name)
                    except:
                        print('Pretrained network %s has fewer layers; The following are not initialized:' % name)
                        not_initialized = set()
                        for k, v in pretrained_dict.items():
                            if v.size() == model_dict[k].size():
                                model_dict[k] = v

                        for k, v in model_dict.items():
                            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                                # not_initialized.add(k)
                                not_initialized.add(k.split('.')[0])
                        print(sorted(not_initialized))
                        net.load_state_dict(model_dict)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()
                if not self.isTrain:
                    net.eval()

    def update_learning_rate(self, epoch=None):
        """Update learning rate"""
        for scheduler in self.schedulers:
            if epoch == None:
                scheduler.step()
            else:
                scheduler.step(epoch)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate=%.7f' % lr)

    def get_current_learning_rate(self):
        lr_G = self.optimizers[0].param_groups[0]['lr']
        lr_D = self.optimizers[1].param_groups[0]['lr']
        return lr_G, lr_D

    def save_results(self, save_data, old_size, data_name='none', data_ext='jpg'):
        """Save the training or testing results to disk"""
        img_paths = self.get_image_paths()

        for i in range(save_data.size(0)):
            print('process image ...... %s' % img_paths[i])
            short_path = ntpath.basename(img_paths[i])  # get image path
            name = os.path.splitext(short_path)[0]
            img_name = '%s_%s.%s' % (name, data_name, data_ext)

            util.mkdir(self.opt.results_dir)
            img_path = os.path.join(self.opt.results_dir, img_name)
            img_numpy = util.tensor2im(save_data[i].data)
            img_numpy = cv2.resize(img_numpy, (old_size[1], old_size[0]))
            util.save_image(img_numpy, img_path)

    def save_chair_results(self, save_data, old_size, img_path, data_name='none', data_ext='jpg'):
        """Save the training or testing results to disk"""
        img_paths = self.get_image_paths()
        print(save_data.shape)
        for i in range(save_data.size(0)):
            print('process image ...... %s' % img_paths[i])
            short_path = ntpath.basename(img_paths[i])  # get image path
            name = os.path.splitext(short_path)[0]
            img_name = '%s_%s.%s' % (name, data_name, data_ext)

            util.mkdir(self.opt.results_dir)
            img_numpy = util.tensor2im(save_data[i].data)
            img_numpy = cv2.resize(img_numpy, (old_size[1], old_size[0]))
            util.save_image(img_numpy, img_path)
