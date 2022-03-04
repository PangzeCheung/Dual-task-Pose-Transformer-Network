import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.iter_start = start_epoch

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
writer = SummaryWriter(comment=opt.name)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        print("epoch: ", epoch, "iter: ", epoch_iter, "total_iteration: ", total_steps, end=" ")
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        save_fake = total_steps % opt.display_freq == display_delta

        model.set_input(data)
        model.optimize_parameters()

        losses = model.get_current_errors()
        for k, v in losses.items():
            print(k, ": ", '%.2f' % v, end=" ")
        lr_G, lr_D = model.get_current_learning_rate()
        print("learning rate G: %.7f" % lr_G, end=" ")
        print("learning rate D: %.7f" % lr_D, end=" ")
        print('\n')


        writer.add_scalar('Loss/app_gen_s', losses['app_gen_s'], total_steps)
        writer.add_scalar('Loss/content_gen_s', losses['content_gen_s'], total_steps)
        writer.add_scalar('Loss/style_gen_s', losses['style_gen_s'], total_steps)
        writer.add_scalar('Loss/app_gen_t', losses['app_gen_t'], total_steps)
        writer.add_scalar('Loss/ad_gen_t', losses['ad_gen_t'], total_steps)
        writer.add_scalar('Loss/dis_img_gen_t', losses['dis_img_gen_t'], total_steps)
        writer.add_scalar('Loss/content_gen_t', losses['content_gen_t'], total_steps)
        writer.add_scalar('Loss/style_gen_t', losses['style_gen_t'], total_steps)
        writer.add_scalar('LR/G', lr_G, total_steps)
        writer.add_scalar('LR/D', lr_D, total_steps)


        ############## Display results and errors ##########
        if total_steps % opt.print_freq == print_delta:
            losses = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, total_steps, losses, lr_G, lr_D, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(total_steps, losses)

        if total_steps % opt.display_freq == display_delta:
            visualizer.display_current_results(model.get_current_visuals(), epoch)
            if hasattr(model, 'distribution'):
                visualizer.plot_current_distribution(model.get_current_dis())

        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save_networks('latest')
            if opt.dataset_mode == 'market':
                model.save_networks(total_steps)
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0 or (epoch > opt.niter and epoch % (opt.save_epoch_freq//2) == 0):
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### linearly decay learning rate after certain iterations
    model.update_learning_rate()
