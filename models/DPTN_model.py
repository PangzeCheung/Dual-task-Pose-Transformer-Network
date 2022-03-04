import numpy as np
import torch
import os
import itertools
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import external_function
from . import base_function


class DPTNModel(BaseModel):
    def name(self):
        return 'DPTNModel'

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--init_type', type=str, default='orthogonal', help='initial type')
        parser.add_argument('--use_spect_g', action='store_false', help='use spectual normalization in generator')
        parser.add_argument('--use_spect_d', action='store_false', help='use spectual normalization in generator')
        parser.add_argument('--use_coord', action='store_true', help='use coordconv')
        parser.add_argument('--lambda_style', type=float, default=500, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--layers_g', type=int, default=3, help='number of layers in G')
        parser.add_argument('--save_input', action='store_true', help="whether save the input images when testing")
        parser.add_argument('--num_blocks', type=int, default=3, help="number of resblocks")
        parser.add_argument('--affine', action='store_true', default=True, help="affine in PTM")
        parser.add_argument('--nhead', type=int, default=2, help="number of heads in PTM")
        parser.add_argument('--num_CABs', type=int, default=2, help="number of CABs in PTM")
        parser.add_argument('--num_TTBs', type=int, default=2, help="number of CABs in PTM")

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--t_s_ratio', type=float, default=0.5, help='loss ratio between dual tasks')
        parser.add_argument('--dis_layers', type=int, default=4, help='number of layers in D')
        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.old_size = opt.old_size
        self.t_s_ratio = opt.t_s_ratio
        self.loss_names = ['app_gen_s', 'content_gen_s', 'style_gen_s', 'app_gen_t', 'ad_gen_t', 'dis_img_gen_t', 'content_gen_t', 'style_gen_t']
        self.model_names = ['G']
        self.visual_names = ['source_image', 'source_pose', 'target_image', 'target_pose', 'fake_image_s', 'fake_image_t']

        self.net_G = networks.define_G(opt, image_nc=opt.image_nc, pose_nc=opt.structure_nc, ngf=64, img_f=512,
                                       encoder_layer=3, norm=opt.norm, activation='LeakyReLU',
                                       use_spect=opt.use_spect_g, use_coord=opt.use_coord, output_nc=3, num_blocks=3, affine=True, nhead=opt.nhead, num_CABs=opt.num_CABs, num_TTBs=opt.num_TTBs)

        # Discriminator network
        if self.isTrain:
            self.model_names = ['G', 'D']
            self.net_D = networks.define_D(opt, ndf=32, img_f=128, layers=opt.dis_layers, use_spect=opt.use_spect_d)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            #self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            self.GANloss = external_function.GANLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function.VGGLoss().to(opt.device)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G.parameters())),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                lr=opt.lr * opt.ratio_g2d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            self.schedulers = [base_function.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        else:
            self.net_G.eval()

        if not self.isTrain or opt.continue_train:
            print('model resumed from latest')
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        source_image, source_pose = input['Xs'], input['Ps']
        target_image, target_pose = input['Xt'], input['Pt']
        if len(self.gpu_ids) > 0:
            self.source_image = source_image.cuda()
            self.source_pose = source_pose.cuda()
            self.target_image = target_image.cuda()
            self.target_pose = target_pose.cuda()

        self.image_paths = []
        for i in range(self.source_image.size(0)):
            self.image_paths.append(os.path.splitext(input['Xs_path'][i])[0] + '_2_' + input['Xt_path'][i])

    def forward(self):
        # Encode Inputs
        self.fake_image_t, self.fake_image_s = self.net_G(self.source_image, self.source_pose, self.target_pose)

    def test(self):
        """Forward function used in test time"""
        fake_image_t, fake_image_s = self.net_G(self.source_image, self.source_pose, self.target_pose, False)
        self.save_results(fake_image_t, self.old_size, data_name='vis')

    def backward_D_basic(self, netD, real, fake):
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        return D_loss

    def backward_D(self):
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen_t = self.backward_D_basic(self.net_D, self.target_image, self.fake_image_t)
        D_loss = self.loss_dis_img_gen_t
        D_loss.backward()

    def backward_G_basic(self, fake_image, target_image, use_d):
        # Calculate reconstruction loss
        loss_app_gen = self.L1loss(fake_image, target_image)
        loss_app_gen = loss_app_gen * self.opt.lambda_rec

        # Calculate GAN loss
        loss_ad_gen = None
        if use_d:
            base_function._freeze(self.net_D)
            D_fake = self.net_D(fake_image)
            loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(fake_image, target_image)
        loss_style_gen = loss_style_gen * self.opt.lambda_style
        loss_content_gen = loss_content_gen * self.opt.lambda_content

        return loss_app_gen, loss_ad_gen, loss_style_gen, loss_content_gen

    def backward_G(self):
        base_function._unfreeze(self.net_D)

        self.loss_app_gen_t, self.loss_ad_gen_t, self.loss_style_gen_t, self.loss_content_gen_t = self.backward_G_basic(self.fake_image_t, self.target_image, use_d = True)

        self.loss_app_gen_s, self.loss_ad_gen_s, self.loss_style_gen_s, self.loss_content_gen_s = self.backward_G_basic(self.fake_image_s, self.source_image, use_d = False)
        G_loss = self.t_s_ratio*(self.loss_app_gen_t+self.loss_style_gen_t+self.loss_content_gen_t) + (1-self.t_s_ratio)*(self.loss_app_gen_s+self.loss_style_gen_s+self.loss_content_gen_s)+self.loss_ad_gen_t
        G_loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

