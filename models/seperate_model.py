import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class SeperateModel(BaseModel):
    def name(self):
        return 'SeperateModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        seq_len = opt.seq_len
        self.seq_len = opt.seq_len
        self.pre_len = opt.pre_len
        shape = (opt.loadSize/8, opt.loadSize/8)
        self.input_seq = self.Tensor(nb, seq_len, opt.input_nc, size, size)
        #self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # Content_Encoder, Motion_Encoder, Motion_Predictor, Overall_Generator, Overall_D
        # opt.input_nc = 3
        # opt.output_nc = 3
        self.netCE = networks.content_E(opt.input_nc, opt.latent_nc,
                                                  opt.ngf, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netME = networks.motion_E(opt.input_nc, opt.latent_nc,
                                                opt.ngf, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids)
        #print 'latent_nc', opt.latent_nc
        self.netMP = networks.motion_P(shape, seq_len, opt.latent_nc,
                                            opt.latent_nc, opt.latent_nc*2, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG = networks.define_G(shape, seq_len, opt.latent_nc*2,
                                            opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netCE, 'CE', which_epoch)
            self.load_network(self.netME, 'ME', which_epoch)
            self.load_network(self.netMP, 'MP', which_epoch)
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_pool = ImagePool(opt.pool_size)
            #self.fake_A_pool = ImagePool(opt.pool_size)
            #self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionPixel = torch.nn.L1Loss()
            self.criterionPre   = torch.nn.L1Loss()
            #self.criterionIdt = torch.nn.L1Loss()
            self.criterionFeat = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_CE = torch.optim.Adam(self.netCE.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_ME = torch.optim.Adam(self.netME.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_MP = torch.optim.Adam(self.netMP.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netCE)
        networks.print_network(self.netME)
        networks.print_network(self.netMP)
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    '''def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']'''

    def set_input(self,input):
        # X: batchsize, seq_len_x, inchains, size(0), size(1)
        # Y: batchsize, seq_len_y, inchains, size(0), size(1)
        input_X = input[0]
        input_Y = input[1]
        #self.input_X.resize_(input_X.size()).copy_(input_X)
        #self.input_Y.resize_(input_Y.size()).copy_(input_Y)
        if len(self.gpu_ids) > 0:
            self.input_X = [x.clone().cuda() for x in input_X]
            self.input_Y = [y.clone().cuda() for y in input_Y]
        else:
            self.input_X = [x.clone() for x in input_X]
            self.input_Y = [y.clone() for y in input_Y]
        #self.image_paths = input['X_paths']

    def forward_G(self):
        self.real_X = [Variable(x) for x in self.input_X]
        self.real_Y = [Variable(y) for y in self.input_Y]

    def forward_P(self):
        self.real_X = [Variable(x) for x in self.input_X]
        self.real_Y = [Variable(y) for y in self.input_Y]

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    # get image paths
    def get_image_paths(self):
        return 0#self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake = self.fake_pool.query(self.fakes)
        self.real = Variable(torch.cat(self.input_Y, 0))
        self.loss_D = self.backward_D_basic(self.netD, self.real, fake)

    def backward_G(self):
        #lambda_idt = self.opt.identity
        lambda_fea = self.opt.lambda_fea
        lambda_pix = self.opt.lambda_pix
        #lambda_pre = self.opt.lambda_pre
        lambda_gan = self.opt.lambda_gan
        # forward
        self.content_codes = [self.netCE(real) for real in self.real_X] # [bs,1, oc, s0, s1]*len(self.real_X)
        self.motion_codes = [self.netME(real) for real in self.real_X] # [bs,1, oc, s0, s1]*len(self.real_X)
        batch_size = self.content_codes[0].size(0)
        #print 'batch_size: ',batch_size
        hidden_state_G = self.netG.init_hidden(batch_size)
        hidden_state_MP = self.netMP.init_hidden(batch_size)
        self.mcode = self.netMP(torch.cat(self.motion_codes,1), hidden_state_MP)
        #self.mcodes = [self.netMP(self.motion_codes[t],hidden_state_MP) for t in xrange(len(self.motion_codes))]
        hidden_state_M = self.mcode[0]
        self.pre_feats = [self.mcode[1]]

        concat_codes = []
        for l in xrange(len(self.content_codes)):
            concat_codes += [torch.cat([self.content_codes[l], self.pre_feats[-1]-self.motion_codes[l]],2)]
        latent_code = torch.cat(concat_codes,1)
        #print 'latent_code shape: ', latent_code.size()
        hidden_state_G_0, fake_0 = self.netG(latent_code, hidden_state_G)
        self.fakes = [fake_0]
        #print 'Prediction size: ', fake_0.size()

        self.loss_gan = self.criterionGAN(fake_0, True)*lambda_gan
        self.loss_pix = self.criterionPixel(fake_0, self.real_Y[0])*lambda_pix
        self.fake_fea = [self.netME(fake_0)]
        self.real_fea = [Variable(self.netME(self.real_Y[0]).data)]
        self.loss_fea = self.criterionFeat(self.fake_fea[-1],self.real_fea[-1])*lambda_fea

        for t in xrange(1, self.pre_len):
            hidden_state_M, outmcode = self.netMP(self.pre_feats[-1],hidden_state_M)
            self.pre_feats += [outmcode]

            concat_codes = []
            for l in xrange(len(self.content_codes)):
                concat_codes += [torch.cat([self.content_codes[l], self.pre_feats[-1] - self.motion_codes[l]],2)]
            latent_code = torch.cat(concat_codes, 1)
            hidden_state_G_t, fake_t = self.netG(latent_code, hidden_state_G)
            self.fakes += [fake_t]
            self.loss_gan += self.criterionGAN(fake_t, True)*lambda_gan
            self.loss_pix += self.criterionPixel(fake_t, self.real_Y[t])*lambda_pix
            self.fake_fea += [self.netME(fake_t)]
            self.real_fea += [Variable(self.netME(self.real_Y[t]).data)]
            self.loss_fea += self.criterionFeat(self.fake_fea[-1],self.real_fea[-1])*lambda_fea

        self.loss_G = self.loss_gan + self.loss_pix #+ self.loss_fea
        self.loss_G.backward()
        #print 'generation loss: ', self.loss_gan.data[0], ' pixel loss: ', self.loss_pix.data[0], ' feature loss: ', self.loss_fea.data[0], ' overall loss: ', self.loss_G.data[0]
    def backward_P(self):
        # forward
        #print len(self.real_X)
        #print self.real_X[0]
        self.motion_codes_p = [self.netME(real) for real in self.real_X] # [bs,1, oc, s0, s1]*len(self.real_X)
        batch_size = self.motion_codes_p[0].size(0)
        #print 'batch_size: ', batch_size
        hidden_state_MP = self.netMP.init_hidden(batch_size)
        #print 'hidden_h shape ',len(hidden_state_MP)
        #print 'hidden_h shape ',hidden_state_MP[0][0].size()
        #print 'motion_codes size: ', self.motion_codes_p[0].size()
        self.mcode_p = self.netMP(torch.cat(self.motion_codes_p,1), hidden_state_MP)
        hidden_state_M = self.mcode_p[0]
        self.pre_feats_p = [self.mcode_p[1]]
        self.real_fea_p = [self.netME(self.real_Y[0])]
        self.loss_pre = self.criterionPre(self.pre_feats_p[-1], Variable(self.real_fea_p[-1].data))

        for t in xrange(1, self.pre_len):
            hidden_state_M, outmcode = self.netMP(self.pre_feats_p[-1],hidden_state_M)
            self.pre_feats_p += [outmcode]
            self.real_fea_p += [self.netME(self.real_Y[t])]
            self.loss_pre += self.criterionPre(self.pre_feats_p[-1],Variable(self.real_fea_p[-1].data))

        self.loss_P = self.loss_pre
        self.loss_P.backward()
        #print 'output shape: ', len(self.pre_feats_p), ' output size: ', self.pre_feats_p[0].size(), 'Loss_P: ', self.loss_P.data[0]

    def optimize_generator(self):
        # forward_G
        self.forward_G()
        #Encoder
        self.optimizer_CE.zero_grad()
        self.optimizer_ME.zero_grad()
        # Decoder
        self.optimizer_G.zero_grad()

        self.backward_G()

        self.optimizer_CE.step()
        self.optimizer_ME.step()
        self.optimizer_G.step()

        # D

    def optimize_discriminator(self):
        #self.optimizer_D.zero_grad()
        #self.backward_D()
        self.optimizer_D.step()

    def optimize_predictor(self):
        # forward_P
        '''self.forward_P()
        # Encoder
        self.optimizer_ME.zero_grad()
        # Predictor
        self.optimizer_MP.zero_grad()

        self.backward_P()
        '''
        self.optimizer_ME.step()
        self.optimizer_MP.step()

    def optimize_parameters(self):
    #    self.optimize_predictor()
        self.optimize_generator()
        self.optimizer_D.zero_grad()
        self.backward_D()
        # Encoder
        self.optimizer_ME.zero_grad()
        # Predictor
        self.optimizer_MP.zero_grad()

        self.backward_P()



    def get_current_errors(self):
        PRE = self.loss_pre.data[0]
        GAN = self.loss_gan.data[0]
        FEA = self.loss_fea.data[0]
        PIX = self.loss_pix.data[0]
        DES = self.loss_D.data[0]
        return OrderedDict([('Prediction', PRE), ('G', GAN), ('Feature', FEA), ('Pixel', PIX), ('D',DES)])

    def get_current_visuals(self):
        #real_Y_0 = util.tensor2im(self.real_Y[0].data)
        #fake_Y_0 = util.tensor2im(self.fakes[0].data)
        #real_Y_E = util.tensor2im(self.real_Y[-1].data)
        #fake_Y_E = util.tensor2im(self.fakes[-1].data)
        images = []
        for i in xrange(self.seq_len):
            name = 'frame_'+str(i)
            image = util.tensor2im(self.real_X[i].data)
            images += [(name,image)]

        for j in xrange(self.pre_len):
            real_name = 'frame_'+str(self.seq_len+j)
            real_image = util.tensor2im(self.real_Y[j].data)
            fake_name = 'fake_'+str(self.seq_len+j)
            fake_image = util.tensor2im(self.fakes[j].data)
            images += [(real_name, real_image),(fake_name,fake_image)]

        return OrderedDict(images)
        #return OrderedDict([('real_Began', real_Y_0), ('Pred_Began', fake_Y_0), ('real_End', real_Y_E),
        #                    ('Pred_End', fake_Y_E)])

    def save(self, label):
        self.save_network(self.netCE, 'CE', label, self.gpu_ids)
        self.save_network(self.netME, 'ME', label, self.gpu_ids)
        self.save_network(self.netMP, 'MP', label, self.gpu_ids)
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_CE.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_ME.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_MP.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
