import datetime
import os
import time

import numpy as np
import torch
from torchvision.utils import save_image

from logger import Logger
from discriminator import Discriminator
from drnet import DRNET
from drunet import DRUNET


class Solver(object):
    def __init__(self, nomakeup_loader, makeup_loader, config):
        #
        # Setup data loaders
        self.nomakeup_loader = nomakeup_loader
        self.makeup_loader = makeup_loader

        #
        # High-Level configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tensorboard = (config.use_tensorboard == 'true')

        #
        # Model configuration
        self.generator = config.generator_type
        self.block = config.block_type
        self.makeup_label_size = config.makeup_label_size
        self.nomakeup_label_size = 0
        self.att_depth = config.att_depth
        self.image_size = config.image_final_size
        self.g_init_conv = config.g_init_conv
        self.d_init_conv = config.d_init_conv
        self.g_std_blocks = config.g_std_blocks
        self.g_res_blocks = config.g_res_blocks
        self.downsampling_factor = config.downsampling_factor
        self.d_depth = config.d_depth
        self.lambda_adv = config.lambda_adv
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_recalib_step = config.lambda_recalib_step
        self.lambda_recalib_factor = config.lambda_recalib_factor

        #
        # Training configuration
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.g_train_step = config.g_train_step
        self.makeup_removal_step = config.makeup_removal_step
        self.iters_bef_decay = config.iters_bef_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_update_step = config.lr_update_step
        self.g_decay_rate = config.g_decay_rate
        self.d_decay_rate = config.d_decay_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iter = config.resume_iter
        self.log_step = config.log_step
        self.debug_step = config.debug_step
        self.model_ckpt_step = config.model_ckpt_step

        #
        # Directory configuration
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.debug_dir = config.debug_dir
        self.test_results_dir = config.test_results_dir

        #
        # Build the model and tensorboard
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        # Expanded size when turned into one hot vector
        # Add two for mask
        exp_mlabel_size = self.makeup_label_size + np.sum(self.att_depth)
        joint_label_size = self.nomakeup_label_size + exp_mlabel_size + 2

        # Implement chosen generator architecture
        module = __import__(self.generator)
        self.block = getattr(module, self.block)

        if self.generator == 'drnet':
            self.G = DRNET(
                block=self.block, 
                downsampling_factor=self.downsampling_factor,
                res_blocks=self.g_res_blocks,
                init_channels=self.g_init_conv, 
                label_dim=joint_label_size
            )
        elif self.generator == 'drunet':
            self.G = DRUNET(
                block=self.block, 
                res_blocks=self.g_res_blocks,
                downsampling_factor=self.downsampling_factor,
                init_channels=self.g_init_conv, 
                label_dim=joint_label_size
            )

        # Build discriminator
        # Exclude mask
        self.D = Discriminator(
            self.image_size, 
            depth=self.d_depth, 
            init_channels=self.d_init_conv,
            label_dim=joint_label_size-2
        )

        # Build network optimisers
        self.g_optimiser = torch.optim.Adam(
            self.G.parameters(), 
            self.g_lr,
            [self.beta1, self.beta2]
        )

        self.d_optimiser = torch.optim.Adam(
            self.D.parameters(), 
            self.d_lr,
            [self.beta1, self.beta2]
        )

        # Setup networks in correct device for processing
        self.G.to(self.device)
        self.D.to(self.device)

    def restore_model(self, resume_iter):
        print('Loading the trained models from step {}...'.format(resume_iter))
        G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(resume_iter))
        D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(resume_iter))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        self.logger = Logger(self.log_dir)

    def log_info(self, curr_iter, param):
        for tag, value in param.items():
            self.logger.scalar_summary(tag, value, curr_iter)

    def print_train_info(self, curr_iter, dataset, param):
        elapsed_time = time.time() - self.start_time
        elapsed_time = str(datetime.timedelta(seconds=elapsed_time))[:-7]

        log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(
            elapsed_time, curr_iter, self.num_iters, dataset
        )

        for tag, value in param.items():
            log += ", {}: {:.4f}".format(tag, value)
        
        print(log)

        if self.use_tensorboard:
            self.log_info(curr_iter, param)


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                   Training Auxiliary Functions

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def update_lr(self, g_lr, d_lr):
        for parameters in self.g_optimiser.param_groups:
            parameters['lr'] = g_lr
        for parameters in self.d_optimiser.param_groups:
            parameters['lr'] = d_lr

    def denorm(self, x):
        # Convert input range from [-1, 1] to [0, 1]
        out = (x + 1) / 2
        out = out.clamp_(0, 1)
        return out

    def shuffle(self, in_label, att_depth):
        np.random.seed(time.time())
        torch.manual_seed(time.time())
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        shuffled_label = []
        non_binary_atts = len(att_depth)

        # Separate classes to ensure non-binary attributes are not
        # simultaneously zero (it would mean no makeup)
        for i in range(non_binary_atts):
            random_order = torch.randperm(in_label.size(0))
            random_column = in_label[random_order, i]
            if i == 0:
                shuffled_label = torch.Tensor(random_column)
            else:
                shuffled_label = torch.Tensor(np.column_stack((shuffled_label, random_column)))

        for i, row in enumerate(shuffled_label):
            if torch.sum(row) == 0:
                # Choose the minimum to guarantee the class exists
                small_max = int(torch.min(torch.Tensor(att_depth)))
                shuffled_label[i] = shuffled_label[i] + torch.randint_like(
                    shuffled_label[i], low=1, high=small_max+1
                )

        for i in range(non_binary_atts, in_label.size(1)):
            random_order = torch.randperm(in_label.size(0))
            random_column = in_label[random_order, i]
            shuffled_label = torch.Tensor(np.column_stack((shuffled_label, random_column)))

        return torch.Tensor(shuffled_label)

    def reset_gradients(self):
        self.g_optimiser.zero_grad()
        self.d_optimiser.zero_grad()

    def onehot_encoding(self, label_list, att_depth):
        # Build a one-hot vector based on label_list activation values
        batch_size = label_list.size(0)
        one_hot = torch.zeros(batch_size, att_depth)
        one_hot[np.arange(batch_size), label_list.long()] = 1
        return one_hot.long()

    def label_convert(self, in_label, att_depth):
        non_binary_atts = len(att_depth)

        # Split list between non-binary classes, that need encoding
        # and binary classes, that do not
        non_bin_label = in_label[:, :non_binary_atts]
        bin_label = in_label[:, non_binary_atts:]

        for i in range(non_bin_label.size(1)):
            label = non_bin_label[:, i]

            # Activation vector
            # 1 if attribute is activated, 0 otherwise
            act_vector = label/label
            act_vector[np.isnan(act_vector)] = 0
            act_vector = act_vector[:, None].long()

            # Build a one-hot vector for attribute in position i
            feature_vector = self.onehot_encoding(label, att_depth[i]+1)

            # Concatenate attribute vector with activation vector
            # Discard column 0 of feature_vector
            # Value of zero (activating column 0) means attribute inactive
            feature_vector = torch.cat((act_vector, feature_vector[:, 1:].long()), dim=1)

            if i == 0:
                non_bin_list = feature_vector
            else:
                non_bin_list = torch.cat((non_bin_list, feature_vector), 1)

        # Concatenate both sets of classes
        att_list = torch.cat((non_bin_list, bin_label.long()), dim=1)

        return att_list

    def gradient_penalty(self, x, y):
        # Penalises model if gradient moves away from target value 1
        weight = torch.ones(y.size()).to(self.device)

        # Sum of the gradients of images wrt probabilities tied to its patches
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )
        dydx = dydx[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        gradient_penalty = torch.mean((dydx_l2norm - 1)**2)
        return gradient_penalty

    def classification_loss(self, logit, target):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction='sum')
        return loss/logit.size(0)


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                                Training Functions

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def train_discriminator(self, x_real, orig_label, trg_label, orig_class, trg_class):
        #
        # Compute loss for real images
        out_src, out_cls = self.D(x_real)
        d_loss_real = torch.mean(out_src)
        d_loss_cls = self.classification_loss(out_cls, orig_label)

        #
        # Compute loss for fake images
        x_fake = self.G(x_real, trg_class)
        out_src, _ = self.D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        #
        # Compute loss for Gradient Penalty
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(x_hat, out_src)

        #
        # Adversarial Loss (Wasserstein GAN objective with Gradient Penalty)
        d_loss_adv = - d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp

        #
        # Overall Loss
        d_loss = self.lambda_adv * d_loss_adv + self.lambda_cls * d_loss_cls

        # Backward and optimise
        self.reset_gradients()
        d_loss.backward()
        self.d_optimiser.step()

        #
        # Logging loss values
        self.loss = {}
        self.loss['D/loss_real'] = d_loss_real.item()
        self.loss['D/loss_fake'] = d_loss_fake.item()
        self.loss['D/loss_gp'] = d_loss_gp.item()
        self.loss['D/loss_adv'] = d_loss_adv.item()
        self.loss['D/loss_cls'] = d_loss_cls.item()
        self.loss['D/loss_tot'] = d_loss.item()

        #
        # For debugging
        self.disc = {}
        with torch.no_grad():
            max_cls, _ = out_cls.max(1)
            max_cls = max(max_cls)
            min_cls, _ = out_cls.min(1)
            min_cls = min(min_cls)
            median_cls = torch.median(out_cls)
            max_src, _ = out_src.max(1)
            max_src, _ = max_src.max(0)
            max_src, _ = max_src.max(0)
            max_src = max(max_src)
            min_src, _ = out_src.min(1)
            min_src, _ = min_src.min(0)
            min_src, _ = min_src.min(0)
            min_src = min(min_src)
            median_src = torch.median(out_src)
        self.disc['RealCls/Max'] = max_cls.item()
        self.disc['RealCls/Min'] = min_cls.item()
        self.disc['RealCls/Median'] = median_cls.item()
        self.disc['RealSrc/Max'] = max_src.item()
        self.disc['RealSrc/Min'] = min_src.item()
        self.disc['RealSrc/Median'] = median_src.item()

    def train_generator(self, x_real, orig_label, trg_label, orig_class, trg_class):
        #
        # Original to target transformation
        x_fake = self.G(x_real, trg_class)
        out_src, out_cls = self.D(x_fake)

        # Adversarial Loss
        g_loss_fake = - torch.mean(out_src)

        # Domain Classification Loss
        g_loss_trn_cls = self.classification_loss(out_cls, smooth_trg_label)

        #
        # Cycle-consistency condition
        x_reconstruct = self.G(x_fake, orig_class)
        g_loss_rec = torch.mean(torch.abs(x_real - x_reconstruct))

        #
        # Reconstruction classification loss
        _, rec_cls = self.D(x_reconstruct)
        g_loss_rec_cls = self.classification_loss(rec_cls, orig_label)

        # Summing both classification losses
        g_loss_cls = g_loss_trn_cls + g_loss_rec_cls

        #
        # Overall loss
        g_loss = self.lambda_adv * g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_rec * g_loss_rec

        # Backward and optimise
        self.reset_gradients()
        g_loss.backward()
        self.g_optimiser.step()

        # Logging
        self.loss['G/loss_adv'] = g_loss_fake.item()
        self.loss['G/loss_trn_cls'] = g_loss_trn_cls.item()
        self.loss['G/loss_rec_cls'] = g_loss_trn_cls.item()
        self.loss['G/loss_cls'] = g_loss_cls.item()
        self.loss['G/loss_rec'] = g_loss_rec.item()
        self.loss['G/loss_tot'] = g_loss.item()

        #
        # For debugging
        self.gen = {}
        with torch.no_grad():
            max_cls, _ = out_cls.max(1)
            max_cls = max(max_cls)
            min_cls, _ = out_cls.min(1)
            min_cls = min(min_cls)
            median_cls = torch.median(out_cls)
            max_src, _ = out_src.max(1)
            max_src, _ = max_src.max(0)
            max_src, _ = max_src.max(0)
            max_src = max(max_src)
            min_src, _ = out_src.min(1)
            min_src, _ = min_src.min(0)
            min_src, _ = min_src.min(0)
            min_src = min(min_src)
            median_src = torch.median(out_src)
        self.gen['FakeCls/Max'] = max_cls.item()
        self.gen['FakeCls/Min'] = min_cls.item()
        self.gen['FakeCls/Median'] = median_cls.item()
        self.gen['FakeSrc/Max'] = max_src.item()
        self.gen['FakeSrc/Min'] = min_src.item()
        self.gen['FakeSrc/Median'] = median_src.item()

    def train(self):
        self.loss = {}

        #
        # Setup data iterators
        makeup_iter = iter(self.makeup_loader)
        nomakeup_iter = iter(self.nomakeup_loader)

        #
        # Fixed inputs for debugging
        fix_makeup, makeup_fix_label = next(makeup_iter)
        fix_makeup = fix_makeup.float().to(self.device)
        fix_nomakeup = next(nomakeup_iter)
        fix_nomakeup = fix_nomakeup.float().to(self.device)

        debug_makeup_org = self.label_convert(makeup_fix_label, self.att_depth)
        debug_nomakeup_org = torch.zeros_like(debug_makeup_org)

        debug_makeup_mask = self.onehot_encoding(torch.zeros(fix_makeup.size(0)), 2)
        debug_nomakeup_mask = self.onehot_encoding(torch.ones(fix_nomakeup.size(0)), 2)

        #
        # Storing learning rate for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        #
        # Start or resume training.
        start_iter = 0
        if self.resume_iter:
            start_iter = self.resume_iter
            self.restore_model(self.resume_iter)

        # Each iteration performs training once for each dataset
        # In essence two iterations run in each actual iteration
        print('Start training...', "\n")
        self.start_time = time.time()

        for i in range(start_iter, self.num_iters):
            # Increase weight of adversarial and domain classification losses
            if (i+1) == self.lambda_recalib_step:
                self.lambda_adv *= self.lambda_recalib_factor
                self.lambda_cls *= self.lambda_recalib_factor

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            #
            # Fetch next batches
            # Restart if dataset finished
            try:
                makeup_real, makeup_orig_label = next(makeup_iter)
            except Exception:
                makeup_iter = iter(self.makeup_loader)
                makeup_real, makeup_orig_label = next(makeup_iter)

            try:
                nomakeup_real = next(nomakeup_iter)
            except Exception:
                nomakeup_iter = iter(self.nomakeup_loader)
                nomakeup_real = next(nomakeup_iter)

            # Convert makeup_orig_label to one hot vector
            makeup_orig_label = self.label_convert(makeup_orig_label, self.att_depth)

            # Create original label for nomakeup dataset
            # All 0's as no attribute activated
            nomakeup_orig_label = torch.zeros_like(makeup_orig_label)

            # Create mask vectors
            makeup_mask = self.onehot_encoding(torch.zeros(makeup_real.size(0)), 2)
            nomakeup_mask = self.onehot_encoding(torch.ones(nomakeup_real.size(0)), 2)

            #
            # Create target labels
            # Shuffle makeup labels for nomakeup
            makeup_trg_label = nomakeup_orig_label.clone()

            random_order = torch.randperm(makeup_orig_label.size(0))
            nomakeup_trg_label = makeup_orig_label[random_order].clone()

            # For makeup dataset
            makeup_orig_class = torch.cat([makeup_orig_label, makeup_mask], dim=1)
            makeup_trg_class = torch.cat([makeup_trg_label, makeup_mask], dim=1)

            # For nomakeup dataset
            nomakeup_orig_class = torch.cat([nomakeup_orig_label, nomakeup_mask], dim=1)
            nomakeup_trg_class = torch.cat([nomakeup_trg_label, nomakeup_mask], dim=1)

            # Joining and shuffling both datasets if nomakeup included
            if (i+1) % self.makeup_removal_step == 0:
                rnd_order = torch.randperm(self.batch_size*2)
                real_inputs = torch.cat((makeup_real, nomakeup_real), dim=0)[rnd_order]
                orig_labels = torch.cat((makeup_orig_label, nomakeup_orig_label), dim=0)[rnd_order]
                trg_labels = torch.cat((makeup_trg_label, nomakeup_trg_label), dim=0)[rnd_order]
                orig_class = torch.cat((makeup_orig_class, nomakeup_orig_class), dim=0)[rnd_order]
                trg_class = torch.cat((makeup_trg_class, nomakeup_trg_class), dim=0)[rnd_order]
            else:
                rnd_order = torch.randperm(self.batch_size)
                real_inputs = nomakeup_real[rnd_order]
                orig_labels = nomakeup_orig_label[rnd_order]
                trg_labels = nomakeup_trg_label[rnd_order]
                orig_class = nomakeup_orig_class[rnd_order]
                trg_class = nomakeup_trg_class[rnd_order]

            # =================================================================================== #
            #                                  2. Training                                        #
            # =================================================================================== #
            #
            # Input images and labels
            real_inputs = real_inputs.float().to(self.device)
            orig_labels = orig_labels.float().to(self.device)
            trg_labels = trg_labels.float().to(self.device)
            orig_class = orig_class.float().to(self.device)
            trg_class = trg_class.float().to(self.device)

            # Train the Discriminator
            self.train_discriminator(real_inputs, orig_labels, trg_labels,
                                     orig_class, trg_class)

            # Train the Generator
            # If condition determines D/G training balance
            if (i+1) % self.g_train_step == 0:
                self.train_generator(
                    real_inputs, orig_labels, trg_labels, orig_class, trg_class)

            # Print and log training information
            if (i+1) % self.log_step == 0:
                self.print_train_info(i+1, 'All', self.loss)
                self.log_info(i+1, self.disc)
                self.log_info(i+1, self.gen)

            # =================================================================================== #
            #                                 3. Varied                                           #
            # =================================================================================== #
            #
            # Saving checkpoints
            if (i+1) % self.model_ckpt_step == 0:
                G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

            # Decaying learning rates
            # At the end of every epoch, and after a certain point in training
            if (i+1) % self.lr_update_step == 0 and (i+1) > self.iters_bef_decay:
                g_lr -= self.g_decay_rate * self.g_lr
                d_lr -= self.d_decay_rate * self.d_lr
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            # Translate fixed inputs for debugging
            if (i+1) % self.debug_step == 0:
                with torch.no_grad():
                    # Build target labels
                    debug_nomakeup_trg = torch.cat([debug_makeup_org, debug_nomakeup_mask], dim=1)
                    debug_nomakeup_trg = debug_nomakeup_trg.float().to(self.device)
                    debug_makeup_trg = torch.cat([debug_nomakeup_org, debug_makeup_mask], dim=1)
                    debug_makeup_trg = debug_makeup_trg.float().to(self.device)

                    # Get original and translated images
                    x_fake_list = [fix_nomakeup]
                    x_fake_list.append(self.G(fix_nomakeup, debug_nomakeup_trg))
                    x_fake_list.append(fix_makeup)
                    x_fake_list.append(self.G(fix_makeup, debug_makeup_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)

                    # Save images onto debug directory
                    sample_path = os.path.join(self.debug_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                                Testing Functions

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def random_label(self, batch_size, att_depth, label_size):
        np.random.seed(time.time())
        torch.manual_seed(time.time())
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        random_label = []
        non_binary_atts = len(att_depth)

        # Separate classes due to different depths
        for i in range(non_binary_atts):
            # Generate a column of random numbers up to number of classes
            random_column = torch.randint(0, att_depth[i]+1, (batch_size, 1))
            if i == 0:
                random_label = torch.Tensor(random_column)
            else:
                random_label = torch.Tensor(np.column_stack((random_label, random_column)))

        for i, row in enumerate(random_label):
            if torch.sum(row) == 0:
                # Choose the minimum to guarantee the class exists
                small_max = int(torch.min(torch.Tensor(att_depth)))
                rnd_term = torch.randint_like(random_label[i], low=1, high=small_max+1)
                random_label[i] = random_label[i] + rnd_term

        for i in range(non_binary_atts, label_size):
            random_column = torch.randint(0, 2, (batch_size, 1))
            random_label = torch.Tensor(np.column_stack((random_label, random_column)))

        return torch.Tensor(random_label)

    def test(self):
        # Load the trained generator
        if self.resume_iter:
            self.restore_model(self.resume_iter)
        else:
            self.restore_model(self.num_iters)

        with torch.no_grad():
            # Test for makeup application
            for i, nomakeup_images in enumerate(self.nomakeup_loader):
                nomakeup_images = nomakeup_images.to(self.device)
                test_list = [nomakeup_images]
                
                random_label = self.random_label(
                    self.batch_size, self.att_depth, self.makeup_label_size
                )

                mask = self.onehot_encoding(torch.ones(nomakeup_images.size(0)), 2).long()

                target_label = self.label_convert(random_label, self.att_depth)
                target_label = torch.cat([target_label, mask], dim=1)
                target_label = target_label.to(self.device)
                test_list.append(self.G(nomakeup_images, target_label.float()))
                test_list = torch.cat(test_list, dim=3)

                # Save images onto test results directory
                result_path = os.path.join(self.test_results_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(test_list.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved test images into {}...'.format(result_path))
