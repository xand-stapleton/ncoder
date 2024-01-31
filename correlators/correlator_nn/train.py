# TODO: Tidy up and document code

from tqdm.auto import tqdm
import torch
import sys

from torch.utils.data import DataLoader
from datasets import *
from torch import nn, optim
import matplotlib.pyplot as plt

# Hacky workaround to prevent relative import error
sys.path.append('..')
from correlator_utils.all_pts_helper import Cumulants, gen_combs
# from utils import diagonalise_weights
import einops

import numpy as np

# Temporarily suppress tqdm output while debugging
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def check_set_gpu(number=None, verbose=True):
    # Sloppy function to check if the GPU is available (CUDA) and if so use it!
    if verbose:
        print('Checking for the GPU...')

    if torch.cuda.is_available():
        if verbose:
            print('CUDA device is available!')
            if number is not None:
                print(f'Attempting to set GPU to device {number}')
                device = torch.cuda.device(f'cuda:{int(number)}')
                # Set to device 'Number'
            else:
                print('Continuing with default GPU')

            print('CUDA device count:', torch.cuda.device_count())
            print(f'Current CUDA device is device'
                  f'{torch.cuda.current_device()},'
                  f'and is'
                  f'{torch.cuda.get_device_name(torch.cuda.current_device())}')

            # Torch memory properties
            tot_mem = torch.cuda.get_device_properties(0).total_memory
            res_mem = torch.cuda.memory_reserved(0)
            alloc_mem = torch.cuda.memory_allocated(0)
            free_mem = res_mem - alloc_mem  # free inside reserved

            print(f'Memory information:\n'
                  f'Free memory{free_mem}\n'
                  f'GPU Memory:{tot_mem}')
        else:
            # Otherwise set the number if needed
            if number is not None:
                device = torch.cuda.device(f'cuda:{int(number)}')
    else:
        if verbose:
            print('No CUDA device found. Falling back to CPU')

        device = torch.device('cpu')
    return device


# Create a class for the dataset
# Define the neural network
class Net(nn.Module):
    def __init__(self, total_mask_shape, reduction, batch_size):
        super(Net, self).__init__()

        self.mask_shape = total_mask_shape
        self.reduction = reduction
        # self.fc = nn.Linear(28*28, self.mask_shape)
        # Use a simple single linear layer initially (nn.Sequential for later)
        self.fc = nn.Sequential(
            # ReLU optional
            nn.ReLU(),
            # nn.Linear(batch_size*28**2, int(self.reduction * self.mask_shape),
            #           bias=False),
            nn.Linear(batch_size*28**2, int(0.015*self.mask_shape) + 32,
                      bias=False),
            nn.ReLU(),
            # Need the +784 for the one point cumulants which are unmaasked
            # nn.Linear(int(self.reduction * self.mask_shape), self.mask_shape +
            #           784, bias=False),
            nn.Linear(int(0.015*self.mask_shape) + 32, self.mask_shape + 784, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        # Forward function is the action of the trained NN on the inputs
        # Flatten the image to use NN. By action of layer we need to be cols.
        return self.fc(x)


def save_checkpoint(filename, state_dict, model_args, batch, running_loss,
                    total_mask):
    print('Saving checkpoint...')
    checkpoint = {
        'model_args': model_args,
        'state_dict': state_dict(),
        'running_loss': running_loss,
        'batch_number': batch,
        'total_mask': total_mask,
    }
    torch.save(checkpoint, filename)
    print('Successfully saved state')


class Model(nn.Module):
    def __init__(self, acceptance_param_2pt=0.001, acceptance_param_3pt=0.001,
                 epochs=1, total_mask=(None, None),
                 mask_method='BKS', mask_batch_samples=20,
                 reduction=0.1, device='cpu',
                 checkpoint_file=None, checkpoint_freq=256,
                 recover_from_file=None):

        # Use slice(None) for no masking in mask method

        # Inherit from nn.Module
        super(Model, self).__init__()

        # Collect all args and retain for checkpointing. Less elegant than
        # using errors but doesn't throw variable not accessed errors
        self.args = {
            'acceptance_param_2pt': acceptance_param_2pt,
            'acceptance_param_3pt': acceptance_param_3pt,
            'epochs': epochs, 'total_mask': total_mask,
            'mask_method': mask_method,
            'mask_batch_samples': mask_batch_samples,
            'reduction': reduction, 'device': device,
            'checkpoint_file': checkpoint_file,
            'checkpoint_freq': checkpoint_freq,
        }

        if recover_from_file is not None:
            checkpoint = torch.load(recover_from_file)
            self.args = checkpoint['model_args']
            self.load_state_dict(checkpoint['state_dict'])
            self.running_loss = checkpoint['running_loss']
            # We need the +1 to account for the fact that checkpoints are saved
            # at the end of batch. Don't want to train again as this introduces
            # part of another epoch
            self.start_batch_number = int(checkpoint['batch_number']) + 1

        # Configure parameters
        self.run_before = False
        self.acceptance_param_2pt = self.args['acceptance_param_2pt']
        self.acceptance_param_3pt = self.args['acceptance_param_3pt']

        # Technical implementation parameters
        self.usejax = False
        self.checkpoint_file = self.args['checkpoint_file']
        self.checkpoint_freq = self.args['checkpoint_freq']

        # Use the KL divergence as a loss function. We want to minimise the
        # error between the approximated cumulants and the true cumulants

        # Use 'batchmean' reduction as that is consistent with Amari's def
        # self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.loss_fn = nn.MSELoss()

        # Set the number of epochs
        self.epochs = self.args['epochs']

        # Set the reduction
        self.reduction = self.args['reduction']

        # Allow a predefined mask
        self.mask_2_pt, self.mask_3_pt = self.args['total_mask']
        self.mask_method = self.args['mask_method']
        self.mask_batch_samples = self.args['mask_batch_samples']

        # This is generated with potentially recovered information so don't
        # need it explicitly in checkpoint. Initialise
        self.masked_combination_indices = None
        self.mask_shape_2_pt = None
        self.mask_shape_3_pt = None
        self.total_mask_shape = None

        # Set device
        self.device = self.args['device']

    def train(self, train_dataloader, fashion_mnist=False,
              exclude_outliers=True):

        # Add the dataloader information to the args dictionary
        self.args.update({
            'no_batches': len(train_dataloader),
            'batch_size': train_dataloader.batch_size,
        })

        self.no_batches = self.args['no_batches']
        self.batch_size = self.args['batch_size']

        # Train the network
        print('Training the model')

        # Iterate over epoch number.
        for epoch in tqdm(range(self.epochs), position=0, desc='Epochs',
                          leave=True, colour='red'):
            running_loss = 0.0

            # Iterate over batches
            for batch_number, data in enumerate(pbar_batch := tqdm(
                train_dataloader, position=1, desc='Batches', leave=True,
                colour='green'
            ), 0):

                # Split our MNIST dataset batch into the images and labels
                # (latter not used). This is the correct input structure 
                # {image_number (batch idx), image_dim}
                image_batch, _ = data
                image_batch = einops.rearrange(image_batch,
                                               'b c x y -> b (c x y)')
                image_batch_numpy = image_batch.numpy()

                # Diagnostic data
                # print('Inputs shape', inputs.shape)
                # print('Image shape', inputs[img].shape) print('Image contents', inputs[img])


                # If not run before, instantiate the cumulants class and
                # generate contractions(faster than calculating each time)
                if not self.run_before:
                    two_pt_cumulants = Cumulants(len(image_batch_numpy), 784, k=2,
                                                 acceptance_parameter=self.acceptance_param_2pt,
                                                 usejax=True)
                    two_pt_comb_indices = gen_combs(
                        784, 2, two_pt_cumulants.combs_with_replacement)

                    three_pt_cumulants = Cumulants(len(image_batch_numpy), 784, k=3,
                                                   acceptance_parameter=self.acceptance_param_3pt,
                                                   usejax=True)
                    three_pt_comb_indices = gen_combs(
                        784, 3, three_pt_cumulants.combs_with_replacement)

                    # Generate a loss history (total)
                    self.loss_hist = []

                # Generate a mask if first time and we don't have one
                # already (not passed in/recovered). We want the mask to be
                # consistent for all cumulants
                if self.mask_2_pt is None and self.mask_method is not None:
                    self.mask_2_pt = two_pt_cumulants.create_mask(
                        mask_method=self.mask_method,
                        dataloader=train_dataloader,
                        batch_size=self.batch_size,
                        num_batch_to_consider=self.mask_batch_samples)

                    self.mask_3_pt = three_pt_cumulants.create_mask(
                        mask_method=self.mask_method,
                        dataloader=train_dataloader,
                        batch_size=self.batch_size,
                        num_batch_to_consider=self.mask_batch_samples)

                    # The shape of the first layer will be the
                    # same shape
                    # as the cumulants
                    self.mask_shape_2_pt = self.mask_2_pt.shape
                    self.mask_shape_3_pt = self.mask_3_pt.shape

                elif self.mask_2_pt is None and self.mask_method is None:
                    self.mask = slice(None)

                    self.mask_shape_2_pt = two_pt_cumulants.combs_with_replacement
                    self.mask_shape_3_pt = three_pt_cumulants.combs_with_replacement


                # Update the mask argument for when the args are saved
                self.args.update({'mask_2_pt': self.mask_2_pt})
                self.args.update({'mask_3_pt': self.mask_3_pt})

                # Create a new array of combination indices from mask
                self.masked_2_pt_comb_indices = two_pt_comb_indices[self.mask_2_pt]
                self.masked_3_pt_comb_indices = three_pt_comb_indices[self.mask_3_pt]

                # If the neural network is not initialised,
                # then initialise it and the optimiser
                if not self.run_before:
                    self.total_mask_shape = len(self.mask_2_pt) + len(self.mask_3_pt)
                    print('Total mask shape (2pt + 3pt)', self.total_mask_shape)
                    self.net = Net(self.total_mask_shape, self.reduction, self.batch_size).to(self.device)
                    print(self.net)
                    if fashion_mnist:
                        self.optimizer = optim.Adam(self.net.parameters(),
                                                    lr=0.00001)
                    else:
                        self.optimizer = optim.Adam(self.net.parameters(),
                                                    lr=0.0001)
                    # The algorithm has now been run before, so don't need
                    # initialisation procedure
                    self.run_before = True

                # Zero the optimiser gradient
                self.optimizer.zero_grad()

                # Mask and evaluate the cumulant sample

                # Evaluate the true masked correlators
                # print('Eval. sample cumulants: ', cumulants.vector_form_eval_correlators_jax(self.masked_combination_indices))
                eval_mask_cumul_2pt = torch.tensor(np.asarray(two_pt_cumulants.vector_form_eval_correlators_jax(self.masked_2_pt_comb_indices, image_batch_numpy))).to(self.device)
                eval_mask_cumul_3pt = torch.tensor(np.asarray(three_pt_cumulants.vector_form_eval_correlators_jax(self.masked_3_pt_comb_indices, image_batch_numpy))).to(self.device)

                # Diagnostic. Check cumulants
                # print(eval_mask_cumul)

                # Generate the NN prediction of the cumulants
                outputs = self.net(image_batch.flatten())

                one_point_cumul = torch.mean(image_batch, dim=0)

                combined_cumul = torch.cat((one_point_cumul,
                                            eval_mask_cumul_2pt,
                                            eval_mask_cumul_3pt))

                # Evaluate the KL divergence between NN and true cumulants
                loss = self.loss_fn(outputs, combined_cumul)

                # Exclude clear outliers (hacky)
                if exclude_outliers:
                    if len(self.loss_hist) > 10:
                        if loss.item() > self.loss_hist[0]:
                            continue

                self.loss_hist.append(loss.item())

                # Update the loss on the end of the progressbar
                pbar_batch.set_postfix({'loss': float(loss)})


                # Backward pass: backprop and update model with optimiser
                loss.backward()
                self.optimizer.step()

                # If we want to checkpoint, then save at the end of each
                # checkpoint_freq batches
                if (self.checkpoint_freq is not None
                        and (batch_number % self.checkpoint_freq)
                        == self.checkpoint_freq-1):

                    # If file isn't specified, generate a sensibly named
                    # one
                    if self.checkpoint_file is None:
                        outfile = (f'checkpoints/checkpoint_e{epoch+1}_'
                                   f'l{running_loss/100:.3f}_'
                                   f'b{batch_number}.pth.tar')
                    else:
                        outfile = self.checkpoint_file

                    print('checkpoint reached')
                    # save_checkpoint(outfile, self.state_dict,
                    #                 self.args, batch_number,
                    #                 running_loss, 
                    #                 (self.masked_2_pt_comb_indices,
                    #                  self.masked_3_pt_comb_indices))

                    # Print the loss at the end of each batch (preceded tqdm)
                    # if img % self.batch_size == self.batch_size - 1:
                    #     print(f'Epoch {epoch+1}, batch {batch_number+1}:',
                    #           f'loss {running_loss/100:.3f}')

                # Reset running loss
                # running_loss = 0.0
            plt.plot(self.loss_hist)
            plt.savefig('reconstr/since_iaifi/loss_enc.pdf')
            torch.save(self.loss_hist, 'reconstr/since_iaifi/enc_loss_hist')
