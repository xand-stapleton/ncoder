# TODO: !Docstrings!

from torch.utils.data import DataLoader
from datasets import MNISTDataset, FashionMNISTDataset

import sys
sys.path.append('..')
import train as cumulant_gen

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import einops
import matplotlib.pyplot as plt

# Suppress error from too many open files
torch.multiprocessing.set_sharing_strategy('file_system')

def bootstrap_approx(cum_batch_size, cum_batch_no, cum_epochs,
                     cum_acceptance_param_2pt, cum_acceptance_param_3pt,
                     cum_mask_method, fish_batch_size, reconstr_epoch,
                     reconstr_lr):
    '''
    Train the cumulant estimator neural network and the reconstruction network
    '''
    
    # Train up the cumulant estimator neural network
    device = cumulant_gen.check_set_gpu()

    train_dataloader = cumulant_gen.MNISTDataset().get_testing_data(
        batches=cum_batch_no, batch_size=cum_batch_size
    )


    gen_model = cumulant_gen.Model(acceptance_param_2pt=cum_acceptance_param_2pt,
                                   acceptance_param_3pt=cum_acceptance_param_3pt,
                                   epochs=cum_epochs, total_mask=(None, None),
                                   mask_method=cum_mask_method, 
                                   mask_batch_samples=20,
                                   reduction=0.1, device=device)
    gen_model.train(train_dataloader, fashion_mnist=False)

    plt.plot(range(0, len(gen_model.loss_hist)), gen_model.loss_hist)
    plt.show()

    print('Passed training approximator')

    reconstr_model = ReconstructorModel(usejax=True)
    reconstr_model.train_test_approx(train_dataloader, gen_model,
                                     cum_acceptance_param_2pt,
                                     cum_acceptance_param_3pt, False,
                                     reconstr_epoch, reconstr_lr)

    print('Passed reconstruction model')

    plt.plot(range(0, len(reconstr_model.reconstr_loss_hist)),
             reconstr_model.reconstr_loss_hist)
    plt.show()

    return gen_model, reconstr_model, None


class ReconstructorNet(nn.Module):
    def __init__(self, masked_cumul_size, batch_size):
        super(ReconstructorNet, self).__init__()
        self.layers = nn.Sequential(nn.ReLU(),
                                    # Need +784 for the one-point cumulants
                                    nn.Linear(masked_cumul_size + 784, 8200, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(8200, 784, bias=False),
                                    nn.ReLU(),
                                    )

    def forward(self, x):
        return self.layers(x)

class ReconstructorModel:
    def __init__(self, usejax=True):
        self.run_before = False
        self.usejax = usejax
        self.reconstr_net = None
        self.mixed_losses = []
        self.stiff_losses = []
        self.reconstr_loss_hist = []
        return None


    def train_test_approx(self, samples, trained_approximator,
                          two_point, three_point, test=False, epochs=1, lr=0.001):
        '''
        Use cumulant approximator and reconstruct images based on cumulants
        '''


        two_point = float(two_point)
        three_point = float(three_point)

        approximator_model = trained_approximator
        approximator_net = approximator_model.net

        # Create reconstructor. Mask shape is same as no of (masked) cumulants
        # Length of training samples is num batches, len of batch is batch size
        print('Batch size: ', samples.batch_size)
        self.reconstr_net = ReconstructorNet(approximator_model.total_mask_shape,
                                             batch_size=samples.batch_size)

        # NN things
        loss_hist = []
        reconstr_hist = []
        batch_avg_hist = []

        self.loss = nn.MSELoss()
        loss = self.loss

        optimizer = optim.Adam(self.reconstr_net.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch_no, batch in enumerate(samples):
                # Want to calculate the approx cumulants of
                # image batch, pass to NN and reconsruct image

                # Prerequisites
                # Unpack the batch as it comes from DataLoader
                batch, _ = batch

                # Rearrange to collapse the image channel and wxh dims
                batch = einops.rearrange(batch, 'b c x y -> b (c x y)')

                # 1. Calculate the forward pass of the approximator
                approx_cumul = approximator_net.forward(batch.flatten())

                # 2. Pass this into the reconstructor
                reconstr_batch = self.reconstr_net(approx_cumul)

                # Since we only have 1 image out, the cumulants of n copies of
                # the same image are the same as for 1 copy. This is needed
                # since that encoder expects a certain sized input.
                reconstr_batch = torch.cat([reconstr_batch.clone() for _ in range(samples.batch_size)])
                # 3. Calculate the cumulants of the pred. batch
                reconstr_batch_cumul = approximator_net.forward(reconstr_batch.flatten())

                # 4. Evaluate the loss of the reconstructed batch (loss from
                # cumulants)
                reconstr_loss = loss(approx_cumul, reconstr_batch_cumul)
                print('Reconstr loss shape: ', reconstr_loss.shape)
                loss_hist.append(reconstr_loss.detach().numpy())
                reconstr_hist.append(reconstr_batch.detach())
                batch_avg_hist.append(batch)

                if not test:
                    optimizer.zero_grad()
                    reconstr_loss.backward()
                    optimizer.step()

                print(f'Epoch: {epoch}, Batch: {batch_no}, Loss: {reconstr_loss.detach()}')
        torch.save(reconstr_hist, f'reconstr/since_iaifi/mixed_reconstr_avg_approx_all_1pt_{two_point}-2pt_{three_point}-3pt_simpleout')
        torch.save(batch_avg_hist, f'reconstr/since_iaifi/batch_avg_approx_all_1pt_{two_point}-2pt_{three_point}-3pt_simpleout')
        torch.save(loss_hist, f'reconstr/since_iaifi/loss_hist_all_1pt_{two_point}-2pt_{three_point}-3pt_simpleout')
        return self


if __name__ == '__main__':
    bootstrap_approx(cum_batch_size=16, cum_batch_no=50, cum_epochs=8,
                     cum_acceptance_param_2pt=0.00, cum_acceptance_param_3pt=0.017,
                     cum_mask_method='BKS', fish_batch_size=16, reconstr_epoch=65,
                     reconstr_lr=0.0001)
    # bootstrap_approx(cum_batch_size=16, cum_batch_no=50, cum_epochs=8,
    #                  cum_acceptance_param_2pt=0.00, cum_acceptance_param_3pt=0.001,
    #                  cum_mask_method='BKS', fish_batch_size=16, reconstr_epoch=65,
    #                  reconstr_lr=0.0001, stiffen=False, cutoff=0.4)
