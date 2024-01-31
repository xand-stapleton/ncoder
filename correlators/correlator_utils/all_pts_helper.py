from jax import vmap
from jaxlib.xla_client import execute_with_python_values_replicated
import numpy as np
import jax.numpy as jnp
from math import factorial
from itertools import combinations_with_replacement
from collections import namedtuple
import gc
from torch.utils.data import Dataset, Subset, DataLoader
from correlator_utils.all_pts_correlator import partition
import einops

from correlator_utils.helper_functions import gen_combs, gen_contractions, partition

''' Logic for the program:

1. Create an array with all possible pixel combinations for each image in batch
2. Find all length-k partitions of the indices corresp. to k point correlators
3. Stack the pixel combinations for image in batch
4. Evaluate correlators along the batch axis 
'''
# Create a named tuple since this allows for more readable code later
CorrelatorsShape = namedtuple('CorrelatorsShape', 'batch_dim length n')


# Create a class for masked correlators
class Cumulants:
	def __init__(self, batch_size, image_dim=784, k=3, acceptance_parameter=1,
			     usejax = False, moments=False):
		'''
		Initialise the cumulants class.

		Arguments:
		image: Array of image pixel values (NDArary).
		n: Number of point correlators (int).
		acceptance_param: Proportion of correlators to consider (float)
		'''

		# Image parameters
		self.image_dim = image_dim

		# Length of correlator
		self.k = k

		self.batch_size = batch_size

		# Generate the number of correlators (faster than dynamically finding)
		combs_numerator = factorial(self.k + self.image_dim - 1)
		combs_denominator = factorial(self.image_dim - 1) * factorial(self.k)
		self.combs_with_replacement = combs_numerator // combs_denominator

		# Number of correlators to keep
		self.acceptance_parameter = acceptance_parameter

		# Correlator parameters
		self.correlators_shape = CorrelatorsShape(self.batch_size,
									self.combs_with_replacement, self.k)

		self.mask_number = round(self.correlators_shape.length * self.acceptance_parameter)
		self.masked = False

		# The number of rows = n, and represents number of correlator points

		if moments:
			# No need to use moment-cumulant formula as we only need one moment
			self.partition_indices = list(range(self.k))
		else:
			# Generate partitioned indices (moment-cumulant formula/Wick's thm blocks)
			self.partition_indices = list(partition(list(range(self.k))))

		# Use jax.numpy instead of numpy. Default=False. Jax is faster for
		# larger datasets
		if usejax:
			self.lib = jnp
		else:
			self.lib = np


	def create_mask(self, mask_method, dataloader, batch_size, num_batch_to_consider):

		# Check that there's a defined acceptance parameter
		if self.acceptance_parameter is None:
			raise SystemExit('Acceptance parameter is None.',
					'Ambiguous masking.')

		if mask_method == 'uniform':
			# This is just one step. NumPy is fine here even if usejax=True
			mask = np.random.choice(self.correlators_shape.length,
						self.mask_number, replace=False)

		elif mask_method == 'BKS':
			training_dataset = dataloader.dataset
			subset = Subset(training_dataset, range(num_batch_to_consider * batch_size))
			subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
			combination_indices = gen_combs(self.image_dim, self.k, self.combs_with_replacement)
			correlator_common_non_zero = np.arange(0, self.combs_with_replacement)
			for batch in subset_loader:
				batch_data, _ = batch
				batch_data = einops.rearrange(batch_data,
									 'b c x y -> b (c x y)')
				batch_data = batch_data.numpy()
				print('Batch length: ', len(batch_data))
				# Find the true cumulants of the batch
				print('Length of comb indices: ', len(combination_indices))
				true_cumulants = self.vector_form_eval_correlators_jax(combination_indices, batch_data)

				# These are the positions of non-zero cumulants
				non_zero_cumulants = np.nonzero(true_cumulants)
				correlator_common_non_zero = np.intersect1d(non_zero_cumulants,
												   correlator_common_non_zero)
				print(f'New number of intersecting non-zero {self.k}-pt correlators: ', len(correlator_common_non_zero))
				del true_cumulants, non_zero_cumulants

			desired_sample_correlators = round(len(correlator_common_non_zero) * self.acceptance_parameter)
			mask = np.random.choice(correlator_common_non_zero,
						   size=desired_sample_correlators, replace=False)

		else:
			raise NotImplementedError('Only implemented masking methods are uniform and BKS')

		return mask

	def evaluate_correlator(self, correlator):
		# print(correlator)
		# Create a blank partition term
		cumulant_expansion = 0
		# for each partition
		for part in self.partition_indices:
			# Reset the constituent block count for each partition
			block_term = 1
			# for each block in part
			for block_indices in part:
				# is this axis correct?
				# print(correlator[:, block_indices])
				# print(np.prod(correlator[:, block_indices], axis=1))
				block_term *= np.mean(np.prod(correlator[:, block_indices], axis=1))
			length = len(part)
			sign = (-1) ** (length - 1)
			cumulant_expansion += block_term * sign * factorial(length-1)
		return cumulant_expansion

	def vector_form_eval_correlators_numpy(self, combination_indices, batch):
		'''Forms the correlator from points and evaluates over a batch

		We expect the batch to be an array of shape:
		{image_number (batch idx), image_dim}
		'''

		# Each combination indices sublist is an n-point correlator, where
		# each number in a correlator is a batch of samples of x_i
		correlators = np.empty(len(combination_indices))

		for idx, combination in enumerate(combination_indices):
			# Takes all x_i all x_j all x_k in batch and finds combs
			# Input to evaluate correlators. Each row is all obs of x_1, x_2, etc 
			correlators[idx] = self.evaluate_correlator(np.take(batch, combination, axis=1))
		return correlators

	def vector_form_eval_correlators_jax(self, combination_indices, batch):
		'''Forms the correlator from points and evaluates over a batch'''

		# Each combination indices sublist is an n-point correlator, where
		# each number in a correlator is a batch of samples of x_i
		correlators = jnp.empty(len(combination_indices))

		eval_correlator = lambda combination: self.evaluate_correlator(jnp.take(batch, combination, axis=1))

		# eval_correlator_batched = vmap(eval_correlator, (0, None), 0)
		eval_correlator_batched = vmap(eval_correlator)
		correlators = eval_correlator_batched(combination_indices)

		return correlators

	def mask_correlators(self, mask):
		# Prevent masking several times
		if not self.masked:
			self.masked = True
			# For each set of correlators in the batch, mask
			for i in range(self.correlators_shape.batch_dim):
				self.correlators[i] = self.correlators[i][mask]
			self.correlators_shape = CorrelatorsShape(*self.correlators.shape)
		return self.correlators
