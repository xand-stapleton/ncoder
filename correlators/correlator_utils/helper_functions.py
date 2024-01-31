import numpy as np
from itertools import combinations_with_replacement

def partition(collection):
	'''
	Helper function to generate set partitions.

		Input:
		- collection: list
			Collection of items to be broken up into partitions

		Output: generator_object
			- Yields first item in collection and smaller subset 
	'''
	if len(collection) == 1:
		yield [collection]
		return

	first = collection[0]
	for smaller in partition(collection[1:]):
		# insert `first` in each of the subpartition's subsets
		for n, subset in enumerate(smaller):
			yield smaller[:n] + [[first] + subset]  + smaller[n+1:]
			# put `first` in its own subset 
			yield [[first]] + smaller

def gen_contractions(length):
	'''
	Generates the set partitions of list containing n integers (1-n)

	Input:
	- n: integer
		Length of set to be partitioned

	Output:
	- partition_list: list
		Partitions formed from the set [i], where 0<i=<n
	'''
	collection = list(range(1, length+1))
	partitions = []
	for n, p in enumerate(partition(collection), 1):
		partitions.append(sorted(p))
	return partitions


def block(image, rows, cols):
	'''Take an image of dimension orig_rows x orig_cols and split into blocks of
	dimension rows x cols. This should work with both Jax and NumPy'''
	# Calculate the dimensions of the image
	# (in the case of MNIST this will be 28,28)
	orig_rows, orig_cols = image.shape
	assert orig_rows % rows == 0, (f'Image row size is not wholly divisible'
								    'by block row size {orig_rows}%{rows}!=0')
	assert orig_cols % cols == 0, (f'Image column size is not wholly divisible'
								    'by block column size {orig_cols}%{rows}!=0')
	return (image.reshape(orig_rows//rows, rows, -1, cols)
               .swapaxes(1,2)
               .reshape(-1, rows, cols))


# TODO: Delete these lines
# def wrapper(correlator):
# 	return perform_contractions(list(partition(correlator)))

# def block_correlator(block, n):
# 	'''Calculates the correlator of a blocked matrix'''
# 	flattened_block = block.flatten()
# 	meshgrid = jnp.meshgrid(*[flattened_block]*n)
# 	correlators = jnp.dstack((meshgrid)).reshape(-1,n)
# 	# return correlators
# 	return jnp.array(vmap(wrapper)(correlators))

def gen_combs(image_dim, k, combs_with_replacement):
	''' Generate indices for all of the combinations with replacement.'''

	# Generate the initial indices
	indices = np.arange(image_dim)

	# Find all combinations

	# Create a generator to save RAM
	combinations_generator = combinations_with_replacement(indices, k)

	# Generator items are tuples. These are no good. Efficiently kreate
	# unpacking generator
	unpacking_generator = (pixel for comb in combinations_generator
					 for pixel in comb)

	# Generate our combination structure. Return an not store to save mem.
	# Float16 is sufficient as pixel precision is overkill (only 256 vals)
	return np.fromiter(unpacking_generator, dtype=np.int64
				 # ,count=self.combs_with_replacement
				 ).reshape(combs_with_replacement, k)

