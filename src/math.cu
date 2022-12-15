#include "def.cuh"

__global__
void _acc_sum(const size_t *in, size_t *out, size_t node_size) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t depth = 1;
	if (tid < node_size) {
		// copy data
		out[tid] = in[tid];

		// scan algorithm
		while (depth < node_size) {
			if (tid >= 1) {
				out[tid] += out[tid-depth];
			}
			depth *= 2;
		}
	}
}

__host__
void acc_sum(const size_t *in, size_t *out, size_t node_size) {
    _acc_sum <<<GRID_DIM(node_size), BLOCK_DIM>>> (in, out, node_size);
}

__global__
void _element_mul(const size_t *in1, const size_t *in2, size_t *out, size_t node_size) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < node_size) {
		out[tid] = in1[tid] * in2[tid];
	}
}

__host__
void element_mul(const size_t *in1, const size_t *in2, size_t *out, size_t node_size) {
    _element_mul <<<GRID_DIM(node_size), BLOCK_DIM>>> (in1, in2, out, node_size);
}

__global__
void _reduce_sum(const size_t *in, size_t *out, size_t node_size) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t buffer_size = node_size;

	if (tid < node_size) {
		// copy data
		out[tid] = in[tid];

		// sum all data
		while (buffer_size > 1) {
			if (tid < buffer_size/2) {
				out[tid] += out[2*tid + buffer_size%2];
			}
			buffer_size += buffer_size % 2;
			buffer_size /= 2;
		}
	}
}

__host__
void reduce_sum(const size_t *in, size_t *out, size_t node_size) {
    _reduce_sum <<<GRID_DIM(node_size), BLOCK_DIM>>> (in, out, node_size);
}