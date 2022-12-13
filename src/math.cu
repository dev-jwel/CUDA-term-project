#include "header.cuh"

__global__
void acc_sum(const Count_t *in, Count_t *out, size_t node_size) {
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

__global__
void element_mul(const Count_t *in1, const Count_t *in2, Count_t *out, size_t node_size) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < node_size) {
		out[tid] = in1[tid] * in2[tid];
	}
}

__global__
void sum(const Count_t *in, Count_t *out, size_t node_size) {
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
