#include "def.cuh"

#include <iostream>
#include <stdlib.h>

__global__
void _acc_sum(const size_t *in, size_t *out, size_t diff, size_t node_size) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (diff <= tid && tid < node_size) {
        out[tid] += out[tid-diff];
	}
}

__host__
void acc_sum(const size_t *in, size_t *out, size_t node_size) {
    cudaMemcpy(out, in, sizeof(size_t)*node_size, cudaMemcpyDeviceToDevice);

    // scan algorithm

    size_t diff = 1;
    while (diff < node_size) {
        _acc_sum <<<GRID_DIM(node_size), BLOCK_DIM>>> (in, out, diff, node_size);
        diff *= 2;
    }
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
void _reduce_sum(const size_t *in, size_t *out, size_t node_size, bool add_last_element) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t buffer_size = node_size;

	if (tid < node_size) {
        if (tid < node_size-1 || add_last_element) {
            out[tid] += out[tid + node_size];
        }
	}
}

__host__
void reduce_sum(const size_t *in, size_t *out, size_t node_size) {
    cudaMemcpy(out, in, sizeof(size_t)*node_size, cudaMemcpyDeviceToDevice);

    while (node_size > 1) {
        bool add_last_element = node_size % 2 == 0;
        node_size = node_size / 2 + node_size % 2;
        _reduce_sum <<<GRID_DIM(node_size), BLOCK_DIM>>> (in, out, node_size, add_last_element);
    }
}