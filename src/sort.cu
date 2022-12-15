#include "def.cuh"
#include "device_functions.cuh"

#include <stdio.h>

__inline__ __device__
void merge(
	const Edge *in, Edge *out,
	size_t middle, size_t edge_size,
	bool is_key_src
) {
	size_t left_idx = 0;
	size_t right_idx = middle;
	size_t sorted_idx = 0;
	int result;

	if (edge_size > 0) {
		while(left_idx < middle && right_idx < edge_size) {
			if (is_key_src) {
				result = compare_src(&in[left_idx], &in[right_idx]);
			} else {
				result = compare_dst(&in[left_idx], &in[right_idx]);
			}

			if (result == 1) {
				out[sorted_idx++] = in[right_idx++];
			} else {
				out[sorted_idx++] = in[left_idx++];
			}
		}

		while (left_idx < middle) {
			out[sorted_idx++] = in[left_idx++];
		}
		while (right_idx < edge_size) {
			out[sorted_idx++] = in[right_idx++];
		}
	}
}

__global__
void merge_sort(
	const Edge *in, Edge *out,
	size_t edge_size, size_t block_size,
	bool is_key_src
) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t max_tid = CEIL_DIV(edge_size, block_size);

	if (tid < max_tid) {
		merge(
			&in[tid * block_size],
			&out[tid * block_size],
			block_size/2,
			tid == max_tid-1 ? edge_size - tid * block_size : block_size,
			is_key_src
		);
		/*
		if (tid == max_tid-1) {
			merge(&in[tid * block_size], &out[tid * block_size], block_size/2, edge_size - tid * block_size);
		} else {
			merge(&in[tid * block_size], &out[tid * block_size], block_size/2, block_size);
		}
		*/
	}
}

__host__
void sort_by_dst(const Edge *in, Edge *out, Edge *buffer, size_t edge_size) {
	Edge *temp;
	size_t block_size = 1;

	Edge *debug = (Edge *) malloc(sizeof(Edge) * edge_size);

	cudaMemcpy(buffer, in, sizeof(Edge) * edge_size, cudaMemcpyDeviceToDevice);

	cudaMemcpy(debug, buffer, sizeof(Edge) * edge_size, cudaMemcpyDeviceToHost);


	while (block_size < edge_size) {
		block_size *= 2;
		merge_sort <<<GRID_DIM(CEIL_DIV(edge_size, block_size)), BLOCK_DIM>>> (buffer, out, edge_size, block_size, false);
		temp = out;
		out = buffer;
		buffer = temp;
	}

	free(debug);
	
	cudaMemcpy(out, buffer, sizeof(Edge) * edge_size, cudaMemcpyDeviceToDevice);
}

__host__
void stable_sort_by_src(const Edge *in, Edge *out, Edge *buffer, size_t edge_size) {
	Edge *temp;
	size_t block_size = 1;
	cudaMemcpy(buffer, in, sizeof(Edge) * edge_size, cudaMemcpyDeviceToDevice);

	while (block_size < edge_size) {
		block_size *= 2;
		merge_sort<<<GRID_DIM(edge_size), BLOCK_DIM>>>(buffer, out, edge_size, block_size, true);
		temp = out;
		out = buffer;
		buffer = temp;
	}
	
	cudaMemcpy(out, buffer, sizeof(Edge) * edge_size, cudaMemcpyDeviceToDevice);
}
