#include "header.cuh"

__inline__ __device__
void merge(
	const Edge *in, Edge *out,
	int (*compare)(const void *edge1, const void *edge2),
	size_t edge_size
) {
	size_t middle = edge_size / 2;
	size_t left_idx = 0;
	size_t right_idx = middle;
	size_t sorted_idx = 0;
	
	if (edge_size > 0) {
		while(left_idx < middle && right_idx < edge_size) {
			if (compare(&in[left_idx], &in[right_idx]) == 1) {
				out[sorted_idx++] = in[right_idx++];
			} else {
				out[sorted_idx++] = in[left_idx++];
			}
		}

		while (left_idx < middle) {
			out[sorted_idx++] = in[left_idx++];
		}
		while (right_idx < edge_size) {
			out[sorted_idx++] = in[left_idx++];
		}
	}
}

__global__
void merge_sort(
	const Edge *in, Edge *out,
	int (*compare)(const void *edge1, const void *edge2),
	size_t edge_size, size_t block_size
) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	size_t max_tid = edge_size / block_size;
	if (edge_size % block_size > 0) {
		max_tid += 1;
	}

	if (tid < max_tid) {
		if (tid == max_tid-1) {
			merge(&in[tid * block_size], &out[tid * block_size], compare, edge_size - tid * block_size);
		} else {
			merge(&in[tid * block_size], &out[tid * block_size], compare, block_size);
		}
	}
}

__host__
void sort_by_dst(const Edge *in, Edge *out, Edge *buffer, size_t edge_size) {
	Edge *temp;
	size_t block_size = 1;
	cudaMemcpy(buffer, in, sizeof(Edge) * edge_size, cudaMemcpyDeviceToDevice);

	while (block_size < edge_size) {
		block_size *= 2;

		size_t GRID_DIM = edge_size / BLOCK_DIM;
		if (edge_size % BLOCK_DIM > 0) {
			GRID_DIM += 1;
		}

		merge_sort <<<GRID_DIM, BLOCK_DIM>>> (buffer, out, compare_dst, edge_size, block_size);

		temp = out;
		out = buffer;
		buffer = temp;
	}
	
	cudaMemcpy(out, buffer, sizeof(Edge) * edge_size, cudaMemcpyDeviceToDevice);
}

__host__
void stable_sort_by_src(const Edge *in, Edge *out, Edge *buffer, size_t edge_size) {
	Edge *temp;
	size_t block_size = 1;
	cudaMemcpy(buffer, in, sizeof(Edge) * edge_size, cudaMemcpyDeviceToDevice);

	while (block_size < edge_size) {
		block_size *= 2;

		size_t GRID_DIM = edge_size / BLOCK_DIM;
		if (edge_size % BLOCK_DIM > 0) {
			GRID_DIM += 1;
		}

		merge_sort<<<GRID_DIM, BLOCK_DIM>>>(buffer, out, compare_src, edge_size, block_size);

		temp = out;
		out = buffer;
		buffer = temp;
	}
	
	cudaMemcpy(out, buffer, sizeof(Edge) * edge_size, cudaMemcpyDeviceToDevice);
}
