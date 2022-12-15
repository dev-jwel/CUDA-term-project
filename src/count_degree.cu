#include "def.cuh"
#include "device_functions.cuh"

__global__
void _count_in_degree(const Edge *in, size_t *out, size_t edge_size, size_t node_size) {
	size_t node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx >= node_size) {
        return;
    }

    Edge target = {0, node_idx};
	size_t edge_idx = binary_search(
        in, compare_dst, sizeof(Edge), edge_size, (void *) &target, false // right most result
    );
	

    if (in[edge_idx-1].dst != node_idx) {
        out[node_idx] = 0;
    } else {
        out[node_idx] = edge_idx - binary_search(
            in, compare_dst, sizeof(Edge), edge_size, (void *) &target, true // left most result
        );
    }
}

__host__
void count_in_degree(const Edge *in, size_t *out, size_t edge_size, size_t node_size) {
	_count_in_degree <<<GRID_DIM(node_size), BLOCK_DIM>>> (in, out, edge_size, node_size);
}

__global__
void _count_out_degree(const Edge *in, size_t *out, size_t edge_size, size_t node_size) {
	size_t node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx >= node_size) {
        return;
    }

    Edge target = {node_idx, 0};
	size_t edge_idx = binary_search(
        in, compare_src, sizeof(Edge), edge_size, (void *) &target, false // right most result
    );

    if (in[edge_idx-1].src != node_idx) {
        out[node_idx] = 0;
    } else {
        out[node_idx] = edge_idx - binary_search(
            in, compare_src, sizeof(Edge), edge_size, (void *) &target, true // left most result
        );
    }
}

__host__
void count_out_degree(const Edge *in, size_t *out, size_t edge_size, size_t node_size) {
	_count_out_degree <<<GRID_DIM(node_size), BLOCK_DIM>>> (in, out, edge_size, node_size);
}
