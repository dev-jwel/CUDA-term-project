#include <stdio.h>
#include "def.cuh"
#include "device_functions.cuh"

__global__
void _count_triangles(
	const Edge *dst_sorted, const Edge *src_sorted,
	const size_t *in_degree, const size_t *out_degree,
	const size_t *accumulated_num_candidates_by_node,
	size_t node_size, size_t edge_size,
	size_t *counter
) {
	Edge edge;
	size_t temp, num_candidates;
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t num_all_threads = gridDim.x * blockDim.x;
    size_t num_all_candidates = accumulated_num_candidates_by_node[node_size-1];

	counter[tid] = 0;

	if (tid >= num_all_candidates) {
		return;
	}
	if (num_all_threads > num_all_candidates) {
		num_all_threads = num_all_candidates;
	}
	
	// split candidates by tid

	size_t candidate_idx_start = tid * num_all_candidates / num_all_threads;
	size_t candidate_idx_end = (tid+1) * num_all_candidates / num_all_threads;

	// get start and end index by tid

	size_t node_idx_start = start_node_of_candidates(
		accumulated_num_candidates_by_node, node_size, tid, num_all_threads
	);
	size_t node_idx_end = end_node_of_candidates(
		accumulated_num_candidates_by_node, node_size, tid, num_all_threads
	);
	size_t dst_idx = start_dst_node_index_of_edge_list(
		dst_sorted, edge_size, node_idx_start
	);
	size_t src_idx = start_src_node_index_of_edge_list(
		src_sorted, edge_size, node_idx_start
	);


	// handle when node range is 1

	if (node_idx_end - node_idx_start == 1) {
		size_t candidate_offset = candidate_idx_start;
		if (node_idx_start > 0) {
			candidate_offset -= accumulated_num_candidates_by_node[node_idx_start-1];
		}
		temp = candidate_idx_end - accumulated_num_candidates_by_node[node_idx_end-2];

		for (; candidate_offset < temp; ++candidate_offset) {
			size_t dst_offset = temp / out_degree[node_idx_start];
			size_t src_offset = temp % out_degree[node_idx_start];
			edge.src = dst_sorted[dst_idx + dst_offset].src;
			edge.dst = src_sorted[src_idx + src_offset].dst;
			if (has_pair(src_sorted, edge, edge_size)) {
				counter[tid] += 1;
			}
		}
		return;
	}

	// candidates of first node

	temp = candidate_idx_start;
	if (node_idx_start > 0) {
		temp -= accumulated_num_candidates_by_node[node_idx_start-1];
	}
	num_candidates = in_degree[node_idx_start] * out_degree[node_idx_start];

	for (size_t candidate_offset=temp; candidate_offset < num_candidates; ++candidate_offset) {
		size_t dst_offset = temp / out_degree[node_idx_start];
		size_t src_offset = temp % out_degree[node_idx_start];

		edge.src = dst_sorted[dst_idx + dst_offset].src;
		edge.dst = src_sorted[src_idx + src_offset].dst;
		if (has_pair(src_sorted, edge, edge_size)) {
			counter[tid] += 1;
		}
	}

	// candidates of all nodes except first and last one

	for (size_t node_idx=node_idx_start+1; node_idx < node_idx_end; ++node_idx) {
		num_candidates = in_degree[node_idx] * out_degree[node_idx];

		for (size_t candidate_offset=0; candidate_offset < num_candidates; ++candidate_offset) {
			size_t dst_offset = temp / out_degree[node_idx];
			size_t src_offset = temp % out_degree[node_idx];

			edge.src = dst_sorted[dst_idx + dst_offset].src;
			edge.dst = src_sorted[src_idx + src_offset].dst;
			if (has_pair(src_sorted, edge, edge_size)) {
				counter[tid] += 1;
			}
		}
	}

	// candidates of last node

	temp = candidate_idx_end - accumulated_num_candidates_by_node[node_idx_end-2];

	for (size_t candidate_offset=0; candidate_offset < temp; ++candidate_offset) {
		size_t dst_offset = temp / out_degree[node_idx_end];
		size_t src_offset = temp % out_degree[node_idx_end];

		edge.src = dst_sorted[dst_idx + dst_offset].src;
		edge.dst = src_sorted[src_idx + src_offset].dst;
		if (has_pair(src_sorted, edge, edge_size)) {
			counter[tid] += 1;
		}
	}
}

__host__
void count_triangles(
	const Edge *dst_sorted, const Edge *src_sorted,
	const size_t *in_degree, const size_t *out_degree,
	const size_t *accumulated_num_candidates_by_node,
	size_t node_size, size_t edge_size,
	size_t *counter
) {
	_count_triangles <<<COUNTER_GRID_DIM, BLOCK_DIM>>> (
		dst_sorted, src_sorted,
		in_degree, out_degree,
		accumulated_num_candidates_by_node,
		node_size, edge_size,
		counter
	);
}
