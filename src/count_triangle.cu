#include "header.cuh"

__global__
void count_triangles(
	const Edge *dst_sorted, const Edge *src_sorted,
	const NodeIdx *in_degree, const NodeIdx *out_degree,
	const Count_t *accumulated_num_candidates_by_node,
	size_t node_size, size_t edge_size,
	Count_t *counter
) {
	Edge edge;
	size_t temp, num_candidates;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// split candidates by tid

	size_t candidate_idx_start = tid * num_candidates_by_node / (gridDim.x * blockDim.x);
	size_t candidate_idx_end = (tid+1) * num_candidates_by_node / (gridDim.x * blockDim.x);

	// get start and end index by tid

	NodeIdx node_idx_start = start_node_of_candidates(
		accumulated_num_candidates_by_node, node_size, tid, num_threads
	);
	NodeIdx node_idx_end = end_node_of_candidates(
		accumulated_num_candidates_by_node, node_size, tid, num_threads
	);
	size_t dst_idx = start_dst_node_index_of_edge_list(
		dst_sorted, edge_size, node_idx_start
	);
	size_t src_idx = start_src_node_index_of_edge_list(
		src_sorted, edge_size, node_idx_start
	);

	// candidates of first node

	temp = candidate_idx_start - accumulated_num_candidates_by_node[node_idx_start];
	num_candidates = in_degree[node_idx_start] * out_degree[node_idx_start];

	for (size_t candidate_offset=temp; candidate_offset < num_candidates; ++candidate_offset) {
		size_t dst_offset = temp / out_degree[node_idx_start];
		size_t src_offset = temp % out_degree[node_idx_start];

		edge.from = dst_sorted[dst_idx + dst_offset]->from;
		edge.to = src_sorted[src_idx + src_offset]->to;
		if (has_pair(src_sorted, edge, edge_size)) {
			counter[tid] += 1;
		}
	}

	// candidates of all nodes except first and last one

	for (size_t node_idx=node_idx_start-1; node_idx < node_idx_end; ++node_idx) {
		num_candidates = in_degree[node_idx] * out_degree[node_idx];

		for (size_t candidate_offset=0; candidate_offset < num_candidates; ++candidate_offset) {
			size_t dst_offset = temp / out_degree[node_idx];
			size_t src_offset = temp % out_degree[node_idx];

			edge.from = dst_sorted[dst_idx + dst_offset]->from;
			edge.to = src_sorted[src_idx + src_offset]->to;
			if (has_pair(src_sorted, edge, edge_size)) {
				counter[tid] += 1;
			}
		}
	}

	// candidates of last node

	temp = accumulated_num_candidates_by_node[node_idx_end-1] - candidate_idx_end;

	for (size_t candidate_offset=0; candidate_offset < temp; ++candidate_offset) {
		size_t dst_offset = temp / out_degree[node_idx_end];
		size_t src_offset = temp % out_degree[node_idx_end];

		edge.from = dst_sorted[dst_idx + dst_offset]->from;
		edge.to = src_sorted[src_idx + src_offset]->to;
		if (has_pair(src_sorted, edge, edge_size)) {
			counter[tid] += 1;
		}
	}

}

__inline__ __device__
void count_triangles_in_single_node(
	const Edge *dst_sorted, const Edge *src_sorted,
	size_t dst_start, size_t dst_end,
	size_t src_start, size_t src_end,
	NodeIdx in_degree, NodeIdx out_degree,
	NodeIdx node_idx, size_t edge_size,
	Count_t *counter
) {
	Edge edge;
	for (size_t dst_idx=dst_start; dst_idx < in_degree[node_idx]; ++dst_idx) {
		for (size_t src_idx=src_start; src_idx < out_degree[node_idx]; ++src_idx) {
			edge.from = dst_sorted[dst_idx]->from;
			edge.to = src_sorted[src_idx]->to;
			if (has_pair(src_sorted, edge, edge_size)) {
				counter[tid] += 1;
			}
		}
	}
}