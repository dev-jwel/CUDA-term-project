#include "header.cuh"

__global__
void count_triangles(
	const Edge *dst_sorted, const Edge *src_sorted,
	const NodeIdx *in_degree, const NodeIdx *out_degree,
	const NodeIdx *num_candidates_by_node,
	size_t node_size, size_t edge_size,
	NodeIdx *buffer
) {
	// TODO
}
