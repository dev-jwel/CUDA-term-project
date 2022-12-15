#include "header.cuh"

__inline__ __device__ size_t binary_search(
	const void *array,
	int (*compare) (const void *val1, const void *val2),
	size_t element_size,
	size_t array_size,
	const void *target,
	bool select_left
) {
	size_t min = 0;
	size_t max = array_size;
	size_t mid;

	while (min < max) {
		mid = (min + max) / 2;
		switch (compare(array + element_size * mid, target)) {
			case -1: min = mid + 1; break;
			case  1: max = mid; break;
			case  0:
				if (select_left) {
					max = mid;
				} else {
					min = mid + 1;
				}
			break;
		}
	}

	return min;
}

__inline__ __device__
int compare_src(const void *edge1, const void *edge2) {
	if (((Edge *)edge1)->from < ((Edge *)edge2)->from) return -1;
	if (((Edge *)edge1)->from > ((Edge *)edge2)->from) return 1;
	return 0;
}

__inline__ __device__
int compare_dst(const void *edge1, const void *edge2) {
	if (((Edge *)edge1)->to < ((Edge *)edge2)->to) return -1;
	if (((Edge *)edge1)->to > ((Edge *)edge2)->to) return 1;
	return 0;
}

__inline__ __device__
int compare_edge(const void *edge1, const void edge2) {
	if (((Edge *)edge1)->from < ((Edge *)edge2)->from) return -1;
	if (((Edge *)edge1)->from > ((Edge *)edge2)->from) return 1;
	if (((Edge *)edge1)->to < ((Edge *)edge2)->to) return -1;
	if (((Edge *)edge1)->to > ((Edge *)edge2)->to) return 1;
	return 0;
}

__inline__ __device__
int compare_count(const void *cnt1, const void cnt2) {
	if (cnt1 < cnt2) {
		return -1;
	} else if (cnt1 > cnt2) {
		return 1;
	} else {
		return 0;
	}
}

__inline__ __device__
size_t start_src_node_index_of_edge_list(
	const Edge *edges,
	size_t edge_size,
	NodeIdx idx
) {
	Edge target = {idx, 0};
	return binary_search(edges, compare_src, sizeof(Edge), edge_size, (void *) &target, true);
}


__inline__ __device__
size_t start_dst_node_index_of_edge_list(
	const Edge *edges,
	size_t edge_size,
	NodeIdx idx
) {
	Edge target = {0, idx};
	return binary_search(edges, compare_dst, sizeof(Edge), edge_size, (void *) &target, true);
}

__inline__ __device__
NodeIdx start_node_of_candidates(
	const Count_t *accumulated_num_candidates_by_node,
	size_t node_size,
	size_t tid,
	size_t num_threads
) {
	Count_t num_all_candidates = accumulated_num_candidates_by_node[node_size-1];
	Count_t target = tid * num_all_candidates / num_threads;

	return binary_search(
		accumulated_num_candidates_by_node, compare_count, sizeof(Count_t), node_size, (void *) &target, true
	);
}

__inline__ __device__
NodeIdx end_node_of_candidates(
	const Count_t *accumulated_num_candidates_by_node,
	size_t node_size,
	size_t tid,
	size_t num_threads
) {
	Count_t num_all_candidates = accumulated_num_candidates_by_node[node_size-1];
	Count_t target = (tid+1) * num_all_candidates / num_threads;

	return binary_search(
		accumulated_num_candidates_by_node, compare_count, sizeof(Count_t), node_size, (void *) &target, false
	);
}

__inline__ __device__
bool has_pair(const Edge *fully_sorted_edge, Edge edge, size_t edge_size) {
	Edge target = edge;
	size_t idx = binary_search(fully_sorted_edge, compare_edge, sizeof(Edge), edge_size, (void *) &target, true);
	return compare_edge(fully_sorted_edge[idx], target) == 0;
}
