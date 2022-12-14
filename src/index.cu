#include "header.cuh"

__inline__ __device__
NodeIdx start_node_of_tid(
	const Count_t *accumulated_num_candidates_by_node,
	size_t node_size,
	size_t tid,
	size_t num_threads
) {
	Count_t num_all_candidates = accumulated_num_candidates_by_node[node_size-1];
	Count_t target = tid * num_all_candidates / num_threads;
	NodeIdx min = 0;
	NodeIdx max = node_size;
	NodeIdx mid;

	while (max - min > 1) {
		mid = (min + max) / 2;
		if (accumulated_num_candidates_by_node[mid] > target) {
			max = mid;
		} else {
			min = mid;
		}
	}

	return min;
}

__inline__ __device__
NodeIdx end_node_of_tid(
	const Count_t *accumulated_num_candidates_by_node,
	size_t node_size,
	size_t tid,
	size_t num_threads
) {
	Count_t num_all_candidates = accumulated_num_candidates_by_node[node_size-1];
	Count_t target = tid * num_all_candidates / num_threads;
	NodeIdx min = 0;
	NodeIdx max = node_size;
	NodeIdx mid;

	while (max - min > 1) {
		mid = (min + max) / 2;
		if (accumulated_num_candidates_by_node[mid] < target) {
			min = mid;
		} else {
			max = mid;
		}
	}

	return max;
}

__inline__ __device__
int compair_edge(Edge edge1, Edge edge2) {
	if (edge1.from < edge2.from) return -1;
	if (edge1.from > edge2.from) return 1;
	if (edge1.to < edge2.to) return -1;
	if (edge1.to > edge2.to) return 1;
	return 0;
}

__inline__ __device__
bool has_pair(const Edge *fully_sorted_edge, Edge edge, size_t edge_size) {
	// TODO
	size_t min = 0;
	size_t max = edge_size;
	size_t mid;

	while (max > min) {
		mid = (min + max) / 2;
		switch (compair_edge(fully_sorted_edge[mid], edge)) {
			case -1: min = mid; break;
			case 1:  max = mid; break;
			case 0:  return true;
		}
	}

	return false;
}
