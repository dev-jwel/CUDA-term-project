#ifndef _DEVICE_FUNCTINOS_CUH_
#define _DEVICE_FUNCTINOS_CUH_

#include "def.cuh"

__device__
size_t binary_search(
	const void *array,
	int (*compare) (const void *val1, const void *val2),
	size_t element_size,
	size_t array_size,
	const void *target,
	bool select_left
);
/**
 * 이 함수는 주어진 정렬된 리스트에서 찾고자 하는 값을 찾는다.
 * 중복된 값이 있으면 select_left에 따라 왼쪽 또는 오른쪽 끝의 위치를 반환한다.
 * 검색하고자 하는 값이 없을 때는 인접한 값의 위치를 반환한다.
 */

__device__
int compare_src(const void *edge1, const void *edge2);
/**
 * 이 함수는 src만을 기준을 간선을 비교한다.
 */

__device__
int compare_dst(const void *edge1, const void *edge2);
/**
 * 이 함수는 dst만을 기준을 간선을 비교한다.
 */

__device__
size_t start_src_node_index_of_edge_list(
	const Edge *edges,
	size_t edge_size,
	size_t idx
);
/**
 * 이 함수는 src 기준 정렬된 간선리스트에서 주어진 노드의 시작 인덱스를 리턴한다.
 */

__device__
size_t start_dst_node_index_of_edge_list(
	const Edge *edges,
	size_t edge_size,
	size_t idx
);
/**
 * 이 함수는 dst 기준 정렬된 간선리스트에서 주어진 노드의 시작 인덱스를 리턴한다.
 */

__device__
size_t start_node_of_candidates(
	const size_t *accumulated_num_candidates_by_node,
	size_t node_size,
	size_t tid,
	size_t num_threads
);
/**
 * 이 함수는 노드마다 삼각형의 후보군의 수(들어오는 차수와 나가는 차수의 곱)의 누적합을 입력으로 받는다.
 * 주어진 tid에 대응되는 시작 노드를 찾는다.
 */

__device__
size_t end_node_of_candidates(
	const size_t *accumulated_num_candidates_by_node,
	size_t node_size,
	size_t tid,
	size_t num_threads
);
/**
 * 이 함수는 노드마다 삼각형의 후보군의 수(들어오는 차수와 나가는 차수의 곱)의 누적합을 입력으로 받는다.
 * 주어진 tid에 대응되는 마지막 노드를 찾는다.
 * 마지막 노드의 인덱스에 1이 더해져 있음을 유의하자.
 */

__device__
bool has_pair(const Edge *fully_sorted_edge, Edge edge, size_t edge_size);
/**
 * 이 함수는 주어진 간선이 정렬된 간선리스트에 존재하는지 확인한다.
 */

#endif // _DEVICE_FUNCTINOS_CUH_