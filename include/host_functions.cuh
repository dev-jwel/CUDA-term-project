#ifndef _HOST_FUNCTIONS_CUH_
#define _HOST_FUNCTIONS_CUH_

#include "def.cuh"
#include <vector>

__host__
std::vector<Edge> load_bitcoin_otc();
/**
 * 이 함수는 그래프 데이터를 메모리로 읽어들인다.
 */

__host__
void sort_by_dest(const Edge *in, Edge *out, Edge *buffer, size_t edge_size);
/**
 * 이 함수는 주어진 간선리스트 in을 src 기준으로 out에 정렬한다.
 * out과 buffer는 in과 같은 크기로 할당되어야 하나, 초기화될 필요는 없다.
 */

__host__
void stable_sort_by_source(const Edge *in, Edge *out, Edge *buffer, size_t edge_size);
/**
 * 이 함수는 주어진 간선리스트 in을 dest 기준으로 out에 정렬한다.
 * out과 buffer는 in과 같은 크기로 할당되어야 하나, 초기화될 필요는 없다.
 */

__host__
void count_in_degree(const Edge *in, size_t *out, size_t edge_size, size_t node_size);
/**
 * 이 함수는 dst 기준으로 정렬된 간선리스트 in에서 각 노드의 들어오는 차수를 out에 기록한다.
 * in과 out의 길이는 각각 edge_size, node_size이다.
 * out은 초기화될 필요는 없다.
 */

__host__
void count_out_degree(const Edge *in, size_t *out, size_t edge_size, size_t node_size);
/**
 * 이 함수는 src 기준으로 정렬된 간선리스트 in에서 각 노드의 나가는 차수를 out에 기록한다.
 * in과 out의 길이는 각각 edge_size, node_size이다.
 * out은 초기화될 필요는 없다.
 */

__host__
void acc_sum(const size_t *in, size_t *out, size_t node_size);
/**
 * 이 함수는 in의 누적합을 구해 out에 기록한다.
 * out[i] = in[0] + in[1] + ... + in[i]
 * out은 in과 같은 크기로 할당되어야 하나, 초기화될 필요는 없다.
 */

__host__
void element_mul(const size_t *in1, const size_t *in2, size_t *out, size_t node_size);
/**
 * 이 함수는 in1과 in2의 각 원소를 곱한 결과를 out에 기록한다.
 * out[i] = in1[i] * in2[i]
 * out은 in1, in2와 같은 크기로 할당되어야 하나, 초기화될 필요는 없다.
 */

__host__
void reduce_sum(const size_t *in, size_t *out, size_t node_size);
/**
 * 이 함수는 in의 모든 원소의 합은 out[0]에 기록한다.
 * 병렬화를 위하여 out은 in과 동일한 크기일 필요가 있다.
 */


__host__
void count_triangles(
	const Edge *dst_sorted, const Edge *src_sorted,
	const size_t *in_degree, const size_t *out_degree,
	const size_t *accumulated_num_candidates_by_node,
	size_t node_size, size_t edge_size,
	size_t *counter
);
/**
 * 이 함수는 졍렬된 간선 리스트, 차수 리스트, 노드별 후보 삼각형 수의 누적합, 노드수, 간선수를 입력으로 받는다.
 * counter는 각 스레드가 센 삼각형의 개수를 기록하며, 이 커널이 실행된 순간의 스레드 수와 길이가 같아야 한다.
 */

__host__
size_t naive_counter(const std::vector<Edge> edges);
/**
 * 이 함수는 무식하게 CPU만으로 삼각형의 개수를 센다.
 */

#endif // _HOST_FUNCTIONS_CUH_
