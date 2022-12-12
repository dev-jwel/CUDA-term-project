#ifndef _HEADER_CUH_
#define _HEADER_CUH_

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

typedef unsigned int NodeIdx;
typedef unsigned int Count_t;

typedef struct {
	NodeIdx from;
	NodeIdx to;
} Edge;

__global__
void sort_by_dest(const Edge *edge, size_t edge_size);
__global__
void stable_sort_by_source(const Edge *edge, size_t edge_size);

__global__
void count_in_degree(const Edge *in, Count_t *out, size_t edge_size, size_t node_size);
__global__
void count_out_degree(const Edge *in, Count_t *out, size_t edge_size, size_t node_size);

__global__
void acc_sum(const Count_t *in, Count_t *out, size_t node_size);
__global__
void element_mul(const Count_t *in1, const Count_t *in2, Count_t *out, size_t node_size);
__global__
void sum(const Count_t *in, Count_t *out, size_t node_size);

__inline__ __device__
NodeIdx start_node_of_tid(const Count_t *num_candidates_by_node, size_t tid);
__inline__ __device__
NodeIdx end_node_of_tid(const Count_t *num_candidates_by_node, size_t tid);
__inline__ __device__
bool has_pair(const Edge *fully_sorted_edge, Edge edge);

__global__
void count_triangles(
	const Edge *dst_sorted, const Edge *src_sorted,
	const NodeIdx *in_degree, const NodeIdx *out_degree,
	const NodeIdx *num_candidates_by_node,
	size_t node_size, size_t edge_size,
	NodeIdx *buffer
);

int naive_counter(const Edge *edges);

#endif // _HEADER_CUH_
