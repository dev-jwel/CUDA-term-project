#include "header.cuh"

__global__
void count_in_degree(const Edge *in, Count_t *out, size_t edge_size, size_t node_size) {
	// TODO
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	for(int i=0; i<node_size; i++) {
		if(in[tid].to == i) out[i] += 1;
	}
}

__global__
void count_out_degree(const Edge *in, Count_t *out, size_t edge_size, size_t node_size) {
	// TODO

	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	for(int i=0; i<node_size; i++) {
		if(in[tid].from == i) out[i] += 1;
	}
}

void initial_out(Count_t *out, size_t node_size) {
	for(i=0; i<node_size; i++) {
		out[i] = 0;
	}
}
