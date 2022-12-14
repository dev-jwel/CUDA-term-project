#include "header.cuh"

__global__
void sort_by_dest(const Edge *edge, size_t edge_size) {
	// TODO
	int temp = 0;
	size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

	for(int i=0; i<edge_size; i++) {
		if (i%2 == 0) {
			if((in[tid].to > in[tid+1].to) && (tid < edge_size) && (tid %2 == 0)) {
				temp = in[tid+1];
				in[tid+1] = in[tid];
				in[tid] = temp;
			} 
		}

		else {
			if((in[tid].to > in[tid+1].to) && (tid < edge_size) && (tid%2 !=0)) {
				temp = in[tid+1];
				in[tid+1] = in[tid];
				in[tid] = temp;
			}
		}
	}
	out[tid] = in[tid];

	/*for(int i=0; i<edge_size; i++) {
		if(in[tid].to > in[tid+1].to) {
			temp = in[tid];
			in[tid] = in[tid + 1];
			in[tid] = temp;
			out[tid] = in[tid];
		}
	}*/

	/*for(int i=0; i<edge_size; i++) {
		for(int j=0; j<edge_size -i; j++) {
			if(edge[j].to > edge[j+1].to) {
				temp = edge[j];
				edge[j] = edge[j+1];
				edge[j+1] = temp;
			}
		}
	}*/
}

__global__
void stable_sort_by_source(const Edge *in, Edge *out, size_t edge_size) {
	// TODO
	int temp = 0;
	size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

	for(int i=0; i<edge_size; i++) {
		if (i%2 == 0) {
			if((in[tid].from > in[tid+1].from) && (tid < edge_size) && (tid %2 == 0)) {
				temp = in[tid+1];
				in[tid+1] = in[tid];
				in[tid] = temp;
			} 
		}

		else {
			if((in[tid].from > in[tid+1].from) && (tid < edge_size) && (tid%2 !=0)) {
				temp = in[tid+1];
				in[tid+1] = in[tid];
				in[tid] = temp;
			}
		}
	}
	out[tid] = in[tid];
	
	/*for(int i=0; i<edge_size; i++) {
		if(in[tid].from > in[tid+1].from) {
			temp = in[tid];
			in[tid] = in[tid+1];
			in[tid+1] = temp;
			out[tid] = in[tid];
		}
	}
	out[tid] = in[tid];*/

	/*for(int i=0; i<edge_size; i++) {
		for(int j=0; j<edge_size-i; j++) {
			if(edge.from[j] > edge.from[j+1]) {
				temp = edge[j];
				edge[j] = edge[j+1];
				edge[j+1] = temp;
			}
		}
	}*/
}
