#include "header.cuh"

__global__
void sort_by_dest(const Edge *edge, size_t edge_size) {
	// TODO
	int temp = 0;
    for(int i=0; i<edge_size; i++) {
        for(int j=0; j<edge_size -i; j++) {
            if(edge[j].to > edge[j+1].to) {
                temp = edge[j];
                edge[j] = edge[j+1];
                edge[j+1] = temp;
            }
        }
    }
}

__global__
void stable_sort_by_source(const Edge *edge, size_t edge_size) {
	// TODO
	int temp = 0;
	for(int i=0; i<edge_size; i++) {
		for(int j=0; j<edge_size-i; j++) {
			if(edge.from[j] > edge.from[j+1]) {
				temp = edge[j];
				edge[j] = edge[j+1];
				edge[j+1] = temp;
			}
		}
	}
}
