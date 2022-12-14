#include <stdio.h>
#include <math.h>
#include "header.cuh"

#define edge_size 512 // Edge 개수

int main(int argc, char *argv[]) {
	srand(time(NULL));
	printf("hello\n");
	double block_size, edge_size, grid_size, node_size;
	struct Edge *dev_in;
	struct Edge *dev_out;
	struct Edge *dev_out2;
	
	block_size = BLOCK_SIZE;
	grid_size = ceil(edge_size / block_size);

	struct Edge *in = (struct Edge *)malloc(sizeof(struct Edge)*edge_size);
	struct Edge *out = (struct Edge *)malloc(sizeof(struct Edge)*edge_size);
	struct Edge *out2 = (struct Edge *)malloc(sizeof(struct Edge)*edge_size);
	Count_t *c_in_out = (Count_t *)malloc(node_size * sizeof(Count_t));
	Count_t *c_out_out = (Count_t *)malloc(node_size * sizeof(Count_t)); 

	cudaMalloc((void**)&dev_in, sizeof(Edge)*edge_size);
	cudaMalloc((void**)&dev_out, sizeof(Edge)*edge_size);
	cudaMalloc((void**)&dev_out2, sizeof(Edge)*edge_size);
	cudaMalloc((void**)&dev_c_in_out, sizeof(Count_t)*node_size);
	cudaMalloc((void**)&dev_c_out_out, sizeof(Count_t)*node_size);

	dim3 Dg(grid_size, 1, 1);
	dim3 Db(block_size, 1, 1)
	for(int i=0; i<edge_size; i++) {
		sort_by_dest<<<Dg, Db>>>(dev_in, dev_out, edge_size);
	}
	cudaMemcpy(&out, dev_out, sizeof(Edge)*edge_size, cudaMemcpyDeviceToHost); // dst에 대해 정렬된 값 src로

	for(int j=0; j<edge_size; j++) {
		stable_sort_by_source<<<Dg, Db>>>(dev_out, dev_out2, edge_size);
	}

	cudaMemcpy(&out2, dev_out2, sizeof(Edge)*edge_size, cudaMemcpyDeviceToHost); // src에 대해 정렬된 값 host로

	initial_out(&cout, node_size); // cout 초기화 과정

	count_in_degree<<Dg, Db>>>(dev_out, dev_c_in_out, edge_size, node_size);
	count_out_degree<<<Dg, Db>>>(dev_out2, dev_c_out_out, edge_size, node_size);

	cudaMemcpy(&c_in_out, dev_c_in_out, sizeof(Count_t)*node_size, cudaMemcpyDeviceToHost); // dst 기준으로 정렬된 outgoing 차수 device에서 host로
	cudaMemcpy(&c_out_out, dev_c_out_out, sizeof(Count_t)*node_size, cudaMemcpyDeviceToHost); // src 기준으로 정렬된 incoming 차수 device에서 host로
	//cudaFree(dev_out);




	cudaDeviceSynchronize(); //device가 작업 완료할 때까지 host는 대기.
	return 0;
}
