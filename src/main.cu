#include <stdio.h>
#include "header.cuh"

int main(int argc, char *argv[]) {
	printf("hello\n");

	struct Edge *dev_c;
    struct Edge dev;

    cudaMalloc((void**)&dev_c, sizeof(Edge)*N);
    sort_by_dest<<< >>>(dev_c, N); // <<< >>> 안에 뭐 넣어야 하는지 모르겠음
    cudaMemcpy(&dev, dev_c, sizeof(Edge)*N, cudaMemcpyDeviceToHost); // dst에 대해 정렬

    cudaMalloc((void**)&dev_c, sizeof(Edge)*N);
    stable_sort_by_source<<< >>>(dev_c, N); // <<< >>> 안에 뭐 넣어야 하는지 모르겠음
    cudaMemcpy(&dev, dev_c, sizeof(Edge)*N, cudaMemcpyDeviceToHost); // src에 대해 정렬
	return 0;
}
