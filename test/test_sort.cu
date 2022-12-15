#include <stdio.h>
#include "def.cuh"
#include "host_functions.cuh"

int test_sort() {
	int ret = 0;
	bool failed;

	Edge host_test_input[10] = {
        {5, 5}, {2, 4}, {4, 6}, {4, 2}, {6, 2}, {2, 6}, {6, 6}, {6, 4}, {2, 2}, {4, 4}
    };
    Edge host_test_dst[10] = {
        {4, 2}, {6, 2}, {2, 2}, {2, 4}, {6, 4}, {4, 4}, {5, 5}, {4, 6}, {2, 6}, {6, 6}
    };
    Edge host_test_src[10] = {
        {2, 2}, {2, 4}, {2, 6}, {4, 2}, {4, 4}, {4, 6}, {5, 5}, {6, 2}, {6, 4}, {6, 6}
    };
	Edge host_test_result[10];

    Edge *dev_buffer_1;
    Edge *dev_buffer_2;
    Edge *dev_buffer_3;

	cudaMalloc((void **) &dev_buffer_1, sizeof(Edge) * 10);
	cudaMalloc((void **) &dev_buffer_2, sizeof(Edge) * 10);
	cudaMalloc((void **) &dev_buffer_3, sizeof(Edge) * 10);

    cudaMemcpy(dev_buffer_1, host_test_input, sizeof(host_test_input), cudaMemcpyHostToDevice);

	sort_by_dst(dev_buffer_1, dev_buffer_2, dev_buffer_3, 10);
    cudaMemcpy(host_test_result, dev_buffer_2, sizeof(host_test_result), cudaMemcpyDeviceToHost);

	failed = false;
	for (int i=0; i<10; i++) {
		if (host_test_result[i].src != host_test_dst[i].src || host_test_result[i].dst != host_test_dst[i].dst) {
			failed = true;
			ret = -1;
		}
	}

	if (failed) {
		printf("failed for sort_by_dst\n");
		for (int i=0; i<10; i++) {
			printf("(%lu, %lu) ", host_test_result[i].src, host_test_result[i].dst);
		}
		printf("\n");
		for (int i=0; i<10; i++) {
			printf("(%lu, %lu) ", host_test_dst[i].src, host_test_dst[i].dst);
		}
		printf("\n");
	}

	stable_sort_by_src(dev_buffer_2, dev_buffer_1, dev_buffer_3, 10);
    cudaMemcpy(host_test_result, dev_buffer_1, sizeof(host_test_result), cudaMemcpyDeviceToHost);

	failed = false;
	for (int i=0; i<10; i++) {
		if (host_test_result[i].src != host_test_src[i].src || host_test_result[i].dst != host_test_src[i].dst) {
			failed = true;
			ret = -1;
		}
	}

	if (failed) {
		printf("failed for stable_sort_by_src\n");
		for (int i=0; i<10; i++) {
			printf("(%lu, %lu) ", host_test_result[i].src, host_test_result[i].dst);
		}
		printf("\n");
		for (int i=0; i<10; i++) {
			printf("(%lu, %lu) ", host_test_src[i].src, host_test_src[i].dst);
		}
		printf("\n");
	}

	cudaFree(dev_buffer_1);
	cudaFree(dev_buffer_2);
	cudaFree(dev_buffer_3);

	return ret;
}
