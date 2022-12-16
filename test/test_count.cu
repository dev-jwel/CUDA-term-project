#include <stdio.h>
#include "def.cuh"
#include "device_functions.cuh"
#include "host_functions.cuh"

int test_count() {
	int ret = 0;
	bool failed;

	Edge host_test_src[10] = {
        {0, 1}, {0, 3}, {0, 4}, {3, 2}, {3, 3}, {5, 9}, {6, 1}, {6, 4}, {9, 1}, {9, 2}
    };
    Edge host_test_dst[10] = {
        {0, 1}, {3, 1}, {0, 2}, {9, 2}, {3, 6}, {1, 6}, {4, 6}, {8, 6}, {9, 8}, {1, 8}
    };
	size_t host_in_degree[10] = {0,2,2,0,0,0,4,0,2,0};
	size_t host_out_degree[10] = {3,0,0,2,0,1,2,0,0,2};
	size_t host_result_degree[10];

    Edge *dev_test_src;
    Edge *dev_test_dst;
	size_t *dev_result_degree;

	cudaMalloc((void **) &dev_test_src, sizeof(host_test_src));
	cudaMalloc((void **) &dev_test_dst, sizeof(host_test_dst));
	cudaMalloc((void **) &dev_result_degree, sizeof(host_result_degree));

    cudaMemcpy(dev_test_src, host_test_src, sizeof(host_test_src), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_test_dst, host_test_dst, sizeof(host_test_dst), cudaMemcpyHostToDevice);

	count_in_degree(dev_test_dst, dev_result_degree, 10, 10);
    cudaMemcpy(host_result_degree, dev_result_degree, sizeof(host_result_degree), cudaMemcpyDeviceToHost);

	failed = false;
	for (int i=0; i<10; i++) {
		if (host_result_degree[i] != host_in_degree[i]) {
			failed = true;
			ret = -1;
		}
	}

	if (failed) {
		printf("failed for count_in_degree\n");
		for (int i=0; i<10; i++) {
			printf("%d ", host_result_degree[i]);
		}
		printf("\n");
	}

	count_out_degree(dev_test_src, dev_result_degree, 10, 10);
    cudaMemcpy(host_result_degree, dev_result_degree, sizeof(host_result_degree), cudaMemcpyDeviceToHost);

	failed = false;
	for (int i=0; i<10; i++) {
		if (host_result_degree[i] != host_out_degree[i]) {
			failed = true;
			ret = -1;
		}
	}

	if (failed) {
		printf("failed for count_out_degree\n");
		for (int i=0; i<10; i++) {
			printf("%d ", host_result_degree[i]);
		}
		printf("\n");
	}

	cudaFree(dev_test_src);
	cudaFree(dev_test_dst);
	cudaFree(dev_result_degree);

	return ret;
}
