#include <stdio.h>
#include "header.cuh"

int test_math() {
	int i, ret = 0;

	Count_t host_test_1[10] = {1,2,3,4,5,6,7,8,9,10};
	Count_t host_test_2[10] = {2,3,4,5,6,7,8,9,10,11};
	Count_t host_acc_sum[10] = {1,3,6,10,15,21,28,36,45,55};
	Count_t host_mul[10] = {2,6,12,20,30,42,56,72,90,110};
	Count_t host_result[10];

	Count_t *dev_test_1, *dev_test_2, *dev_result;

	cudaMalloc((void **) &dev_test_1, sizeof(host_test_1));
	cudaMalloc((void **) &dev_test_2, sizeof(host_test_2));
	cudaMalloc((void **) &dev_result, sizeof(host_result));

	cudaMemcpy(dev_test_1, host_test_1, sizeof(host_test_1), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_test_2, host_test_2, sizeof(host_test_2), cudaMemcpyHostToDevice);

	acc_sum <<<1, BLOCK_DIM>>> (dev_test_1, dev_result, 10);
	cudaMemcpy(host_result, dev_result, sizeof(host_result), cudaMemcpyDeviceToHost);
	for (i=0; i<10; ++i) {
		if (host_result[i] != host_acc_sum[i]) {
			printf("error on accsum\n");
			ret = -1;
			break;
		}
	}

	element_mul <<<1, BLOCK_DIM>>> (dev_test_1, dev_test_2, dev_result, 10);
	cudaMemcpy(host_result, dev_result, sizeof(host_result), cudaMemcpyDeviceToHost);
	for (i=0; i<10; ++i) {
		if (host_result[i] != host_mul[i]) {
			printf("error on element_mul\n");
			ret = -1;
			break;
		}
	}

	reduce_sum <<<1, BLOCK_DIM>>> (dev_test_1, dev_result, 10);
	cudaMemcpy(host_result, dev_result, sizeof(host_result), cudaMemcpyDeviceToHost);
	if (host_result[0] != 55) {
		printf("error on sum\n");
		ret = -1;
	}

	cudaFree(dev_test_1);
	cudaFree(dev_test_2);
	cudaFree(dev_result);

	return ret;
}