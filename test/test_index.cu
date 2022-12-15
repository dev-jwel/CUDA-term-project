#include <stdio.h>
#include "header.cuh"

__device__
int compare_int(const void *int1, const void *int2) {
	if (int1 < int2) return -1;
	if (int1 > int2) return 1;
	return 0;
}

__global__
void test_binary_search(
    int *array, size_t size, int _target, size_t left, size_t right, int *result
) {
    int target = _target;
    *result = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int _left = binary_search(
            array, compare_int, sizeof(int), size, &target, true
        );

        if (left != _left) {
            *result += 1;
        }

        int _right = binary_search(
            array, compare_int, sizeof(int), size, &target, false
        );

        if (_right != _right) {
            *result += 2;
        }
    }
}

int test_index() {
	int i, ret = 0;

	int host_test_int[10] = {1,1,4,4,4,6,6,6,9,9};
	Edge host_test_src[10] = {
        {0, 1}, {0, 3}, {0, 2}, {3, 2}, {3, 3}, {6, 9}, {6, 1}, {6, 0}, {9, 1}, {9, 2}
    };
    Edge host_test_dst[10] = {
        {0, 1}, {3, 1}, {0, 2}, {9, 2}, {3, 6}, {1, 6}, {4, 6}, {8, 6}, {9, 8}, {1, 8}
    };
	int host_result_int[1];

	int *dev_test_int;
    Edge *dev_test_src;
    Edge *dev_test_dst;
    int *dev_result_int;

	cudaMalloc((void **) &dev_test_int, sizeof(host_test_int));
	cudaMalloc((void **) &dev_test_src, sizeof(host_test_src));
	cudaMalloc((void **) &dev_test_dst, sizeof(host_test_dst));
	cudaMalloc((void **) &dev_result_int, sizeof(host_result_int));

	cudaMemcpy(dev_test_int, host_test_int, sizeof(host_test_int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_test_src, host_test_src, sizeof(host_test_src), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_test_dst, host_test_dst, sizeof(host_test_dst), cudaMemcpyHostToDevice);

	test_binary_search <<<1, 1>>> (dev_test_int, 10, 6, 5, 7, dev_result_int);
	cudaMemcpy(host_result_int, dev_result_int, sizeof(host_result_int), cudaMemcpyDeviceToHost);
	if ((*host_result_int) & 1) {
        printf("error on left element selection\n");
        ret = -1;
    }
    if ((*host_result_int) & 2) {
        printf("error on right element selection\n");
        ret = -1;
    }

    printf("index test done\n");

	cudaFree(dev_test_int);
	cudaFree(dev_test_src);
	cudaFree(dev_test_dst);
    cudaFree(dev_result_int);

	return ret;
}