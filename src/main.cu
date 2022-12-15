#include <iostream>
#include <vector>
#include "def.cuh"
#include "host_functions.cuh"
#include <sys/time.h>

#define edge_size 512 // Edge 개수

using namespace std;

void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time);
float timevalToFloat(struct timeval* time);

int main(int argc, char *argv[]) {
	srand(time(NULL));
	struct timeval htod_start, htod_end;
	struct timeval gpu_start, gpu_end;
	struct timeval dtoh_start, dtoh_end;

	size_t max_node_idx = 0;
	size_t *counter[edge_size]; //edge_size만큼 ..?????
	Edge *dev_dst, *dev_dst_final, *buffer;
	Edge *dev_src, *dev_src_final;
	Edge *host_src_result;
	vector<Edge> edge_list = load_bitcoin_otc();

	gettimeofday(&start, NULL);
	for (auto &edge : edge_list) {
		if (edge.src > max_node_idx) {
			max_node_idx = edge.src;
		}
		if (edge.dst > max_node_idx) {
			max_node_idx = edge.dst;
		}
	}

	//size_t host_result_in_degree[max_node_idx], host_result_out_degree[max_node_idx];
	size_t *dev_result_in_degree, *dev_result_out_degree;
	size_t *dev_result_mul, *dev_result_sum;
	size_t *dev_counter;

	cudaMalloc((void **)&dev_dst, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&dev_src, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&dev_dst_final, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&buffer, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&dev_src_final, sizeof(Edge)*edge_size);
	cudaMalloc((void **)&dev_result_in_degree, sizeof(int)*max_node_idx);
	cudaMalloc((void **)&dev_result_out_degree, sizeof(int)*max_node_idx);

	gettimeofday(&htod_start, NULL);
	cudaMemcpy(dev_dst, edge_list, sizeof(Edge)*edge_size, cudaMemcpyHostToDevice);
	gettimeofday(&htod_end, NULL);
	struct timeval htod_gap;
	getGapTime(&htod_start, &htod_end, &htod_gap);
	float f_htod_gap = timevalToFloat(&htod_gap);
	sort_by_dst(dev_dst, dev_dst_final, buffer, edge_list.size());

	gettimeofday(&gpu_start, NULL);
	cudaMemcpy(dev_src, dev_dst_final, sizeof(Edge)*edge_size, cudaMemcpyDeviceToDevice);
	stable_sort_by_src(dev_src, dev_src_final, buffer, edge_list.size());

	//host_src_result = (Edge*)malloc(sizeof(Edge)*edge_size);
	//cudaMemcpy(host_src_result, dev_result, sizeof(Edge)*edge_size, cudaMemcpyDeviceToHost);
	// 위에까지 sort 함수

	count_in_degree(dev_dst_final, dev_result_in_degree, edge_list.size(), max_node_idx);
	//cudaMemcpy(host_result_in_degree, dev_result_degree, sizeof(host_result_in_degree), cudaMemcpyDeviceToHost);
	
	count_out_degree(dev_src_final, dev_result_out_degree, edge_list.size(), max_node_idx);
	//cudaMemcpy(host_result_out_degree, dev_result_degree, sizeof(host_result_out_degree), cudaMemcpyDeviceToHost);
	//위에까지 count_degree 함수

	cudaMalloc((void **) &dev_result_mul, sizeof(int)*max_node_idx);
	element_mul(dev_result_in_degree, dev_result_out_degree, dev_result_mul, max_node_idx);
	// 각 노드마다 확인해야하는 개수 구하는 부분까지

	
	cudaMalloc((void **) &dev_result_sum, sizeof(int)*max_node_idx);
	acc_sum(dev_result_mul, dev_result_sum, max_node_idx); // 누적합 구하기

	cudaMalloc((void **) &dev_counter, sizeof(counter));
	count_triangles(dev_dst_final, dev_src_final, dev_result_in_degree, dev_result_out_degree, dev_result_sum, max_node_idx, edge_list.size(), dev_counter);
	cudaDeviceSynchronize();

	gettimeofday(&gpu_end, NULL);
	struct timeval gpu_gap;
	getGapTime(&gpu_start, &gpu_end, &gpu_gap);
	float f_gpu_gap = timevalToFloat(&gpu_gap);

	gettimeofday(&dtoh_start, NULL);
	cudaMemcpy(counter, dev_counter, sizeof(counter), cudaMemcpyDeviceToHost);
	gettimeofday(&dtoh_end, NULL);
	struct timeval dtoh_gap;
	getGapTime(&dtoh_start, &dtoh_end, &dtoh_gap);
	float f_dtoh_gap = timevalToFloat(&dtoh_gap);

	float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;

	printf("total time : %.6f\n", total_gap);


	cudaFree(dev_dst);
	cudaFree(dev_src);
	cudaFree(dev_dst_final);
	cudaFree(buffer);
	cudaFree(dev_src_final);
	cudaFree(dev_result_in_degree);
	cudaFree(dev_result_out_degree);
	cudaFree(dev_result_mul);
	cudaFree(dev_result_sum);
	cudaFree(dev_counter);

	cout << "max_node_idx: " << max_node_idx << ", num_edge: " << edge_list.size() << endl;
    cout << "count: " << naive_counter(edge_list) << endl;

	return 0;
}

void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time)
{
	gap_time->tv_sec = end_time->tv_sec - start_time->tv_sec;
	gap_time->tv_usec = end_time->tv_usec - start_time->tv_usec;
	if(gap_time->tv_usec < 0){
		gap_time->tv_usec = gap_time->tv_usec + 1000000;
		gap_time->tv_sec -= 1;
}

}

float timevalToFloat(struct timeval* time){
	double val;
	val = time->tv_sec;
	val += (time->tv_usec * 0.000001);
	return val;
}