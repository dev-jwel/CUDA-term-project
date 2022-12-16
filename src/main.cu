#include <iostream>
#include <vector>
#include "def.cuh"
#include "host_functions.cuh"
#include <sys/time.h>

using namespace std;

void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time);
float timevalToFloat(struct timeval* time);

int main(int argc, char *argv[]) {
	srand(time(NULL));
	struct timeval cpu_start, cpu_end, cpu_gap;
	struct timeval gpu_start, gpu_end, gpu_gap;
	float cpu_time, gpu_time;
	size_t cpu_count, gpu_count;

	vector<Edge> edges = load_bitcoin_otc();

	gettimeofday(&cpu_start, NULL);
	cpu_count = naive_counter(edges);
	gettimeofday(&cpu_end, NULL);

	gettimeofday(&gpu_start, NULL);
	gpu_count = cuda_counter(edges);
	gettimeofday(&gpu_end, NULL);

	getGapTime(&cpu_start, &cpu_end, &cpu_gap);
	cpu_time = timevalToFloat(&cpu_gap);
	getGapTime(&gpu_start, &gpu_end, &gpu_gap);
	gpu_time = timevalToFloat(&gpu_gap);

	cout << "cpu count: " << cpu_count << ", gpu count: " << gpu_count << endl;
	cout << "cpu time: " << cpu_time << ", gpu time:" << gpu_time << endl;

	return 0;
}

void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time) {
	gap_time->tv_sec = end_time->tv_sec - start_time->tv_sec;
	gap_time->tv_usec = end_time->tv_usec - start_time->tv_usec;
	if (gap_time->tv_usec < 0) {
		gap_time->tv_usec = gap_time->tv_usec + 1000000;
		gap_time->tv_sec -= 1;
	}
}

float timevalToFloat(struct timeval* time) {
	double val;
	val = time->tv_sec;
	val += (time->tv_usec * 0.000001);
	return val;
}
