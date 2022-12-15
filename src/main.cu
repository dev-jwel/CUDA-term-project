#include <iostream>
#include <vector>
#include "def.cuh"
#include "host_functions.cuh"

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	size_t max_node_idx = 0;
	vector<Edge> edge_list = load_bitcoin_otc();

	for (auto &edge : edge_list) {
		if (edge.src > max_node_idx) {
			max_node_idx = edge.src;
		}
		if (edge.dst > max_node_idx) {
			max_node_idx = edge.dst;
		}
	}

	cout << "max_node_idx: " << max_node_idx << ", num_edge: " << edge_list.size() << endl;
    cout << "count: " << naive_counter(edge_list) << endl;

	return 0;
}
