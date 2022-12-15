#include <iostream>
#include <vector>
#include "header.cuh"

using namespace std;

int main(int argc, char *argv[]) {
	srand(time(NULL));

	size_t max_node_idx = 0;
	vector<Edge> edge_list = load_bitcoin_otc();

	for (auto &edge : edge_list) {
		if (edge.from > max_node_idx) {
			max_node_idx = edge.from;
		}
		if (edge.to > max_node_idx) {
			max_node_idx = edge.to;
		}
	}

	cout << "max_node_idx: " << max_node_idx << ", num_edge: " << edge_list.size() << endl;

	return 0;
}
