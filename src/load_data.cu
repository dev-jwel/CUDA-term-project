#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "header.cuh"

using namespace std;

vector<Edge> load_bitcoin_otc() {
	ifstream ifs("soc-sign-bitcoinotc.csv");
	string line;
	string delim = ",";
	vector<Edge> edge_list;

	while (getline(ifs, line)) {
		string src = line.substr(0, line.find(delim));
		line.erase(0, line.find(delim)+1);
		string dst = line.substr(0, line.find(delim));
		edge_list.push_back({stoi(src), stoi(dst)});
	}

	return edge_list;
}