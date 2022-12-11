#include <bits/stdc++.h>
using namespace std;

__global__
void findInOutDegree(vector<vector<int>> adjlist,
                     int n)
{   vector<int> iN(n,0);
    vector<int> ouT(n,0);

    for(int i=0;i<n;i++)
    {
        // Out degree 계산
       ouT[i] = adjlist[i].size();
           for(int j=0;j<adjlist[i].size();j++)
          iN[adjlist[i][j]]++;
     }

    cout << "Vertex\t\tIn\t\tOut" << endl;
    for(int k = 0; k < n; k++)
    {cout << k << "\t\t"
             << iN[k] << "\t\t"
             << ouT[k] << endl;
    }
}
