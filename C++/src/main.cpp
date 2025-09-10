#include <iostream>
#include "../include/utilities.h"
#include "../include/seqKmeans.h"
#include "../include/parKmeans.h"
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <gperftools/profiler.h>
#include <cmath>

int main() {
    int nc = 10000000;
    int c = 3;
    int dim = 3;
    double start, end, start1, end1;

    size_t testSize = 11;
    std::vector<double> speedUps(testSize);

    for (size_t i = 0; i < testSize; i++) {
        auto dataset = SOAdataGenerator(nc, c, dim);
        auto centers = SOAcentersGenerator(c, dataset);
        auto sCenters = centers;

        std::cout<<i <<"\n";
        omp_set_num_threads(8);
        //std::cout <<"\n"<< "PARALLEL SOA Kmeans\n";
        start1 = omp_get_wtime();
        auto pClusters = SOApKmeans(dataset, centers);
        end1 = omp_get_wtime();
        double timePar = (end1 - start1) * 1e9;
        //std::cout <<"time PARALLEL: "<< timePar << " ns\n";

        //std::cout <<"\n"<< "SEQUENTIAL SOA Kmeans\n";
        start = omp_get_wtime();
        auto sClusters = SOAsKmeans(dataset, sCenters);
        end = omp_get_wtime();
        double timeSeq = (end - start) * 1e9;
        //std::cout <<"time SEQUENTIAL: "<< timeSeq << " ns\n";

        double speedUp = timeSeq / timePar;
        speedUps[i] = speedUp;
        //std::cout << "\n" <<"SPEEDUP: "<< speedUp << "\n";
    }
    sort(speedUps.begin(), speedUps.end());
    std::cout<<(double)(speedUps[(testSize - 1) / 2] + speedUps[testSize / 2]) / 2.0;


    return 0;
}