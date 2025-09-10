//
// Created by Lapo Bisognin on 16/07/25.
//

#include "../include/utilities.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <random>
#include <vector>

bool operator==(const SOAdataset& a, const SOAdataset& b) {
    for (size_t i = 0; i < a.nPoints; i++) {
        for (size_t j = 0; j < a.nDims; j++)
        {
            if (a.at(i,j) != b.at(i,j))
                return false;
        }
    }
    return true;
}


SOAdataset SOAdataGenerator(size_t nc, size_t c, size_t dims) {
    SOAdataset ds(nc * c, dims);
    const size_t n = nc * c;
    float distance = 4;
    float stddev = 2;

    std::vector<std::normal_distribution<float>> gaussDists(c);
    for (size_t j = 0; j < c; ++j)
        gaussDists[j] = std::normal_distribution<float>(j * distance, (j + 1) * stddev);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);

    for (size_t j = 0; j < c; ++j) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < dims; ++k) {
                ds.data[k * n + i] = gaussDists[j](gen);
            }
        }
    }
    return ds;
}

SOAdataset SOAcentersGenerator(size_t c, SOAdataset ds) {
    SOAdataset centers(c, ds.nDims);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution (0,ds.nPoints);
    int cenInd;
    cenInd = distribution(generator);
    for (int i=0; i<c; ++i)
        {
        for (int k=0; k<ds.nDims; k++)
            centers.at(i,k) = ds.at(cenInd,k);
        }
    return centers;
}


std::ostream& operator<<(std::ostream& os, const SOAdataset& d) {
    for (size_t i = 0; i < d.nPoints; i++) {
        os << "dataset" << i << "(";
        for (size_t j = 0; j < d.nDims; j++) {
            os << d.at(i, j);
            if (j + 1 < d.nDims) os << ", ";
        }
        os << ")";
        if (i + 1 < d.nPoints) os << "\n";
    }
    return os;
}

