//
// Created by Lapo Bisognin on 15/07/25.
//

#ifndef KMEANS_H
#define KMEANS_H
#include <iosfwd>
#include <vector>
#include "utilities.h"


std::vector<size_t> SOAsKmeans(const SOAdataset& ds, SOAdataset& centers);
#endif //KMEANS_H
