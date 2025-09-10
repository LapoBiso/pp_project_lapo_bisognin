//
// Created by Lapo Bisognin on 15/07/25.
//

#include "../include/seqKmeans.h"
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include "../include/utilities.h"


std::vector<size_t> SOAsKmeans(const SOAdataset& ds, SOAdataset& centers)
{
    const size_t n = ds.nPoints, d = ds.nDims, k = centers.nPoints, max_iter = 30;
    std::vector<size_t> assign(n);
    SOAdataset sum(k, d);
    std::vector<size_t> count(k, 0);
    for (size_t it = 0; it < max_iter; ++it) {
        for(size_t i = 0; i < k*d; i++)
            sum.data[i] = 0;
        for (size_t i = 0; i < n; i++) {
            size_t best;
            float bestDist = std::numeric_limits<float>::max();
            for (size_t j = 0; j < k; j++) {
                float dist = 0.0f;
                float diff = 0.0f;
                for (size_t g = 0; g < d; g++) {
                    diff = ds.data[g*n+i] - centers.data[g*k+j];
                    dist += diff * diff;
                }
                if (dist < bestDist) {
                    bestDist = dist;
                    best = j;
                }
            }
            assign[i] = best;
            for (size_t g = 0; g < d; ++g)
                sum.data[g*k+best] += ds.data[g*n+i];
            count[best] += 1;
        }

        for (size_t j = 0; j < k; ++j)
         {
             if (count[j] > 0)
             {
                 for (size_t g = 0; g < d; ++g)
                     centers.data[g*k+j] = sum.data[g*k+j] / count[j];
             }
            count[j] = 0;
         }
    }
    return assign;
}