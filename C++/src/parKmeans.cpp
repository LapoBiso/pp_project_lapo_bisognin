//
// Created by Lapo Bisognin on 17/07/25.
//

#include "../include/parKmeans.h"
#include"../include/utilities.h"
#include <cstdio>
#include <iostream>
#include <omp.h>


std::vector<size_t> SOApKmeans(const SOAdataset& ds, SOAdataset& centers)
{

    const size_t n = ds.nPoints, d = ds.nDims, k = centers.nPoints, max_iter = 30;
    std::vector<size_t> assign(n);
    SOAdataset sum(k, d);
    std::vector<size_t> count(k, 0);
    float* __restrict sum_ptr = sum.data.data();
    size_t*   __restrict count_ptr = count.data();

#pragma omp parallel default(none) shared(ds, centers, assign, sum_ptr, count_ptr, sum, count) firstprivate(n, d, k, max_iter)
    {
        for (size_t it = 0; it < max_iter; ++it)
            {
            for(size_t i = 0; i < k*d; i++)
                sum.data[i] = 0;
#pragma omp for schedule(static) reduction(+:sum_ptr[:k*d], count_ptr[:k])
            for (size_t i = 0; i < n; i++) {
                size_t best = -1;
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
                for (size_t g = 0; g < d; g++)
                    sum_ptr[g*k+best] += ds.data[g*n+i];
                count_ptr[best] += 1;
            }

#pragma omp single
            {
                for (size_t j = 0; j < k; ++j)
                {
                    if (count[j])
                    {
                        for (size_t g = 0; g < d; ++g)
                            centers.data[g*k+j] = sum.data[g*k+j] / count[j];
                    }
                    count[j] = 0;
                }
            }
        }
    }
    return assign;
}
