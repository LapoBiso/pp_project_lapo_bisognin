//
// Created by Lapo Bisognin on 16/07/25.
//

#ifndef UTILITIES_H
#define UTILITIES_H
#include <iosfwd>
#include <vector>
struct SOAdataset {
    size_t nPoints;
    size_t nDims;
    std::vector<float> data;

    SOAdataset(size_t nPoints, size_t nDims): nPoints(nPoints), nDims(nDims) {
        data.resize(nPoints * nDims);
    }
    void resize(size_t points, size_t dims)
    {
        nPoints = points;
        nDims = dims;
        data.resize(points * dims);
    }

    float &at(size_t point, size_t dim) {
        return data[dim * nPoints + point];
    }


    [[nodiscard]] float const &at(size_t point, size_t dim) const {
        return data[dim * nPoints + point];
    }
};


bool operator==(const SOAdataset& a, const SOAdataset& b);
SOAdataset SOAdataGenerator(size_t nc, size_t c, size_t dims);
std::ostream& operator<<(std::ostream& os, const SOAdataset& d);
SOAdataset SOAcentersGenerator(size_t c, SOAdataset ds);

#endif //UTILITIES_H
