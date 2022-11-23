
#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseSubsetParameterization : public ceres::LocalParameterization
{
public:
    explicit PoseSubsetParameterization(const std::vector<int>& constant_parameters);
private:
    virtual bool Plus(const double *x, const double *delta_, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
    std::vector<char> constancy_mask_;
};
