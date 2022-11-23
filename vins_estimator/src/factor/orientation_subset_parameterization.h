
#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class OrientationSubsetParameterization : public ceres::LocalParameterization
{
public:
    explicit OrientationSubsetParameterization(const std::vector<int> &constant_parameters);

private:
    virtual bool Plus(const double *x, const double *delta_, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 4; };
    virtual int LocalSize() const { return 3; };
    std::vector<char> constancy_mask_;
};
