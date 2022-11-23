#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"
#include <mutex>
#include <vector>

class OdomScaleFactor : public ceres::SizedCostFunction<3,              //num of residual
                                                        7,              //parameter of pose
                                                        7>              //parameter of pose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static double GetResidual(const Eigen::Vector3d& Pi, const Eigen::Quaterniond& Qi,
                              const Eigen::Vector3d& Pj, const Eigen::Quaterniond& Qj,
                              const Eigen::Vector3d& T_ij);

    typedef Eigen::Matrix<double, 3, 3> covariance_t;

    explicit OdomScaleFactor(const Eigen::Vector3d& T_ij);

    virtual ~OdomScaleFactor() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    inline const Eigen::Vector3d& GetTij() const;

private:
    Eigen::Vector3d T_ij_;

    mutable covariance_t covariance_;
};

inline const Eigen::Vector3d& OdomScaleFactor::GetTij() const
{
    return T_ij_;
}
