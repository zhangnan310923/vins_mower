#include "odom_factor.h"

OdomScaleFactor::OdomScaleFactor(const Eigen::Vector3d& T_ij) : T_ij_(T_ij)
{
    const double weight = 400;
    covariance_ << weight, 0, 0, 0, weight * 0.5, 0, 0, 0, weight;
}

double OdomScaleFactor::GetResidual(const Eigen::Vector3d& Pi, const Eigen::Quaterniond& Qi,
                                    const Eigen::Vector3d& Pj, const Eigen::Quaterniond& Qj,
                                    const Eigen::Vector3d& T_ij)
{
    const Eigen::Vector3d pts_imu_i = T_ij;
    const Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    const Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);

    const double weight = 400;
    Eigen::Matrix3d covariance;
    covariance << weight, 0, 0, 0, weight * 0.5, 0, 0, 0, weight;
    const double residual = (covariance * pts_imu_j.head<3>()).squaredNorm();

    return residual;
}

bool OdomScaleFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    const Eigen::Vector3d pts_imu_i = T_ij_;
    const Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    const Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);

    //residual
    Eigen::Map<Eigen::Vector3d> residual(residuals);
    residual.head<3>() = pts_imu_j.head<3>();
    residual = covariance_ * residual;

    //Compute Jacobians
    if(jacobians)
    {
        const Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        const Eigen::Matrix3d Rj = Qj.toRotationMatrix();

        if(jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            Eigen::Matrix<double, 3, 6> jacobian_i;
            jacobian_i.leftCols<3>() = Rj.transpose();
            jacobian_i.rightCols<3>() = Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.block<3, 6>(0, 0) = covariance_ * jacobian_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if(jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            Eigen::Matrix<double, 3, 6> jacobian_j;
            jacobian_j.leftCols<3>() = -Rj.transpose();
            jacobian_j.rightCols<3>() = Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.block<3, 6>(0, 0) = covariance_ * jacobian_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
    }
    return true;
}