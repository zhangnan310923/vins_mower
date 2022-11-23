#ifndef YAWALIGN_H
#define YAWALIGN_H

#include <iostream>
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <math.h>

class OdomPoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Eigen::Map<const Eigen::Vector3d> _p(x + 1);
        Eigen::Map<const Eigen::Quaterniond> _q(x + 4);
        Eigen::Map<const Eigen::Vector3d> dp(delta + 1);
        Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 4));
        Eigen::Map<Eigen::Vector3d> p(x_plus_delta + 1);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 4);

        x_plus_delta[0] = x[0] + delta[0];
        p = _p + dp;
        q = (_q * dq).normalized();

        return true;
    };

    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 8, 7, Eigen::RowMajor>> j(jacobian);

        j.topRows<7>().setIdentity();
        j.bottomRows<1>().setZero();

        return true;
    };

    virtual int GlobalSize() const { return 8; };
    virtual int LocalSize() const {return 7; };
};

class OdomScalarCostFunctor 
{
public:
    OdomScalarCostFunctor(Eigen::Vector3d pw, Eigen::Vector3d pc): pw_(pw), pc_(pc) {}

    template <typename T>
    bool operator()(const T* const par, T* residuals) const 
    {
        Eigen::Matrix<T, 3, 1> pww, pcc;
        pww[0] = T(pw_[0]);
        pww[1] = T(pw_[1]);
        pww[2] = T(pw_[2]);

        pcc[0] = T(pc_[0]);
        pcc[1] = T(pc_[1]);
        pcc[2] = T(pc_[2]);

        T s = par[0];

        Eigen::Matrix<T, 3, 1> tt(par[1], par[2], par[3]);
        Eigen::Quaternion<T> qq(par[7], par[4], par[5], par[6]);
        Eigen::Matrix<T, 3, 1> pre = pww - (s * (qq * pcc) + tt);

        residuals[0] = pre[0];
        residuals[1] = pre[1];
        residuals[2] = pre[2];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d pww, const Eigen::Vector3d pcc)
    {
        return (new ceres::AutoDiffCostFunction<OdomScalarCostFunctor, 3, 8>(
            new OdomScalarCostFunctor(pww, pcc)));
    }

private:
    Eigen::Vector3d pw_;
    Eigen::Vector3d pc_;        
};

inline int getRwc(const vector<Eigen::Vector3d> &pws, const vector<Eigen::Vector3d> &pcs, Eigen::Matrix3d &Rwc, Eigen::Vector3d &Twc, double &Swc)
{
    double par[8] = {1, 0, 0, 0, 0, 0, 0, 1};
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    ceres::LocalParameterization *local_parameterization = new OdomPoseLocalParameterization();
    problem.AddParameterBlock(par, 8, local_parameterization);

    for(size_t i = 0; i < pws.size(); i++)
    {
        ceres::CostFunction* cost_function = OdomScalarCostFunctor::Create(pws[i], pcs[i]);
        problem.AddResidualBlock(cost_function, loss_function, par);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 4;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    Eigen::Quaterniond q(par[7], par[4], par[5], par[6]);
    Rwc = q.toRotationMatrix();
    Twc[0] = par[1]; 
    Twc[1] = par[2]; 
    Twc[2] = par[3];
    Swc = par[0];

    double sum = 0;
    int num = pws.size();
    for(size_t i = 0; i < pws.size(); i++)
    {
        Eigen::Vector3d pww = Swc * (Rwc * pcs[i]) + Twc;
        Eigen::Vector3d distance = pws[i] - pww; 
        std::cout << "-------------" << distance.transpose() << std::endl;
        double dd  = sqrt(distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2]);
        sum += dd;
    }

    if(sum / num > 0.05)
    {
        return false;
    }
    return true;
}

#endif