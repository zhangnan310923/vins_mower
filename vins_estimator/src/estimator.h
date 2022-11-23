#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"
#include "factor/odom_factor.h"
//#include "factor/wheel_integration_base.h"

#include <nav_msgs/Odometry.h>

#include <unordered_map>
#include <queue>
#include <deque>
#include <opencv2/core/eigen.hpp>


class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processIMUEncoder(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity, const Vector3d &encoder_velocity); // encoder
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    void setStartValue();
    bool failureDetection();
    bool processSynchronizedENU(const std_msgs::Header &header);   //TODO(ZNN)
    double getSfromOdom();          //todo znn
    bool processSynchronizedOdom(const std_msgs::Header &header);
    void fastPredictWheel(double t, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity);  //wheel 2
    void processWheel(double t, double dt, const Vector3d &linear_velocity, const Vector3d &angular_velocity);   //wheel 2
    void fastPredictPureWheel(double t, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity, Eigen::Vector3d &P, Eigen::Quaterniond &Q, Eigen::Vector3d &V);
    void initPlane();
    void updateLastWheelStates();
    //void ZUPT(const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    //todo(znn) add wheel predict
    std::mutex mProcess;
    std::mutex mWheelBuf;
    std::mutex mWheelPropagate;
    queue<pair<double, Eigen::Vector3d>> wheelVelBuf;
    queue<pair<double, Eigen::Vector3d>> wheelGyrBuf;
    //todo wheel
    double latest_time_wheel;
    Eigen::Vector3d latest_P_wheel, latest_V_wheel, latest_vel_wheel_0, latest_gyr_wheel_0;
    double latest_sx, latest_sy, latest_sw;
    Eigen::Quaterniond latest_Q_wheel;
    bool openExWheelEstimation;
    bool openIxEstimation;
    //todo plane
    double para_plane_R[1][SIZE_ROTATION];
    double para_plane_Z[1][1];
    bool openPlaneEstimation;
    //solver
    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Matrix3d rio;       //encoder
    Vector3d tio;       //encoder

    Matrix3d rpw;       //wheel 2
    double zpw;         //wheel 2
    double sx = 1, sy = 1, sw = 1;      //wheel 2
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];

    Vector3d Po;        //encoder
    Vector3d Vo;        //encoder
    Matrix3d Ro;        //encoder

    //TODO(ZNN) cache odom P Q
    Vector3d Cache_Ps[(WINDOW_SIZE + 1)];
    Matrix3d Cache_Rs[(WINDOW_SIZE + 1)];
    std::deque<nav_msgs::Odometry> Cache_Odom;
    Eigen::Isometry3d tmp_TF = Eigen::Isometry3d::Identity();
    bool addOdomFactor;
    double td;
    double td_wheel;            //wheel 2

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];       
    //WheelIntegrationBase *pre_integrations_wheel[(WINDOW_SIZE + 1)];    //wheel 2
    Vector3d acc_0, gyr_0;
    Vector3d vel_0_wheel, gyr_0_wheel;      //wheel 2
    Vector3d enc_v_0;       //encoder  1

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> encoder_velocity_buf[(WINDOW_SIZE + 1)];           //encoder 1
    vector<double> dt_buf_wheel[(WINDOW_SIZE + 1)];                     //wheel 2
    vector<Vector3d> linear_velocity_buf_wheel[(WINDOW_SIZE + 1)];      //wheel 2
    vector<Vector3d> angular_velocity_buf_wheel[(WINDOW_SIZE + 1)];     //wheel 2

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    
    int imu_count = 0;                  //todo(znn) Add ZUPT  
    
    std::deque<Eigen::Vector3d> Acc_Zero_Check;    //todo(znn) Add ZUPT
    std::deque<Eigen::Vector3d> Gyr_Zero_Check;    //todo(znn) Add ZUPT
    Eigen::Vector3d acc_sum_;
    Eigen::Vector3d gyr_sum_; 

    bool first_imu;
    bool first_wheel;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Ex_Pose_enc[SIZE_POSE]; // encoder
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];
    double para_Ex_Pose_wheel[1][SIZE_POSE];            //wheel
    double para_Ix_sx_wheel[1][1];                      //wheel
    double para_Ix_sy_wheel[1][1];                      //wheel
    double para_Ix_sw_wheel[1][1];                      //wheel
    double para_Td_wheel[1][1];                         //wheel


    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;
    //WheelIntegrationBase *tmp_wheel_pre_integration;       //wheel 2

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;


};
