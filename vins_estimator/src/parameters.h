#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 185.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;
extern int ESTIMATE_EXTRINSIC_WHEEL;        //wheel 2
extern int ESTIMATE_INTRINSIC_WHEEL;        //wheel 2
extern double VEL_N_wheel;                  //wheel 2
extern double GYR_N_wheel;                  //wheel 2
extern double SX;                           //wheel 2
extern double SY;                           //wheel 2
extern double SW;                           //wheel 2
extern std::queue<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>> encoder_buf;    //todo(znn) wheel 2 node 2

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
extern double ENC_N;            //encoder
extern int ENCODER;                //use encoder or not
extern double LEFT_D, RIGHT_D;     //diameter of left, right wheel
extern double ENC_RESOLUTION;      //resulution of encoder
extern double WHEELBASE;           //distance between two wheels


extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;
extern Eigen::Matrix3d RIO;         //wheel
extern Eigen::Vector3d TIO;         //wheel

extern double ROLL_N, PITCH_N, ZPW_N;               //plane
extern double ROLL_N_INV, PITCH_N_INV, ZPW_N_INV;   //plane

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string GRONUD_TRUTH_PATH;
extern std::string EXTRINSIC_WHEEL_ITERATE_PATH;    //wheel
extern std::string TD_WHEEL_PATH;
extern std::string IMU_TOPIC;
extern std::string ODOM_TOPIC;
extern std::string TF_TOPIC;
extern std::string ENCODER_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ESTIMATE_TD_WHEEL;                   //wheel
extern int ROLLING_SHUTTER;
extern double ROW, COL, DOWN_SAMPLE;
extern int USE_WHEEL;                       //wheel
extern int USE_PLANE;                       //plane
extern int ONLY_INITIAL_WITH_WHEEL;         //wheel
extern bool is_stop;                    //todo(znn) Add ZUPT
void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1,
    SIZE_ROTATION = 4
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

enum WheelExtrinsicAdjustType{
    ADJUST_WHEEL_TRANSLATION,
    ADJUST_WHEEL_ROTATION,
    ADJUST_WHEEL_ALL,
    ADJUST_WHEEL_NO_Z,
    ADJUST_WHEEL_NO_ROTATION_NO_Z
};
extern WheelExtrinsicAdjustType WHEEL_EXT_ADJ_TYPE;

