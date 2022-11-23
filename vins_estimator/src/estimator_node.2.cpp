#include <stdio.h>
#include <queue>
#include <deque>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <nav_msgs/Odometry.h>
#include <tf2_msgs/TFMessage.h>
#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#include <custom_msgs/Encoder.h>
#include <sensor_msgs/Imu.h>



#define __ADDENCODER__
//#define __NOENCODER__
//todo(znn) gb set
bool first_odom = true;
bool first_tf = true;
Eigen::Vector3d p_from_first_odom;
Eigen::Quaterniond q_from_first_odom;
double last_odom_t = 0;
double sin_lamb, cos_lamb, sin_phi, cos_phi;
Eigen::Vector3d StartPoint = Eigen::Vector3d::Zero();
//todo(znn) get tf
bool get_tf_cb_state = true;

Estimator estimator;

std::condition_variable con;
double current_time = -1;
double wheel_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
//queue<sensor_msgs::ImuPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
/* queue<std::pair<double, Eigen::Vector3d>> wheelVelBuf;
queue<std::pair<double, Eigen::Vector3d>> wheelGyrBuf; */
queue<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>> encoder_buf;    //todo(znn) 
std::deque<std::pair<std_msgs::Header, Eigen::Isometry3d>> Cache_TF;    //todo(znn)
std::pair<std_msgs::Header, Eigen::Isometry3d> cache_tf;        //todo(znn)

int sum_of_wait = 0;

std::mutex m_buf;           //imu buf
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;
std::mutex e_buf;           //encoder buf
std::mutex e_state;
std::mutex o_buf;
std::mutex t_buf;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
bool init_wheel = 1;
double last_imu_t = 0;
//wheel
double last_encoder_t = 0; 
double latest_encoder_time;
custom_msgs::Encoder last_wheel;
//Wheel
Eigen::Vector3d latest_P_wheel, latest_V_wheel, latest_vel_wheel_0, latest_gyr_wheel_0;
Eigen::Quaterniond latest_Q_wheel;


//todo(znn) gb set
Eigen::Vector3d setStartPoint(double lat, double lon, double height) {
    const double a = 6378137.0;             //
    const double b = 6356752.314245;        //
    const double f = (a - b) / a;           //
    const double e = f * (2 - f);
    double lamb = lat * M_PI / 180.0;
    double phi = lon * M_PI / 180.0;
    sin_lamb = sin(lamb);
    cos_lamb = cos(lamb);
    sin_phi = sin(phi);
    cos_phi = cos(phi);
    double N_tmp = a / sqrt(1 - e * sin_lamb * sin_lamb);

    Eigen::Vector3d StartPoint_tmp;
    StartPoint_tmp.x() = (height + N_tmp) * cos_lamb * cos_phi;
    StartPoint_tmp.y() = (height + N_tmp) * cos_lamb * sin_phi;
    StartPoint_tmp.z() = (height + (1 - e) * N_tmp) * sin_lamb;

    return StartPoint_tmp;
}

void ECEF2ENU(const Eigen::Vector3d &P_ECEF, const Eigen::Vector3d &StartPoint, Eigen::Vector3d &P_ENU) {

    double delta_x = P_ECEF.x() - StartPoint.x();
    double delta_y = P_ECEF.y() - StartPoint.y();
    double delta_z = P_ECEF.z() - StartPoint.z();

    double t = -cos_phi * delta_x - sin_phi * delta_y;

    double E = -sin_phi * delta_x + cos_phi * delta_y;
    double N = t * sin_lamb + cos_lamb * delta_z;
    double U = cos_lamb * cos_phi * delta_x + cos_lamb * sin_phi * delta_y + sin_lamb * delta_z;

    P_ENU << E, N, U;
}

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

/* todo(znn) add gb */
void odom_callback(const nav_msgs::Odometry &odom_msg)
{
    if(odom_msg.header.stamp.toSec() <= last_odom_t)
    {
        ROS_WARN("odom message in disorder!");
        return;
    }
    last_odom_t = odom_msg.header.stamp.toSec();
    Vector3d T = Vector3d(odom_msg.pose.pose.position.x,
                          odom_msg.pose.pose.position.y,
                          odom_msg.pose.pose.position.z);
    Eigen::Quaterniond Q = Quaterniond(odom_msg.pose.pose.orientation.w,
                          odom_msg.pose.pose.orientation.x,
                          odom_msg.pose.pose.orientation.y,
                          odom_msg.pose.pose.orientation.z);
    
    Vector3d T_ENU = Eigen::Vector3d::Zero();
    if(first_odom){
        StartPoint = setStartPoint(40.046074, 116.349485, 52.91);
        ECEF2ENU(T, StartPoint, p_from_first_odom);
        q_from_first_odom = Q;
        q_from_first_odom = q_from_first_odom.inverse();
        first_odom = false;
    } else{
        ECEF2ENU(T, StartPoint, T_ENU);
        T_ENU = T_ENU - p_from_first_odom;  
        Q = Q * q_from_first_odom; 
    }

    std_msgs::Header header = odom_msg.header;
    header.frame_id = "world";
    pubGbOdometry(T_ENU, Q, header);

    //cache odom
    nav_msgs::Odometry odom_cache;
    odom_cache.header = odom_msg.header;
    odom_cache.pose.pose.position.x = T_ENU.x();
    odom_cache.pose.pose.position.y = T_ENU.y();
    odom_cache.pose.pose.position.z = T_ENU.z();
    odom_cache.pose.pose.orientation = odom_msg.pose.pose.orientation;
    o_buf.lock();
    estimator.Cache_Odom.push_back(odom_cache);
    o_buf.unlock();
    con.notify_one();
}

/* todo(znn) add tf */
void tf_callback(const tf2_msgs::TFMessage &tf_msg)
{
    
    for(std::vector<geometry_msgs::TransformStamped>::const_iterator it = tf_msg.transforms.begin(); it != tf_msg.transforms.end(); ++it){
        if(it->header.frame_id == "BODY" && get_tf_cb_state) {
            Eigen::Vector3d T_body_camera = Eigen::Vector3d(it->transform.translation.x,
                                                            it->transform.translation.y,
                                                            it->transform.translation.z);
            Eigen::Matrix3d R_body_camera = Eigen::Quaterniond(it->transform.rotation.w,
                                                               it->transform.rotation.x,
                                                               it->transform.rotation.y,
                                                               it->transform.rotation.z).toRotationMatrix();
            Eigen::Isometry3d TF_body_camera = Eigen::Isometry3d::Identity();
            TF_body_camera.matrix().topLeftCorner(3, 3) = R_body_camera;
            TF_body_camera.matrix().topRightCorner(3, 1) = T_body_camera;  
            //TF_body_camera = TF_camera_body.inverse();
            get_tf_cb_state = false;                             
        }
        std_msgs::Header tf_header = it->header;
        if(it->header.frame_id == "ENU" && first_tf)
        {       //body2enu
            Eigen::Vector3d T_enu_body = Eigen::Vector3d(it->transform.translation.x,
                                                         it->transform.translation.y,
                                                         it->transform.translation.z);
            Eigen::Matrix3d R_enu_body = Eigen::Quaterniond(it->transform.rotation.w,
                                                               it->transform.rotation.x,
                                                               it->transform.rotation.y,
                                                               it->transform.rotation.z).toRotationMatrix();
            Eigen::Isometry3d TF_enu_body = Eigen::Isometry3d::Identity();
            TF_enu_body.matrix().topLeftCorner(3, 3) = R_enu_body;
            TF_enu_body.matrix().topRightCorner(3, 1) = T_enu_body;

            //todo(znn) add mutex for odom
            t_buf.lock();       
            Cache_TF.emplace_back(tf_header, TF_enu_body);
            t_buf.unlock();
            con.notify_one();
            cache_tf = Cache_TF.back();
            first_tf = false;
        }

        if(it->header.frame_id == "ENU")
        {
            Eigen::Vector3d T_enu_body = Eigen::Vector3d(it->transform.translation.x,
                                                         it->transform.translation.y,
                                                         it->transform.translation.z);
            Eigen::Matrix3d R_enu_body = Eigen::Quaterniond(it->transform.rotation.w,
                                                            it->transform.rotation.x,
                                                            it->transform.rotation.y,
                                                            it->transform.rotation.z).toRotationMatrix();
            Eigen::Isometry3d TF_enu_body = Eigen::Isometry3d::Identity();
            TF_enu_body.matrix().topLeftCorner(3, 3) = R_enu_body;
            TF_enu_body.matrix().topRightCorner(3, 1) = T_enu_body;

            //todo(znn) add mutex for odom
            t_buf.lock();       
            Cache_TF.emplace_back(tf_header, TF_enu_body);
            t_buf.unlock();
            con.notify_one();

            //T_body
/*             auto it = estimator.Cache_Odom.begin();
            Eigen::Isometry3d cache_tf_front = Eigen::Isometry3d::Identity();
            Eigen::Isometry3d cache_tf_back = Eigen::Isometry3d::Identity();
            double t1, t2;
            while(it != estimator.Cache_Odom.end())
            {
                double t = it->header.stamp.toSec();
                Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
                T_body.matrix().topLeftCorner(3, 3) = Eigen::Quaterniond(it->pose.pose.orientation.w,
                                                                         it->pose.pose.orientation.x,
                                                                         it->pose.pose.orientation.y,
                                                                         it->pose.pose.orientation.z).toRotationMatrix();
                T_body.matrix().topRightCorner(3, 1) = Eigen::Vector3d(it->pose.pose.position.x, 
                                                                       it->pose.pose.position.y,
                                                                       it->pose.pose.position.z);

                //Eigen::Vector3d T = Eigen::Vector3d::Zero();
                //Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
                while((Cache_TF.front().first.stamp.toSec() < it->header.stamp.toSec()) && !Cache_TF.empty())
                {
                    t1 = Cache_TF.front().first.stamp.toSec();
                    cache_tf_front = Cache_TF.front().second;
                    ROS_WARN_STREAM("Cache_TF front time = " << t1 << "\n" << "odom time = " << it->header.stamp.toSec() << endl);
                    Cache_TF.pop_front();

                }
                if(Cache_TF.empty()){
                        ROS_WARN("can not interpolation. wait for more tf.");
                        break;
                }else{
                    //interpolate
                    t2 = Cache_TF.front().first.stamp.toSec();
                    cache_tf_back = Cache_TF.front().second;
                    double alpha = (t - t1) / (t2 - t1);
                    //T = (1 - alpha) * cache_tf_front.topRightCorner(3, 1) + alpha * cache_tf_back.topRightCorner(3, 1);
                    //R = (Eigen::Quaterniond(cache_tf_front.topLeftCorner(3, 3)).slerp(alpha, Eigen::Quaterniond(cache_tf_back.topLeftCorner(3, 3)))).toRotationMatrix();
                    Eigen::Isometry3d T_odom_enu_body = Eigen::Isometry3d::Identity();
                    T_odom_enu_body.matrix().topRightCorner(3, 1) = (1 - alpha) * cache_tf_front.matrix().topRightCorner(3, 1) + alpha * cache_tf_back.matrix().topRightCorner(3, 1);
                    //Eigen::Matrix3d R1 = cache_tf_front.matrix().topLeftCorner(3, 3);
                    //Eigen::Matrix3d R2 = cache_tf_back.matrix().topLeftCorner(3, 3);
                    Eigen::Quaterniond Q1 = Eigen::Quaterniond(Eigen::Matrix3d(cache_tf_front.matrix().topLeftCorner(3, 3)));
                    Eigen::Quaterniond Q2 = Eigen::Quaterniond(Eigen::Matrix3d(cache_tf_back.matrix().topLeftCorner(3, 3)));
                    Eigen::Quaterniond Q = Q1.slerp(alpha, Q2);
                    //T_odom_enu_body.matrix().topLeftCorner(3, 3) = (Eigen::Quaterniond(cache_tf_front.matrix().topLeftCorner(3, 3)).slerp(alpha, Eigen::Quaterniond(cache_tf_back.matrix().topLeftCorner(3, 3)))).toRotationMatrix();
                    T_odom_enu_body.matrix().topLeftCorner(3, 3) = Q.toRotationMatrix();
                    T_body = cache_tf.second * T_body * T_odom_enu_body.inverse();
                    Eigen::Vector3d T_ = T_body.matrix().topRightCorner(3, 1);
                    Eigen::Matrix3d R_ = T_body.matrix().topLeftCorner(3, 3);
                    Eigen::Quaterniond Q_ = Eigen::Quaterniond(R_);
                    pubCacheOdometry(T_, Q_, it->header);
                    it++;
                    estimator.Cache_Odom.pop_front();                   
                }      
            } */
        }
    }
}

//todo(znn) check fixp format 
void encoder_callback(const custom_msgs::EncoderConstPtr &encoder_msg)
{
    double t = encoder_msg->header.stamp.toSec();
    if (t <= last_encoder_t)
    {
        ROS_WARN("encoder message in disorder!");
    }
    double dt = t - last_encoder_t;    
    last_encoder_t = encoder_msg->header.stamp.toSec();
    
    double encoder_right_vel = (encoder_msg->right_encoder + last_wheel.right_encoder) / 2;
    double encoder_left_vel = (encoder_msg->left_encoder + last_wheel.left_encoder) / 2;
    double encoder_omega = (encoder_right_vel - encoder_left_vel) / WHEELBASE;              //todo(znn) deg or rad? should be rad
    double encoder_vel = 0.5 * (encoder_right_vel + encoder_left_vel); 
    Eigen::AngleAxisd tmp_rot_vec(encoder_omega * dt, Eigen::Vector3d::UnitY());            //todo(znn) : UnitZ() should be?
    //todo(znn) wx wy set to zero? or use imu's wx wy?
    //Eigen::Vector3d angular_velocity << 0, 0, encoder_omega;
    Eigen::Vector3d angular_velocity = {gyr_0.x(), gyr_0.y(), encoder_omega};            //todo use tmp_rot_vec * angular_velocity
    Eigen::Vector3d linear_velocity =  {encoder_vel, tmp_V.y(), tmp_V.z()};
    
    estimator.mWheelBuf.lock();
    encoder_buf.push(make_pair(t, make_pair(linear_velocity, angular_velocity)));
    estimator.mWheelBuf.unlock();

    if (estimator.solver_flag == estimator.NON_LINEAR)
    {
        estimator.mWheelPropagate.lock();
        estimator.fastPredictWheel(t, linear_velocity, angular_velocity);
        pubWheelLatestOdometry(latest_P_wheel, latest_Q_wheel, latest_V_wheel, t);
        Eigen::Quaterniond q;
        Eigen::Vector3d p;
        Eigen::Vector3d v;
        estimator.fastPredictPureWheel(t, linear_velocity, angular_velocity, p, q, v);
        //pubPureWheelLatestOdometry(p, q, v, t);
        estimator.mWheelPropagate.unlock();
    }
    return;
}

/* void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
} */

//
void imu_callback(const sensor_msgs::Imu &imu_msg)
{
    
    sensor_msgs::ImuPtr imu_msg_ptr(new sensor_msgs::Imu);
    *imu_msg_ptr = imu_msg;
    if (imu_msg_ptr->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("IMU message in disorder!");
        return;
    }
    last_imu_t = imu_msg_ptr->header.stamp.toSec();
        
    m_buf.lock();
    imu_buf.push(imu_msg_ptr);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg_ptr->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg_ptr);
        std_msgs::Header header = imu_msg.header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

#ifdef __ADDENCODER__
std::vector<std::tuple<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr, 
            std::vector<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>>>> getMeasurements()
{
    std::vector<std::tuple<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr, 
                std::vector<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>>>> measurements;    
    while(true)
    {
        if(imu_buf.empty() || feature_buf.empty() || encoder_buf.empty())
            return measurements;

        if(!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("WAIT FOR IMU.");
            sum_of_wait++;
            return measurements;
        }

        if(!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("IMAGE WITHOUT IMU, THROW.");
            feature_buf.pop();
            continue;
        }

        if(!(encoder_buf.back().first > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("WAIT FOR ENCODER.");
            sum_of_wait++;
            return measurements;
        }

        if(!(encoder_buf.front().first < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("IMAGE WIHTOUT ENCODER, THROW.");
            feature_buf.pop();
            continue;
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while(imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if(IMUs.empty())
            ROS_WARN("NO IMU BETWEEN IMAGE.");

        std::vector<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>> encoders;
        while(encoder_buf.front().first < img_msg->header.stamp.toSec() + estimator.td)
        {
            encoders.emplace_back(encoder_buf.front());
            encoder_buf.pop();
        }
        encoders.emplace_back(encoder_buf.front());
        if(encoders.empty())
            ROS_WARN("NO ENCODER BETWEEN IMAGE.");

        measurements.emplace_back(IMUs, img_msg, encoders);
    }
    return measurements;
}

// todo(znn) visual-inertial-wheel odometry
void process()
{
    while(true)
    {
        std::vector<std::tuple<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr, 
                               std::vector<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>>>> measurements;
                                                     
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]{return (measurements = getMeasurements()).size() != 0;});
        lk.unlock();
        m_estimator.lock();

        for(auto &measurement : measurements)
        {
            auto img_msg = std::get<1>(measurement);
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0, vx = 0, vy = 0, vz = 0;
            Eigen::Vector3d wheel_angle_vel = Eigen::Vector3d::Zero();
            Eigen::Vector3d wheel_linear_vel = Eigen::Vector3d::Zero();
            //wheel process
            for(auto &encoder_msg : std::get<2>(measurement))
            {
                double t = encoder_msg.first;
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if(t <= img_t)
                {
                    if(wheel_time < 0)
                        wheel_time = t;
                    double dt = t - wheel_time;
                    wheel_angle_vel = encoder_msg.second.second;
                    wheel_linear_vel = encoder_msg.second.first;
                    estimator.processWheel(t, dt, encoder_msg.second.first, encoder_msg.second.second);
                }else
                {
                    double dt_1 = img_t - wheel_time;
                    double dt_2 = t - img_t;
                    wheel_time = img_t;
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    wheel_linear_vel = w1 * wheel_linear_vel + w2 * encoder_msg.second.first;
                    wheel_angle_vel = w1 * wheel_angle_vel + w2 * encoder_msg.second.second;
                    estimator.processWheel(t, dt_1, wheel_linear_vel, wheel_angle_vel);
                }
            }

            for (auto &imu_msg : std::get<0>(measurement))
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            

            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }
            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            // map<feature_id, []<camera_id, xyz_uv_velocity>>
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);

            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();        
    }
}
#endif


#ifdef __NOENCODER__
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);

        //todo(znn) add body P from vrtk (groundtruth VIO)
    }
    return measurements;
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {   
        TicToc t_s;
        TicToc t_imu;
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            double imu_t = t_imu.toc();
            ROS_WARN("IMU INTERG costs: %fms", imu_t);
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            //TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}
#endif


int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);
    ros::Subscriber sub_odom = n.subscribe(ODOM_TOPIC, 2000, odom_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_tf = n.subscribe(TF_TOPIC, 2000, tf_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_encoder = n.subscribe(ENCODER_TOPIC, 2000, encoder_callback, ros::TransportHints().tcpNoDelay());
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
