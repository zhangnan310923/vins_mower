#include "estimator.h"
#include "utility/visualization.h"
#include "yawalign.h"
#include "factor/imu_encoder_factor.h"

//#define __ADDODOMFACTOR__

bool is_stop = false;           //add ZUPT

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;

    if(ENCODER)
    {
        tio = TIO;
        rio = RIO;
    }
}

void Estimator::clearState()
{
    //todo(znn) add encoder
    if(ENCODER)
    {
        Ro.setIdentity();
        Po.setZero();
        Vo.setZero();  
        for(int i = 0; i < WINDOW_SIZE + 1; i++)
        {
            encoder_velocity_buf[i].clear();
        }    
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }
    

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    //todo(znn) add ZUPT
    for(auto iter = Acc_Zero_Check.begin(); iter != Acc_Zero_Check.end(); ++iter)
    {
        Acc_Zero_Check.pop_front();
    }
    for(auto iter = Gyr_Zero_Check.begin(); iter != Gyr_Zero_Check.end(); iter++)
    {
        Gyr_Zero_Check.pop_front();
    }

    is_stop = false;                    //todo(znn) ZUPT
    imu_count = 0;                      //todo(znn) ZUPT
    acc_sum_.setZero();
    gyr_sum_.setZero();

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/* void Estimator::ZUPT(const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if(imu_count < 20)
    {
        Acc_Zero_Check.push_back(linear_acceleration);
        Gyr_Zeor_Check.push_back(angular_velocity);
        acc_sum_ += linear_acceleration;
        gyr_sum_ += angular_velocity;
        imu_count++;
    }else
    {
        Acc_Zero_Check.push_back(linear_acceleration);
        Gyr_Zeor_Check.push_back(angular_velocity);
        acc_sum_ = acc_sum_ + Acc_Zero_Check.back() - Acc_Zero_Check.front();
        gyr_sum_ = gyr_sum_ + Gyr_Zeor_Check.back() - Gyr_Zeor_Check.front();
        Acc_Zero_Check.pop_front();
        Gyr_Zeor_Check.pop_front();
        Eigen::Vector3d avr_acc = acc_sum_ / imu_count;
        Eigen::Vector3d avr_gyr = gyr_sum_ / imu_count;
        double accnorm = avr_acc.squaredNorm();
        double gyrnorm = avr_gyr.squaredNorm();

        Eigen::Vector3d var_acc;
        Eigen::Vector3d var_gyr;
        int num = 0;
        for(auto &acc_iter : Acc_Zero_Check)
        {
            var_acc.x() += ((acc_iter.x() - avr_acc.x()) * (acc_iter.x() - avr_acc.x()));            //todo
            var_acc.y() += ((acc_iter.y() - avr_acc.y()) * (acc_iter.y() - avr_acc.y()));            //todo
            var_acc.z() += ((acc_iter.z() - avr_acc.z()) * (acc_iter.z() - avr_acc.z()));            //todo
            
            num++;
            
        }
        ROS_WARN_STREAM("NUM = "<< num);
        for(auto &gyr_iter : Gyr_Zeor_Check)
        {
            var_gyr.x() += ((gyr_iter.x() - avr_gyr.x()) * (gyr_iter.x() - avr_gyr.x()));
            var_gyr.y() += ((gyr_iter.y() - avr_gyr.y()) * (gyr_iter.y() - avr_gyr.y()));
            var_gyr.z() += ((gyr_iter.z() - avr_gyr.z()) * (gyr_iter.z() - avr_gyr.z()));
        }

        ROS_WARN_STREAM("var_acc_0 = " << var_acc);
        ROS_WARN_STREAM("var_gyr_0 = " << var_gyr); 
        var_acc /= imu_count;
        var_gyr /= imu_count;
        
        ROS_WARN_STREAM("gyrnorm = " << gyrnorm);
        ROS_WARN_STREAM("var_acc = " << var_acc);
        ROS_WARN_STREAM("var_gyr = " << var_gyr); 
        //if((gyrnorm <= 0.005) && (var_acc.x() <= 8.0e-6) && (var_acc.y() <= 8.0e-6) && \
           (var_gyr.x() <= 0.001) && (var_gyr.y() <= 0.001) && (var_gyr.z() <= 0.001))
        if((gyrnorm <= 0.05) && (var_acc.x() <= 2) && (var_acc.y() <= 1))
           //(var_gyr.x() <= 0.05) && (var_gyr.y() <= 0.05))
        {
            is_stop = true;
        }else
        {
            is_stop = false;
        }     
    }
} */

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;

        //ZUPT
        //ZUPT(acc_0, gyr_0);
        
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/* //midpoint
void Estimator::fastPredictWheel(double t, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time_wheel;
    latest_time_wheel = t;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity);
    Eigen::Vector3d un_vel_0 = latest_Q_wheel * latest_vel_wheel_0;
    //these latest_values will fresh after inintialStructure()
    latest_Q_wheel = latest_Q_wheel * Utility::deltaQ(un_gyr * dt);
    latest_V_wheel = 0.5 * (latest_Q_wheel * linear_velocity + un_vel_0);
    latest_P_wheel = latest_P_wheel + dt * latest_V_wheel ;
    latest_vel_wheel_0 = linear_velocity;
    latest_gyr_wheel_0 = angular_velocity; 
}

//midpoint
void Estimator::fastPredictPureWheel(double t, Eigen::Vector3d linear_velocity, Eigen::Vector3d angular_velocity, 
                                               Eigen::Vector3d &P, Eigen::Quaterniond &Q, Eigen::Vector3d &V)
{
    static bool first_time = false;
    static Eigen::Quaterniond Q_latest;
    static Eigen::Vector3d V_latest = Eigen::Vector3d::Zero();
    static Eigen::Vector3d P_latest = Eigen::Vector3d::Zero();
    static Eigen::Vector3d vel_latest_0 = Eigen::Vector3d::Zero();
    static Eigen::Vector3d gyr_latest_0 = Eigen::Vector3d::Zero();
    static double t_latest;
    if(!first_time){
        first_time = true;
        Q_latest = latest_Q_wheel;
        V_latest = latest_V_wheel;
        P_latest = latest_P_wheel;
        vel_latest_0 = latest_vel_wheel_0;
        gyr_latest_0 = latest_gyr_wheel_0;
        t_latest = latest_time_wheel;
        //std::cout<<"fastPredictPureWheel initial pose: \n"<<P_latest.transpose()<<std::endl<<Q_latest.coeffs().transpose()<<std::endl;
    }

    double dt = t - t_latest;
    t_latest = t;
    Eigen::Vector3d un_gyr = 0.5 * (gyr_latest_0 + angular_velocity);
    Eigen::Vector3d un_vel_0 = Q_latest * vel_latest_0;
    
    //these latest_values will fresh after inintialStructure()
    Q_latest = Q_latest * Utility::deltaQ(un_gyr * dt);
    V_latest = 0.5 * (Q_latest * linear_velocity + un_vel_0);
    P_latest = P_latest + dt * V_latest ;
    vel_latest_0 = linear_velocity;
    gyr_latest_0 = angular_velocity;

    P = P_latest;
    Q = Q_latest;
    V = V_latest;
}

//todo(znn) add Wheel
void Estimator::processWheel(double t, const Vector3d &linear_velocity, const Vector3d &angular_velocity)
{
    mWheelBuf.lock();
    wheelVelBuf.push(make_pair(t, linear_velocity));
    wheelGyrBuf.push(make_pair(t, angular_velocity));
    //printf("input imu with time %f \n", t);
    mWheelBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mWheelPropagate.lock();
        fastPredictWheel(t, linear_velocity, angular_velocity);
        pubWheelLatestOdometry(latest_P_wheel, latest_Q_wheel, latest_V_wheel, t);
        Eigen::Quaterniond q;
        Eigen::Vector3d p;
        Eigen::Vector3d v;
        fastPredictPureWheel(t, linearVelocity, angularVelocity, p, q, v);
        pubPureWheelLatestOdometry(p, q, v, t);
        mWheelPropagate.unlock();
    }
} */

//todo(znn) add IMUencoder
void Estimator::processIMUEncoder(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity, const Vector3d &encoder_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
        enc_v_0 = encoder_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count], enc_v_0};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity, encoder_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity, encoder_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        encoder_velocity_buf[frame_count].push_back(encoder_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;

        // 
        Vo = encoder_velocity;
        Po += dt * Vo;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
    enc_v_0 = encoder_velocity;
}                                

//todo(znn) get corresponding odom
bool Estimator::processSynchronizedENU(const std_msgs::Header &header) {
    if(Cache_Odom.empty()){
        ROS_WARN("groudtruth enu cache is empty.");
        return false;
    }else {
        double t = header.stamp.toSec();
        auto it = Cache_Odom.begin();
        while (it != Cache_Odom.end()) {
            auto it_next = next(it);
            if(it->header.stamp.toSec() > t) {
                //ROS_WARN("can not synchronize enu");
                return false;
            } else if(it->header.stamp.toSec() <= t && it_next->header.stamp.toSec() >= t) {
                //interpolate
                double t1 = it->header.stamp.toSec();
                double t2 = it_next->header.stamp.toSec();
                double alpha = (t - t1) / (t2 - t1);
                Vector3d T1 = Vector3d(it->pose.pose.position.x, 
                                       it->pose.pose.position.y,
                                       it->pose.pose.position.z);
                Vector3d T2 = Vector3d(it_next->pose.pose.position.x,
                                       it_next->pose.pose.position.y,
                                       it_next->pose.pose.position.z);
                Vector3d T = (1 - alpha) * T1 + alpha * T2;
                Quaterniond Q1 = Quaterniond(it->pose.pose.orientation.w,
                                             it->pose.pose.orientation.x,
                                             it->pose.pose.orientation.y,
                                             it->pose.pose.orientation.z);
                Quaterniond Q2 = Quaterniond(it_next->pose.pose.orientation.w,
                                             it_next->pose.pose.orientation.x,
                                             it_next->pose.pose.orientation.y,
                                             it_next->pose.pose.orientation.z);
                Quaterniond interpolated_quat = Q1.slerp(alpha, Q2);
                Matrix3d R = interpolated_quat.toRotationMatrix();
                Cache_Ps[frame_count] = T;
                Cache_Rs[frame_count] = R;
                pubCacheENU(T, interpolated_quat, header);
                return true;
            } else {
                it++;
                Cache_Odom.pop_front();
            }
        }       
    }
    return true;
}

double Estimator::getSfromOdom()
{
    double s = 0;
    double sum = 0;
    Eigen::Vector3d Ps_tmp =  Eigen::Vector3d::Zero();
    Eigen::Vector3d Cache_Ps_tmp =  Eigen::Vector3d::Zero();
    for(int i=0; i < WINDOW_SIZE - 1; i++)
    {
        Ps_tmp = Ps[i] - Ps[i+1];
        Cache_Ps_tmp  = Cache_Ps[i] - Cache_Ps[i+1];
        s = abs(Ps_tmp.norm() - Cache_Ps_tmp.norm());
        sum += s;
    }
    s = sum / WINDOW_SIZE;
    return s;
}

/* bool Estimator::processSynchronizedOdom(const std_msgs::Header &header) 
{   
    Eigen::Isometry3d T_enu = Eigen::Isometry3d::Identity();
    T_enu.matrix().topLeftCorner(3, 3) = Cache_Rs[frame_count];             //use a new framcount 
    T_enu.matrix().topRightCorner(3, 1) = Cache_Ps[frame_count];
    double t = header.stamp.toSec();

    if(Cache_TF.empty()){
        ROS_WARN("tf cache is empty.");
    }else 
    {   
        Vector3d T1 = Eigen::Vector3d::Zero();
        Matrix3d R1 = Eigen::Isometry3d::Identity();
        //todo(znn) new tf slerp 
        auto it = Cache_TF.begin();
        while(it != Cache_TF.end())
        {   
            while(it->first.stamp.toSec() < t && it != Cache_TF.end())
            {
                //t1
                T1 = it->second.matrix().topRightCorner(3, 1);
                R1 = it->second.matrix().topLeftCorner(3, 3);
                it++;
            }
            if(it->first.stamp.toSec() >= t)
            {
                Vector3d T2 = Eigen::Vector3d::Zero();
                Matrix3d R2 = Eigen::Isometry3d::Identity();
                T2 = it->second.matrix().topRightCorner(3, 1);
                R2 = it->second.matrix().topLeftCorner(3, 3);
                Vector3d T = (1 - alpha) * T1 + alpha * T2;
                Quaterniond interpolated_quat = Quaterniond(R1).slerp(alpha, Quaterniond(R2));
                Eigen::Isometry3d T_enu_body = Eigen::Isometry3d::Identity();
                T_enu_body.matrix().topLeftCorner(3, 3) = interpolated_quat.toRotationMatrix();
                T_enu_body.matrix().topRightCorner(3, 1) = T;
                Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
                T_body = tmp_TF * T_enu * T_enu_body.inverse();
                return true;                
            }
            if(it == Cache_TF.end())
            {
                ROS_WARN("this monment can not get sync tf.");
                return false;
            }

        }

        double t = header.stamp.toSec();
        auto it = Cache_TF.begin();
        while (it != Cache_TF.end()) {
            auto it_next = next(it);
            if(it->first.stamp.toSec() > t) {
                ROS_WARN("can not synchronize tf");
                tmp_TF = it->second;
                return false;
            } else if(it->first.stamp.toSec() <= t && it_next->first.stamp.toSec() >= t) {
                //interpolate
                double t1 = it->first.stamp.toSec();
                double t2 = it_next->first.stamp.toSec();
                double alpha = (t - t1) / (t2 - t1);
                Vector3d T1 = it->second.matrix().topRightCorner(3, 1);
                Vector3d T2 = it_next->second.matrix().topRightCorner(3, 1);
                Vector3d T = (1 - alpha) * T1 + alpha * T2;
                Matrix3d R1 = it->second.matrix().topLeftCorner(3, 3);
                Matrix3d R2 = it_next->second.matrix().topLeftCorner(3, 3);
                Quaterniond Q1 = Eigen::Quaterniond(R1);
                Quaterniond Q2 = Eigen::Quaterniond(R2);
                Quaterniond interpolated_quat = Q1.slerp(alpha, Q2);
                Matrix3d R = interpolated_quat.toRotationMatrix();

                Eigen::Isometry3d T_enu_body = Eigen::Isometry3d::Identity();
                T_enu_body.matrix().topLeftCorner(3, 3) = R;
                T_enu_body.matrix().topRightCorner(3, 1) = T;

                Eigen::Isometry3d T_enu = Eigen::Isometry3d::Identity();
                T_enu.matrix().topLeftCorner(3, 3) = Cache_Rs[frame_count];
                T_enu.matrix().topRightCorner(3, 1) = Cache_Ps[frame_count];

                Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
                T_body = tmp_TF * T_enu * T_enu_body.inverse();

                Eigen::Vector3d T_ = T_body.matrix().topRightCorner(3, 1);
                Eigen::Matrix3d R_ = T_body.matrix().topLeftCorner(3, 3);
                Quaterniond Q_ = Quaterniond(R_);
                pubCacheOdometry(T_, Q_, header);
                tmp_TF = it->second;
                return true;
            } else {    
                cache_tf = Cache_TF.front().second;
                Cache_TF.pop_front();
                it++;
            }
        }
    }
        Vector3d T = cache_tf.matrix().topRightCorner(3, 1);
        Matrix3d R = cache_tf.matrix().topLeftCorner(3, 3);
                
        Eigen::Isometry3d T_enu_body = Eigen::Isometry3d::Identity();
        T_enu_body.matrix().topLeftCorner(3, 3) = R;
        T_enu_body.matrix().topRightCorner(3, 1) = T;

        Eigen::Isometry3d T_enu = Eigen::Isometry3d::Identity();
        T_enu.matrix().topLeftCorner(3, 3) = Cache_Rs[frame_count];
        T_enu.matrix().topRightCorner(3, 1) = Cache_Ps[frame_count];

        Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
        T_body = tmp_TF * T_enu * T_enu_body.inverse();  
        

        Eigen::Vector3d T_ = T_body.matrix().topRightCorner(3, 1);
        Eigen::Matrix3d R_ = T_body.matrix().topLeftCorner(3, 3);
        Quaterniond Q_ = Quaterniond(R_);
        //pubCacheOdometry(T_, Q_, header);
          
        //no new tf at this image frame, choose to use last newest tf
    return true;    
} */

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    if(ENCODER)
    {   
        tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count], enc_v_0};
    }else
    {
        tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    }
      
    //todo(znn) get corresponding odom
    bool sync_odom_state = processSynchronizedENU(header);
    if(!sync_odom_state) {
        ROS_WARN("failed to get sync enu.");
    } 

/*     bool body_odom_state = processSynchronizedOdom(header);
    if(body_odom_state) {
        ROS_WARN("fresh  sync odom.");
    }  */

    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            if(result)
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                //setStartValue();
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];             
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        //ROS_WARN("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_WARN("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_WARN("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_WARN("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_WARN("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }

    //todo(znn) use encoder to get s to check with imualign result.
    //s = getSfromOdom();
    //s = 1.0;
    ROS_WARN_STREAM("S =      " << getSfromOdom());
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(g);                  
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {   
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_WARN_STREAM("g0     " << g.transpose());
    ROS_WARN_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    /* todo(znn) yaw align */
/*     vector<Eigen::Vector3d> pws, pcs;
    for(int i = 0; i < frame_count; i++)
    {
        pws.push_back(Cache_Ps[i]);
        pcs.push_back(Ps[i]);
    }
    Eigen::Matrix3d Rwc;
    Eigen::Vector3d Twc;
    double Swc;
    if(!getRwc(pws, pcs, Rwc, Twc, Swc))
    {
        std::cout << "failed to align w and c." << std::endl;
        return false;
    } */

    /* todo(znn) fresh R P V and deep */
/*     for(int i = 0; i < frame_count; i++)
    {
        Rs[i] = Rwc * Rs[i];
        Vs[i] = Swc * (Rwc * Vs[i]);
        Ps[i] = Swc * (Rwc * Ps[i]) + Twc;
    }

    ROS_WARN_STREAM("Swc =     " << Swc);
    for(auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= Swc;
    } 
 */
    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            //todo(znn) change FOCAL_LENGTH = 460 to FOCAL_LENGTH = 185, the Corresponding parallax = 15 (from 30) 
            if(average_parallax * FOCAL_LENGTH > 12 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_WARN("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * FOCAL_LENGTH, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    //todo (znn) add encoder
    if(ENCODER)
    {
        para_Ex_Pose_enc[0] = tio.x();
        para_Ex_Pose_enc[1] = tio.y();
        para_Ex_Pose_enc[2] = tio.z();
        Quaterniond qio(rio);
        para_Ex_Pose_enc[3] = qio.x();
        para_Ex_Pose_enc[4] = qio.y();
        para_Ex_Pose_enc[5] = qio.z();
        para_Ex_Pose_enc[6] = qio.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

//todo(znn) set start value by cache_odom
void Estimator::setStartValue()
{
    Vector3d origin_R0 = Utility::R2ypr(Cache_Rs[0]);
    Vector3d origin_P0 = Cache_Ps[0];

    Vector3d origin_R00 = Utility::R2ypr(Rs[0]);
    double y_diff = origin_R0.x() - origin_R00.x();
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Cache_Rs[0] * Rs[0].transpose();
    }
 
/*     Vector3d delta_P = Ps[10] - Ps[0];
    double yaw_vins =  atan(delta_P.y() / delta_P.x()); 
    delta_P = Cache_Ps[10] - Cache_Ps[0];
    double yaw_odom = atan(delta_P.y() / delta_P.x());
    double delta_yaw = yaw_odom - yaw_vins;
    ROS_WARN_STREAM("delta_yaw = " << delta_yaw); */
    
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
/*         //check why
        Vector3d R = Utility::R2ypr(Rs[i] * Cache_Rs[i].transpose());
        ROS_WARN_STREAM("PYR[" << i << "] = " << R.transpose()); */
        
        Rs[i] = rot_diff * Rs[i];
        
        Ps[i] = rot_diff * (Ps[i] - Ps[0]) + origin_P0;

        Vs[i] = rot_diff * Vs[i];
       
    }
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    //todo(znn) add encoder
    if(ENCODER)
    {
        tio = Vector3d(para_Ex_Pose_enc[0],
                       para_Ex_Pose_enc[1],
                       para_Ex_Pose_enc[2]);
        rio = Quaterniond(para_Ex_Pose_enc[6],
                          para_Ex_Pose_enc[3],
                          para_Ex_Pose_enc[4],
                          para_Ex_Pose_enc[5]).toRotationMatrix();
        TIO = tio;
        RIO = rio;
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        return true;
    }
    return false;
}


void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }

    //todo(znn) add ZUPT
/*     if(is_stop)
    {   
        ROS_WARN("MOWER IS STOP.");
        for(int i = 0; i < WINDOW_SIZE; i++)
        {
            problem.SetParameterBlockConstant(para_Pose[i]);
            problem.SetParameterBlockConstant(para_SpeedBias[i]);
        }  
    }  */
    //TODO(ZNN) add encoder para
    if(ENCODER)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose_enc, SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            problem.SetParameterBlockConstant(para_Ex_Pose_enc);
        }        
    }

    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    vector2double();

    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    //todo(znn) add encoder or not
    if(!ENCODER)
    {   
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }

    }else
    {
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            /* ZUPT */
/*             if(is_stop)
            {
                loss_function = new ceres::HuberLoss(0.01);
            }else
            {
                loss_function = new ceres::HuberLoss(1.0);
            } */
            if(!is_stop)
            {
                IMUEncoderFactor* imu_encoder_factor = new IMUEncoderFactor(pre_integrations[j]);
                problem.AddResidualBlock(imu_encoder_factor, loss_function, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j], para_Ex_Pose_enc);
            }
        }
    }
    
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                    /* ZUPT */
                    if(is_stop)
                    {
                        loss_function = new ceres::HuberLoss(100.0);
                    }else
                    {
                        loss_function = new ceres::HuberLoss(1.0);
                    }
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {   
                /* ZUPT */
                if(is_stop)
                {
                    loss_function = new ceres::HuberLoss(100.0);
                }else
                {
                    loss_function = new ceres::HuberLoss(1.0);
                }
                
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    /* todo(znn) add rtk or P_enu */

#ifdef __ADDODOMFACTOR__
    ceres::LossFunction* odom_loss_function;
    odom_loss_function = new ceres::HuberLoss(1.0);

    for(int i = 0; i < WINDOW_SIZE; i++)
    {   
        Eigen::Vector3d delta_P = Cache_Ps[i + 1] - Cache_Ps[i];
        OdomScaleFactor *odom_scale_factor = new OdomScaleFactor(delta_P);
        problem.AddResidualBlock(odom_scale_factor, odom_loss_function, para_Pose[i], para_Pose[i + 1]);
    }
#endif

    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {   
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //todo(znn) add encoder
        {
            if(ENCODER)
            {
                if (pre_integrations[1]->sum_dt < 10.0)
                {
                    IMUEncoderFactor* imu_encoder_factor = new IMUEncoderFactor(pre_integrations[1]);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_encoder_factor, NULL, vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1], para_Ex_Pose_enc}, vector<int>{0, 1});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }else
            {
                if (pre_integrations[1]->sum_dt < 10.0)
                {
                    IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                                    vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                                    vector<int>{0, 1});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }               
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        //todo(znn)
        if(ENCODER)
        {
           addr_shift[reinterpret_cast<long>(para_Ex_Pose_enc)] = para_Ex_Pose_enc; // encoder 
        }
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if(ENCODER)
            {
                addr_shift[reinterpret_cast<long>(para_Ex_Pose_enc)] = para_Ex_Pose_enc; // encoder 
            }
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
                //TODO(ZNN) add encoder
                if(ENCODER)
                {
                    encoder_velocity_buf[i].swap(encoder_velocity_buf[i + 1]);
                }

                Cache_Ps[i].swap(Cache_Ps[i + 1]);                      //todo(znn) slide window
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            Cache_Ps[WINDOW_SIZE] = Cache_Ps[WINDOW_SIZE - 1];          //todo(znn)

            delete pre_integrations[WINDOW_SIZE];
            //todo(znn) add encoder
            if(ENCODER)
            {
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE], enc_v_0};
                encoder_velocity_buf[WINDOW_SIZE].clear();
            }else
            {
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            }
            

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];
                //todo(znn) add encoder
                if(ENCODER)
                {
                    Vector3d tmp_encoder_velocity = encoder_velocity_buf[frame_count][i];
                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity, tmp_encoder_velocity);
                    encoder_velocity_buf[frame_count - 1].push_back(tmp_encoder_velocity);
                }else
                {
                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
                }
                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            Cache_Ps[frame_count - 1] = Cache_Ps[frame_count];          //todo(znn)

            delete pre_integrations[WINDOW_SIZE];
            //todo(znn) add encoder
            if(ENCODER)
            {
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE], enc_v_0};
                encoder_velocity_buf[WINDOW_SIZE].clear();
            }else
            {
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            }
            
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

