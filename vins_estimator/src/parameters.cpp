#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;
double ENC_N;           //encoder noise

double VEL_N_wheel;     //wheel  linear_velocity noise
double GYR_N_wheel;     //wheel  angular_velocity noise 
double SX;              //wheel  x_scale fix
double SY;              //wheel  y_scale fix
double SW;              //wheel  w_scale fix
Eigen::Matrix3d RIO;      //R from encoder to body
Eigen::Vector3d TIO;      //T from encoder to body

double ROLL_N, PITCH_N ,ZPW_N;                  //plane
double ROLL_N_INV, PITCH_N_INV, ZPW_N_INV;      //plane


std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;

int ESTIMATE_EXTRINSIC_WHEEL;       //wheter to estimate extrinsic of wheel  
int ESTIMATE_INTRINSIC_WHEEL;       //wheter to estimate intrinsic of wheel  
int ESTIMATE_TD_WHEEL;              //wheter to estimate td of wheel  

std::string VINS_RESULT_PATH;
std::string GRONUD_TRUTH_PATH;
std::string EX_CALIB_RESULT_PATH;           //wheel
std::string IN_CALIB_RESULT_PATH;           //wheel
std::string INTRINSIC_ITERATE_PATH;         //wheel

std::string IMU_TOPIC;
std::string ODOM_TOPIC;
std::string TF_TOPIC;

std::string ENCODER_TOPIC;                          //wheel
std::string EXTRINSIC_WHEEL_ITERATE_PATH;           //wheel extrinsic para


double ROW, COL;
double DOWN_SAMPLE;     //TODO inorder to fit mower
double TD, TR;

int ENCODER;                //use encoder or not
double LEFT_D, RIGHT_D;     //diameter of left, right wheel
double ENC_RESOLUTION;      //resulution of encoder
double WHEELBASE;           //distance between two wheels

int USE_WHEEL;              //use wheel or not
int USE_PLANE;              //use plane or not 
int ONLY_INITIAL_WITH_WHEEL;    //initial with wheel  esimator.2

WheelExtrinsicAdjustType WHEEL_EXT_ADJ_TYPE;
template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["odom_topic"] >> ODOM_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    fsSettings["tf_topic"] >> TF_TOPIC;
    fsSettings["encoder_topic"] >> ENCODER_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.txt";
    GRONUD_TRUTH_PATH = OUTPUT_PATH + "/ground_truth_from_vrtk.txt";
    
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;

    // create folder if not exists
    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());

    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    std::ofstream fout_gd(GRONUD_TRUTH_PATH, std::ios::out);
    fout_gd.close();

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ENC_N = fsSettings["enc_n"];        //encoder noise
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    DOWN_SAMPLE = fsSettings["down_sample"];
    ROW = ROW / DOWN_SAMPLE;
    COL = COL / DOWN_SAMPLE;
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    ENCODER = fsSettings["is_encoder"]; // whether to use encoder
    ENC_RESOLUTION = fsSettings["encoder_resolution"]; // encode resolution
    LEFT_D = fsSettings["left_wheel_diameter"];       //left_wheel_diameter
    RIGHT_D = fsSettings["right_wheel_diameter"];     //right_wheel_diameter
    WHEELBASE = fsSettings["wheelbase"];                //wheelbase

    USE_WHEEL = fsSettings["wheel"];
    ROS_WARN("USE_WHEEL: %d", USE_WHEEL);

    USE_PLANE = fsSettings["plane"];
    ROS_WARN("USE_PLAENE: %d", USE_PLANE);

    ONLY_INITIAL_WITH_WHEEL = fsSettings["noly_initial_with_wheel"];
    ROS_WARN("ONLY_INITIAL_WITH_WHEEL: %d", ONLY_INITIAL_WITH_WHEEL);

    if(USE_WHEEL)
    {
        VEL_N_wheel = fsSettings["wheel_velocity_noise_sigma"];
        GYR_N_wheel = fsSettings["wheel_gyro_noise_sigma"];
        SX = static_cast<double>(fsSettings["sx"]);
        SY = static_cast<double>(fsSettings["sy"]);
        SW = static_cast<double>(fsSettings["sw"]);
        ESTIMATE_EXTRINSIC_WHEEL = fsSettings["estimate_wheel_extrinsic"];
        if(ESTIMATE_EXTRINSIC_WHEEL == 2)
        {
            ROS_WARN("have no prior about wheel extrinsic param, calibrate extrinsic param");
            RIO = Eigen::Matrix3d::Identity();
            TIO = Eigen::Vector3d::Zero();
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }else
        {
            if (ESTIMATE_EXTRINSIC_WHEEL == 1)
            {
                ROS_WARN(" Optimize wheel extrinsic param around initial guess!");
                EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
            }
            if (ESTIMATE_EXTRINSIC_WHEEL == 0)
                ROS_WARN(" fix extrinsic param ");

            cv::Mat cv_T;
            fsSettings["body_T_wheel"] >> cv_T;
            Eigen::Matrix4d T;
            cv::cv2eigen(cv_T, T);
            RIO = T.block<3, 3>(0, 0);
            TIO = T.block<3, 1>(0, 3);
            //normalization
            Eigen::Quaterniond QIO(RIO);
            QIO.normalize();
            RIO = QIO.toRotationMatrix();
//            RIO = RIO.eval() * Utility::ypr2R(Eigen::Vector3d(-10, 10, -10));
        }

        if(ESTIMATE_EXTRINSIC_WHEEL){
            EXTRINSIC_WHEEL_ITERATE_PATH = OUTPUT_PATH + "/extrinsic_iterate_wheel.csv";
            std::ofstream fout(EXTRINSIC_WHEEL_ITERATE_PATH, std::ios::out);
            fout.close();
        }
        if(ESTIMATE_EXTRINSIC_WHEEL)
        {
            int extrinsic_type = static_cast<int>(fsSettings["extrinsic_type_wheel"]);
            switch(extrinsic_type){
                case 0:
                    WHEEL_EXT_ADJ_TYPE = WheelExtrinsicAdjustType::ADJUST_WHEEL_ALL;
                    ROS_INFO("adjust translation and rotation of cam extrinsic");
                    break;
                case 1:
                    WHEEL_EXT_ADJ_TYPE = WheelExtrinsicAdjustType::ADJUST_WHEEL_TRANSLATION;
                    ROS_INFO("adjust only translation of cam extrinsic");
                    break;
                case 2:
                    WHEEL_EXT_ADJ_TYPE = WheelExtrinsicAdjustType::ADJUST_WHEEL_ROTATION;
                    ROS_INFO("adjust only rotation of cam extrinsic");
                    break;
                case 3:
                    WHEEL_EXT_ADJ_TYPE = WheelExtrinsicAdjustType::ADJUST_WHEEL_NO_Z;
                    ROS_INFO("adjust without Z of translation of wheel extrinsic");
                    break;
                case 4:
                    WHEEL_EXT_ADJ_TYPE = WheelExtrinsicAdjustType::ADJUST_WHEEL_NO_ROTATION_NO_Z;
                    ROS_INFO("adjust without rotation and Z of translation of wheel extrinsic");
                    break;
                default:
                    ROS_WARN("the extrinsic type range from 0 to 4");
            }

        }

        ESTIMATE_INTRINSIC_WHEEL = static_cast<int>(fsSettings["estimate_wheel_intrinsic"]);
        if(ESTIMATE_INTRINSIC_WHEEL == 2){
            ROS_WARN("have no prior about wheel intrinsic param, calibrate intrinsic param");
            IN_CALIB_RESULT_PATH = OUTPUT_PATH + "/intrinsic_parameter.csv";
            INTRINSIC_ITERATE_PATH = OUTPUT_PATH + "/intrinsic_iterate.csv";
            std::ofstream fout(INTRINSIC_ITERATE_PATH, std::ios::out);
            fout.close();
        }else{
            if (ESTIMATE_INTRINSIC_WHEEL == 1)
            {
                ROS_WARN(" Optimize wheel intrinsic param around initial guess!");
                IN_CALIB_RESULT_PATH = OUTPUT_PATH + "/intrinsic_parameter.csv";
                INTRINSIC_ITERATE_PATH = OUTPUT_PATH + "/intrinsic_iterate.csv";
                std::ofstream fout(INTRINSIC_ITERATE_PATH, std::ios::out);
                fout.close();
            }
            if (ESTIMATE_INTRINSIC_WHEEL == 0)
                ROS_WARN(" fix intrinsic param ");
        }

    }

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());

        //todo(znn) extrinsic from wheel to body
        fsSettings["extrinsicRotation_io"] >> cv_R;
        fsSettings["extrinsicTranslation_io"] >> cv_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Qio(eigen_R);
        eigen_R = Qio.normalized();
        RIO = eigen_R;
        TIO = eigen_T;
        ROS_INFO_STREAM("Extrinsic_Rio : " << std::endl << RIO);
        ROS_INFO_STREAM("Extrinsic_Tic : " << std::endl << TIO.transpose());
        
    } 

    if(USE_PLANE)
    {
        ROLL_N = static_cast<double>(fsSettings["roll_n"]);
        PITCH_N = static_cast<double>(fsSettings["pitch_n"]);
        ZPW_N = static_cast<double>(fsSettings["zpw_n"]);
        ROLL_N_INV = 1.0 / ROLL_N;
        PITCH_N_INV = 1.0 / PITCH_N;
        ZPW_N_INV = 1.0 / ZPW_N;
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }
    
    fsSettings.release();
}
