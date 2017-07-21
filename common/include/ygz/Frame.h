#ifndef YGZ_FRAME_H
#define YGZ_FRAME_H

#include <opencv2/core/core.hpp>
#include <vector>

#include "ygz/NumTypes.h"
#include "ygz/Settings.h"
#include "ygz/Camera.h"
#include "ygz/IMUData.h"
#include "ygz/IMUPreIntegration.h"

using cv::Mat;
using namespace std;

namespace ygz {

    struct Feature;
    struct MapPoint;

    // 帧结构
    // The basic frame struct
    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // copy constructor
        Frame(const Frame &frame);

        /**
         * constructor from two images
         * @param left the left image
         * @param right the right image
         * @param timestamp the time stamp
         * @param cam the camera object
         * @param imuSinceLastFrame imu data since last frame
         */
        // 请注意左右图必须是已经校正过的
        Frame(
                const cv::Mat &left, const cv::Mat &right,
                const double &timestamp,
                shared_ptr<CameraParam> cam,
                const VecIMU &imuSinceLastFrame);

        /**
         * default constructor
         */
        Frame() {
            mnId = nNextId++;
        }

        virtual ~Frame();

        // 计算图像金字塔
        // compute the image pyramid
        void ComputeImagePyramid();

        /**
         * 计算场景场景的深度中位数
         * @param q 百分比
         */
        double ComputeSceneMedianDepth(const int &q);

        // set the pose of this frame and update the inner variables, give Twb
        // 设置位姿，并更新Rcw, tcw等一系列量
        void SetPose(const SE3d &Twb);

        // 从TCW来实现SetPose
        void SetPoseTCW(const SE3d &Tcw);

        // 获得位姿 Twb
        inline SE3d GetPose() {
            unique_lock<mutex> lock(mMutexPose);
            return SE3d(mRwb, mTwb);
        }

        // get Tcw
        SE3d GetTCW() {
            unique_lock<mutex> lock(mMutexPose);
            return SE3d(mRcw, mtcw);
        }

        // Check if a MapPoint is in the frustum of the camera
        // and fill variables of the MapPoint to be used by the tracking
        /**
         * 判断路标点是否在视野中
         * 会检测像素坐标、距离和视线角
         * @param pMP   被检测的地图点
         * @param viewingCosLimit   视线夹角cos值，如0.5为60度，即视线夹角大于60度时判定为负
         * @param boarder   图像边界
         * @return
         */
        bool isInFrustum(shared_ptr<MapPoint> pMP, float viewingCosLimit, int boarder = 20);

        /** 计算一个特征是否在grid里
         * @param feature 需要计算的特征
         * @param posX 若在某网格内，返回该网格的x坐标
         * @param posY 若在某网格内，返回该网格的y坐标
         * @return
         */
        bool PosInGrid(const shared_ptr<Feature> feature, int &posX, int &posY);

        /** 获取一定区域内的地图点
         * @param x 查找点的x坐标
         * @param y 查找点的y坐标
         * @param r 半径
         * @param minLevel 最小层数
         * @param maxLevel 最大层数
         * @return
         */
        vector<size_t> GetFeaturesInArea(
                const float &x, const float &y, const float &r, const int minLevel = -1,
                const int maxLevel = -1);

        /**
         * 将第i个特征投影到世界坐标
         * @param i 特征点的下标
         * @return
         */
        Vector3d UnprojectStereo(const int &i);

        /**
         * 计算某个指定帧过来的预积分
         * @param pLastF 上一个帧（需要知道位移和速度）
         * @param imupreint 预积分器
         */
        void ComputeIMUPreIntSinceLastFrame(shared_ptr<Frame> pLastF, IMUPreIntegration &imupreint);

        /**
         * compute the imu preintegration from last kf
         * 计算从参考帧过来的预积分
         * 结果存储在自带的预积分器中
         */
        inline void ComputeIMUPreInt() {
            if (mpReferenceKF.expired() == false)
                ComputeIMUPreIntSinceLastFrame(mpReferenceKF.lock(), mIMUPreInt);
        }

        // update the navigation status using preintegration
        /**
         * @param imupreint 已经积分过的预积分器
         * @param gw 重力向量
         */
        void UpdatePoseFromPreintegration(const IMUPreIntegration &imupreint, const Vector3d &gw);

        // Assign keypoints to the grid for speed up feature matching (called in the constructor).
        // 将特征点分配到网格中
        void AssignFeaturesToGrid();

        // 获取所有特征点的描述子
        vector<Mat> GetAllDescriptor();

        // 判定是否关键帧
        bool IsKeyFrame() const { return mbIsKeyFrame; }

        // 将当前帧设成关键帧
        bool SetThisAsKeyFrame();

        // 计算词袋
        void ComputeBoW();

        // count the well tracked map points in this frame
        int TrackedMapPoints(const int &minObs);

        // coordinate transform: world, camera, pixel
        inline Vector3d World2Camera(const Vector3d &p_w, const SE3d &T_c_w) const {
            return T_c_w * p_w;
        }

        inline Vector2d World2Camera2(const Vector3d &p_w, const SE3d &T_c_w) const {
            Vector3d v = T_c_w * p_w;
            return Vector2d(v[0] / v[2], v[1] / v[2]);
        }

        inline Vector3d Camera2World(const Vector3d &p_c, const SE3d &T_c_w) const {
            return T_c_w.inverse() * p_c;
        }

        inline Vector2d Camera2Pixel(const Vector3d &p_c) const {
            return Vector2d(
                    mpCam->fx * p_c(0, 0) / p_c(2, 0) + mpCam->cx,
                    mpCam->fy * p_c(1, 0) / p_c(2, 0) + mpCam->cy
            );
        }

        inline Vector3d Pixel2Camera(const Vector2d &p_p, float depth = 1) const {
            return Vector3d(
                    (p_p(0, 0) - mpCam->cx) * depth / mpCam->fx,
                    (p_p(1, 0) - mpCam->cy) * depth / mpCam->fy,
                    depth
            );
        }

        inline Vector2d Pixel2Camera2(const Vector2f &p_p, float depth = 1) const {
            return Vector2d(
                    (p_p[0] - mpCam->cx) * depth / mpCam->fx,
                    (p_p[1] - mpCam->cy) * depth / mpCam->fy
            );
        }

        inline Vector3d Pixel2World(const Vector2d &p_p, const SE3d &TCW, float depth = 1) const {
            return Camera2World(Pixel2Camera(p_p, depth), TCW);
        }

        inline Vector2d World2Pixel(const Vector3d &p_w, const SE3d &TCW) const {
            return Camera2Pixel(World2Camera(p_w, TCW));
        }

        // accessors
        inline Vector3d Speed() {
            unique_lock<mutex> lock(mMutexPose);
            return mSpeedAndBias.segment<3>(0);
        }

        inline Vector3d BiasG() {
            unique_lock<mutex> lock(mMutexPose);
            return mSpeedAndBias.segment<3>(3);
        }

        inline Vector3d BiasA() {
            unique_lock<mutex> lock(mMutexPose);
            return mSpeedAndBias.segment<3>(6);
        }

        inline Matrix3d Rwb() {
            unique_lock<mutex> lock(mMutexPose);
            return mRwb.matrix();
        }

        inline Vector3d Twb() {
            unique_lock<mutex> lock(mMutexPose);
            return mTwb;
        }

        inline Matrix3d Rwc() {
            unique_lock<mutex> lock(mMutexPose);
            return mRwc;
        }

        inline Matrix3d Rcw() {
            unique_lock<mutex> lock(mMutexPose);
            return mRcw;
        }

        inline Vector3d Tcw() {
            unique_lock<mutex> lock(mMutexPose);
            return mtcw;
        }

        inline Vector3d Ow() {
            unique_lock<mutex> lock(mMutexPose);
            return mOw;
        }

        inline void SetSpeedBias(
                const Vector3d &speed, const Vector3d &biasg, const Vector3d &biasa) {
            unique_lock<mutex> lock(mMutexPose);
            mSpeedAndBias.segment<3>(0) = speed;
            mSpeedAndBias.segment<3>(3) = biasg;
            mSpeedAndBias.segment<3>(6) = biasa;
        }

        inline void SetSpeed(const Vector3d &speed) {
            unique_lock<mutex> lock(mMutexPose);
            mSpeedAndBias.segment<3>(0) = speed;
        }

        inline void SetBiasG(const Vector3d &biasg) {
            unique_lock<mutex> lock(mMutexPose);
            mSpeedAndBias.segment<3>(3) = biasg;
        }

        inline void SetBiasA(const Vector3d &biasa) {
            unique_lock<mutex> lock(mMutexPose);
            mSpeedAndBias.segment<3>(6) = biasa;
        }

        // 取得P+R的6维向量
        inline Vector6d PR() {
            unique_lock<mutex> lock(mMutexPose);
            Vector6d pr;
            pr.head<3>() = mTwb;
            pr.tail<3>() = mRwb.log();
            return pr;
        }

        const IMUPreIntegration &GetIMUPreInt() { return mIMUPreInt; }

        // ---------------------------------------------------------------
        // Data
        // time stamp
        double mTimeStamp = 0;

        // left and right image
        Mat mImLeft, mImRight;    // 左/右图像，显示用
        shared_ptr<CameraParam> mpCam = nullptr; // 相机参数

        // features of left and right, basically we use the left ones.
        std::mutex mMutexFeature;       // 特征锁
        std::vector<shared_ptr<Feature>> mFeaturesLeft;    // 左眼特征点容器
        std::vector<shared_ptr<Feature>> mFeaturesRight;   // 右眼特征点容器

        // 左右目的金字塔
        // left and right pyramid
        std::vector<cv::Mat> mPyramidLeft;
        std::vector<cv::Mat> mPyramidRight;

        long unsigned int mnId = 0;             // id
        static long unsigned int nNextId;       // next id
        long unsigned int mnKFId = 0;           // keyframe id
        static long unsigned int nNextKFId;     // next keyframe id

        // pose, speed and bias
        // 这些都是状态量，优化时候的中间量给我放到优化相关的struct里去
        std::mutex mMutexPose;            // lock the pose related variables
        SO3d mRwb;  // body rotation
        Vector3d mTwb = Vector3d(0, 0, 0); // body translation
        SpeedAndBias mSpeedAndBias = Vector9d::Zero(); // V and bias
        // 位姿相关的中间量
        Matrix3d mRcw = Matrix3d::Identity();   ///< Rotation from world to camera
        Vector3d mtcw = Vector3d::Zero();       ///< Translation from world to camera
        Matrix3d mRwc = Matrix3d::Identity();   ///< Rotation from camera to world
        Vector3d mOw = Vector3d::Zero();        ///< =mtwc,Translation from camera to world

        // 参考的关键帧
        weak_ptr<Frame> mpReferenceKF;    // the reference keyframe

        // 是否关键帧
        bool mbIsKeyFrame = false;

        // Bag of Words Vector structures.
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // 从上一个帧到这里的IMU
        VecIMU mvIMUDataSinceLastFrame;         // 也可能是上一个关键帧
        IMUPreIntegration mIMUPreInt;           // 至上个关键帧的预积分器

        // 划分特征点的网格
        std::vector<std::vector<std::size_t>> mGrid;

        // 词典
        static shared_ptr<ORBVocabulary> pORBvocabulary;

        // ----------------------------------------
        // Debug stuffs
        // 设置Ground truth的位姿，debug用
        void SetPoseGT(const SE3d &TwbGT) {
            mTwbGT = TwbGT;
            mbHaveGT = true;
        }

        SE3d GetPoseGT() const {
            if (mbHaveGT)
                return mTwbGT;
            return SE3d();
        }

        SE3d mTwbGT;            // Ground truth pose
        bool mbHaveGT = false;  // 有没有Ground Truth的位姿
    };

}

#endif
