#include "ygz/Frame.h"
#include "ygz/Settings.h"
#include "ygz/MapPoint.h"
#include "ygz/Feature.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace ygz {

    // static variables
    long unsigned int Frame::nNextId = 0;
    long unsigned int Frame::nNextKFId = 0;
    shared_ptr<ORBVocabulary> Frame::pORBvocabulary = nullptr;

    // copy constructor
    Frame::Frame(const Frame &frame)
            :
            mTimeStamp(frame.mTimeStamp), mpCam(frame.mpCam),
            mFeaturesLeft(frame.mFeaturesLeft), mFeaturesRight(frame.mFeaturesRight), mnId(frame.mnId),
            mpReferenceKF(frame.mpReferenceKF), mImLeft(frame.mImLeft), mImRight(frame.mImRight) {
        SetPose(SE3d(frame.mRwb, frame.mTwb));
        mGrid.resize(setting::FRAME_GRID_ROWS * setting::FRAME_GRID_COLS);
    }

    // normal constructor
    Frame::Frame(const cv::Mat &left, const cv::Mat &right, const double &timestamp, shared_ptr<CameraParam> cam,
                 const VecIMU &IMUSinceLastFrame)
            : mTimeStamp(timestamp), mImLeft(left.clone()), mImRight(right.clone()), mpCam(cam),
              mvIMUDataSinceLastFrame(IMUSinceLastFrame) {
        SetPose(SE3d());
        mnId = nNextId++;
        mGrid.resize(setting::FRAME_GRID_ROWS * setting::FRAME_GRID_COLS);
    }

    Frame::~Frame() {
        mFeaturesLeft.clear();
        mFeaturesRight.clear();
    }

    void Frame::SetPose(const SE3d &Twb) {
        unique_lock<mutex> lock(mMutexPose);
        mRwb = Twb.rotationMatrix();
        mTwb = Twb.translation();
        SE3d TWC = Twb * setting::TBC;
        SE3d TCW = TWC.inverse();
        mRcw = TCW.rotationMatrix();
        mtcw = TCW.translation();
        mOw = TWC.translation();
        mRwc = TWC.rotationMatrix();
    }

    void Frame::SetPoseTCW(const SE3d &Tcw) {
        unique_lock<mutex> lock(mMutexPose);
        mRcw = Tcw.rotationMatrix();
        mtcw = Tcw.translation();
        SE3d Twc = Tcw.inverse();
        mRwc = Twc.rotationMatrix();
        mOw = Twc.translation();
        SE3d Twb = Twc * setting::TBC.inverse();
        mRwb = Twb.rotationMatrix();
        mTwb = Twb.translation();
    }

    bool Frame::SetThisAsKeyFrame() {
        if (mbIsKeyFrame == true)
            return true;

        // 置关键帧
        mbIsKeyFrame = true;
        mnKFId = nNextKFId++;
        return true;
    }

    void Frame::ComputeIMUPreIntSinceLastFrame(const shared_ptr<Frame> pLastF, IMUPreIntegration &IMUPreInt) {
        // Reset pre-integrator first
        unique_lock<mutex> lock(mMutexPose);
        IMUPreInt.reset();

        const VecIMU &vIMUSInceLastFrame = mvIMUDataSinceLastFrame;

        Vector3d bg = pLastF->BiasG();
        Vector3d ba = pLastF->BiasA();

        // remember to consider the gap between the last KF and the first IMU
        {
            const IMUData &imu = vIMUSInceLastFrame.front();
            double dt = std::max(0., imu.mfTimeStamp - pLastF->mTimeStamp);
            IMUPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);
        }
        // integrate each imu
        for (size_t i = 0; i < vIMUSInceLastFrame.size(); i++) {
            const IMUData &imu = vIMUSInceLastFrame[i];
            double nextt;
            if (i == vIMUSInceLastFrame.size() - 1)
                nextt = mTimeStamp;         // last IMU, next is this KeyFrame
            else
                nextt = vIMUSInceLastFrame[i + 1].mfTimeStamp;  // regular condition, next is imu data

            // delta time
            double dt = std::max(0., nextt - imu.mfTimeStamp);
            // update pre-integrator
            IMUPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);
        }
    }

    void Frame::UpdatePoseFromPreintegration(const IMUPreIntegration &imupreint, const Vector3d &gw) {
        unique_lock<mutex> lock(mMutexPose);

        Matrix3d dR = imupreint.getDeltaR();
        Vector3d dP = imupreint.getDeltaP();
        Vector3d dV = imupreint.getDeltaV();
        double dt = imupreint.getDeltaTime();

        Vector3d Pwbpre = mTwb;     // 平移
        Matrix3d Rwbpre = mRwb.matrix();
        Vector3d Vwbpre = mSpeedAndBias.segment<3>(0);

        Matrix3d Rwb = Rwbpre * dR;
        Vector3d Pwb = Pwbpre + Vwbpre * dt + 0.5 * gw * dt * dt + Rwbpre * dP;
        Vector3d Vwb = Vwbpre + gw * dt + Rwbpre * dV;

        // Here assume that the pre-integration is re-computed after bias updated, so the bias term is ignored
        mRwb = Rwb;
        mTwb = Pwb;
        SE3d TWC = SE3d(Rwb, Pwb) * setting::TBC;
        SE3d TCW = TWC.inverse();
        mRcw = TCW.rotationMatrix();
        mtcw = TCW.translation();
        mOw = TWC.translation();
        mRwc = TWC.rotationMatrix();
        mSpeedAndBias.segment<3>(0) = Vwb;
    }

    vector<size_t> Frame::GetFeaturesInArea(
            const float &x, const float &y, const float &r, const int minLevel,
            const int maxLevel) {
        unique_lock<mutex> lock(mMutexFeature);
        vector<size_t> vIndices;
        vIndices.reserve(mFeaturesLeft.size());

        const int nMinCellX = max(0, (int) floor((x - r) * setting::GridElementWidthInv));
        if (nMinCellX >= setting::FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) setting::FRAME_GRID_COLS - 1,
                                  (int) ceil((x + r) * setting::GridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - r) * setting::GridElementHeightInv));
        if (nMinCellY >= setting::FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) setting::FRAME_GRID_ROWS - 1,
                                  (int) ceil((y + r) * setting::GridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[iy * setting::FRAME_GRID_COLS + ix];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {

                    const shared_ptr<Feature> feature = mFeaturesLeft[vCell[j]];
                    if (bCheckLevels) {
                        if (int(feature->mLevel) < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (int(feature->mLevel) > maxLevel)
                                continue;
                    }

                    const float distx = feature->mPixel[0] - x;
                    const float disty = feature->mPixel[1] - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }
        return vIndices;
    }

    void Frame::AssignFeaturesToGrid() {
        if (mFeaturesLeft.empty())
            return;

        for (auto g: mGrid)
            g.clear();

        unique_lock<mutex> lock(mMutexFeature);
        for (size_t i = 0; i < mFeaturesLeft.size(); i++) {
            shared_ptr<Feature> f = mFeaturesLeft[i];
            if (f == nullptr)
                continue;
            int nGridPosX, nGridPosY;
            if (PosInGrid(f, nGridPosX, nGridPosY)) {
                mGrid[nGridPosX + nGridPosY * setting::FRAME_GRID_COLS].push_back(i);
            }
        }
    }

    bool Frame::isInFrustum(shared_ptr<MapPoint> pMP, float viewingCosLimit, int boarder) {

        // 3D in absolute coordinates
        Vector3d P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Vector3d Pc = mRcw * P + mtcw;
        const float &PcX = Pc[0];
        const float &PcY = Pc[1];
        const float &PcZ = Pc[2];

        // Check valid depth
        if (PcZ < setting::minPointDis || PcZ > setting::maxPointDis) {
            return false;
        }

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = mpCam->fx * PcX * invz + mpCam->cx;
        const float v = mpCam->fy * PcY * invz + mpCam->cy;

        if (u < boarder || u > (setting::imageWidth - boarder))
            return false;
        if (v < boarder || v > (setting::imageHeight - boarder))
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const Vector3d PO = P - mOw;
        const float dist = PO.norm();

        // Check viewing angle
        Vector3d Pn = pMP->GetNormal();
        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit) {
            return false;
        }

        // Data used by the tracking
        pMP->mTrackProjX = u;
        pMP->mTrackProjY = v;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    void Frame::ComputeBoW() {
        if (pORBvocabulary && mBowVec.empty() && !mFeaturesLeft.empty()) {
            vector<cv::Mat> vCurrentDesc = this->GetAllDescriptor();
            pORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    vector<Mat> Frame::GetAllDescriptor() {
        unique_lock<mutex> lock(mMutexFeature);
        vector<Mat> ret;
        ret.reserve(mFeaturesLeft.size());
        for (size_t i = 0; i < mFeaturesLeft.size(); i++) {
            ret.push_back(cv::Mat(1, 32, CV_8UC1, mFeaturesLeft[i]->mDesc));
        }
        return ret;
    }

    Vector3d Frame::UnprojectStereo(const int &i) {
        unique_lock<mutex> lock(mMutexFeature);

        const float z = 1.0 / mFeaturesLeft[i]->mfInvDepth;
        if (z > 0) {
            const float u = mFeaturesLeft[i]->mPixel[0];
            const float v = mFeaturesLeft[i]->mPixel[1];
            const float x = (u - mpCam->cx) * z * mpCam->fxinv;
            const float y = (v - mpCam->cy) * z * mpCam->fyinv;
            Vector3d x3Dc(x, y, z);
            return mRwc * x3Dc + mOw;
        } else
            return Vector3d(0, 0, 0);
    }

    void Frame::ComputeImagePyramid() {

        mPyramidLeft.resize(setting::numPyramid);
        mPyramidRight.resize(setting::numPyramid);

        for (size_t level = 0; level < setting::numPyramid; ++level) {
            float scale = setting::invScaleFactors[level];
            Size sz(cvRound((float) mImLeft.cols * scale), cvRound((float) mImLeft.rows * scale));
            Size wholeSize(sz.width + setting::EDGE_THRESHOLD * 2, sz.height + setting::EDGE_THRESHOLD * 2);

            Mat tempL(wholeSize, mImLeft.type()), masktempL;
            Mat tempR(wholeSize, mImRight.type()), masktempR;

            mPyramidLeft[level] = tempL(Rect(setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD, sz.width, sz.height));

            mPyramidRight[level] = tempR(Rect(setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if (level != 0) {
                resize(mPyramidLeft[level - 1], mPyramidLeft[level], sz, 0, 0, INTER_LINEAR);

                resize(mPyramidRight[level - 1], mPyramidRight[level], sz, 0, 0, INTER_LINEAR);

                copyMakeBorder(mPyramidLeft[level], tempL, setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               BORDER_REFLECT_101 + BORDER_ISOLATED);

                copyMakeBorder(mPyramidRight[level], tempR, setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               BORDER_REFLECT_101 + BORDER_ISOLATED);
            } else {
                copyMakeBorder(mImLeft, tempL, setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD, BORDER_REFLECT_101);
                copyMakeBorder(mImRight, tempR, setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD, BORDER_REFLECT_101);
            }

        }
    }

    bool Frame::PosInGrid(const shared_ptr<Feature> feature, int &posX, int &posY) {
        posX = int(feature->mPixel[0] * setting::GridElementWidthInv);
        posY = int(feature->mPixel[1] * setting::GridElementHeightInv);
        if (posX < 0 || posX >= setting::FRAME_GRID_COLS
            || posY < 0 || posY >= setting::FRAME_GRID_ROWS)
            return false;
        return true;
    }

    double Frame::ComputeSceneMedianDepth(const int &q) {
        unique_lock<mutex> lock(mMutexFeature);
        vector<double> vDepth;
        for (auto feat: mFeaturesLeft) {
            if (feat && feat->mfInvDepth > 0)
                vDepth.push_back(1.0 / feat->mfInvDepth);
        }

        if (vDepth.empty())
            return 0;

        sort(vDepth.begin(), vDepth.end());
        return vDepth[(vDepth.size() - 1) / q];
    }

    int Frame::TrackedMapPoints(const int &minObs) {
        int nPoints = 0;
        const bool bCheckObs = minObs > 0;
        unique_lock<mutex> lock(mMutexFeature);
        int N = mFeaturesLeft.size();
        for (int i = 0; i < N; i++) {
            if (mFeaturesLeft[i] == nullptr)
                continue;
            shared_ptr<MapPoint> pMP = mFeaturesLeft[i]->mpPoint;
            if (pMP) {
                if (!pMP->isBad()) {
                    if (bCheckObs) {
                        if (pMP->Observations() >= minObs)
                            nPoints++;
                    } else
                        nPoints++;
                }
            }
        }

        return nPoints;
    }

} //namespace ygz


