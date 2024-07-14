#include <boost/algorithm/string.hpp>

#include <gtsam/geometry/Pose2.h>

#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam/nonlinear/Marginals.h>

#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ExpressionFactor.h>

#include <gtsam/nonlinear/expressions.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

using namespace gtsam;
using namespace std;

namespace
{
  const double kGravityAcc = 9.81;
  const double kRearTrackWidth = 1.545;
  const double kWheelBase = 2.778;

  enum class LocMsgType {
    None,
    Imu,
    Icp,
    Odo
  };

  struct LocMsgBase {
    LocMsgBase() {}
    LocMsgBase(LocMsgType arg_type): type(arg_type) {}

    double ts;
    LocMsgType type = LocMsgType::None;

    virtual double getTs() const { return ts; }
    LocMsgType getType() const { return type; }
    virtual ~LocMsgBase() {}
  };

  struct LocImuMsg: LocMsgBase {
    LocImuMsg(): LocMsgBase(LocMsgType::Imu) {}

    Vector3 linear_acc;
    Vector3 ang_vel;
  };

  struct LocOdoMsg: LocMsgBase {
    LocOdoMsg(): LocMsgBase(LocMsgType::Odo) {}

    double fl, fr, rl, rr;
  };

  struct LocIcpMsg: LocMsgBase {
    LocIcpMsg(): LocMsgBase(LocMsgType::Icp) {}

    Vector3 xyz;
    Vector3 rpy;
  };

  std::vector<std::shared_ptr<LocMsgBase>> parseImuData(const std::string& fname) {
    std::vector<std::shared_ptr<LocMsgBase>> ret;

    std::fstream fd(fname.c_str(), std::ios_base::in);

    std::string line;
    std::getline(fd, line);

    while (std::getline(fd, line)) {
      vector<string> strs;
      boost::split(strs, line, boost::is_any_of(","));

      auto imu_msg = std::make_shared<LocImuMsg>();
      imu_msg->ts = stod(strs[0]);
      imu_msg->linear_acc = Vector3(stod(strs[1]), stod(strs[2]), stod(strs[3]));
      imu_msg->ang_vel = Vector3(stod(strs[4]), stod(strs[5]), stod(strs[6]));

      ret.push_back(imu_msg);
    }

    return ret;
  }

  std::vector<std::shared_ptr<LocMsgBase>> parseIcpData(const std::string& fname) {
    std::vector<std::shared_ptr<LocMsgBase>> ret;

    std::fstream fd(fname.c_str(), std::ios_base::in);

    std::string line;
    std::getline(fd, line);

    while (std::getline(fd, line)) {
      vector<string> strs;
      boost::split(strs, line, boost::is_any_of(","));

      auto icp_msg = std::make_shared<LocIcpMsg>();
      icp_msg->ts = stod(strs[0]);
      icp_msg->xyz = Vector3(stod(strs[1]), stod(strs[2]), stod(strs[3]));
      icp_msg->rpy = Vector3(stod(strs[4]), stod(strs[5]), stod(strs[6]));
      ret.push_back(icp_msg);
    }

    return ret;
  }  
}


using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::C;


std::shared_ptr<PreintegratedCombinedMeasurements::Params> getImuParams() {
  // We use the sensor specs to build the noise model for the IMU factor.
  double accel_noise_sigma = 0.00048333333333333334;
  double gyro_noise_sigma = 8.726646259971647e-05;
  double accel_bias_rw_sigma = 0.004905;
  double gyro_bias_rw_sigma = 0.000001454441043;
  Matrix33 measured_acc_cov = I_3x3 * pow(accel_noise_sigma, 2);
  Matrix33 measured_omega_cov = I_3x3 * pow(gyro_noise_sigma, 2);
  Matrix33 integration_error_cov =
      I_3x3 * 1e-14;  // error committed in integrating position from velocities
  Matrix33 bias_acc_cov = I_3x3 * pow(accel_bias_rw_sigma, 2);
  Matrix33 bias_omega_cov = I_3x3 * pow(gyro_bias_rw_sigma, 2);
  Matrix66 bias_acc_omega_init =
      I_6x6 * 1e-5;  // error in the bias used for preintegration

  auto p = PreintegratedCombinedMeasurements::Params::MakeSharedU(kGravityAcc);
  // PreintegrationBase params:
  p->accelerometerCovariance =
      measured_acc_cov;  // acc white noise in continuous
  p->integrationCovariance =
      integration_error_cov;  // integration uncertainty continuous
  // should be using 2nd order integration
  // PreintegratedRotation params:
  p->gyroscopeCovariance =
      measured_omega_cov;  // gyro white noise in continuous
  // PreintegrationCombinedMeasurements params:
  p->biasAccCovariance = bias_acc_cov;      // acc bias in continuous
  p->biasOmegaCovariance = bias_omega_cov;  // gyro bias in continuous
  p->biasAccOmegaInt = bias_acc_omega_init;

  return p;
}


int main(int argc, char** argv) {

  auto imu_params = getImuParams();

  auto imu_msgs = parseImuData("imu.txt");
  assert(imu_msgs.size() > 0);
  auto icp_msgs = parseIcpData("icp.txt");
  assert(icp_msgs.size() > 0);

  cerr << "imu_msgs: " << imu_msgs.size() << endl;
  cerr << "icp_msgs: " << icp_msgs.size() << endl;

  auto tot_msgs = imu_msgs;
  std::copy(icp_msgs.begin(), icp_msgs.end(), std::back_inserter(tot_msgs));
  std::sort(tot_msgs.begin(), tot_msgs.end(), [&](const std::shared_ptr<LocMsgBase>& lhs, 
                                                  const std::shared_ptr<LocMsgBase>& rhs) 
  { return lhs->getTs() < rhs->getTs(); });

  std::shared_ptr<LocIcpMsg> init_icp_msg = std::dynamic_pointer_cast<LocIcpMsg>(icp_msgs[0]);

  auto noise_calib_model = noiseModel::Diagonal::Sigmas(
    (Vector6() << Vector3::Constant(0.005), Vector3::Constant(0.005))
        .finished());

  auto noise_model_icp = noiseModel::Diagonal::Sigmas(
    (Vector6() << Vector3::Constant(0.1), Vector3::Constant(0.1))
        .finished());

  auto noise_model_vel_icp = noiseModel::Diagonal::Sigmas(
    (Vector3() << Vector3::Constant(0.1)).finished());

  Pose3 prior_pose(Rot3::Ypr(init_icp_msg->rpy[2], init_icp_msg->rpy[1], init_icp_msg->rpy[0]), init_icp_msg->xyz);
  Vector3 prior_velocity = 10.0 * (std::dynamic_pointer_cast<LocIcpMsg>(icp_msgs[1])->xyz - std::dynamic_pointer_cast<LocIcpMsg>(icp_msgs[0])->xyz);
  imuBias::ConstantBias prior_imu_bias;

  std::shared_ptr<PreintegrationType> preintegrated =
      std::make_shared<PreintegratedCombinedMeasurements>(imu_params, prior_imu_bias);

  auto prev_bias = prior_imu_bias;
  NavState prev_state(prior_pose, prior_velocity);

  Pose3 prior_calib;

  Values initial_values;
  int correction_count = 0;
  initial_values.insert(X(correction_count), prior_pose);
  initial_values.insert(V(correction_count), prior_velocity);
  initial_values.insert(B(correction_count), prior_imu_bias);
  initial_values.insert(C(0), prior_calib);

  auto pose_noise_model = noiseModel::Diagonal::Sigmas(
      (Vector(6) << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5)
          .finished());
  auto velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.1);
  auto bias_noise_model = noiseModel::Isotropic::Sigma(6, 1e-3);  

  NonlinearFactorGraph nonlinear_graph;
  nonlinear_graph.addPrior(X(correction_count), prior_pose, pose_noise_model);
  nonlinear_graph.addPrior(V(correction_count), prior_velocity, velocity_noise_model);
  nonlinear_graph.addPrior(B(correction_count), prior_imu_bias, bias_noise_model);
  nonlinear_graph.addPrior(C(0), prior_calib, noise_calib_model);

  ++correction_count;

  double prev_imu_ts = -1;

  std::shared_ptr<LocIcpMsg> last_icp_msg;
  for (const auto& msg: tot_msgs) {
    if (msg->getType() == LocMsgType::Imu) {
      if (!last_icp_msg) continue;

      std::shared_ptr<LocImuMsg> imu_msg = std::dynamic_pointer_cast<LocImuMsg>(msg);

      Vector3 rpy = last_icp_msg->rpy;

      auto dt = prev_imu_ts < 0 ? 1e-5 : (imu_msg->ts - prev_imu_ts);

      preintegrated->integrateMeasurement(imu_msg->linear_acc, imu_msg->ang_vel, dt);

      prev_imu_ts = imu_msg->ts;

      continue;
    }

    if (msg->getType() == LocMsgType::Icp) {
      auto pre_last_icp_msg = last_icp_msg;
      last_icp_msg = std::dynamic_pointer_cast<LocIcpMsg>(msg);

      if (!pre_last_icp_msg) {
        continue;
      }

      auto preint_imu = dynamic_cast<const PreintegratedCombinedMeasurements&>(*preintegrated);
      CombinedImuFactor imu_factor(X(correction_count - 1), V(correction_count - 1),
                                   X(correction_count), V(correction_count),
                                   B(correction_count - 1),  B(correction_count),
                                   preint_imu);
      nonlinear_graph.add(imu_factor);

      auto prev_icp_pose = Pose3(Rot3::Ypr(pre_last_icp_msg->rpy[2], pre_last_icp_msg->rpy[1], pre_last_icp_msg->rpy[0]),
                                 pre_last_icp_msg->xyz);
      auto icp_pose = Pose3(Rot3::Ypr(last_icp_msg->rpy[2], last_icp_msg->rpy[1], last_icp_msg->rpy[0]), last_icp_msg->xyz);

      using Pose3_ = Expression<Pose3>;
      gtsam::Pose3 icp_movement = prev_icp_pose.between(icp_pose);
      Expression<Pose3> icp_movement_prediction = between(
          Pose3_(compose(Pose3_(X(correction_count - 1)), Pose3_(C(0)))),
          Pose3_(compose(Pose3_(X(correction_count)), Pose3_(C(0))))
        );
      nonlinear_graph.addExpressionFactor(noise_calib_model, icp_movement, icp_movement_prediction);

      NavState prop_state = preintegrated->predict(prev_state, prev_bias);
      
      initial_values.insert(X(correction_count), prop_state.pose());
      initial_values.insert(V(correction_count), prop_state.v());
      initial_values.insert(B(correction_count), prev_bias);

      const int kIntermediateOptFrequency = 100;
      if (correction_count % kIntermediateOptFrequency == 0) {
        LevenbergMarquardtOptimizer optimizer(nonlinear_graph, initial_values);
        auto result = optimizer.optimize();

        prev_state = NavState(result.at<Pose3>(X(correction_count)),
                              result.at<Vector3>(V(correction_count)));
        prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));

      } else {
        prev_state = prop_state;
        
        prev_bias = initial_values.at<imuBias::ConstantBias>(B(correction_count));
      }

      preintegrated->resetIntegrationAndSetBias(prev_bias);

      ++correction_count;

      continue;
    }
  }

  LevenbergMarquardtParams opt_params;
  opt_params.setVerbosityLM("SUMMARY");
  // opt_params.setVerbosity("ERROR");

  LevenbergMarquardtOptimizer optimizer(nonlinear_graph, initial_values, opt_params);
  Values result = optimizer.optimize();

  // std::cout << "ts,x,y,z" << std::endl;
  // for (int i = 0; i < correction_count; ++i) {
  //   auto pnt = result.at<Pose3>(X(i)).translation();

  //   std::cout << i << fixed << setprecision(10) << "," << pnt[0] << "," << pnt[1] << "," << pnt[2] << std::endl;
  // }

  std::cerr << "solver: error=" << optimizer.error() << " iterations=" << optimizer.iterations() << std::endl;

  auto result_calib = result.at<Pose3>(C(0));
  auto result_rot = result_calib.rotation();

  cerr << "Calibration:" << endl;
  cerr << "rotation: " << fixed << setprecision(10) << "r=" << result_rot.roll() << ", p=" << result_rot.pitch() << ", y=" << result_rot.yaw() << endl;
  cerr << "translation: " << fixed << setprecision(10) << "x=" << result_calib.x() << ", y=" << result_calib.y() << ", z=" << result_calib.z() << endl;

  return 0;
}
