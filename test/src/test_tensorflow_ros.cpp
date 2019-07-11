/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ********************
 *  v1.0: drwnz (david.wong@tier4.jp)
 *
 * test_tensorflow_ros.cpp
 *
 *  Created on: May 16th 2019
 */

#include <tensorflow_ros/tensorflow_ros.hpp>
#include <gtest/gtest.h>
#include <ros/ros.h>

class RosTensorFlowInstantiationTest : public ::testing::Test
{
protected:
  RosTensorFlowInstantiationTest()
  {
    std::string test_model_file;
    private_nh_.getParam("test_model_file", filename_);
  }
  ~RosTensorFlowInstantiationTest()
  {
  }
  ros::NodeHandle private_nh_;
  std::string filename_;
};

class RosTensorFlowOperationTest : public RosTensorFlowInstantiationTest
{
protected:
  RosTensorFlowOperationTest()
  {
    tf_session = new tensorflow_ros::TensorFlowSession(filename_.c_str(), "inputs", "predictions");
  }
  ~RosTensorFlowOperationTest()
  {
    delete tf_session;
  }

  tensorflow_ros::TensorFlowSession* tf_session;
};

TEST_F(RosTensorFlowInstantiationTest, loadModel)
{
  try
  {
    tensorflow_ros::TensorFlowSession test_tf_session(filename_.c_str(), "inputs", "predictions");
  }
  catch (std::runtime_error& e)
  {
    FAIL() << "Model loading problem, exception thrown: " << e.what();
  }
  catch (...)
  {
    FAIL() << "Model loading problem, exception thrown";
  }
}

TEST_F(RosTensorFlowInstantiationTest, loadModelIncorrectly)
{
  try
  {
    tensorflow_ros::TensorFlowSession test_tf_session("no_model.pb", "inputs", "predictions");
    FAIL() << "Model loading problem, expected std::runtime_error when incorrect graph file provided";
  }
  catch (std::runtime_error& e)
  {
    EXPECT_STREQ(e.what(), "Unable to start TensorFlow session.")
        << "Model loading problem, incorrect std::runtime_error when incorrect graph file provided";
  }
  catch (...)
  {
    ADD_FAILURE() << "Model loading problem, std::runtime_error when incorrect graph file provided";
  }

  try
  {
    tensorflow_ros::TensorFlowSession test_tf_session(filename_.c_str(), "wrong_inputs", "predictions");
    FAIL() << "Model loading problem, expected std::runtime_error when incorrect input operation provided";
  }
  catch (std::runtime_error& e)
  {
    EXPECT_STREQ(e.what(), "Unable to start TensorFlow session.")
        << "Model loading problem, expected different std::runtime_error when incorrect input operation provided";
  }
  catch (...)
  {
    ADD_FAILURE() << "Model loading problem, expected std::runtime_error when incorrect input operation provided";
  }

  try
  {
    tensorflow_ros::TensorFlowSession test_tf_session(filename_.c_str(), "inputs", "wrong_predictions");
    FAIL() << "Model loading problem, expected std::runtime_error when incorrect output operation provided";
  }
  catch (std::runtime_error& e)
  {
    EXPECT_STREQ(e.what(), "Unable to start TensorFlow session.")
        << "Model loading problem, expected different std::runtime_error when incorrect output operation provided";
  }
  catch (...)
  {
    ADD_FAILURE() << "Model loading problem, expected std::runtime_error when incorrect output operation provided";
  }
}

TEST_F(RosTensorFlowOperationTest, runInference)
{
  /*
   * Inference is tested with a simple model that predicts whether the sum of a vecotr of 10 floats is greate than 50.
   * If the sum of the elements of the input vector is less than 50, it should return FALSE.
   * If the sum of the elements of the input vectoris  more than 50, it shoud return TRUE.
   * The model is trained on 10,000 random ten digit vectors with each element between 0 and 10.
   */

  const std::vector<std::int64_t> input_data_dimensions = { 1, 10 };
  const std::vector<float> input_data_1 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  const std::vector<float> input_data_2 = { 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0 };
  std::vector<std::vector<unsigned char>> output_data_1;
  std::vector<std::vector<unsigned char>> output_data_2;
  EXPECT_TRUE(tf_session->AddInputVector(input_data_1, input_data_dimensions));
  EXPECT_TRUE(tf_session->RunInference());
  tf_session->GetOutputVectors(output_data_1);
  EXPECT_FALSE(output_data_1.size() == 0);
  EXPECT_FALSE(output_data_1[0][0]);

  EXPECT_TRUE(tf_session->AddInputVector(input_data_2, input_data_dimensions));
  EXPECT_TRUE(tf_session->RunInference());
  tf_session->GetOutputVectors(output_data_2);
  EXPECT_FALSE(output_data_2.size() == 0);
  EXPECT_TRUE(output_data_2[0][0]);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "test_tensorflow_ros");
  return RUN_ALL_TESTS();
}
