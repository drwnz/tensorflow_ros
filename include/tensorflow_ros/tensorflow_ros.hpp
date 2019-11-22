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
 *  v0.1.0: drwnz (david.wong@tier4.jp)
 *
 * tensorflow_ros.hpp
 *
 *  Created on: February 6th 2019
 */

#ifndef TENSORFLOW_ROS_HPP
#define TENSORFLOW_ROS_HPP

#define __APP_NAME__ "tensorflow_ros"

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/c_api_experimental.h>
#include <cstdio>
#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>
#include <ros/ros.h>

namespace tensorflow_ros
{
void DeallocateBuffer(void* data, size_t);

/** TensorFlowSession
*
* A class for creating a TensorFlow session from a frozen graph file, including:
* 1. Adding input data in std::vector format
* 2. Running inference on input data
* 3. Output data in std::vector format
*
*/
class TensorFlowSession
{
private:
  TF_Graph* graph_;
  TF_Status* status_;
  std::vector<TF_Tensor*> input_tensors_;
  std::vector<TF_Tensor*> output_tensors_;
  std::vector<TF_Output> input_operation_;
  std::vector<TF_Output> output_operation_;
  TF_Session* session_;

  void DeleteTensorVector(std::vector<TF_Tensor*>& tensor_vector);

  TF_Buffer* ReadBufferFromFile(const char* filename);

  bool AddInputTensor(TF_DataType input_tf_type, const std::vector<std::int64_t>& input_dimensions,
                      const void* input_data, std::size_t data_length);

public:
  /**
  * @brief TensorFlowSession
  *
  * @details Class constructor, which creates a new TensorFlow session from a frozen graph file.
  * Input and output operations must be identified by the user.
  * @param[in] graph_filename             File name of the frozen graph
  * @param[in] input_operation_name       Entry operation name within the graph
  * @param[in] input_operation_name       Exit operation name within the graph
  */
  TensorFlowSession(const char* graph_filename, const char* input_operation_name, const char* output_operation_name);

  ~TensorFlowSession();

  /**
  * @brief AddInputVector
  *
  * @details Adds a new input vector to the TensorFlow session, for use in inference.
  * @tparam input_type                    Input data type
  * @param[in] param input_data           Input data vector
  * @param[in] input_dimensions           Dimensions of input vector data
  * @return                               Status of the function (success or failure)
  */
  template <typename input_type>
  bool AddInputVector(const std::vector<input_type>& input_data, const std::vector<std::int64_t>& input_dimensions)
  {
    TF_DataType input_tf_type;
    if (std::is_same<input_type, float>::value)
    {
      input_tf_type = TF_FLOAT;
    }
    else if (std::is_same<input_type, double>::value)
    {
      input_tf_type = TF_DOUBLE;
    }
    else if (std::is_same<input_type, std::int32_t>::value)
    {
      input_tf_type = TF_INT32;
    }
    else if (std::is_same<input_type, std::uint8_t>::value)
    {
      input_tf_type = TF_UINT8;
    }
    else if (std::is_same<input_type, std::int16_t>::value)
    {
      input_tf_type = TF_INT16;
    }
    else if (std::is_same<input_type, std::int8_t>::value)
    {
      input_tf_type = TF_INT8;
    }
    else if (std::is_same<input_type, std::int64_t>::value)
    {
      input_tf_type = TF_INT64;
    }
    else if (std::is_same<input_type, std::uint16_t>::value)
    {
      input_tf_type = TF_UINT16;
    }
    else if (std::is_same<input_type, std::uint32_t>::value)
    {
      input_tf_type = TF_UINT32;
    }
    else if (std::is_same<input_type, std::uint64_t>::value)
    {
      input_tf_type = TF_UINT64;
    }
    else
    {
      ROS_ERROR("[%s] Error in assigning output type: Requested cast not possible", __APP_NAME__);
      std::cout << "Error in assigning output type: Requested cast not possible" << std::endl;
      return false;
    }
    return AddInputTensor(input_tf_type, input_dimensions, input_data.data(), input_data.size() * sizeof(input_type));
  }

  /**
  * @brief RunInference
  *
  * @details Runs inference on the input vectors.
  * @return                               Status of the function (success or failure)
  */
  bool RunInference(void);

  /**
  * @brief GetOutputVectors
  *
  * @details Retrieves the results of inference.
  * Important - only supports outputs that are all of the same type! If they have multiple types, please read them
  * manually using GetOutputTensors() from the TensorFlow C API.
  * @tparam output_type                   Output data type
  * @param[out] output_vectors            Vector of vectors for storing output result data
  */
  template <typename output_type>
  void GetOutputVectors(std::vector<std::vector<output_type>>& output_vectors)
  {
    output_vectors.reserve(output_tensors_.size());
    for (const auto tensor : output_tensors_)
    {
      const auto tensor_data = static_cast<output_type*>(TF_TensorData(tensor));
      if (tensor_data != nullptr)
      {
        output_vectors.push_back(
            { tensor_data, tensor_data + (TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor))) });
      }
      else
      {
        ROS_ERROR("[%s] Error in inference: Empty output vector", __APP_NAME__);
        std::cout << "Error in inference: Empty output vector" << std::endl;
      }
    }
  }
};
}  // namespace tensorflow_ros

#endif
