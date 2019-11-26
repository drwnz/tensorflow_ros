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
 * tensorflow_ros.cpp
 *
 *  Created on: February 6th 2019
 */

#include <tensorflow_ros/tensorflow_ros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace tensorflow_ros
{
TensorFlowSession::TensorFlowSession(const char* graph_filename, const char* input_operation_name,
                                     const char* output_operation_name)
{
  graph_ = TF_NewGraph();
  status_ = TF_NewStatus();
  input_tensors_.clear();
  output_tensors_ = { nullptr };

#if TENSORFLOW_NEW_EXPERIMENTAL_API
  TF_Buffer* options_buffer = TF_CreateConfig(0, 1, sysconf(_SC_NPROCESSORS_ONLN));
#else
  TF_Buffer* options_buffer = TF_CreateConfig(0, 1);
#endif

  TF_ImportGraphDefOptions* graph_options = TF_NewImportGraphDefOptions();
  TF_SessionOptions* session_options = TF_NewSessionOptions();
  TF_SetConfig(session_options, options_buffer->data, options_buffer->length, status_);

  TF_Buffer* buffer = ReadBufferFromFile(graph_filename);
  if (buffer == nullptr)
  {
    ROS_ERROR("[%s] Failed to read frozen graph file: %s", __APP_NAME__, graph_filename);
    throw std::runtime_error("Unable to start TensorFlow session.");
  }

  TF_GraphImportGraphDef(graph_, buffer, graph_options, status_);
  TF_DeleteImportGraphDefOptions(graph_options);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(status_) != TF_OK)
  {
    TF_DeleteGraph(graph_);
    ROS_ERROR("[%s] Failed to load frozen graph file: %s", __APP_NAME__, graph_filename);
    throw std::runtime_error("Unable to start TensorFlow session.");
  }

  session_ = TF_NewSession(graph_, session_options, status_);
  TF_DeleteSessionOptions(session_options);
  if (TF_GetCode(status_) != TF_OK)
  {
    ROS_ERROR("[%s] Failed to start TensorFlow session using frozen graph file: %s", __APP_NAME__, graph_filename);
    throw std::runtime_error("Unable to start TensorFlow session.");
  }
  TF_Operation* input_operation = TF_GraphOperationByName(graph_, input_operation_name);
  TF_Operation* output_operation = TF_GraphOperationByName(graph_, output_operation_name);
  if (input_operation == nullptr)
  {
    ROS_ERROR("[%s] Failed to start TensorFlow session using input operation: %s", __APP_NAME__, input_operation_name);
    throw std::runtime_error("Unable to start TensorFlow session.");
  }
  if (output_operation == nullptr)
  {
    ROS_ERROR("[%s] Failed to start TensorFlow session using output operation: %s", __APP_NAME__,
              output_operation_name);
    throw std::runtime_error("Unable to start TensorFlow session.");
  }

  input_operation_ = { { input_operation, 0 } };
  output_operation_ = { { output_operation, 0 } };
}

TensorFlowSession::~TensorFlowSession()
{
  if (output_tensors_.size() > 0)
  {
    DeleteTensorVector(output_tensors_);
  }
  if (input_tensors_.size() > 0)
  {
    DeleteTensorVector(input_tensors_);
  }
  TF_CloseSession(session_, status_);
  TF_DeleteSession(session_, status_);
  TF_DeleteStatus(status_);
  TF_DeleteGraph(graph_);
}

bool TensorFlowSession::AddInputTensor(TF_DataType input_tf_type, const std::vector<std::int64_t>& input_dimensions,
                                       const void* input_data, std::size_t data_length)
{
  if (input_dimensions.data() == nullptr || input_data == nullptr)
  {
    ROS_ERROR("[%s] Unable to read input data for inference in TensorFlow", __APP_NAME__);
    std::cout << "Unable to read input data for inference in TensorFlow" << std::endl;
    return false;
  }

  TF_Tensor* input_tensor =
      TF_AllocateTensor(input_tf_type, input_dimensions.data(), static_cast<int>(input_dimensions.size()), data_length);

  if (input_tensor == nullptr)
  {
    ROS_ERROR("[%s] Unable to create input tensor for inference in TensorFlow", __APP_NAME__);
    std::cout << "Unable to create input tensor for inference in TensorFlow" << std::endl;
    return false;
  }

  void* input_tensor_data = TF_TensorData(input_tensor);
  if (input_tensor_data == nullptr)
  {
    ROS_ERROR("[%s] Unable to create input tensor for inference in TensorFlow", __APP_NAME__);
    std::cout << "Unable to create input tensor for inference in TensorFlow" << std::endl;
    TF_DeleteTensor(input_tensor);
    return false;
  }
  // Try a direct pointer swap here later.
  std::memcpy(input_tensor_data, input_data, std::min(data_length, TF_TensorByteSize(input_tensor)));
  input_tensors_.push_back(input_tensor);
  return true;
}

bool TensorFlowSession::RunInference(void)
{
  if (input_operation_.data() == nullptr || output_operation_.data() == nullptr)
  {
    ROS_ERROR("[%s] Invalid frozen graph operation", __APP_NAME__);
    std::cout << "Invalid frozen graph operation" << std::endl;
    return false;
  }

  if (input_tensors_.size() == 0)
  {
    ROS_ERROR("[%s] No input data provided for inference in TensorFlow", __APP_NAME__);
    std::cout << "No input data provided for inference in TensorFlow" << std::endl;
    return false;
  }

  if (output_tensors_.size() > 0)
  {
    DeleteTensorVector(output_tensors_);
  }
  output_tensors_ = { nullptr };
  TF_SessionRun(session_,
                nullptr,  // Run options.
                input_operation_.data(), input_tensors_.data(),
                static_cast<int>(input_tensors_.size()),  // Input tensors, input tensor values, number of inputs.
                output_operation_.data(), output_tensors_.data(),
                static_cast<int>(output_tensors_.size()),  // Output tensors, output tensor values, number of outputs.
                nullptr, 0,                                // Target operations, number of targets.
                nullptr,                                   // Run metadata.
                status_                                    // Output status.
                );

  // Careful here - delete reference to data only?
  DeleteTensorVector(input_tensors_);
  if (TF_GetCode(status_) != TF_OK)
  {
    ROS_ERROR("[%s] Inference in TensorFlow failed", __APP_NAME__);
    std::cout << "Inference in TensorFlow failed" << std::endl;
    DeleteTensorVector(output_tensors_);
    DeleteTensorVector(input_tensors_);
    return false;
  }
  return true;
}

void TensorFlowSession::DeleteTensorVector(std::vector<TF_Tensor*>& tensor_vector)
{
  for (auto tensor : tensor_vector)
  {
    if (tensor != nullptr)
    {
      TF_DeleteTensor(tensor);
    }
  }
  tensor_vector.clear();
}

void DeallocateBuffer(void* data, size_t)
{
  std::free(data);
}

TF_Buffer* TensorFlowSession::ReadBufferFromFile(const char* filename)
{
  const auto file = std::fopen(filename, "rb");
  if (file == nullptr)
  {
    ROS_ERROR("[%s] Specified file: %s does not exist", __APP_NAME__, filename);
    std::cout << "Specified file: " << filename << "does not exist" << std::endl;
    return nullptr;
  }

  std::fseek(file, 0, SEEK_END);
  const auto filesize = ftell(file);
  std::fseek(file, 0, SEEK_SET);

  if (filesize < 1)
  {
    ROS_ERROR("[%s] Specified file: %s is empty", __APP_NAME__, filename);
    std::cout << "Specified file: " << filename << "is empty" << std::endl;
    std::fclose(file);
    return nullptr;
  }

  const auto data = std::malloc(filesize);
  const auto readsize = std::fread(data, filesize, 1, file);
  if (readsize < 1)
  {
    ROS_ERROR("[%s] Unable to read contents of file: %s", __APP_NAME__, filename);
    std::cout << "Unable to read contents of file: " << filename << std::endl;
    std::free(data);
    std::fclose(file);
    return nullptr;
  }

  std::fclose(file);

  TF_Buffer* buffer = TF_NewBuffer();
  buffer->data = data;
  buffer->length = filesize;
  buffer->data_deallocator = DeallocateBuffer;

  return buffer;
}
}  // namespace tensorflow_ros
