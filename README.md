# TensorFlow interface for ROS

This ROS package provides a simple interface to the TensorFlow C API.

## Requirements

#### For GPU support:
* NVIDIA GPU with compute capability of 6.0 or higher.
* CUDA: Tested with v9.0 and v10.0 -> [How to install](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* CUDNN: Tested with 7.4.1 -> [How to install](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
* TensorFlow C API (GPU version): Tested with 1.12.1 and 1.13.1 -> [How to install](https://www.tensorflow.org/install/lang_c)

#### For CPU only:
* TensorFlow C API (CPU version): Tested with 1.12.1 -> [How to install](https://www.tensorflow.org/install/lang_c)

## How to setup

#### For GPU support:
1. Install CUDA
2. Install CUDNN
3. Install TensorFlow C API (GPU version)
4. Build:
```
$ colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Or just the `tensorflow_ros` package:
```
$ colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to tensorflow_ros
```

#### For CPU only:
1. Install TensorFlow C API (CPU version)
2. Build:
```
$ colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Or just the `tensorflow_ros` package:
```
$ colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to tensorflow_ros
```

## API
#### `TensorFlowSession` constructor
```
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
```

#### Add input data vectors to the `TensorFlowSession`

```
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
```

#### Running inference

```
/**
* @brief RunInference
*
* @details Runs inference on the input vectors.
* @return                               Status of the function (success or failure)
*/
bool RunInference(void);
```

#### Reading inference result output vectors

```
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
```

## Usage

This library implements a class,`TensorFlowSession` which loads a TensorFlow graph.

```cpp
#include <tensorflow_ros/tensorflow_ros.hpp>

const std::vector<float> input_data_vector = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
const std::vector<std::int64_t> input_data_dimensions = { 1, 10 };
std::vector<std::vector<unsigned char>> output_data_vector;
bool success;

/*
Create a new session:
"test_graph.pb"        const char*                              Frozen graph file containing model and weights
"input_operation"      const char*                              Name of input operation within the graph
"output operation"     const char*                              Name of output operation within the graph

This will load the graph and start the TensorFlow session.
*/
TensorFlowSession test_session =
    tensorflow_ros::TensorFlowSession("test_graph.pb", "input_operation", "output_operation");

/*
Create input tensor which match the inputs of the loaded graph:
input_data_vector      std::vector<input_type>&                 Flattened vector of input values
input_data_dimensions  std::vector<std::int64_t>&               Vector of dimensions of the input

Multiple input tensors can be added by calling this function multiple times, noting that order is important.
All created input tensors will be deallocated after inference.
*/
test_session.AddInputVector(input_data_vector, input_data_dimensions);

/*
Run inference using on the input tensors:
success                bool                                     True if successful, False if failed

This function also clears the input tensor,ready for the next input.
*/
success = test_session.RunInference();

/*
Retrieve the output data vectors:
output_data_vector      std::vector<std::vector<output_type>>&  Vector of flattened output vectors

Output vectors are flattened.
*/
test_session.GetOutputVectors(output_data_vectors);

std::cout << "Result: " << +output_data_vectors[0][0] << std::endl; // Display result

```
