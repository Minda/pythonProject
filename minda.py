import os

# os.environ["NUMBA_ENABLE_CUDASIM"] = "1" # needs to appear before `from numba import cuda`
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "1" # set to "1" for more debugging, but slower performance

from numba import cuda, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import math
import time
import random

NUM_ROWS = 1000
NUM_ITERATIONS = 100
TEST_ITERATIONS = 10
LINE_SIZE = 100
NUM_CLUSTERS = 5
MAX_INPUT_VALUE = 100
NUM_SEEDS = 32
EPSILON_PERCENT = 0.002

class TimingStats:
  def __init__(self):
    self.sum_of_runtimes = 0
    self.first_runtime = None
    self.num_measurements = 0

  def add_runtime(self, runtime):
    if self.first_runtime is None:
      self.first_runtime = runtime

    self.sum_of_runtimes += runtime
    self.num_measurements += 1

  def get_average_runtime(self):
    if self.num_measurements == 0:
      return 0
    return self.sum_of_runtimes / self.num_measurements

  def get_average_runtime_after_first(self):
    if self.num_measurements <= 1:
      return 0
    return (self.sum_of_runtimes - self.first_runtime) / (self.num_measurements-1)

  def get_first_runtime(self):
    if self.num_measurements <= 1:
      return 0
    return self.first_runtime

  def get_number_of_runtimes(self):
    return self.num_measurements

  def get_compile_time(self):
    return self.get_first_runtime() - self.get_average_runtime_after_first()

  def get_total_runtime(self):
    return self.sum_of_runtimes

  def print_report(self):
    print("Total Runtime on -", self.get_number_of_runtimes(),
          "- Iterations: ", self.get_total_runtime())
    print("First Runtime:", self.get_first_runtime())
    print("Average Runtime:", self.get_average_runtime())
    print("Average Runtime (ignoring compile time):", self.get_average_runtime_after_first())
    print("Estimated Compile Time:", self.get_compile_time())

#TESTING: Expose the device function to test functionality
@cuda.jit
def test_yard_stick(centroids, double_value_device):
  double_value_device[0] = find_yard_stick(centroids)

@cuda.jit
def test_find_max(centroids, double_value_device):
  double_value_device[0] = find_max(centroids)

@cuda.jit
def test_find_min(centroids, double_value_device):
  double_value_device[0] = find_min(centroids)

@cuda.jit
def test_get_initial_centroids(centroids, row, rng_states):
  get_initial_centroids(row, centroids, rng_states)

@cuda.jit
def test_find_largest_smallest_distance(input, output_centroids, iter_val):
  if iter_val.ndim < 1:
    return
  iter = iter_val[0] #unpack our int value
  largest_smallest_distance(input, output_centroids, iter)

@cuda.jit
def test_random_index_generation(output, input, rng_states):
  random_index = get_random_index(input, rng_states)

  #Save the result in the device array (can be passed in)
  if output.shape[0] > 0:
    output[0] = random_index

@cuda.jit
def test_sort_centroids(centroids):
  sort_centroids(centroids)

@cuda.jit
def test_converged(converged_arr, centroids, old_centroids, epsilon_arr):
  #converged_arr : use an int array with 0 = False, 1 = True to pass boolean
  # value beween GPU/CPU
  epsilon = epsilon_arr[0] #extract epsilon value
  converged_arr[0] = converged(centroids, old_centroids, epsilon)

@cuda.jit
def test_get_centroids(row_data, labels, centroids):

  #error = error_arr[0] #unpack our int value
  #error_arr[0] = calc_error(row_data, error, labels, centroids)
  get_centroids(row_data, labels, centroids)


@cuda.jit
def test_get_labels(labels, row_data, centroids):
  get_labels(labels, row_data, centroids)

@cuda.jit
def test_fill_row(row_data, empty_array_device):
  shared_input = cuda.shared.array(shape=(LINE_SIZE), dtype=np.float32)
  fill_row(row_data, shared_input)
  for x in range(LINE_SIZE):
    empty_array_device[0][x] = shared_input[x]

#---HOST FUNCTIONS---

def host_test_converged(centroids_sorted_3, centroids_to_compare, converged_arr,
                        yard_stick_arr, epsilon_arr):
  #GPU/device variables
  centroids_sorted_3_device = cuda.to_device(centroids_sorted_3)
  centroids_sorted_3_copy_device = cuda.to_device(centroids_to_compare)
  yard_stick_arr_device = cuda.to_device(yard_stick_arr)
  converged_arr_device = cuda.to_device(converged_arr)

  #get yard stick
  test_yard_stick[1,1](centroids_sorted_3_device, yard_stick_arr_device)
  cuda.synchronize()

  #use yard stick value to get epsilon
  yard_stick_arr = yard_stick_arr_device.copy_to_host()
  yard_stick = round(float(yard_stick_arr[0]), 2)
  epsilon = yard_stick*EPSILON_PERCENT
  epsilon_arr[0]: epsilon

  print("\nRUN test for convergence")
  print("Yard Stick: ", yard_stick, " & epsilon: ", epsilon)
  print("Convergence before: ", converged_arr)
  print("Centroids: ", centroids_sorted_3)
  print("Compare to Centroids: ", centroids_sorted_3)

  #check convergence with epsilon
  epsilon_arr_device = cuda.to_device(epsilon_arr)
  centroids_coverged = test_converged[1, 1](converged_arr_device,
      centroids_sorted_3_device, centroids_sorted_3_copy_device, epsilon_arr_device) #single thread
  cuda.synchronize()

  converged_arr = converged_arr_device.copy_to_host()

  print("Were centroids were found to be converged? Convergence= ", converged_arr)

def host_test_sort_centroids(centroids_filled, centroids_result):
  print("\nRUN Sort centroids test")
  print("Centroids to test: \n", centroids_filled)
  print("Shape of centroids: ", centroids_filled.shape)
  centroids_filled_device = cuda.to_device(centroids_filled)

  test_sort_centroids[1, 1](centroids_filled_device) #single thread
  cuda.synchronize()

  centroids_filled = centroids_filled_device.copy_to_host()

  print("Centroids after: \n", centroids_filled)

  message = ("The sorted cetroids should be ", str(centroids_result))

  assert np.allclose(centroids_filled, centroids_result, atol=1e-6), message
  print("Centroids array after sort:", centroids_filled)

def host_test_get_initial_centroids(row_array, output_centroids_1, rng_states):
  output_centroids_device = cuda.to_device(output_centroids_1)
  row_array_device = cuda.to_device(row_array)

  print("Row to test: ", row_array)
  print("Shape of row: ", row_array.shape)
  print("Centroids for row: \n", output_centroids_1)
  print("Shape of centroids: ", output_centroids_1.shape)

  #get initial centroids should take a randomly chosen centroid and get values
  #the furthest away up to k values
  #TODO: get this to select a different seed each time
  test_get_initial_centroids[1, NUM_CLUSTERS](output_centroids_device,
                                             row_array_device, rng_states)
  cuda.synchronize()
  output_centroids_1 = output_centroids_device.copy_to_host()
  print("Centroids array after:", output_centroids_1)

#Test Yard Stick device function with a set centroids against a result
def host_yard_stick(centroids, dim, result):
  centroids = np.sort(centroids)
  double_value = np.array([0.0], dtype=np.float64)  # 'float64' is equivalent to 'double' in C

  centroids_device = cuda.to_device(centroids)
  double_value_device = cuda.to_device(double_value)
  test_yard_stick[1,1](centroids_device, double_value_device)
  cuda.synchronize()

  double_value = double_value_device.copy_to_host()
  yard_stick = round(float(double_value[0]), 2)

  message = ("The yard stick result should be ", str(result))

  print("Test of centroids ", centroids, " with yard stick results: ", yard_stick)
  assert round(yard_stick, 2) == result, message

#Test the min or max function (pass in which one with a boolean)
def host_min_max(centroids, dim, result, find_min = True):
  double_value = np.array([0.0], dtype=np.float64)
  label = "Minimum"

  #GPU/device arrays
  centroids_device = cuda.to_device(centroids)
  double_value_device = cuda.to_device(double_value)

  if find_min:
    test_find_min[dim[0], dim[1]](centroids_device, double_value_device)
  else:
    test_find_max[dim[0], dim[1]](centroids_device, double_value_device)
    label = "Maximum"
  cuda.synchronize()

  double_value = double_value_device.copy_to_host()
  val = round(float(double_value[0]), 2)

  print("Test of centroids ", centroids, " results in ", label, " value: ", val)
  message = "The "+ str(label)+ " value should be "+ str(result)
  assert round(val, 2) == result, message

def host_test_rand_index(int_value, row_array, dim, rng_states):

  #GPU/device arrays
  int_value_device = cuda.to_device(int_value)
  row_array_device = cuda.to_device(row_array)

  #Run kernel to get random index
  test_random_index_generation[1,1](int_value_device, row_array_device, rng_states)
  cuda.synchronize()

  int_value = int_value_device.copy_to_host()
  rand_num = row_size = None

  if int_value.shape[0] > 0:
    rand_num = int_value[0]
  row_size = row_array.shape[0]

  #rng_states_host = rng_states.copy_to_host()
  #print("RNG states: ", rng_states_host)
  print("Result of random index generation: ", rand_num)
  message = "The generated value "+ str(rand_num)+ " should be less than "+ str(row_size)
  assert rand_num <= row_size, message
  message = "The generated value "+ str(rand_num)+ " should greater than 0"
  assert rand_num > 0, message

def host_test_largest_smallest(row_array, output_centroids_2, seed_val, iter):
  print("\n RUN test")
  print("Iter value=", iter[0])
  print("Seed value= ", seed_val)
  print("Row to test: ", row_array)

  output_centroids_2[0] = seed_val
  print("Centroids array to start:", output_centroids_2)

  output_centroids_device = cuda.to_device(output_centroids_2)
  row_array_device = cuda.to_device(row_array.copy())
  iter_device = cuda.to_device(iter)

  test_find_largest_smallest_distance[1, NUM_CLUSTERS](
      row_array_device, output_centroids_device, iter_device)
  cuda.synchronize()

  output_centroids_2 = output_centroids_device.copy_to_host()
  iter = iter_device.copy_to_host()
  print("Centroids array after:", output_centroids_2)
  return output_centroids_2

#Test the components used by kmeans algorithm separately
def test_components(grid_dim, block_dim, rng_states):
  #centroids = np.random.uniform(1.0, 100.0, NUM_CLUSTERS).astype(np.float32)

  print("START Testing ALL device functions...")

  print("\nTEST\n- Testing Yard Stick...")
  dim = [grid_dim, block_dim]

  #Test for min distance at start
  centroids_1 = np.array([32.09, 37.28, 71.26], dtype=np.float32)
  host_yard_stick(centroids_1, dim, 5.19)

  #Test for min distance at end
  centroids_2 = np.array([32.09, 67.28, 71.26], dtype=np.float32)
  host_yard_stick(centroids_2, dim, 3.98)

  print("\nTEST\n- Testing Min / Max Functions...")
  centroids_3 = np.array([24.09, 17.28, 83.26], dtype=np.float32)
  centroids_4 = np.array([24.09, 87.34, 1.26], dtype=np.float32)
  print("Min:")
  host_min_max(centroids_1, dim, 32.09)
  host_min_max(centroids_3, dim, 17.28)
  host_min_max(centroids_4, dim, 1.26)
  print("Max:")
  host_min_max(centroids_1, dim, 71.26, False)
  host_min_max(centroids_3, dim, 83.26, False)
  host_min_max(centroids_4, dim, 87.34, False)

  row = [14.94, 32.76, 9.94, 82.51, 4.78, 13.51, 65.01, 30.05, 34.93, 77.66, 95.78, 35.50]
  row_array = np.array(row, dtype=np.float32)
  double_value = np.array([0.0], dtype=np.float64)
  int_value = np.array([0], dtype=np.int32)

  print("\nTEST\n-Testing if we can get a random index on the GPU...")
  host_test_rand_index(int_value.copy(), row_array, dim, rng_states)

  print("\nTEST\n-Testing largest smallest distance...")
  #TODO: plug into getting the init centroids..

  output_centroids_2 = np.zeros(NUM_CLUSTERS, dtype=np.float64)

  iter = np.array([0], dtype=np.int32) #iter is a reference of a particular cluster index
  iter[0] = output_centroids_2.shape[0]-1 #get some value for iter

  #print("row array shape: ", row_array.shape)
  #print("index half way: ", num_elements/2)
  #print("item half way: ", row_array[int(num_elements/2)])
  num_elements = row_array.shape[0]
  seed_val = row_array[int(num_elements/2)]
  output_centroids_2 = host_test_largest_smallest(row_array, output_centroids_2, seed_val, iter)
  print("Centroids array after first test:", output_centroids_2)

  iter[0] = iter[0] - 1

  if iter[0] > 0:
    output_centroids_2 = host_test_largest_smallest(row_array, output_centroids_2, seed_val, iter)
    print("Centroids array after second test:", output_centroids_2)

  print("\nTEST\n-Testing code to get initial centroids...")
  output_centroids_1 = np.zeros((NUM_CLUSTERS), dtype=np.float64)
  host_test_get_initial_centroids(row_array.copy(), output_centroids_1.copy(), rng_states)

  print("\nTEST\n-Testing code to sort centroids...")
  centroids_filled_3 = np.array([33.50, 95.78, 4.78], dtype=np.float32)
  centroids_result_3 = np.array([4.78, 33.5, 95.78], dtype=np.float32)
  host_test_sort_centroids(centroids_filled_3, centroids_result_3)
  centroids_filled_4 = np.array([33.50, 95.78, 4.78, 9.25], dtype=np.float32)
  centroids_result_4 = np.array([4.78, 9.25, 33.5, 95.78], dtype=np.float32)
  host_test_sort_centroids(centroids_filled_4, centroids_result_4)
  centroids_filled_5 = np.array([33.50, 95.78, 4.78, 9.25, 98.95], dtype=np.float32)
  centroids_result_5 = np.array([4.78, 9.25, 33.5, 95.78, 98.95], dtype=np.float32)
  host_test_sort_centroids(centroids_filled_5, centroids_result_5)

  print("\nTEST\n-Testing code to test convergence...")

  #host variables
  centroids_sorted_3 = np.array(np.sort(centroids_filled_3.copy()), dtype=np.float32)
  non_converged_centroids = np.array([34.50, 96.8, 5.9], dtype=np.float32)
  nc_centroids_sorted = np.array(np.sort(non_converged_centroids.copy()), dtype=np.float32)
  yard_stick_arr = epsilon_arr = np.array([0.0], dtype=np.float64) #double
  #single int value for use as a boolean value
  converged_arr = np.array([0], dtype=np.int32) #Initialized to 0 / False

  #test converged
  print("Test converged centroids")
  host_test_converged(centroids_sorted_3.copy(), centroids_sorted_3.copy(),
                      converged_arr, yard_stick_arr, epsilon_arr)
  #test not converged
  print("\nTest non-converged centroids")
  host_test_converged(centroids_sorted_3.copy(), nc_centroids_sorted.copy(),
                      converged_arr, yard_stick_arr, epsilon_arr)

  print("\nTEST\n-Testing calc error...")
  #host variables
  row = [14.94, 32.76, 9.94, 82.51, 4.78, 13.51, 65.01, 30.05, 34.93, 77.66, 95.78, 35.50]
  row_array = np.array(row, dtype=np.float32)
  centroids = [4.78, 33.5, 95.78]
  three_centroids = np.array(centroids, dtype=np.float32)
  labels = [0, 1, 0, 2, 0, 0, 2, 1, 1, 2, 2, 1]
  labels_array = np.array(labels, dtype=np.float32)
  error_arr = np.full(row_array.shape[0], -1) #initialize to -1
  error_arr = np.array(error_arr, dtype=np.int32)
  error_value = np.array([0], dtype=np.int32)

  #GPU/device variables
  row_data_device = cuda.to_device(row_array.copy())
  error_value_device = cuda.to_device(error_value.copy())
  labels_array_device = cuda.to_device(labels_array.copy())
  three_centroids_device = cuda.to_device(three_centroids)

  print("\nRUN test to calculate error")
  print("Row data: ", row_array)
  print("shape of row: ", row_array.shape)
  print("labels: ", labels_array)
  print("shape of labels: ", labels_array.shape)
  print("centroids: ", three_centroids)
  print("shape of centroids: ", three_centroids.shape)
  print("error value before: ", error_value)

  # test_calc_error[1,1](row_data_device, error_value_device, labels_array_device,
  #                     three_centroids_device)
  cuda.synchronize()

  error_value = error_value_device.copy_to_host()
  print("error value after: ", error_value)

  print("\nTEST\n-Testing get labels...")

  #host variables
  five_sorted_centroids = np.array(np.sort(centroids_filled_5.copy()), dtype=np.float32)
  labels_arr = np.full(row_array.shape[0], -1)
  labels_arr = np.array(labels_arr, dtype=np.int32)

  #GPU/device variables
  five_sorted_centroids_device = cuda.to_device(five_sorted_centroids)
  row_array_device = cuda.to_device(row_array.copy())
  labels_arr_device = cuda.to_device(labels_arr.copy())

  print("\nRUN test to get labels")
  print("Centroids: ", five_sorted_centroids.copy())
  print("Row Data: ", row_array)
  print("Labels before:, ", labels_arr)
  test_get_labels[1,1](labels_arr_device, row_array_device, five_sorted_centroids_device)
  cuda.synchronize()

  #print results
  labels_arr = labels_arr_device.copy_to_host()
  print("Labels after: ", labels_arr)
  labels_after = [1, 2, 1, 3, 0, 1, 3, 2, 2, 3, 3, 2]
  labels_arr_after = np.array(labels_after)
  message = "Array results are not as expected"
  assert np.allclose(labels_arr_after, labels_arr, atol=1e-6), message

# @cuda.jit
# def test_calc_error(row_data, error_arr, labels, centroids):
#   if error_arr.ndim < 1:
#     return

#   error = error_arr[0] #unpack our int value
#   error_arr[0] = calc_error(row_data, centroids, error, labels)


def test_new_code(grid_dim, block_dim, rng_states):
  print("\nTEST\n-Testing get centroids...")

  #host variables
  four_sorted_centroids = np.array([14.78, 19.78, 37.45, 95.78], dtype=np.float32)
  labels_after = [1, 2, 1, 3, 0, 1, 3, 2, 2, 3, 3, 2]
  labels_arr_after = np.array(labels_after)
  row = [14.94, 32.76, 9.94, 82.51, 4.78, 13.51, 65.01, 30.05, 34.93, 77.66, 95.78, 97.50]
  row_array = np.array(row, dtype=np.float32)

  #GPU/device variables
  four_sorted_centroids_device = cuda.to_device(four_sorted_centroids.copy())
  row_array_device = cuda.to_device(row_array.copy())
  labels_arr_device = cuda.to_device(labels_arr_after.copy())

  print("\nRUN test to get centroids")
  print("Centroids: ", four_sorted_centroids)
  print("Row Data: ", row_array)
  print("Labels before:, ", labels_arr_after)
  test_get_centroids[1,1](row_array_device, labels_arr_device, four_sorted_centroids_device)
  cuda.synchronize()

  #print results
  centroids = four_sorted_centroids_device.copy_to_host()
  print("centroids after: ", centroids)

  print("\nTEST\n-Testing KMEANS...")

  #host variables
  #four_sorted_centroids = np.array([14.78, 19.78, 37.45, 95.78], dtype=np.float32)
  labels = np.zeros_like(row_array, dtype=np.int32)
  centroids = np.zeros_like(four_sorted_centroids, dtype=np.float32)

  #GPU/device variables
  four_sorted_centroids_device = cuda.to_device(centroids.copy())
  row_array_device = cuda.to_device(row_array.copy())
  labels_arr_device = cuda.to_device(labels.copy())

  print("\nRUN test on Kmeans")
  print("Row Data: ", row_array)
  print("Centroids: ", centroids)
  print("Labels before:, ", labels)

  test_kmeans[1,1](row_array_device, labels_arr_device,
                   four_sorted_centroids_device, rng_states)

  print("\nPOST KMEANS Results")
  print("Centroids after: ", four_sorted_centroids_device.copy_to_host())
  print("Labels after:, ", labels_arr_device.copy_to_host())


  print("\nTEST\n-Testing Fill Row...")

  empty_array = np.zeros((2, 12), dtype=np.int32)
  data_array = np.zeros((2, 12), dtype=np.int32)
  data_array[0] = [14.94, 32.76, 9.94, 82.51, 4.78, 13.51, 65.01, 30.05, 34.93, 77.66, 95.78, 35.50]
  data_array[1] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 20.0, 80.0]

  empty_array_device = cuda.to_device(empty_array)
  row_array_device = cuda.to_device(data_array.copy())
  print("\nRUN test to fill a row")
  print("Row Data Shape: ", data_array.shape)
  print("Row Data before: ", data_array)

  #GPU/device variables

  test_fill_row[2,1](row_array_device, empty_array_device)

  print("Row Data after: ", empty_array_device.copy_to_host())


  print("\nTEST\n-Testing GPU_KMEANS...")

  #host variables
  data_array = np.zeros((2, 12), dtype=np.int32)
  data_array[0] = [14.94, 32.76, 9.94, 82.51, 4.78, 13.51, 65.01, 30.05, 34.93, 77.66, 95.78, 35.50]
  data_array[1] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 20.0, 80.0]
  labels_array = np.zeros((2, 12), dtype=np.int32)
  centroids_array = np.zeros((2, 4), dtype=np.float32)

  print("\nRUN test on CUDA KMEANS")
  print("Input Row Data Shape: ", data_array.shape)
  print("Input Row Data: ", data_array)
  print("Input Row has ndim: ", hasattr(data_array, "ndim"))
  print("Input Row ndim: ", data_array.ndim)
  print("Centroids Shape: ", centroids_array.shape)
  print("Centroids: ", centroids_array)
  print("Labels Shape: ", labels_array.shape)
  print("Labels before:, ", labels_array)


  device_centroids = cuda.to_device(centroids_array.copy())
  input_rows = cuda.to_device(data_array.copy())
  output_labels = cuda.to_device(labels_array.copy())

  # number of rows, number of seeds
  cuda_kmeans[(2, NUM_SEEDS)](input_rows, output_labels, device_centroids, rng_states)

  print("\nPOST CUDA KMEANS")
  #print("Input after: ", output_centroids.copy_to_host())
  print("Centroids after: ", device_centroids.copy_to_host())
  print("Labels after:, ", output_labels.copy_to_host())




###-----------------------------------------------------------------------------
###START OF CODE BEING ACTIVELY DEVELOPED
###-----------------------------------------------------------------------------

@cuda.jit
def test_fill_row(row_data, empty_array_device):
  row = cuda.blockIdx.x
  row_data_length = row_data.shape[1]
  shared_input = cuda.shared.array(shape=(LINE_SIZE,), dtype=np.float32)
  #print("GPU test row =", row)
  # if row == 0:
  #   from pdb import set_trace
  #   set_trace()
  fill_row(row_data[row], shared_input)
  for x in range(0, row_data_length-1):
    empty_array_device[row][x] = shared_input[x]

@cuda.jit
def test_kmeans(input, output_labels, output_centroids, rng_states):

  shared_input = cuda.shared.array(shape=(12,), dtype=np.float32)
  shared_centroids = cuda.shared.array(shape=(NUM_CLUSTERS,), dtype=np.float32)
  shared_labels = cuda.shared.array(shape=(LINE_SIZE,), dtype=np.int32)

  for x in range(LINE_SIZE):
    shared_input[x] = input[x]

  kmeans(shared_input, shared_labels, shared_centroids, rng_states)

  for x in range(LINE_SIZE):
    output_labels[x] = shared_labels[x]
  for x in range(NUM_CLUSTERS):
    output_centroids[x] = shared_centroids[x]

@cuda.jit(device=True)
def copy_shared(shared_arr, device_arr):
  shared_arr_len = shared_arr.shape[0]
  device_arr_len = device_arr.shape[0]
  for s_index in range(shared_arr_len):
    shared_element = shared_arr[s_index]
    device_arr[s_index] = shared_element

@cuda.jit(device=True)
def fill_row(input, shared_input):
  row = cuda.blockIdx.x
  seed = cuda.threadIdx.x

  input_length = input.shape[0]
  shared_input_length = shared_input.shape[0]
  #print("GPU input length: ", input_length)
  #print("GPU shared input length: ", shared_input_length)
  # if row == 1 and seed == 1:
  #   from pdb import set_trace
  #   set_trace()
  for x in range(0, input_length-1):
    if x < shared_input.shape[0] and x < input.shape[0]:
      shared_input[x] = input[x]


@cuda.jit(device=True)
def find_max(centroids):
  res = -1.0
  index = 0
  for centroid in centroids:
    if res == -1.0 or res < centroid:
      res = centroid
  return res

@cuda.jit(device=True)
def find_min(centroids):
  res = -1.0
  index = 0
  for centroid in centroids:
    if res == -1.0 or res > centroid:
      res = centroid
  return res

@cuda.jit(device=True)
def get_random_index(input, rng_states):
  seed = cuda.threadIdx.x
  row = cuda.blockIdx.x
  max_index = input.shape[0] - 1
  random_index = int(LINE_SIZE/2)

  #rand_scaled = (seed * row) % max_index
  #random_index = int(rand_scaled)
  #random_index = min(max(rand_scaled, 0), max_index)

  # if row == 1 and seed == 1:
  #  from pdb import set_trace
  #  set_trace()
  #Generate a random float in the range [0, 1)
  #rand = xoroshiro128p_uniform_float32(rng_states, seed)

  #rand_scaled = int(max_index * rand)

  #random_index = rand_indx
  #print("Random Index: ", random_index)
  return random_index

@cuda.jit(device=True)
def find_yard_stick(centroids):

  min = MAX_INPUT_VALUE
  yard_stick = prev_centroid = None

  #first centroid
  if len(centroids) > 0:
    prev_centroid = min = centroids[0]

  #compare remaining centroids
  for centroid in centroids[1:]:
    distance = centroid - prev_centroid
    if distance < min:
      min = distance
    prev_centroid = centroid

  yard_stick = min
  return yard_stick #return a float

@cuda.jit(device=True)
def all_same(input, output_labels, output_centroids):
  row = cuda.blockIdx.x
  seed = cuda.threadIdx.x

  return True #return a boolean


@cuda.jit(device=True)
def sort_centroids(centroids):
  row = cuda.blockIdx.x
  seed = cuda.threadIdx.x

  if centroids.ndim < 1: #ensure we are in-bounds
    return

  centroids_length = centroids.shape[0]
  for i in range(1, centroids_length):
      key = centroids[i]
      j = i - 1
      while j >= 0 and centroids[j] > key:
          centroids[j + 1] = centroids[j]
          j -= 1
      centroids[j + 1] = key

@cuda.jit(device=True)
def converged(centroids, old_centroids, epsilon):
  if centroids.ndim < 1 or old_centroids.ndim < 1: #ensure we are in-bounds
    return

  centroids_length = centroids.shape[0]
  old_centroids_length = old_centroids.shape[0]

  if centroids_length != old_centroids_length:
    return

  for centroid_index in range(centroids_length):
    centroid = centroids[centroid_index]
    old_centroid = old_centroids[centroid_index]
    if epsilon < abs_sub(centroid, old_centroid):
      return False

  return True #return a boolean

#Function to take a randomly chosen centroid and
#is called one time for each centroid that needs to be selected from the dataset
@cuda.jit(device=True)
def largest_smallest_distance(line, centroids, iter):

  min = 0
  index = -1

  if iter > centroids.shape[0]: #iter must be in range to execute rest of code
    return

  #our line is a single row from the original input array
  for i in range(line.shape[0]): #iterate all values in row
    min_distance = MAX_INPUT_VALUE

    for j in range(iter): #loop only up until our current centroid
      row_element = line[i]
      centroid = centroids[j]
      distance = abs_sub(row_element, centroid)
      if (distance < min_distance):
        min_distance =  distance
    if min_distance > min:
      min = min_distance
      index = i
  centroids[iter] = line[index]

#This function subtracts two values and returns an absolute value of the difference
@cuda.jit(device=True)
def abs_sub(val1, val2):
  diff = val2 - val1
  if diff >= 0:
    return diff
  return -diff

@cuda.jit(device=True)
def get_initial_centroids(row_data, seed_centroids, rng_states):
  #remember, seed_centroids is of size [NUM_CLUSTERS]
  row = cuda.blockIdx.x
  seed = cuda.threadIdx.x

  #check further references to shape will be in-bounds
  if row_data.ndim < 1 or seed_centroids.ndim < 1:
    return

  row_length = row_data.shape[0]
  centroids_length = seed_centroids.shape[0]

  #select a first centroid from the row data at random
  first_centroid_index = get_random_index(row_data, rng_states)
  comp = (row * seed) % (row_length - 1)
  first_centroid_index = int32(comp)
  print("first_centroid_index: ", first_centroid_index)
  cuda.syncthreads()

  # if row == 1 and seed == 1:
  #  from pdb import set_trace
  #  set_trace()

  #get the furthest distance from the random centroid for remaining centroids
  for centroid_index in range(centroids_length):
    centroid = seed_centroids[centroid_index]

    if centroid_index == 0: #assign first centroid using random index
      seed_centroids[centroid_index] = row_data[first_centroid_index]
    else: #find furthest centroid
      largest_smallest_distance(row_data, seed_centroids, centroid_index)

@cuda.jit(device=True)
def get_labels(labels, row_data, centroids):
  #check passed in arrays to make sure accessing shape will work
  if (labels.ndim < 1 or row_data.ndim < 1 or centroids.ndim < 1):
    return

  labels_length = labels.shape[0]
  row_data_length = row_data.shape[0]
  centroids_length = centroids.shape[0]

  #these need to be same size to access elements with an index
  if (labels_length != row_data_length):
    return

  for row_index in range(row_data_length):
    element = row_data[row_index]
    best_centroid_index = 0
    best_distance = abs_sub(element, centroids[best_centroid_index])
    labels[row_index] = int(element)
    for centroid_index in range(centroids_length):
      centroid = centroids[centroid_index]
      curr_dist = abs_sub(element, centroid)
      if curr_dist < best_distance:
        best_centroid_index = centroid_index
        best_distance = curr_dist
    labels[row_index] = best_centroid_index

@cuda.jit(device=True)
def fill_ones(arr):
  for i in range(arr.shape[0]):
    arr[i] = 1

@cuda.jit(device=True)
def get_centroids(row_data, labels, centroids):
  row = cuda.blockIdx.x
  seed = cuda.threadIdx.x

  '''on the first call, get centroids will be called on the inital centroids
  we selected. Then we need to re-compute the centroids based on the row data.
  Then '''

  if row_data.ndim < 1 or labels.ndim < 1 or centroids.ndim < 1:
    return

  labels_length = labels.shape[0]
  row_data_length = row_data.shape[0]
  centroids_length = centroids.shape[0]

  num_mapped = 0
  #num_of_data_for_centroid = cuda.shared.array(shape=(5), dtype=np.float32)
  #shared_error = cuda.shared.array(shape=(NUM_SEEDS), dtype=np.float32)

  if (row_data_length != labels_length): #ensure we stay in range for assigments
    return

  for centroids_idx in range(centroids_length):
    count = 0
    sum = 0
    centroid = centroids[centroids_idx]
    for labels_index in range(labels_length):
      label = labels[labels_index]
      if np.int32(centroids_idx) == np.int32(label):
        sum += row_data[labels_index]
        count += 1
    cuda.syncthreads()
    if count != 0:
      centroids[centroids_idx] = (np.float32(sum) / np.float32(count))


# @cuda.jit(device=True)
# def calc_error(row_data, centroids, error, labels):
#   sse = 0.0;

#   if (labels.ndim < 1 or row_data.ndim < 1 or centroids.ndim < 1):
#     return

#   labels_length = labels.shape[0]
#   row_data_length = row_data.shape[0]
#   centroids_length = centroids.shape[0]

#   for i in range(centroids_length):
#     for j in range(row_data_length):
#       cuda.syncthreads()
#       centroid = centroids[i]
#       if hasattr(centroid, "ndim"):
#         if centroid.ndim > 0:
#           centroid = centroid[0]
#       row = row_data[j]
#       if hasattr(row, "ndim"):
#         if row.ndim > 0:
#           row = row[0]
#       sse += abs_sub(centroid, row) # make this squared to match the C version
#   return sse

@cuda.jit(device=True)
def kmeans(input, output_labels, output_centroids, rng_states): # these are already shared memory and a single row for each
  seed = cuda.threadIdx.x

  get_initial_centroids(input, output_centroids, rng_states)

  sort_centroids(output_centroids)

  yard_stick = find_yard_stick(output_centroids)
  old_centroids = cuda.local.array(NUM_CLUSTERS, dtype=output_centroids.dtype)
  for i in range(NUM_CLUSTERS):
      old_centroids[i] = output_centroids[i]

  # #Loop until 100 iterations or convergence
  for iteration in range(NUM_ITERATIONS):
    if (iteration != 0):
      if converged(output_centroids, old_centroids, yard_stick * EPSILON_PERCENT):
        break

    for i in range(NUM_CLUSTERS):
      old_centroids[i] = output_centroids[i]

    get_labels(output_labels, input, output_centroids)
    get_centroids(input, output_labels, output_centroids)

@cuda.jit()
def cuda_kmeans(input, output_labels, output_centroids, rng_states):
  row = cuda.blockIdx.x
  seed = cuda.threadIdx.x

  #if row == 1 and seed == 1:
  #  from pdb import set_trace
  #  set_trace()

  if (hasattr(input, "ndim") and input.ndim > 1):
    row_data_length = input.shape[1]

  #Shared memory should only run once per block, 2d arrays have shape NUM ROWS x NUM COLS
  shared_input = cuda.shared.array(shape=(LINE_SIZE,), dtype=np.float32)
  shared_error = cuda.shared.array(shape=(NUM_SEEDS,), dtype=np.float32)
  shared_centroids = cuda.shared.array(shape=(NUM_SEEDS, NUM_CLUSTERS), dtype=np.float32)
  shared_labels = cuda.shared.array(shape=(NUM_SEEDS, LINE_SIZE), dtype=np.int32)

  #Copy row data into shared memory
  fill_row(input[row], shared_input)

  #Run kmeans to convergence on row and seed
  kmeans(shared_input, shared_labels[seed], shared_centroids[seed], rng_states)

  #TODO: add error calculation back in
  #calc_error(shared_input, output_centroids, shared_error, output_labels)
  cuda.syncthreads()

  #Do rest of work on a single thread for the blcok
  if (seed == 0):
    #TODO: start with the first seed and find the least error

    #Copy the results into the final labels and centroids that will be passed back to CPU
    copy_shared(shared_labels[seed], output_labels[row])
    copy_shared(shared_centroids[seed], output_centroids[row])


# Host code:

#Set up all host code to run kmeans
def test_old_code(input_data, printouts=True):
    results_host = []
    centroids = []
    centroids_arr = []
    try:
        max_iterations = 100  #TODO: remove after new kmeans Set a maximum number of iterations

        # TODO: we ideally want to generate these on GPU Generate random centroids using k-means++
        centroids = np.random.uniform(1.0, 100.0, NUM_CLUSTERS).astype(np.float32)
        centroids = np.sort(centroids)

        # Create a labels array with appropriate dimensions
        labels = np.zeros_like(input_data, dtype=np.int32) #TODO: remove after new kmeans
        prev_labels = np.zeros_like(labels, dtype=np.int32) #TODO: remove after new kmeans

        # Check data format
        if input_data.shape[0] <= 0 or input_data.shape[1] <= 0:
            raise ValueError("Invalid data format. The data array must have a positive shape.")

        # Check if the number of clusters is valid
        if NUM_CLUSTERS <= 0:
            raise ValueError("Invalid number of clusters. k must be greater than 0.")

        # Check centroid format
        if centroids.shape[0] != NUM_CLUSTERS:
            raise ValueError("Invalid centroid format. The centroids array must have k elements.")

        # Create device arrays for data, centroids, and labels
        data_device = cuda.to_device(input_data.copy())
        centroids_device = cuda.to_device(centroids) #TODO: remove after new kmeans
        centroidsarr_device = cuda.to_device(np.zeros((input_data.shape[0], centroids.shape[0]), dtype=np.float32))
        results_device = cuda.to_device(np.zeros_like(input_data, dtype=np.float32))
        prev_labels_device = cuda.to_device(prev_labels)  # TODO: remove after new kmean - Allocate memory for prev_labels on the GPU

        grid_dim = NUM_ROWS
        block_dim = NUM_SEEDS

        # Create an array of RNG states
        rng_states = cuda.random.create_xoroshiro128p_states(NUM_SEEDS, seed=0)

        #test_components(grid_dim, block_dim, rng_states)
        #test_new_code(grid_dim, block_dim, rng_states)

        # Profile for optimization
        sum_of_runtimes = 0
        first_runtime = 0

        old_code_stats = TimingStats()
        new_code_stats = TimingStats()

        for test_iteration in range(TEST_ITERATIONS):


          start = time.time()
          # cuda_kmeans[grid_dim, block_dim](data_device, results_device, centroidsarr_device)
          # print("\nRUN test on CUDA KMEANS")
          # print("Input Row Data Shape: ", data_array.shape)
          # print("Input Row Data: ", data_array)
          # print("Input Row has ndim: ", hasattr(data_array, "ndim"))
          # print("Input Row ndim: ", data_array.ndim)
          # print("Centroids Shape: ", centroids_array.shape)
          # print("Centroids: ", centroids_array)
          # print("Labels Shape: ", labels_array.shape)
          # print("Labels before:, ", labels_array)


          # device_centroids = cuda.to_device(centroids_array.copy())
          # input_rows = cuda.to_device(data_array.copy())
          # output_labels = cuda.to_device(labels_array.copy())

          # # number of rows, number of seeds
          # cuda_kmeans[(grid_dim, block_dim)](input_rows, output_labels, device_centroids, rng_states)

          # print("\nPOST CUDA KMEANS")
          # #print("Input after: ", output_centroids.copy_to_host())
          # print("Centroids after: ", device_centroids.copy_to_host())
          # print("Labels after:, ", output_labels.copy_to_host())
          # cuda.synchronize()
          end = time.time()
          new_code_stats.add_runtime(end - start)


          #Time an entire kmeans computation
          start = time.time()
          ###This is the current working version
          kmeans_iteration[grid_dim, block_dim](data_device, centroids_device, centroidsarr_device, labels, max_iterations, prev_labels_device)
          # Wait for the kernel to finish
          cuda.synchronize()
          end = time.time()

          # Copy the results back from the GPU to CPU
          results_host = labels.copy()
          centroids_arr = centroidsarr_device.copy_to_host()

          #Report on Results
          if printouts:
            print("Report on OLD CODE, ITERATION=", test_iteration)
            print("Results:\n", results_host[:1, :])
            print("Centroids:\n", centroids)
            print("Centroid Array:\n", centroids_arr[:1, :])
            #print("Input Array:\n", input_data[:1, :])

          #Record Report Statistics
          runtime = end - start #runtime
          old_code_stats.add_runtime(runtime)



        #Report on Runtime Statistics
        print("\n---OLD CODE---\n")
        old_code_stats.print_report()

    except Exception as e:
        print("An error occurred:", str(e))

    return results_host, centroids, centroids_arr

#Set up all host code to run kmeans
def test_new_kmeans(input_data, new_code_stats, rng_states, printouts=True):
    results_host = []
    centroids_arr = []
    try:

      # Check data format
      if input_data.shape[0] <= 0 or input_data.shape[1] <= 0:
          raise ValueError("Invalid data format. The data array must have a positive shape.")

      centroids_array = np.zeros((input_data.shape[0], NUM_CLUSTERS), dtype=np.float32)
      labels_array = np.zeros_like(input_data, dtype=np.int32)

      # Create device arrays for data, centroids, and labels
      data_device = cuda.to_device(input_data.copy())
      centroidsarr_device = cuda.to_device(centroids_array)
      results_device = cuda.to_device(labels_array)

      input_data = np.random.rand(NUM_ROWS, LINE_SIZE).astype(np.float32) * MAX_INPUT_VALUE
      data_device = cuda.to_device(input_data.copy())
      #cuda_kmeans[grid_dim, block_dim](data_device, results_device, centroidsarr_device)
      if printouts:
        print("Report on NEW CODE, ITERATION=", test_iteration)
        print("Input Row Data Shape: ", input_data.shape)
        print("Input Row Data: ", input_data[:1, :])
        print("Input Row has ndim: ", hasattr(input_data, "ndim"))
        print("Input Row ndim: ", input_data.ndim)
        print("Centroids Shape: ", centroids_array.shape)
        print("Centroids: ", centroids_array[:1, :])
        print("Labels Shape: ", labels_array.shape)
        print("Labels before:, ", labels_array[:1, :])

      device_centroids = cuda.to_device(centroids_array.copy())
      input_rows = cuda.to_device(input_data.copy())
      output_labels = cuda.to_device(labels_array.copy())

      # number of rows, number of seeds
      start = time.time()
      cuda_kmeans[(NUM_ROWS, NUM_SEEDS)](input_rows, output_labels, device_centroids, rng_states)
      cuda.synchronize()
      end = time.time()
      new_code_stats.add_runtime(end - start)

      results_host = output_labels.copy_to_host()
      centroids_arr = device_centroids.copy_to_host()

      if printouts:
        print("Centroids after: ", centroids_arr[:1, :])
        print("Labels after:, ", results_host[:1, :])

    except Exception as e:
        print("An error occurred:", str(e))

    return results_host, centroids_arr

if __name__ == "__main__":

  # Check if the number of clusters is valid
  if NUM_CLUSTERS <= 0:
    raise ValueError("Invalid number of clusters. k must be greater than 0.")

  new_code_stats = TimingStats()
  # Create an array of RNG states
  rng_states = cuda.random.create_xoroshiro128p_states(NUM_SEEDS, seed=1)

  #print("random numbers. ", rng_states)

  for test_iteration in range(TEST_ITERATIONS):

    # Generate some fake input data
    input_data = np.random.rand(NUM_ROWS, LINE_SIZE).astype(np.float32) * MAX_INPUT_VALUE

    labels, centroids_arr = test_new_kmeans(input_data, new_code_stats,
                                            rng_states, True)

  #Report on Runtime Statistics
  print("\n---NEW CODE---\n")
  new_code_stats.print_report()

  # Run kmeans on the host
  #labels, centroids, centroids_arr = test_old_code(input_data, False)

  grid_dim = NUM_ROWS
  block_dim = NUM_SEEDS
  #test_components(grid_dim, block_dim, rng_states)
  #test_new_code(grid_dim, block_dim, rng_states)
