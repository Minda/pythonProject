from numba import cuda
import numba

print(numba.__version__)
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import math
import time

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

NUM_ROWS = 2
NUM_ITERATIONS = 3
TEST_ITERATIONS = 1
LINE_SIZE = 100
NUM_CLUSTERS = 3
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
        return (self.sum_of_runtimes - self.first_runtime) / (self.num_measurements - 1)

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


@cuda.jit(device=True)
def copy_shared_float(shared_arr, device_arr):
    shared_arr_len = shared_arr.shape[0]
    device_arr_len = device_arr.shape[0]
    for s_index in range(shared_arr_len):
        device_arr[s_index] = np.float32(shared_arr[s_index])


@cuda.jit(device=True)
def copy_shared_int(shared_arr, device_arr):
    shared_arr_len = shared_arr.shape[0]
    device_arr_len = device_arr.shape[0]
    for s_index in range(shared_arr_len):
        device_arr[s_index] = np.int32(shared_arr[s_index])


@cuda.jit(device=True)
def fill_row(input, shared_input):
    input_length = input.shape[0]
    for x in range(input_length):
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
    for centroid in centroids:
        if res == -1.0 or res > centroid:
            res = centroid
    return res


@cuda.jit(device=True)
def find_min_index(error_array):
    res = math.inf
    index = 0
    for i in range(NUM_SEEDS):
        if error_array[i] < res:
            res = error_array[i]
            index = i
    return index


@cuda.jit(device=True)
def find_yard_stick(centroids):
    min = MAX_INPUT_VALUE
    yard_stick = prev_centroid = None

    # first centroid
    if len(centroids) > 0:
        prev_centroid = min = centroids[0]

    # compare remaining centroids
    for centroid in centroids[1:]:
        distance = centroid - prev_centroid
        if distance < min:
            min = distance
        prev_centroid = centroid

    yard_stick = min
    return yard_stick  # return a float


@cuda.jit(device=True)
def sort_centroids(centroids):
    row = cuda.blockIdx.x
    seed = cuda.threadIdx.x

    if centroids.ndim < 1:  # ensure we are in-bounds
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
    if centroids.ndim < 1 or old_centroids.ndim < 1:  # ensure we are in-bounds
        return

    centroids_length = centroids.shape[0]
    old_centroids_length = old_centroids.shape[0]

    if centroids_length != old_centroids_length:
        return
    # TODO: Check if creating centroid is automatically freed. This should be true.
    for centroid_index in range(centroids_length):
        centroid = centroids[centroid_index]
        old_centroid = old_centroids[centroid_index]
        if epsilon < abs_sub(centroid, old_centroid):
            return False

    return True  # return a boolean


# Function to take a randomly chosen centroid and
# is called one time for each centroid that needs to be selected from the dataset
@cuda.jit(device=True)
def largest_smallest_distance(line, centroids, iter):  # row_data, seed_centroids, centroid_index
    min = 0
    index = 0
    if iter > centroids.shape[0]:  # iter must be in range to execute rest of code
        return
    # our line is a single row from the original input array
    for i in range(line.shape[0]):  # iterate all values in row
        min_distance = MAX_INPUT_VALUE
        for j in range(iter):  # loop only up until our current centroid
            row_element = line[i]
            centroid = centroids[j]
            distance = abs_sub(row_element, centroid)
            if distance < min_distance:
                min_distance = distance
        if min_distance > min:
            min = min_distance
            index = i
    centroids[iter] = line[index]


# This function subtracts two values and returns an absolute value of the difference
@cuda.jit(device=True)
def abs_sub(val1, val2):
    diff = val2 - val1
    if diff >= 0:
        return diff
    return -diff


@cuda.jit(device=True)
def get_random_index(input, rng_states):
    seed = cuda.threadIdx.x
    row = cuda.blockIdx.x
    row_length = input.shape[0]
    random_index = -1

    # Generate a random float in the range [0, 1)
    rand = cuda.random.xoroshiro128p_uniform_float32(rng_states, seed * row)

    # Transform the float to an int in the range of the row length
    random_index = int(rand * row_length)

    return random_index


@cuda.jit(device=True)
def get_initial_centroids(row_data, seed_centroids, rng_states):
    # remember, seed_centroids is of size [NUM_CLUSTERS]
    row = cuda.blockIdx.x
    seed = cuda.threadIdx.x

    # check further references to shape will be in-bounds
    if row_data.ndim < 1 or seed_centroids.ndim < 1:
        return

    row_length = row_data.shape[0]
    centroids_length = seed_centroids.shape[0]

    # select a first centroid from the row data at random
    first_centroid_index = get_random_index(row_data, rng_states)

    # get the furthest distance from the random centroid for remaining centroids
    for centroid_index in range(centroids_length):
        if centroid_index == 0:  # assign first centroid using random index
            seed_centroids[centroid_index] = row_data[first_centroid_index]
        else:
            largest_smallest_distance(row_data, seed_centroids, centroid_index)


@cuda.jit(device=True)
def get_labels(labels, row_data, centroids):
    # check passed in arrays to make sure accessing shape will work
    if (labels.ndim < 1 or row_data.ndim < 1 or centroids.ndim < 1):
        return

    labels_length = labels.shape[0]
    row_data_length = row_data.shape[0]
    centroids_length = centroids.shape[0]

    # these need to be same size to access elements with an index
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
    # num_of_data_for_centroid = cuda.shared.array(shape=(5), dtype=np.float32)
    # shared_error = cuda.shared.array(shape=(NUM_SEEDS), dtype=np.float32)

    if (row_data_length != labels_length):  # ensure we stay in range for assigments
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
        else:
            centroids[centroids_idx - 1] = MAX_INPUT_VALUE * 2


@cuda.jit(device=True)
def calc_error(centroids, row_data, labels):
    sse = 0.0;

    if labels.ndim < 1 or row_data.ndim < 1 or centroids.ndim < 1:
        return

    for i in range(NUM_CLUSTERS):
        centroid = centroids[i]
        if centroid == MAX_INPUT_VALUE + 1:
            return math.inf
        for j in range(LINE_SIZE):
            data = row_data[j]
            if labels[j] == i:
                diff = abs_sub(data, centroid)
                sse += diff * diff
    if sse == 0:
        return math.inf
    return sse


@cuda.jit(device=True)
def kmeans(input, output_labels, output_centroids,
           rng_states):  # these are already shared memory and a single row for each
    seed = cuda.threadIdx.x

    get_initial_centroids(input, output_centroids, rng_states)

    sort_centroids(output_centroids)

    yard_stick = find_yard_stick(output_centroids)
    old_centroids = cuda.local.array(NUM_CLUSTERS, dtype=output_centroids.dtype)
    for i in range(NUM_CLUSTERS):
        old_centroids[i] = output_centroids[i]

    # #Loop until 100 iterations or convergence
    for iteration in range(NUM_ITERATIONS):
        if iteration != 0:
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

    # Shared memory should only run once per block, 2d arrays have shape NUM ROWS x NUM COLS
    shared_input = cuda.shared.array(shape=(LINE_SIZE,), dtype=np.float32)
    shared_error = cuda.shared.array(shape=(NUM_SEEDS,), dtype=np.float32)
    shared_centroids = cuda.shared.array(shape=(NUM_SEEDS, NUM_CLUSTERS), dtype=np.float32)
    shared_labels = cuda.shared.array(shape=(NUM_SEEDS, LINE_SIZE), dtype=np.int32)

    # Copy row data into shared memory
    fill_row(input[row], shared_input)

    # Run kmeans to convergence on row and seed
    kmeans(shared_input, shared_labels[seed], shared_centroids[seed], rng_states)

    # TODO: add error calculation back in
    shared_error[seed] = calc_error(shared_centroids[seed], shared_input, shared_labels[seed])
    cuda.syncthreads()

    # Do rest of work on a single thread for the blcok
    if (seed == 0):
        # TODO: start with the first seed and find the least error
        min = find_min_index(shared_error)

        # Copy the results into the final labels and centroids that will be passed back to CPU
        copy_shared_int(shared_labels[min], output_labels[row])
        copy_shared_float(shared_centroids[min], output_centroids[row])


def compare_clusters(labels1, labels2):
    """
    Compare two clustering label sets using the Adjusted Rand Index (ARI).

    Parameters:
    labels1 (array-like): The first set of clustering labels.
    labels2 (array-like): The second set of clustering labels.

    Returns:
    float: The ARI score, a value between -1 and 1. A score close to 1 indicates
           a high similarity between the two label sets, while a score close to 0
           or negative indicates a low similarity.
    """
    ari_score = adjusted_rand_score(labels1, labels2)
    return ari_score


# Host code:
# Set up all host code to run kmeans
def test_new_kmeans(input_data, new_code_stats, rng_states, printouts=True):
    results_host = []
    centroids_arr = []
    try:

        # Check data format
        if input_data.shape[0] <= 0 or input_data.shape[1] <= 0:
            raise ValueError("Invalid data format. The data array must have a positive shape.")

        centroids_array = np.zeros((input_data.shape[0], NUM_CLUSTERS), dtype=np.float32)
        labels_array = np.zeros_like(input_data, dtype=np.int32)

        input_data = np.round(np.random.rand(NUM_ROWS, LINE_SIZE).astype(np.float32) * MAX_INPUT_VALUE, 2)

        if printouts:
            print("Report on NEW CODE, ITERATION=", test_iteration)
            print("Input Row Data Shape: ", input_data.shape)
            print("Input Row Data: ", np.round(input_data[:1, :], 2))

        device_centroids = cuda.to_device(centroids_array.copy())
        input_rows = cuda.to_device(input_data.copy())
        output_labels = cuda.to_device(labels_array.copy())

        # number of rows, number of seeds
        start = time.time()
        cuda_kmeans[(NUM_ROWS,), (NUM_SEEDS,)](input_rows, output_labels, device_centroids, rng_states)
        cuda.synchronize()
        end = time.time()
        new_code_stats.add_runtime(end - start)

        results_host = output_labels.copy_to_host()
        centroids_arr = np.round(device_centroids.copy_to_host(), 2)

        if printouts:
            print("Centroids after: \n", np.round(centroids_arr[:2, :], 2))
            print("Labels after:, \n", results_host[:2, :])
        print("\n \n \n")

        # SKLEARN KMEANS
        ari_array = []
        for i in range(NUM_ROWS):
            if i > 5:
                break
            output1 = KMeans(n_clusters=3, random_state=0, n_init="auto").fit((input_data[i]).reshape(-1, 1))
            output2 = KMeans(n_clusters=3, random_state=0, n_init="auto").fit((input_data[i]).reshape(-1, 1))
            sklearn_output_centroids1 = np.sort(output1.cluster_centers_.flatten())
            error_array = np.linalg.norm(
                np.array(centroids_arr[i:i + 1, :].flatten()) - np.array(sklearn_output_centroids1))
            print(
                f"SKLearn kmeans row {i}: {sklearn_output_centroids1},  {centroids_arr[i:i + 1, :].flatten()}, {error_array}")

    except Exception as e:
        print("An error occurred:", str(e))

    return results_host, centroids_arr


if __name__ == "__main__":

    # Check if the number of clusters is valid
    if NUM_CLUSTERS <= 0:
        raise ValueError("Invalid number of clusters. k must be greater than 0.")

    new_code_stats = TimingStats()
    # Create an array of RNG states
    rng_states = cuda.random.create_xoroshiro128p_states(LINE_SIZE, seed=0)

    for test_iteration in range(TEST_ITERATIONS):
        # Generate some fake input data
        input_data = np.random.rand(NUM_ROWS, LINE_SIZE).astype(np.float32) * (MAX_INPUT_VALUE)
        labels, centroids_arr = test_new_kmeans(input_data, new_code_stats,
                                                rng_states, True)

    # Report on Runtime Statistics
    print("\n---NEW CODE---\n")
    new_code_stats.print_report()

    # Run kmeans on the host
    # labels, centroids, centroids_arr = test_old_code(input_data, False)

    grid_dim = NUM_ROWS
    block_dim = NUM_SEEDS
    # test_components(grid_dim, block_dim, rng_states)
    # test_new_code(grid_dim, block_dim, rng_states)
