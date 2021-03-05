#!/usr/bin/python

import pyopencl as cl
import numpy

# Write down our kernel as a multiline string.
kernel = """
__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    const unsigned int count)
{
    unsigned int i = get_global_id(0);
   if (i < count)
        c[i] = a[i] + b[i];
}
"""

# The size of the vectors to be added together.
vector_size = 4096

# Step 1: Create a context.
# This will ask the user to select the device to be used.
platform = cl.get_platforms()  # gets all platforms that exist on this machine
devices = platform[0].get_devices()  # gets all GPU's that exist on first platform from platform list
context = cl.Context(devices=devices)  # devices=[device[:]] # Creates context for all devices in the list of "device" from above. context.num_devices give number of devices in this context

# Create a queue to the device.
queue = cl.CommandQueue(context, devices[0])

# Create the program.
program = cl.Program(context, kernel).build()

# Create the input vector A, B, C and D.
h_a = numpy.random.rand(vector_size).astype(numpy.float32)
h_b = numpy.random.rand(vector_size).astype(numpy.float32)
h_c = numpy.random.rand(vector_size).astype(numpy.float32)
h_d = numpy.random.rand(vector_size).astype(numpy.float32)

# Create the result vectors X, Y and Z.
h_x = numpy.empty(vector_size).astype(numpy.float32)
h_y = numpy.empty(vector_size).astype(numpy.float32)
h_z = numpy.empty(vector_size).astype(numpy.float32)

# Send the data to the guest memory.
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)
d_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_d)

# Create the memory on the device to put the result into.
# Notice that the first two buffers are read and write. This is because they are used to store intermediat results.
# Only the final buffer (z) is write-only because it is not used to read back the intermediate result.
d_x = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_x.nbytes)
d_y = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_y.nbytes)
d_z = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_z.nbytes)

# Initiate the kernel.
vadd = program.vadd
vadd.set_scalar_arg_dtypes([None, None, None, numpy.uint32])

# Execute X = A + B
vadd(queue, h_a.shape, None, d_a, d_b, d_x, vector_size)

# Execute Y = X + C
vadd(queue, h_a.shape, None, d_x, d_c, d_y, vector_size)

# Execute Z = Y + D
vadd(queue, h_a.shape, None, d_y, d_d, d_z, vector_size)

# Wait for the queue to be completely processed.
queue.finish()

# Read the array from the device.
cl.enqueue_copy(queue, h_z, d_z)

# Verify the solution.
correct = 0
tolerance = 0.001

for i in range(vector_size):
    # Expected result
    expected = h_a[i] + h_b[i] + h_c[i] + h_d[i]
    actual = h_z[i]
    # Compute the relative error
    relative_error = numpy.absolute((actual - expected) / expected)

    # Print the index if it's wrong.
    if relative_error < tolerance:
        correct += 1
    else:
        print(i, " is wrong")
print(h_c)