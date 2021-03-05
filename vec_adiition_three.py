#!/usr/bin/python

import pyopencl as cl
import numpy

# Write down our kernel as a multiline string.
kernel = """
__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    __global float* d,
    const unsigned int count)
{
    unsigned int i = get_global_id(0);
    if (i < count)
        d[i] = a[i] + b[i] + c[i];
}
"""

# The size of the vectors to be added together.
vector_size = 4096

# Step 1: Create a context.
# This will ask the user to select the device to be used.
context = cl.create_some_context()

# Create a queue to the device.
queue = cl.CommandQueue(context)

# Create the program.
program = cl.Program(context, kernel).build()

# Create the input vector A, B and C.
h_a = numpy.random.rand(vector_size).astype(numpy.float32)
h_b = numpy.random.rand(vector_size).astype(numpy.float32)
h_c = numpy.random.rand(vector_size).astype(numpy.float32)

# Create the result vector D.
h_d = numpy.empty(vector_size).astype(numpy.float32)

# Send the data to the guest memory.
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)

# Create an array on the device for the result.
d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)

# Initiate the kernel.
vadd = program.vadd
vadd.set_scalar_arg_dtypes([None, None, None, None, numpy.uint32])

# Execute D = A + B + C
vadd(queue, h_a.shape, None, d_a, d_b, d_c, d_d, vector_size)

# Wait for the queue to be completely processed.
queue.finish()

# Read the array from the device.
cl.enqueue_copy(queue, h_d, d_d)

# Verify the solution.
correct = 0
tolerance = 0.001

for i in range(vector_size):
    # Expected result
    expected = h_a[i] + h_b[i] + h_c[i]
    actual = h_d[i]
    # Compute the relative error
    relative_error = numpy.absolute((actual - expected) / expected)

    # Print the index if it's wrong.
    if relative_error < tolerance:
        correct += 1
    else:
        print(i, " is wrong")
print(h_d)