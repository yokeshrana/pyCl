import pyopencl as cl
import numpy

# The kernel in Python can be written down as an inline string, or it can be an external file.
# In most exercises the kernel will be inlined into the file.
kernel = """
// Each kernel will take a few arguments. In this case we need to add two vectors together.
// This means that we will have at least two inputs, but also one output vector to store our result in.
// Of course you could add into one of both input vectors, too.
// Additionally we add the count which holds the size of the vector.
// This value can be a parameter, or it can be inlined into the string as a constant.
// This is where string formatting in python comes in handy.
// However, the goal of this course is not to make you a better python programmer, but a better opencl programmer.
__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    const unsigned int count)
{
    // get_global_id gives you the details which identifier the current workitem has received.
    // This is unique for every work item and can be used to index into an array.
    // Here each work item will add the ith element from the vector.
    // All work items outside of the 0-count range will not do anything.
    unsigned int i = get_global_id(0);
    if (i < count)
        c[i] = a[i] + b[i];
}

"""

# The size of the vectors to be added together.
vector_size = 1024

# Step 1: Create a context.
# This will ask the user to select the device to be used.
# Can be automatic, too by setting the environment variable PYOPENCL_CTX.
# Execute your program like this: PYOPENCL_CTX=0 python main.py
context = cl.create_some_context()

# Create a queue to the device.
queue = cl.CommandQueue(context)

# Create the program.
program = cl.Program(context, kernel).build()

# Create two vectors to be added.
h_a = numpy.random.rand(vector_size).astype(numpy.float32)
h_b = numpy.random.rand(vector_size).astype(numpy.float32)

# Create the result vector.
h_c = numpy.empty(vector_size).astype(numpy.float32)

# Send the data to the guest memory.
# COPY_HOST_PTR     :: If specified, it indicates that the application wants the
#                      OpenCL implementation to allocate memory for the memory
#                      object and copy the data from memory referenced by host_ptr
# CL_MEM_READ_ONLY  :: This flag specifies that the memory object is a read-only
#                      memory object when used inside a kernel.Writing to a buffer
#                      or image object created with CL_MEM_READ_ONLY inside a kernel is undefined.
# CL_MEM_WRITE_ONLY :: This flags specifies that the memory object will be written but not
#                      read by a kernel.Reading from a buffer or image object created with
#                      CL_MEM_WRITE_ONLY inside a kernel is undefined.
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

# Create the memory on the device to put the result into.
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

# Execute the kernel.
# Here you can reference multiple kernels. For didactic purposes I copied the kernel and gave it slightly different name.
# You can reference both kernels here.
vadd = program.vadd

# https://documen.tician.de/pyopencl/runtime_program.html?highlight=set_scalar_arg_dtypes#pyopencl.Kernel.set_scalar_arg_dtypes
vadd.set_scalar_arg_dtypes([None, None, None, numpy.uint32])

# https://documen.tician.de/pyopencl/runtime_program.html?highlight=set_scalar_arg_dtypes#pyopencl.Kernel.__call__
vadd(queue, h_a.shape, None, d_a, d_b, d_c, vector_size)

# Wait for the queue to be completely processed.
# https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clFinish.html
queue.finish()

# Read the array from the device.
# https://documen.tician.de/pyopencl/runtime_memory.html?highlight=enqueue_copy#pyopencl.enqueue_copy
cl.enqueue_copy(queue, h_c, d_c)

# Verify the solution.
correct = 0
tolerance = 0.001

# This part of the code will verify our solution. We are dealing with floating point numbers so we have to keep in mind that there can be small deviations from the actual result.
# Therefore we compute the so-called fault-tolerance.
# We take the absolute difference between the expected result and actual result, and divide it by the expected result. This yields the relative error rate.
# It should not be more than 0.001.
for i in range(vector_size):
    # Expected result
    expected = h_a[i] + h_b[i]
    actual = h_c[i]
    # Compute the relative error
    relative_error = numpy.absolute((actual - expected) / expected)

    # Print the index if it's wrong.
    if relative_error < tolerance:
        correct += 1
    else:
        print(i, " is wrong")
print(h_a[:10])
print(h_b[:10])

print(h_c[:10])