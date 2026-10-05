"""Runs on the rented GPU: how much device memory one process can actually allocate, via the CUDA driver API."""
import ctypes, json, subprocess, sys

cuda = ctypes.CDLL('libcuda.so.1')
MIB = 2**20


def check(result, what):
    if result != 0:
        sys.exit(f'{what} failed with CUDA error {result}')


def alloc(size):
    pointer = ctypes.c_uint64()
    return pointer.value if cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(size)) == 0 else None


def info():
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    check(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), 'cuMemGetInfo')
    return free.value, total.value


check(cuda.cuInit(0), 'cuInit')
device, context = ctypes.c_int(), ctypes.c_void_p()
check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'cuDeviceGet')
check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), 'cuDevicePrimaryCtxRetain')
check(cuda.cuCtxSetCurrent(context), 'cuCtxSetCurrent')
free, total = info()

# Largest single allocation: binary search at 2 MiB granularity, freeing each success.
low, high = 0, free // (2 * MIB) + 1
while high - low > 1:
    middle = (low + high) // 2
    pointer = alloc(middle * 2 * MIB)
    if pointer is None:
        high = middle
    else:
        cuda.cuMemFree_v2(ctypes.c_uint64(pointer))
        low = middle
largest = low * 2 * MIB

# Fill: as many 1 GiB blocks as fit, then 2 MiB blocks; their sum is what one process can hold at once.
held, filled = [], 0
for size in [1024 * MIB, 2 * MIB]:
    while (pointer := alloc(size)) is not None:
        held.append(pointer)
        filled += size
free_after, _ = info()
for pointer in held:
    cuda.cuMemFree_v2(ctypes.c_uint64(pointer))

smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True).stdout.strip()
print('PROBE ' + json.dumps({'nvidia_smi': smi, 'smi_total_bytes': int(smi.split(',')[1]) * MIB, 'cuda_total_bytes': total,
                             'free_after_context_bytes': free, 'largest_single_bytes': largest, 'fillable_bytes': filled,
                             'free_after_fill_bytes': free_after}))
