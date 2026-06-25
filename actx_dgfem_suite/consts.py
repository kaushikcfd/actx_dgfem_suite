# Mapping from device name to peak flop rate in GFlOps/s
DEV_TO_PEAK_F64_GFLOPS = {
    "NVIDIA TITAN V": 6144,
    # forcing vector only (for now) for loopy.
    "NVIDIA H200 NVL": 30000,  # https://www.nvidia.com/en-in/data-center/h200/
}

# Mapping from device name to peak bandwidth in GB/s
DEV_TO_PEAK_BW = {
    "NVIDIA TITAN V": 652.8,
    "NVIDIA H200 NVL": 4800,  # https://www.nvidia.com/en-in/data-center/h200/
}
