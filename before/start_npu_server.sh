#!/bin/bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
exec python -m rtp_llm.start_server "$@"
