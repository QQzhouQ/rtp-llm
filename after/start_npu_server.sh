#!/bin/bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
TORCH_LIBS="$(python -c 'import torch; import os; print(os.path.join(torch.__path__[0], "libs"))' 2>/dev/null)"
TORCH_LIB="$(python -c 'import torch; import os; print(os.path.join(torch.__path__[0], "lib"))' 2>/dev/null)"
TORCH_NPU_LIB="$(python -c 'import torch_npu; import os; print(os.path.join(torch_npu.__path__[0], "lib"))' 2>/dev/null)"
export LD_LIBRARY_PATH=${TORCH_LIBS}:${TORCH_LIB}:${TORCH_NPU_LIB}:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
exec python -m rtp_llm.start_server "$@"
