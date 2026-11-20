TASK: run my training on Kaggle's GPU using the Kaggle CLI (already installed and logged in as pavan01fe23bec156). Do NOT print, store or share any tokens/credentials. Never delete my Kaggle datasets. Ask before pushing anything other than the kernel in this folder.

FOLDER: this folder holds kernel-metadata.json and train_kaggle.py. The kernel attaches my datasets `pavan01fe23bec156/results` (OpenPCDet code, irp_results scripts, joint_start.pth) and `pavan01fe23bec156/object-detect` (KITTI).

STEPS
1. Verify kernel-metadata.json (enable_gpu true, enable_internet true, both datasets in dataset_sources). Check `kaggle kernels --help` for the right syntax of this CLI version.
2. Push: `kaggle kernels push -p .` and note the kernel URL/version.
3. Poll status every ~5 minutes with `kaggle kernels status pavan01fe23bec156/irp-fusion-train` until it is complete or errored. Report the status to me at each change.
4. Download the log/output: `kaggle kernels output pavan01fe23bec156/irp-fusion-train -p ./out`. Show me the last 60 lines of the log and the checkpoint list.
5. If it fails, read the error and tell me the cause. Typical suspects: (a) KITTI folders not found -> show me the folder listing it printed; (b) `python setup.py develop` fails (nvcc/torch/CUDA mismatch) -> show the compile error; (c) out-of-memory -> suggest BATCH=1; (d) missing python package -> add it to the pip line. You MAY fix train_kaggle.py for (a),(c),(d) and re-push once; for (b) stop and show me the error.
6. Confirm from the log that it really ran on the GPU (nvidia-smi section and "cuda True") and report seconds per epoch.

IMPORTANT: never change RUN/EPOCHS/the training command unless I ask. Do not start the control run (RUN="control_lidar") until I say so.
