import argparse
import subprocess
import time

print(f"==== Run starting from 0 to 299 ======")

val_ids = [1, 2, 3, 9, 23, 26, 27, 33, 36, 42, 47, 62, 68, 76, 78, 83, 87, 99,
           101, 102, 107, 113, 117, 118, 122, 124, 131, 133, 134, 138, 142, 146,
           147, 149, 154, 156, 159, 162, 164, 173, 178, 181, 184, 186, 187, 188,
           191, 197, 201, 207, 208, 209, 212, 214, 216, 218, 221, 224, 228, 231,
           234, 239, 248, 249, 251, 253, 281, 283, 284, 293, 298, 299]

RETRIES = 1
for j in val_ids:
# for j in range(270, 300, 5):
    print(f"==== Running world {j} ====")
    for attempt in range(RETRIES):  # Retry up to 3 times
        print(f"==== Attempt {attempt + 1} ====")
        result = subprocess.run(["python", "./scripts/run_rviz_imit.py", "--world_idx", str(j), "--out", "imit_val_ddim_1000_5_m16.txt"])
        # if result.returncode == 200:  # Break the loop if the return code is 200
        #     print("==== Success ====")
        #     break
        # print(f"Attempt {attempt + 1}, retrying...")
        time.sleep(1)

    print(f"==== Done with world {j} ====")