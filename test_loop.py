import time
import argparse
import subprocess
import os


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--start','-s', type = int, default = 0)
    parser.add_argument('--to', '-t', type = int, default = 0)
    parser.add_argument('--eps', '-e', type = int, default = 1)
    parser.add_argument('--rviz', '-r', action="store_true")
    # parser.add_argument('--gui', '-g', action="store_true")
    parser.add_argument('--type', required=True)

    args = parser.parse_args()
    
    print(f"==== Run {args.eps} times starting from {args.start} to {args.to} ======")
    print(f"With RVIZ {args.rviz}")
    print(f"Record into CSV file: {args.type}.csv")
    
    if input("Check before proceed. Press Y to continue. ").lower() != "y":
        exit()
    
    for i in range(args.eps):
        for j in range(args.start, args.to +1):
            if args.rviz:
                process = subprocess.run(['python', "run.py",  "--world_idx", f"{j}", "--out", f"{args.type}.csv", "--type", f"{args.type}","--rviz"])
            else:
                process = subprocess.run(['python', "run.py",  "--world_idx", f"{j}", "--out", f"{args.type}.csv", "--type", f"{args.type}"])

    with open("test_log.txt", "a") as f:
        f.write(f"=============================\n")
        f.write(f"Run {args.eps} times in WORLDS {args.start} to {args.to}\n")
        f.write(f"With RVIZ {args.rviz}\n")
        f.write(f"Record into CSV file: {args.type}.csv\n")
        duration = round(time.time()- start_time,2)
        f.write(f"{(args.to - args.start +1)*args.eps} runs takes {duration} s\n")
        f.write(f"Or {round(duration/60,2)} minutes or {round(duration/3600,4)} hours\n")
        f.write("\n\n")


