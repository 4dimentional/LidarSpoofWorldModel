import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

#ros2 domain rmw brute scanner to find active topics

DEFAULT_RMWS = [
    "rmw_fastrtps_cpp",
    "rmw_fastrtps_dynamic_cpp",
    "rmw_cyclonedds_cpp",
    "rmw_connextdds",
    "rmw_connext_cpp",
]

#topics always present in ros2
IGNORED_TOPICS = {"/rosout", "/parameter_events"}

def list_topics(domain_id: int, rmw: str, timeout: float):
    #returns list of topics for given domain and rmw none on timeout or error

    #clone environment and inject ros domain and rmw values
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["RMW_IMPLEMENTATION"] = rmw
    try:
        #run ros2 topic list command to list topics
        proc = subprocess.run(
            ["ros2", "topic", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=timeout,
            text=True,
        )

        #clean output lines
        topics = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

        #filter out default topics
        useful = [t for t in topics if t not in IGNORED_TOPICS]
        return useful
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        raise RuntimeError("ros2 CLI not found. Make sure ROS 2 environment is sourced.")

def scan_single(domain_id, rmw, timeout):
    #test single domain rmw and return details when topics appear
    topics = list_topics(domain_id, rmw, timeout)
    if topics:
        return {
            "domain_id": domain_id,
            "rmw": rmw,
            "topics": topics
        }
    return None

def main():
    #parse cli args build pairs and scan with thread pool
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=0)
    parser.add_argument("--max", type=int, default=232)
    parser.add_argument("--timeout", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rmws", type=str, default=",".join(DEFAULT_RMWS))
    args = parser.parse_args()

    #prepare combinations of domain ids and rmw vendors
    rmws = [r.strip() for r in args.rmws.split(",") if r.strip()]
    pairs = [(d, r) for d in range(args.min, args.max + 1) for r in rmws]

    print(f"Scanning domains {args.min}–{args.max} across {len(rmws)} DDS vendors...")

    #thread pool to scan pairs concurrently
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_single, d, r, args.timeout): (d, r) for d, r in pairs}
        try:
            for i, fut in enumerate(as_completed(futures)):
                res = fut.result()
                if res:
                    #print details of found ros2 domain
                    print("\nFound active ROS 2 domain")
                    print(f"ROS_DOMAIN_ID: {res['domain_id']}")
                    print(f"RMW: {res['rmw']}")
                    print("Topics:")
                    for t in res['topics']:
                        print(f"{t}")
                    return
                #progress update every 100 checks
                if i % 100 == 0:
                    print(f"...checked {i}/{len(pairs)}")
        except KeyboardInterrupt:
            print("\nScan interrupted.")
            return

    print("\nNo active domains found in range.")

if __name__ == "__main__":
    main()
