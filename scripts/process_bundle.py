"""Script to process launch bundles from a bundle dictionary with processing tqdm."""

import glob
import json
import os
import pickle
from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH, DATA_PATH


def worker(task_queue: Queue, done_queue: Queue):
    """Worker function to process each task in the task queue."""
    while True:
        item = task_queue.get()
        if item is None:
            break  # Sentinel received

        token_address, bundle_info = item
        try:
            creator = bundle_info["maker"]
            launch_block = bundle_info["launch_block"]

            result = {
                "bundle": bundle_info["bundle"],
                "bundle_launch": {},
                "bundle_creator_buy": {},
            }

            sol_transfer_path = (
                PROCESSED_DATA_PATH / "bundle" / f"{token_address}.jsonl"
            )
            if os.path.exists(sol_transfer_path):

                # Process Solana transfer data
                with open(sol_transfer_path, "r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        if row["tx_from"] == creator:
                            for block, bundle in bundle_info["bundle"].items():
                                if row["block_id"] <= block:
                                    if row["tx_to"] in set(
                                        [_["maker"] for _ in bundle]
                                    ) - set([creator]):
                                        if block == launch_block:
                                            if block not in result["bundle_launch"]:
                                                result["bundle_launch"][block] = {
                                                    "block_id": block,
                                                    "bundle": bundle,
                                                    "transfer": [row],
                                                }
                                            else:
                                                result["bundle_launch"][block][
                                                    "transfer"
                                                ].append(row)
                                        else:
                                            if (
                                                block
                                                not in result["bundle_creator_buy"]
                                            ):
                                                result["bundle_creator_buy"][block] = {
                                                    "block_id": block,
                                                    "bundle": bundle,
                                                    "transfer": [row],
                                                }
                                            else:
                                                result["bundle_creator_buy"][block][
                                                    "transfer"
                                                ].append(row)
            else:
                done_queue.put(token_address)  # Mark as done even if skipped
                continue

            output_path = (
                PROCESSED_DATA_PATH / "launch_bundle" / f"{token_address}.pickle"
            )
            with open(output_path, "wb") as f:
                pickle.dump(result, f)

        except Exception as e:
            print(f"[ERROR] {token_address}: {e}")
        finally:
            done_queue.put(token_address)  # Signal completion


if __name__ == "__main__":
    with open(PROCESSED_DATA_PATH / "bundle.pkl", "rb") as f:
        bundle_dict = pickle.load(f)

    os.makedirs(PROCESSED_DATA_PATH / "launch_bundle", exist_ok=True)

    finished_tasks = {
        os.path.splitext(os.path.basename(file))[0]
        for file in glob.glob(str(PROCESSED_DATA_PATH / "launch_bundle" / "*.pickle"))
    }

    # Filter tasks to process
    tasks = [
        (token_address, bundle_info)
        for token_address, bundle_info in bundle_dict.items()
        if token_address not in finished_tasks
    ]
    num_tasks = len(tasks)
    num_workers = max(1, min(cpu_count() - 5, num_tasks))

    # Multiprocessing queues
    task_queue = Queue(maxsize=1000)
    done_queue = Queue()

    # Start workers
    workers = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(task_queue, done_queue))
        p.start()
        workers.append(p)

    # Enqueue tasks
    for item in tasks:
        task_queue.put(item)

    # Send sentinel to each worker
    for _ in range(num_workers):
        task_queue.put(None)

    # tqdm for process completion
    with tqdm(total=num_tasks, desc="Processing launch bundles") as pbar:
        for _ in range(num_tasks):
            done_queue.get()
            pbar.update(1)

    # Wait for all workers to complete
    for p in workers:
        p.join()

    print("All bundler tasks processed.")
