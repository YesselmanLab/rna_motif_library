import concurrent.futures
from typing import List, Any, Callable, Optional
from tqdm import tqdm
from rna_motif_library.logger import get_logger

log = get_logger("PARALLEL")


def run_w_threads_in_batches(
    items: List[Any],
    func: Callable,
    threads: int,
    batch_size: int = 100,
    desc: str = "Processing",
    show_progress: bool = True,
) -> List[Any]:
    """
    Process items in batches with parallel threading, using a single thread pool for all batches.

    Args:
        items (list): List of items to process
        func (callable): Function to process each item
        threads (int): Number of threads to use
        batch_size (int, optional): Size of each batch. Defaults to 100.
        desc (str, optional): Description for progress bar. Defaults to "Processing".
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        list: Combined results from all batches
    """
    all_results = []

    # Create a single thread pool for all batches
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for i in range(0, len(items), batch_size):
            log.info(
                f"Processing batch {i//batch_size + 1} of {(len(items) + batch_size - 1)//batch_size}"
            )
            batch = items[i : i + batch_size]

            try:
                # Submit all tasks in the batch
                future_to_item = {executor.submit(func, item): item for item in batch}

                # Process results as they complete
                if show_progress:
                    iterator = tqdm(
                        concurrent.futures.as_completed(future_to_item),
                        total=len(batch),
                        desc=desc,
                    )
                else:
                    iterator = concurrent.futures.as_completed(future_to_item)

                for future in iterator:
                    try:
                        result = future.result()
                        if result is not None:  # Skip None results
                            all_results.append(result)
                    except Exception as e:
                        item = future_to_item[future]
                        log.error(f"Error processing item {item}: {e}")
                    if show_progress:
                        iterator.update(1)

            except Exception as e:
                log.error(f"Error processing batch starting at {i}: {e}")
                continue

    return all_results


def run_w_processes_in_batches(
    items: List[Any],
    func: Callable,
    processes: int,
    batch_size: int = 100,
    desc: str = "Processing",
    show_progress: bool = True,
) -> List[Any]:
    """
    Process items in batches with parallel processing, using a single process pool for all batches.

    Args:
        items (list): List of items to process
        func (callable): Function to process each item
        processes (int): Number of processes to use
        batch_size (int, optional): Size of each batch. Defaults to 100.
        desc (str, optional): Description for progress bar. Defaults to "Processing".
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        list: Combined results from all batches
    """
    all_results = []

    # Create a single process pool for all batches
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        for i in range(0, len(items), batch_size):
            log.info(
                f"Processing batch {i//batch_size + 1} of {(len(items) + batch_size - 1)//batch_size}"
            )
            batch = items[i : i + batch_size]

            try:
                # Submit all tasks in the batch
                future_to_item = {executor.submit(func, item): item for item in batch}

                # Process results as they complete
                if show_progress:
                    iterator = tqdm(
                        concurrent.futures.as_completed(future_to_item),
                        total=len(batch),
                        desc=desc,
                    )
                else:
                    iterator = concurrent.futures.as_completed(future_to_item)

                for future in iterator:
                    try:
                        result = future.result()
                        if result is not None:  # Skip None results
                            all_results.append(result)
                    except Exception as e:
                        item = future_to_item[future]
                        log.error(f"Error processing item {item}: {e}")
                    if show_progress:
                        iterator.update(1)

            except Exception as e:
                log.error(f"Error processing batch starting at {i}: {e}")
                continue

    return all_results


def run_w_threads(
    items: List[Any],
    func: Callable,
    threads: int,
    desc: str = "Processing",
    show_progress: bool = True,
) -> List[Any]:
    """
    Process items with parallel threading.

    Args:
        items (list): List of items to process
        func (callable): Function to process each item
        threads (int): Number of threads to use
        desc (str, optional): Description for progress bar. Defaults to "Processing".
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        list: Results from processing
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): item for item in items}

        # Process results as they complete
        if show_progress:
            iterator = tqdm(
                concurrent.futures.as_completed(future_to_item),
                total=len(items),
                desc=desc,
            )
        else:
            iterator = concurrent.futures.as_completed(future_to_item)

        for future in iterator:
            try:
                result = future.result()
                if result is not None:  # Skip None results
                    results.append(result)
            except Exception as e:
                item = future_to_item[future]
                log.error(f"Error processing item {item}: {e}")
            if show_progress:
                iterator.update(1)

    return results


def run_w_processes(
    items: List[Any],
    func: Callable,
    processes: int,
    desc: str = "Processing",
    show_progress: bool = True,
) -> List[Any]:
    """
    Process items with parallel processing.

    Args:
        items (list): List of items to process
        func (callable): Function to process each item
        processes (int): Number of processes to use
        desc (str, optional): Description for progress bar. Defaults to "Processing".
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        list: Results from processing
    """
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): item for item in items}

        # Process results as they complete
        if show_progress:
            iterator = tqdm(
                concurrent.futures.as_completed(future_to_item),
                total=len(items),
                desc=desc,
            )
        else:
            iterator = concurrent.futures.as_completed(future_to_item)

        for future in iterator:
            try:
                result = future.result()
                if result is not None:  # Skip None results
                    results.append(result)
            except Exception as e:
                item = future_to_item[future]
                log.error(f"Error processing item {item}: {e}")
            if show_progress:
                iterator.update(1)

    return results
