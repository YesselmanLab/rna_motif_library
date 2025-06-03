import concurrent.futures
from typing import List, Any, Callable, Optional
from tqdm import tqdm
import pandas as pd
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
    # if only one process, just run sequentially, required for cluster jobs
    if processes == 1:
        for item in items:
            all_results.append(func(item))
        return all_results

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
    # if only one process, just run sequentially, required for cluster jobs
    if processes == 1:
        for item in items:
            results.append(func(item))
        return results

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


def concat_dataframes_from_files(
    file_paths: List[str]
) -> pd.DataFrame:
    """
    Read multiple JSON or CSV files and concatenate them into a single DataFrame
    using parallel threads. File type is automatically detected from file extension.

    Args:
        file_paths (List[str]): List of paths to the files to be read

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all data from the input files
    """

    def read_file(file_path: str) -> Optional[pd.DataFrame]:
        try:
            # Detect file type from extension
            if file_path.lower().endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.lower().endswith('.json'):
                return pd.read_json(file_path)
            else:
                log.error(f"Unsupported file type for {file_path}. Must end in .csv or .json")
                return None
        except Exception as e:
            log.error(f"Error reading file {file_path}: {e}")
            return None

    # Use run_w_threads to read files in parallel
    dfs = run_w_threads(
        items=file_paths,
        func=read_file,
        threads=min(32, len(file_paths)),  # Limit max threads to 32
        desc="Reading files",
        show_progress=True,
    )
    # Filter out None results and concatenate
    dfs = [df for df in dfs if df is not None]

    if not dfs:
        log.warning("No dataframes were successfully loaded")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
