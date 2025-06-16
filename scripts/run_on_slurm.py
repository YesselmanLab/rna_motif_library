import subprocess
import glob
import os
from typing import List, Dict
import time
from collections import Counter

import pandas as pd
from simple_slurm import Slurm

from rna_motif_library.motif_analysis import (
    split_non_redundant_set,
    get_unique_motifs,
    get_unique_residues,
)
from rna_motif_library.parallel_utils import concat_dataframes_from_files


def get_job_status(job_id: int) -> str:
    """
    Get the status of a SLURM job by its job ID.

    Args:
        job_id (int): The SLURM job ID

    Returns:
        str: The status of the job (e.g., 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED')
    """
    try:
        # Use sacct command to get job status
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "--noheader"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Get the first line of output and strip whitespace
        status = result.stdout.strip().split("\n")[0]
        return status.strip().upper()

    except subprocess.CalledProcessError as e:
        print(f"Error getting job status: {e}")
        return "UNKNOWN"


def wait_for_jobs(job_ids: List[int], check_interval: int = 60) -> Dict[int, str]:
    """
    Wait for a list of SLURM jobs to complete or fail.

    Args:
        job_ids (list[int]): List of SLURM job IDs to monitor
        check_interval (int): Time in seconds between status checks (default: 60)

    Returns:
        dict[int, str]: Dictionary mapping job IDs to their final status
    """

    # Initialize status dictionary
    job_statuses = {job_id: "UNKNOWN" for job_id in job_ids}
    active_jobs = set(job_ids)

    while active_jobs:
        for job_id in list(active_jobs):
            status = get_job_status(job_id)
            job_statuses[job_id] = status
            # Remove job from active jobs if it's completed or failed
            if status == "FAILED":
                raise ValueError(f"Job {job_id} failed")
            if status in ["COMPLETED", "CANCELLED", "TIMEOUT"]:
                active_jobs.remove(job_id)

        # Count current statuses
        status_counts = Counter(job_statuses.values())
        # Print status summary
        print("\nCurrent job status summary:")
        for status, count in status_counts.items():
            print(f"{status}: {count} jobs")
        print(f"Total active jobs: {len(active_jobs)}")

        if active_jobs:
            time.sleep(check_interval)

    return job_statuses


def check_job_completion(job_ids: List[int]):
    n_jobs = len(job_ids)
    job_statuses = wait_for_jobs(job_ids, check_interval=200)
    completed_jobs = [
        job_id for job_id, status in job_statuses.items() if status == "COMPLETED"
    ]
    if len(completed_jobs) != n_jobs:
        raise ValueError(f"Not all jobs completed: {len(completed_jobs)}/{n_jobs}")
    print("All jobs completed")


def generate_slurm_jobs_for_splits():
    pass


def demo():
    # Create a Slurm object
    slurm = Slurm(
        job_name="simple_job",
        output="output_%j.log",
        error="error_%j.log",
        time="01:00:00",
        mem="4G",
        cpus_per_task=1,
    )

    # Define the command to run
    command = 'echo "Hello from SLURM!"'

    # Submit the job
    job_id = slurm.sbatch(command)
    print(f"Submitted job with ID: {job_id}")
    # can check status every 10 seconds since it's a small job
    job_statuses = wait_for_jobs([job_id], check_interval=10)
    print(job_statuses)


# check if a step was completed #########################################################


def are_generate_chains_completed() -> bool:
    df = pd.read_csv("data/csvs/rna_structures.csv")
    print("number of RNA structures in csv:", len(df))
    cif_files = glob.glob("data/pdbs/*.cif")
    print("number of cif files downloaded:", len(cif_files))
    if len(cif_files) < len(df):
        return False
    processed_cif_files = glob.glob("data/pdbs_dfs/*.parquet")
    print("number of processed cif files:", len(processed_cif_files))
    if len(processed_cif_files) < len(df):
        return False
    residues_json_files = glob.glob("data/jsons/residues/*.json")
    print("number of residues json files:", len(residues_json_files))
    if len(residues_json_files) < len(df):
        return False
    chains_json_files = glob.glob("data/jsons/chains/*.json")
    print("number of chains json files:", len(chains_json_files))
    if len(chains_json_files) < len(df):
        return False
    return True


# steps to generate database #########################################################


def generate_splits_and_download_cifs():
    pass


def generate_chains():
    os.makedirs("slurm_job_outputs/generate_chains", exist_ok=True)
    template_path = "scripts/slurm_templates/generate_chains.txt"
    template_str = open(template_path, "r").read()
    # Find all csvs
    csv_files = glob.glob("splits/*.csv")
    job_ids = []
    for i, csv_file in enumerate(csv_files):
        job_name = f"generate_chains_{i}"
        output_path = f"slurm_job_outputs/generate_chains/{job_name}_%j.out"
        error_path = f"slurm_job_outputs/generate_chains/{job_name}_%j.err"
        slurm_job = Slurm(
            job_name=job_name,
            output=output_path,
            error=error_path,
            time="8:00:00",
            mem="4G",
            cpus_per_task=1,
        )
        slurm_job.add_cmd(template_str.format(csv_path=csv_file))
        job_ids.append(slurm_job.sbatch())
    job_statuses = wait_for_jobs(job_ids, check_interval=200)
    completed_jobs = [
        job_id for job_id, status in job_statuses.items() if status == "COMPLETED"
    ]
    if len(completed_jobs) != len(csv_files):
        raise ValueError(
            f"Not all jobs completed: {len(completed_jobs)}/{len(csv_files)}"
        )
    print("All jobs completed")


def generate_ligand_data():
    # needs only one job try to run this locally
    pass


def generate_motifs():
    os.makedirs("slurm_job_outputs/generate_motifs", exist_ok=True)
    template_path = "scripts/slurm_templates/generate_motifs.txt"
    template_str = open(template_path, "r").read()
    csv_files = glob.glob("splits/*.csv")
    job_ids = []
    for i, csv_file in enumerate(csv_files):
        job_name = f"generate_motifs_{i}"
        output_path = f"slurm_job_outputs/generate_motifs/{job_name}_%j.out"
        error_path = f"slurm_job_outputs/generate_motifs/{job_name}_%j.err"
        slurm_job = Slurm(
            job_name=job_name,
            output=output_path,
            error=error_path,
            time="8:00:00",
            mem="4G",
            cpus_per_task=1,
        )
        slurm_job.add_cmd(template_str.format(csv_path=csv_file))
        job_ids.append(slurm_job.sbatch())
    check_job_completion(job_ids)


def generate_non_redundant_set_motifs():
    split_non_redundant_set("data/csvs/nrlist_3.369_3.5A.csv")
    os.makedirs("slurm_job_outputs/generate_non_redundant_set_motifs", exist_ok=True)
    template_path = "scripts/slurm_templates/generate_non_redundant_set_motifs.txt"
    template_str = open(template_path, "r").read()
    csv_files = glob.glob("splits/non_redundant_set_splits/*.csv")
    job_ids = []
    for i, csv_file in enumerate(csv_files):
        job_name = f"generate_non_redundant_set_motifs_{i}"
        output_path = (
            f"slurm_job_outputs/generate_non_redundant_set_motifs/{job_name}_%j.out"
        )
        error_path = (
            f"slurm_job_outputs/generate_non_redundant_set_motifs/{job_name}_%j.err"
        )
        slurm_job = Slurm(
            job_name=job_name,
            output=output_path,
            error=error_path,
            time="2:00:00",
            mem="4G",
            cpus_per_task=1,
        )
        slurm_job.add_cmd(template_str.format(csv_path=csv_file))
        job_ids.append(slurm_job.sbatch())
    check_job_completion(job_ids)


def run_analysis():
    os.makedirs("slurm_job_outputs/analysis", exist_ok=True)
    template_path = "scripts/slurm_templates/analysis.txt"
    template_str = open(template_path, "r").read()
    csv_files = glob.glob("splits/*.csv")
    job_ids = []
    for i, csv_file in enumerate(csv_files):
        job_name = f"analysis_{i}"
        output_path = f"slurm_job_outputs/analysis/{job_name}_%j.out"
        error_path = f"slurm_job_outputs/analysis/{job_name}_%j.err"
        slurm_job = Slurm(
            job_name=job_name,
            output=output_path,
            error=error_path,
            time="2:00:00",
            mem="4G",
            cpus_per_task=1,
        )
        slurm_job.add_cmd(template_str.format(csv_path=csv_file))
        job_ids.append(slurm_job.sbatch())
    check_job_completion(job_ids)
    df = concat_dataframes_from_files(
        glob.glob("data/dataframes/atlas_motifs_compared/*.json")
    )
    df.to_json(
        "data/summaries/other_motifs/atlas_motifs_compared.json", orient="records"
    )
    df = concat_dataframes_from_files(
        glob.glob("data/dataframes/dssr_motifs_compared/*.json")
    )
    df.to_json(
        "data/summaries/other_motifs/dssr_motifs_compared.json", orient="records"
    )


def main():
    os.makedirs("slurm_job_outputs", exist_ok=True)
    # are_generate_chains_completed()
    # generate_chains()
    # generate_ligand_data()
    # generate_motifs()
    # generate_non_redundant_set_motifs()
    run_analysis()


if __name__ == "__main__":
    main()
