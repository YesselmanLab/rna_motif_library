#!/usr/bin/env python3
import os
import glob


def main():
    slurm_jobs_dir = "slurm_jobs"
    os.makedirs(slurm_jobs_dir, exist_ok=True)

    # Find all csvs
    csv_files = glob.glob("splits/*.csv")

    submit_all_script = os.path.join(slurm_jobs_dir, "submit_all_jobs.sh")
    with open(submit_all_script, "w") as submit_script:
        submit_script.write("#!/bin/bash\n\n")

        for idx, csv_file in enumerate(csv_files):
            dir_name = os.path.basename(csv_file)
            job_name = f"job_{dir_name}_{idx+1}"
            slurm_script_path = os.path.join(slurm_jobs_dir, f"{job_name}.slurm")
            generate_slurm_script(slurm_script_path, csv_file, job_name)
            submit_script.write(f"sbatch {slurm_script_path}\n")

    # Make the submit_all_jobs.sh executable
    print(f"Slurm job scripts created in '{slurm_jobs_dir}' csv_file.")
    print(f"To submit all jobs, run: ./{submit_all_script}")


def generate_slurm_script(slurm_script_path, csv, job_name):
    with open("scripts/slurm_template.txt", "r") as template_file:
        template = template_file.read()

    # Format the template with the provided values
    slurm_script_content = template.format(job_name=job_name, csv=csv)

    with open(slurm_script_path, "w") as slurm_file:
        slurm_file.write(slurm_script_content)


if __name__ == "__main__":
    main()
