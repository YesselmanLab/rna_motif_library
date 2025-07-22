#!/usr/bin/env python3
import os
import glob
from run_on_slurm import create_job_script


def generate_slurm_jobs(
    csv_dir: str = "splits",
    output_dir: str = "slurm_jobs",
    time_limit: str = "8:00:00",
    memory: str = "4G",
    cpus: int = 1,
    module_load: str = "module load anaconda",
    environment_setup: str = "conda activate rna_motif_env",
    template_path: str = "scripts/slurm_template.txt",
) -> None:
    """
    Generate SLURM job scripts for processing CSV files.

    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory to store SLURM scripts
        time_limit: Time limit for jobs
        memory: Memory allocation
        cpus: Number of CPUs per job
        module_load: Commands to load modules
        environment_setup: Commands to set up environment
        template_path: Path to SLURM template file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    # Create submit all script
    submit_all_script = os.path.join(output_dir, "submit_all_jobs.sh")
    with open(submit_all_script, "w") as f:
        f.write("#!/bin/bash\n\n")

        for idx, csv_file in enumerate(csv_files, 1):
            # Create job name from CSV filename
            csv_basename = os.path.basename(csv_file)
            job_name = f"process_{csv_basename}_{idx}"

            # Create command for processing
            command = f"""# python rna_motif_library/setup_database.py download-cifs {csv_file}
python rna_motif_library/setup_database.py process-cifs {csv_file}
python rna_motif_library/setup_database.py process-residues {csv_file}
python rna_motif_library/setup_database.py process-chains {csv_file}
python rna_motif_library/setup_database.py process-interactions {csv_file}
python rna_motif_library/setup_database.py generate-motifs {csv_file}"""

            # Generate SLURM script
            script_path = create_job_script(
                template_path=template_path,
                job_name=job_name,
                command=command,
                time_limit=time_limit,
                cpus=cpus,
                memory=memory,
                module_load=module_load,
                environment_setup=environment_setup,
            )

            # Add to submit all script
            f.write(f"sbatch {script_path}\n")

    # Make submit script executable
    os.chmod(submit_all_script, 0o755)

    print(f"✓ Generated {len(csv_files)} SLURM job scripts in '{output_dir}'")
    print(f"✓ Created submit script: {submit_all_script}")
    print(f"To submit all jobs, run: ./{submit_all_script}")


def main():
    """Example usage"""
    generate_slurm_jobs(
        csv_dir="splits",
        output_dir="slurm_jobs",
        time_limit="8:00:00",
        memory="4G",
        cpus=1,
        module_load="module load anaconda",
        environment_setup="conda activate rna_motif_env",
    )


if __name__ == "__main__":
    main()
