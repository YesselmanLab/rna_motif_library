import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


from rna_motif_library.util import get_pdb_ids
from rna_motif_library.x3dna import X3DNAResidueFactory
from rna_motif_library.chain import get_cached_chains


class AngleDistribution:
    def __init__(self):
        """Initialize angle distribution with bins from -180 to 180 in 5 degree intervals"""
        self.bins = list(range(-180, 181, 5))  # Create bins edges
        self.counts = [0] * (len(self.bins) - 1)  # Initialize counts for each bin
        self.total_count = 0

    def add_measurement(self, angle: float):
        """Add a single angle measurement to the distribution"""
        # Find the bin index for this angle
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= angle < self.bins[i + 1]:
                self.counts[i] += 1
                self.total_count += 1
                break

    def get_population(self, angle: float) -> float:
        """Get the population (fraction) of measurements in the bin containing this angle"""
        if self.total_count == 0:
            return 0.0

        # Find the bin containing this angle
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= angle < self.bins[i + 1]:
                return self.counts[i] / self.total_count
        return 0.0

    def get_outlier_score(self, angle: float) -> float:
        """
        Calculate an outlier score for a given angle based on how far it deviates from the distribution.
        Returns a score between 0 and 1, where higher values indicate more extreme outliers.
        Angles beyond the 98th percentile are considered outliers.

        Args:
            angle (float): The angle to evaluate

        Returns:
            float: Outlier score between 0-1, where >0.98 indicates an outlier
        """
        if self.total_count == 0:
            return 0.0

        # Calculate cumulative distribution up to this angle
        cum_sum = 0
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= angle < self.bins[i + 1]:
                # Add half of current bin to get approximate position within bin
                cum_sum += self.counts[i] * 0.5 / self.total_count
                break
            cum_sum += self.counts[i] / self.total_count

        # Convert to outlier score (0-1)
        # Values closer to edges of distribution will have scores closer to 1
        if cum_sum <= 0.5:
            return 1.0 - (cum_sum * 2)
        else:
            return (cum_sum - 0.5) * 2

    def is_outlier(self, angle: float) -> bool:
        """
        Determine if an angle is an outlier based on the 98th percentile threshold.

        Args:
            angle (float): The angle to evaluate

        Returns:
            bool: True if the angle is an outlier, False otherwise
        """
        return self.get_outlier_score(angle) > 0.98

    def to_dict(self):
        """Convert distribution to dictionary for JSON serialization"""
        return {
            "bins": self.bins,
            "counts": self.counts,
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create distribution from dictionary"""
        dist = cls()
        dist.bins = data["bins"]
        dist.counts = data["counts"]
        dist.total_count = data["total_count"]
        return dist

    def to_json(self, filename: str):
        """Save distribution to JSON file"""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json(cls, filename: str):
        """Load distribution from JSON file"""
        with open(filename) as f:
            data = json.load(f)
        return cls.from_dict(data)


def generate_angle_distros():
    keys = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "chi",
        "v1",
        "v2",
        "v3",
        "v4",
    ]
    angles = {key: AngleDistribution() for key in keys}
    pdb_ids = get_pdb_ids()
    for pdb_id in pdb_ids:
        print(pdb_id)
        d = json.load(open(f"data/dssr_output/{pdb_id}.json"))
        torsions = {}
        for k in d["nts"]:
            data = {}
            for key in keys:
                if k[key] is not None:
                    angles[key].add_measurement(float(k[key]))
                    data[key] = k[key]
            r = X3DNAResidueFactory.create_from_string(k["nt_id"])
            torsions[r.get_str()] = data
        json.dump(torsions, open(f"data/jsons/torsions/{pdb_id}.json", "w"))

    for k, v in angles.items():
        v.to_json(f"data/angle_distributions/{k}_distribution.json")


def plot_score_histogram(ax, torsions, angles, bins=100):
    """
    Plot histogram of scores on the provided axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    torsions : dict
        Dictionary containing torsion data
    angles : dict
        Dictionary containing angle distributions
    bins : int, optional
        Number of bins for histogram (default: 100)
    """
    scores = []
    for k, v in torsions.items():
        score = 0
        count = 0
        for a, v1 in v.items():
            if v1 is None:
                continue
            if a not in angles:
                continue
            score += angles[a].get_outlier_score(v1)
            count += 1
        if count > 0:
            scores.append(score / count)

    ax.hist(scores, bins=bins)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")


def load_angle_distributions(keys):
    """
    Load angle distributions from files based on provided keys.

    Parameters:
    -----------
    keys : list
        List of angle names to load

    Returns:
    --------
    dict
        Dictionary mapping angle names to AngleDistribution objects
    """
    distribution_files = glob.glob("data/angle_distributions/*_distribution.json")

    angles = {}
    for file_path in distribution_files:
        # Extract angle name from filename (e.g., "alpha_distribution.json" -> "alpha")
        angle_name = os.path.basename(file_path).replace("_distribution.json", "")
        if angle_name not in keys:
            continue
        angles[angle_name] = AngleDistribution.from_json(file_path)
    return angles


def score_motifs():
    """
    Score motifs based on their torsion angles.
    """
    pdb_id = "6V3A"
    df = pd.read_json(f"data/dataframes/motifs/{pdb_id}.json")
    angles = load_angle_distributions(["alpha", "beta"])
    torsions = json.load(open(f"data/jsons/torsions/{pdb_id}.json"))
    df = df.query("motif_type == 'HAIRPIN'").copy()
    df["score"] = 0
    for i, row in df.iterrows():
        res_ids = row["residues"]
        scores = []
        for res_id in res_ids:
            if res_id not in torsions:
                continue
            score = 0
            count = 0
            for a, v1 in torsions[res_id].items():
                if v1 is None:
                    continue
                if a not in angles:
                    continue
                score += angles[a].get_population(v1)
                count += 1
            if count > 0:
                scores.append(score / count)
        df.loc[i, "score"] = np.mean(scores)
    avg = np.mean(df["score"])
    plt.figure(figsize=(10, 6))
    plt.scatter(df["num_residues"], df["score"], alpha=0.6)
    plt.xlabel("Number of Residues")
    plt.ylabel("Score")
    plt.title(f"Torsion Angle Score vs Number of Residues for {pdb_id}")
    plt.grid(True, alpha=0.3)
    plt.show()


def score_structures():
    pdb_ids = get_pdb_ids()
    angles = load_angle_distributions(["alpha", "beta"])
    data = []
    for pdb_id in pdb_ids:
        torsions = json.load(open(f"data/jsons/torsions/{pdb_id}.json"))
        scores = []
        for k, v in torsions.items():
            score = 0
            count = 0
            for a, v1 in v.items():
                if v1 is None:
                    continue
                if a not in angles:
                    continue
                score += angles[a].get_population(v1)
                count += 1
            if count > 0:
                scores.append(score / count)
        print(pdb_id, np.mean(scores), len(torsions))
        data.append(
            {
                "pdb_id": pdb_id,
                "score": np.mean(scores),
                "num_residues": len(torsions),
            }
        )
    df = pd.DataFrame(data)
    df.to_csv("torison_scores.csv", index=False)


def score_chains():
    pdb_id = "6V3A"
    chains = get_cached_chains(pdb_id)
    angles = load_angle_distributions(["alpha", "beta"])
    torsions = json.load(open(f"data/jsons/torsions/{pdb_id}.json"))
    scores = []
    for chain in chains:
        for res in chain:
            if res.get_str() not in torsions:
                continue
            score = 0
            count = 0
            for a, v1 in torsions[res.get_str()].items():
                if v1 is None:
                    continue
                if a not in angles:
                    continue
                score += angles[a].get_population(v1)
                count += 1
            if count > 0:
                scores.append(score / count)
            if len(scores) > 500:
                break
    fig, ax = plt.subplots(figsize=(30, 2.5))
    ax.plot(scores)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Score")
    ax.set_title(f"{pdb_id}")
    plt.show()


def main():
    # score_motifs()
    score_motifs()
    exit()
    angles = {}
    distribution_files = glob.glob("data/angle_distributions/*_distribution.json")

    keys = ["alpha", "beta"]

    # Call the function
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("4V9F")
    ax[1].set_title("7ASO")
    torsions_1 = json.load(open("data/jsons/torsions/4V9F.json"))
    plot_score_histogram(ax[0], torsions_1, angles)
    torsions_2 = json.load(open("data/jsons/torsions/7ASO.json"))
    plot_score_histogram(ax[1], torsions_2, angles)
    plt.show()


if __name__ == "__main__":
    main()
