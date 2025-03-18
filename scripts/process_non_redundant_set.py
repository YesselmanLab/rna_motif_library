import click
import pandas as pd


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
def main(csv_path):
    df = pd.read_csv(csv_path, header=None)
    data = []
    for _, row in df.iterrows():
        row = row.tolist()
        spl = row[1].split("|")
        if spl[0] not in data:
            data.append(spl[0])
    df = pd.DataFrame(data, columns=["pdb_id"])
    df.to_csv("non_redundant_set.csv", index=False)


if __name__ == "__main__":
    main()
