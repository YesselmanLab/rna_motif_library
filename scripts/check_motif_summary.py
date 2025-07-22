import pandas as pd


def main():
    """
    main function for script
    """
    df = pd.read_json("data/dataframes/motifs/1GID.json")
    df = df.query("is_isolatable == 1")
    print(df)


if __name__ == "__main__":
    main()
