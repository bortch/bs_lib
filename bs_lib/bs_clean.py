import pandas as pd
import numpy as np
from unidecode import unidecode


def normalize_string(s: str) -> str:
    """
    Normalize a string

    Args:
        s (str): the string to normalize

    Returns:
        str: the normalized string
    """
    # remove leading and trailing whitespaces
    # convert to lowercase
    # replace spaces with underscores
    s = s.lower().strip().replace(" ", "_")
    s = unidecode(s)
    return s


def conditional_fill(
    df: pd.DataFrame, columns_values: dict, conditions: dict
) -> pd.DataFrame:
    """Fill empty value in the dataframe based on conditions

    Args:
        df (pd.DataFrame): The dataframe to fill
        columns_values (dict): The columns to fill and their values to fill with
        conditions (dict): The conditions to check before filling the values

    Raises:
        ValueError: If the column to fill does not exist in the dataframe
        ValueError: If the column to check does not exist in the dataframe

    Returns:
        pd.DataFrame: The dataframe with filled values
    """
    # columns_values dict example:
    # {
    # "counterparty_account": "unknown",
    # "transaction": "paiement"
    # }
    # this will fill "unknown" in counterparty_account
    # and "paiement" in transaction

    # condition dict example:
    # {
    # "transaction": "paiement",
    # "counterparty_account": "unknown"
    # }
    # this will check if the column "transaction" contains "paiement"
    # and if the counterparty_account is "unknown"

    # check if the columns to fill exist in the dataframe
    for key, value in conditions.items():
        if key not in df.columns:
            raise ValueError(
                f"Column {key} to test does not exist in the dataframe"
            )

    # check if the columns to fill exist in the dataframe
    for key, value in columns_values.items():
        if key not in df.columns:
            raise ValueError(
                f"Column {key} to fill does not exist in the dataframe"
            )

    # build each condition mask and combine them with &
    mask = np.ones(len(df), dtype=bool)
    for key, value in conditions.items():
        # print(f"key: {key}/ value: {value}")
        # Case np.nan
        if pd.isna(value):
            temp_mask = df[key].isna()
        # case string
        elif isinstance(value, str):
            temp_mask = df[key].str.contains(value, case=False)
        # case numeric
        elif isinstance(value, (int, float)):
            temp_mask = df[key] == value
        else:
            raise ValueError(f"Unsupported type for value {value}")
        # print(f"temp_mask: {temp_mask}")
        mask = mask & temp_mask
        # print(f"mask: {mask}")

    # fill the value
    df.loc[mask, list(columns_values.keys())] = list(columns_values.values())

    return df


def normalize_column_names(df) -> pd.DataFrame:
    """
    Normalize the column names of the dataset

    Args:

        df (pandas.DataFrame): the dataset to normalize

    Returns:

        df (pandas.DataFrame): the dataset with normalized column names
    """
    return df.rename(columns=normalize_string)


if __name__ == "__main__":

    # test the conditional_fill function
    df = pd.DataFrame(
        {
            "counterpart_account": [
                "unknown",
                "account1",
                "account2",
                "account3",
            ],
            "transaction": ["paiement", "retrait", "paiement", "retrait"],
            "amount": [np.NaN, 200, 300, 400],
        }
    )
    print("Initial dataframe")
    print(df)
    print("\n")
    df = conditional_fill(
        df,
        {"amount": 100, "counterpart_account": "account0"},
        {"transaction": "paiement", "counterpart_account": "unknown"},
    )
    print("Filled dataframe")
    print(df)
    print("\n")
