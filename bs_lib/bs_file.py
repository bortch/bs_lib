from os import listdir
from os.path import isfile, join
import pandas as pd

import bs_lib.bs_string as bstring


def get_list_dir(
    in_directory_path,
    with_extension=None,
    match_terms=None,
    exclude_terms=None,
    separator=".",
    verbose=False,
):
    """Get all files from in_directory_path without or with_extension, matching given match_term or not in exclude_term.
    It creates a dict with the filename as value and as key the first part of the filename splitted around a given separator or '.'

    Args:
        in_directory_path (String): The path to the directory
        with_extension (list, optional): a list of all extensions needed. Defaults to None.
        match_terms (list, optional): a list of terms to match. Defaults to [].
        exclude_terms (list, optional): a list of term to exclude. Defaults to [].
        separator (str, optional): the separator used to split filename in key and value. Defaults to ".".
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        dict: A dict where 'value' contains the whole filename and 'keys' are the first part of the splitted filename around the given separator
    """
    files = {}
    if not isinstance(match_terms, list):
        match_terms = []
    if not isinstance(exclude_terms, list):
        exclude_terms = []
    all_files_in_directory = listdir(in_directory_path)
    for filename in sorted(all_files_in_directory):
        if verbose:
            print(f"\npath:{join(in_directory_path, filename)}")
            print(f"is a file? {isfile(join(in_directory_path, filename))}")
            print(
                f"has the right extension? {( not with_extension or filename.endswith(with_extension))}"
            )
            print(
                f"matches terms? {len(match_terms)<1 or any(x in filename for x in match_terms) or filename in match_terms}"
            )
            print(
                f"matches excluded terms? {len(exclude_terms)<1 or not any(x in filename for x in exclude_terms)}"
            )
        # fetch model filename
        if (
            isfile(join(in_directory_path, filename))
            and (not with_extension or filename.endswith(with_extension))
            and (
                len(exclude_terms) < 1
                or not any(x in filename for x in exclude_terms)
            )
            and (
                len(match_terms) < 1 or any(x in filename for x in match_terms)
            )
        ):
            file_name = filename.split(separator)[0]
            files[file_name] = filename
    return files


def load_all_csv(
    dataset_path="dataset", exclude=None, index=None, verbose=False
):
    """Read every csv files from a directory and return a dictionnay of pandas DataFrames.
    Dictionnary's keys are the filename without extension.

    Args:
        dataset_path (str, optional): the directory's path were csv files are stored. Defaults to "dataset".
        exclude (str,optional): A list of filename to exclude with extension. Defaults to None
        index (int,optional): which column is an index. Defaults to None.
        verbose (bool,optional) Default to False.

    Returns:
        dict: a dictionnay of pandas DataFrames. Dictionnary's keys are the filename in snake case and without extension.
    """
    if not isinstance(exclude, list):
        exclude = []
    files = [
        join(dataset_path, f)
        for f in listdir(dataset_path)
        if (
            isfile(join(dataset_path, f))
            and f.endswith(".csv")
            and f not in exclude
        )
    ]
    if verbose:
        print(f"loading:{files}")
    all_data = load_csv_files_as_dict(files, index=index, verbose=verbose)
    return all_data


def load_csv_file(file_path, index=None):
    """Load a csv file as a dataset. Columns name are sanitized as lower snake case

    Args:
        file_path (string): the file's path

    Returns:
        Dataframe: a Pandas DataFrame
    """
    df = pd.read_csv(file_path, index_col=index)
    df.columns = bstring.to_snake_case(df.columns.tolist())
    return df


def load_csv_files_as_dict(files, index=None, verbose=False):
    all_data = {}
    for f in files:
        df = load_csv_file(f, index=index)
        # add dataframe to dict
        key = bstring.to_snake_case(f[:-4])
        key = key.split("/")[-1]
        if verbose:
            print(f"Parsing {f} key:{key}")
        all_data[key] = df

    return all_data


def load_csv(dataset_path):
    """Load a csv file as a dataset. Columns name are sanitized as lower snake case

    Args:
        dataset_path (string): the file's path

    Returns:
        Dataframe: a Pandas DataFrame
    """
    df = pd.read_csv(dataset_path)
    df.columns = bstring.to_snake_case(df.columns.tolist())
    return df


def concat_csv_files_as_dataframe(directory_path):
    dict_of_dataframes = load_all_csv(directory_path)
    all_data = pd.concat(dict_of_dataframes.values())
    return all_data
