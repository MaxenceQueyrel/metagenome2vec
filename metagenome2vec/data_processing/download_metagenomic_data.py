import subprocess
import os


def download_from_tsv_file(
    path_input: str, path_output: str, index_sample_id: int = 1, index_url: int = 10
) -> None:
    """
    Downloads metagenomic data from a TSV file containing sample IDs and URLs to download the data.

    Args:
        path_input (str): The path to the input TSV file.
        path_output (str): The path to the output directory where the downloaded data will be saved.
        index_sample_id (int): The index of the column in the TSV file containing the sample IDs (default: 1).
        index_url (int): The index of the column in the TSV file containing the download URLs (default: 10).
    """
    path_script: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "download_metagenomic_data_from_tsv_file.sh",
    )
    subprocess.call(
        [
            path_script,
            "--path-input",
            path_input,
            "--path-output",
            path_output,
            "--index-sample-id",
            str(index_sample_id),
            "--index-url",
            str(index_url),
        ]
    )
