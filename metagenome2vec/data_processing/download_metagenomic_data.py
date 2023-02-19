import subprocess
import os


def download_from_tsv_file(
    path_input: str, path_output: str, index_sample_id: int = 1, index_url: int = 10
) -> None:
    """

    Args:
        path_input (str): _description_
        path_output (str): _description_
        index_sample_id (int): _description_
        index_sample_url (int): _description_
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
