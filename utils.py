from pathlib import Path
import requests
from tqdm import tqdm
from zipfile import ZipFile

def download_file_with_progress_bar(url: str, dest_path: Path):
    """
    Download a file with a progress bar.

    Parameters
    ----------
    url : str
        URL of the file to download.
    dest_path : Path
        Path to save the downloaded file.

    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses
    total_size = int(response.headers.get('content-length', 0))  # Get the file size from headers
    block_size = 1024  # 1 Kilobyte per block

    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {dest_path.name}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for data in response.iter_content(block_size):
            file.write(data)
            progress.update(len(data))


def prepare_data_directory(procedure=None, url=None, data_dir=None):
    """
    Prepare the local 'data/' directory by downloading and extracting data from Zenodo,
    or from a custom URL if provided.

    Parameters
    ----------
    procedure (str, optional)
        One of ['procedure_1', 'procedure_2', 'procedure_3', 'procedure_4', 'all'].
        If None or 'all', downloads 'data.zip'.
    url (str, optional)
        Advanced use only: provide a custom URL pointing to a .zip file.
        This overrides the default Zenodo links.
    data_dir (str or Path, optional)
        Custom base path where data should be downloaded and extracted.
        Defaults to Path.cwd() / "data".

    Returns
    -------
    Path
        Path to the extracted dataset inside the 'data/' directory.
    """
    # Set base data directory
    base_data_dir = Path(data_dir) if data_dir is not None else Path.cwd() / "data"
    base_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Default filename mapping if no custom URL is given
    default_zip_map = {
        'procedure_1': 'Procedure_1_data.zip',
        'procedure_2': 'Procedure_2_and_3_data.zip',
        'procedure_3': 'Procedure_2_and_3_data.zip',
        'procedure_4': 'Procedure_4_data.zip',
        'all': 'data.zip',
        None: 'data.zip'
    }

    # Determine the zip filename and URL
    if url is not None:
        zip_filename = url.split("/")[-1]
    else:
        if procedure not in default_zip_map:
            raise ValueError(f"Invalid procedure: {procedure}. Must be one of {list(default_zip_map.keys())}")
        zip_filename = default_zip_map[procedure]
        url = f"https://zenodo.org/record/15213970/files/{zip_filename}"

    # Paths for zip and extracted data
    zip_path = base_data_dir / zip_filename
    expected_folder = zip_filename.replace('_data.zip', '').replace('.zip', '')
    extracted_path = base_data_dir / str(expected_folder + "_data")

    # Skip if data already exists
    if extracted_path.exists():
        print(f"Data already exists at: {extracted_path}")
        return extracted_path

    # Download and extract
    print(f"Downloading {zip_filename} from {url}...")
    download_file_with_progress_bar(url, zip_path)

    print(f"Extracting {zip_filename} into {base_data_dir}...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_data_dir)

    zip_path.unlink()
    print(f"Removed zip file: {zip_path}")

    return extracted_path




def resolve_figure_directory(save_figures, filename, default_folder="Figures"):
    """
    Resolves the output directory and base filename for figure saving.

    Parameters:
    ----------------
    save_figures (bool):
        Whether figures are to be saved.
    filename (str or None):
        Optional filename or path prefix for saved outputs.
    default_folder (str):
        Default folder name if no filename is provided.

    Returns:
    ----------------
    output_dir (str):
        Path to the folder where figures will be saved.
    base_filename (str):
        Base name used to generate individual filenames.
    """
        
    if not save_figures:
        return None, None

    if filename:
        filename = Path(filename)
        output_dir = filename.parent if filename.parent != Path('.') else Path(default_folder)
        base_filename = filename.stem
    else:
        output_dir = Path(default_folder)
        base_filename = "figure"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, base_filename

def generate_filename(base, index, extension):
    """
    Generate a sequential filename with a numeric suffix.

    Parameters
    ----------
    base (str)
        Base string for the filename (e.g., "power_map").
    index : int
        Index to append to the filename (start from index 0).
    extension (str)
        File extension (e.g., 'svg', 'png').

    Returns
    -------
    str
        Constructed filename with numeric suffix and extension.
    """
    return f"{base}_{index + 1:02d}.{extension}"

def override_dict_defaults(default_dict, override_dict=None):
    """
    Merges a default dictionary with user-specified overrides.

    Parameters:
    --------------
    default_dict (dict):
        Dictionary containing default key-value pairs.
    override_dict (dict, optional):
        Dictionary of user-defined key-value pairs that override defaults.

    Returns:
    --------------
    dict:
        Merged dictionary where user-defined keys replace defaults.
    """
        
    if override_dict is None:
        override_dict = {}
    return {**default_dict, **override_dict}
