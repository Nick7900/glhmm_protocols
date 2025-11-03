from pathlib import Path
import requests
from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
import sys
import subprocess
import urllib.request

def _in_colab() -> bool:
    """
    Detects whether the code is running on Google Colab.

    Returns:
    ----------------
    bool:
        True if running on Google Colab, otherwise False.
    """
    return "google.colab" in sys.modules


def _pip_install(*pkgs, quiet=True):
    """
    Installs one or more Python packages using pip.

    Parameters:
    ----------------
    *pkgs (str):
        Package names to install.
    quiet (bool):
        If True, suppresses installation output.
    """
    args = [sys.executable, "-m", "pip", "install"]
    if quiet:
        args.append("-q")
    subprocess.check_call(args + list(pkgs))


def _clone_repo_if_needed(REPO_URL = "https://github.com/Nick7900/glhmm_protocols.git",
                          REPO_DIR = "glhmm_protocols"):
    """
    Clones the GLHMM Protocols repository if running in Colab
    and the repository folder does not already exist.
    """
    if not _in_colab():
        return

    if not Path(REPO_DIR).exists():
        print("Cloning the repository...")
        subprocess.check_call(["git", "clone", "-q", REPO_URL])

    import os
    if Path(REPO_DIR).exists() and Path.cwd().name != REPO_DIR:
        os.chdir(REPO_DIR)


def _ensure_utils_present():
    """
    Ensures that the 'utils' module is available.
    If it is missing, downloads a minimal version from GitHub.
    """
    if Path("utils").exists() or Path("utils.py").exists():
        return

    print("utils not found → downloading from GitHub...")
    Path("utils").mkdir(exist_ok=True)
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/Nick7900/glhmm_protocols/main/utils/__init__.py",
        "utils/__init__.py",
    )
    print("utils ready.")


def _fix_pythonpath():
    """
    Adds the repository root to the Python path to enable imports
    from any notebook location.
    """
    here = Path.cwd().resolve()
    repo_root = here.parent if here.name == "Procedures" else here

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _install_dependencies():
    """
    Installs the required Python dependencies if they are not already available.
    This includes 'requests', and the 'glhmm' package.
    """
    try:
        import glhmm  # noqa: F401
    except Exception:
        _pip_install("git+https://github.com/vidaurre/glhmm")
    try:
        import requests  # noqa: F401
    except Exception:
        _pip_install("requests")



def setup_environment():
    """
    Sets up the environment so that notebooks can run both locally
    and on Google Colab.

    The setup performs the following actions:
    ----------------
    • Detects if running on Google Colab.
    • In Colab: clones the repository and installs dependencies.
    • Locally: ensures the 'utils' module exists.
    • Adds the repository root to the Python path.

    Returns:
    ----------------
    None
    """
    if _in_colab():
        print("Google Colab detected → setting up environment...")
        _clone_repo_if_needed()
    else:
        _ensure_utils_present()

    _install_dependencies()
    _fix_pythonpath()
    print("✅ Environment is ready.")
    return _in_colab()


__all__ = [
    "download_file_with_progress_bar",
    "prepare_data_directory",
    "resolve_figure_directory",
    "generate_filename",
    "override_dict_defaults",
]

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
