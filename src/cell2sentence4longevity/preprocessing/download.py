"""Dataset download module."""

from pathlib import Path
from urllib.parse import urlparse

import requests
from eliot import start_action
from tqdm import tqdm


def download_dataset(
    url: str, 
    output_dir: Path, 
    filename: str | None = None,
    force: bool = False
) -> Path:
    """Download a dataset from a URL.
    
    Args:
        url: URL to download from
        output_dir: Directory to save the downloaded file
        filename: Optional filename. If not provided, extracted from URL
        force: If True, re-download even if file exists. Default: False
        
    Returns:
        Path to the downloaded file
    """
    with start_action(
        action_type="download_dataset", 
        url=url, 
        output_dir=str(output_dir),
        force=force
    ) as action:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if filename is None:
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename or filename == '/':
                filename = "dataset.h5ad"
        
        output_path = output_dir / filename
        
        # Check if file already exists and we're not forcing re-download
        if output_path.exists() and not force:
            file_size = output_path.stat().st_size
            action.log(
                message_type="file_exists_skipping_download", 
                path=str(output_path),
                size_mb=file_size / (1024 * 1024)
            )
            print(f"✓ File already exists: {output_path} ({file_size / (1024 * 1024):.2f} MB)")
            return output_path
        
        action.log(message_type="starting_download", filename=filename)
        
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {filename}') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # No content-length header, download without progress bar
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        action.log(
            message_type="download_complete",
            path=str(output_path),
            size_bytes=output_path.stat().st_size
        )
        
        return output_path

