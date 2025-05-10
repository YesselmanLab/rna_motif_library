import requests
import re


def get_latest_rna3dhub_release():
    """
    Fetches the latest RNA 3D Hub release number from the current release page.

    Returns:
        str: The latest release number (e.g., '3.382') or None if not found.
    """
    url = "https://rna.bgsu.edu/rna3dhub/nrlist/release/current"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")
        return None

    # Search for the release number pattern in the page content
    match = re.search(r"Release\s+(\d+\.\d+)", response.text)
    if match:
        return match.group(1)
    else:
        print("Release number not found in the page content.")
        return None


# Example usage
if __name__ == "__main__":
    latest_release = get_latest_rna3dhub_release()
    if latest_release:
        print(f"Latest RNA 3D Hub release: {latest_release}")
    else:
        print("Could not retrieve the latest release number.")
