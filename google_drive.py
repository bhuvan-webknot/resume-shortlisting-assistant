import os
import io
import re
import logging
import tempfile
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TOKEN_FILE = 'drive_token.json'
CREDENTIALS_FILE = 'credentials.json'


def extract_folder_id(folder_input: str) -> str:
    """Extract Google Drive folder ID from a URL or direct ID."""
    folder_input = folder_input.strip()

    if re.match(r'^[a-zA-Z0-9_-]{25,}$', folder_input):
        return folder_input

    m = re.search(r'/drive/folders/([a-zA-Z0-9_-]+)', folder_input)
    if m:
        return m.group(1)

    m = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', folder_input)
    if m:
        return m.group(1)

    raise ValueError(f"Could not extract Google Drive folder ID from: {folder_input}")


def authenticate() -> Credentials:
    """Obtain valid Google Drive API credentials via OAuth 2.0."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"OAuth credentials file '{CREDENTIALS_FILE}' not found.\n"
                    "1. Go to https://console.cloud.google.com/apis/credentials\n"
                    "2. Create an OAuth 2.0 Client ID (Desktop App)\n"
                    "3. Download the JSON and save it as 'credentials.json' in the project root"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())
        logger.info(f"OAuth token saved to {TOKEN_FILE}")

    return creds


def list_files(service, folder_id: str) -> List[dict]:
    """List all non-trashed files inside a Google Drive folder."""
    files = []
    page_token = None

    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token
        ).execute()

        files.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if not page_token:
            break

    return files


def download_file(service, file_id: str, file_name: str, dest_dir: str) -> str:
    """Download a single file from Google Drive to a local directory."""
    request = service.files().get_media(fileId=file_id)
    dest_path = os.path.join(dest_dir, file_name)

    with io.FileIO(dest_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return dest_path


def download_resumes_from_drive(folder_input: str, dest_dir: Optional[str] = None) -> str:
    """Download all .txt / .pdf files from a Google Drive folder to a local directory.

    Returns the path to the directory containing the downloaded files.
    """
    folder_id = extract_folder_id(folder_input)
    logger.info(f"Google Drive folder ID: {folder_id}")

    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="gdrive_resumes_")

    os.makedirs(dest_dir, exist_ok=True)

    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    files = list_files(service, folder_id)
    logger.info(f"Found {len(files)} files in the folder")

    downloaded = []
    allowed = ('.txt', '.pdf')
    for f in files:
        name = f.get('name', '')
        if any(name.lower().endswith(ext) for ext in allowed):
            dest_path = download_file(service, f['id'], name, dest_dir)
            downloaded.append(dest_path)
            logger.info(f"Downloaded: {name}")

    if not downloaded:
        logger.warning("No .txt or .pdf resume files found in the specified Google Drive folder")
    else:
        logger.info(f"Downloaded {len(downloaded)} resume file(s) to {dest_dir}")

    return dest_dir
