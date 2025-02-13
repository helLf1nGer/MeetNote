import os
import json
import requests
import time
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

class ColabGPUService:
    def __init__(self):
        # URL of your deployed Colab notebook service
        self.colab_url = os.getenv('COLAB_SERVICE_URL')
    
    def diarize_with_colab(self, audio_data, n_speakers):
        """
        Send audio data to Colab for GPU-accelerated diarization
        """
        try:
            payload = {
                'audio_data': audio_data,
                'n_speakers': n_speakers
            }
            response = requests.post(f"{self.colab_url}/diarize", json=payload)
            return response.json()
        except Exception as e:
            raise Exception(f"Colab GPU service error: {str(e)}")

class ColabServiceManager:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.SCOPES = [
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]
        self.notebook_name = 'colab_diarization.ipynb'
        self.colab_folder_name = 'Colab Notebooks'
        self.credentials = None
        self._setup_credentials()

    def _setup_credentials(self):
        """Setup Google API credentials"""
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        self.credentials = creds

    def _get_or_create_notebook(self):
        """Get existing notebook in Drive's Colab Notebooks folder"""
        try:
            service = build('drive', 'v3', credentials=self.credentials)
            
            # First, find the Colab Notebooks folder
            logger.info(f"Looking for '{self.colab_folder_name}' folder...")
            folder_results = service.files().list(
                q=f"name='{self.colab_folder_name}' and mimeType='application/vnd.google-apps.folder'",
                spaces='drive',
                fields="files(id, name)"
            ).execute()
            
            folders = folder_results.get('files', [])
            if not folders:
                logger.error(f"Could not find '{self.colab_folder_name}' folder in Drive")
                raise FileNotFoundError(f"Could not find '{self.colab_folder_name}' folder")
            
            folder_id = folders[0]['id']
            logger.info(f"Found Colab folder: {self.colab_folder_name} (ID: {folder_id})")
            
            # Now search for the notebook in that folder
            logger.info(f"Looking for {self.notebook_name} in {self.colab_folder_name}...")
            notebook_results = service.files().list(
                q=f"name='{self.notebook_name}' and '{folder_id}' in parents and mimeType='application/x-ipynb+json'",
                spaces='drive',
                fields="files(id, name)"
            ).execute()
            
            notebooks = notebook_results.get('files', [])
            if not notebooks:
                logger.error(f"Could not find {self.notebook_name} in {self.colab_folder_name}")
                
                # List all files in the folder for debugging
                all_files = service.files().list(
                    q=f"'{folder_id}' in parents",
                    spaces='drive',
                    fields="files(id, name)"
                ).execute().get('files', [])
                
                logger.info(f"Files in {self.colab_folder_name}:")
                for file in all_files:
                    logger.info(f"  - {file['name']}")
                    
                raise FileNotFoundError(f"Could not find {self.notebook_name}")
            
            self.notebook_id = notebooks[0]['id']
            logger.info(f"Found notebook: {self.notebook_name} (ID: {self.notebook_id})")
            return self.notebook_id
            
        except Exception as e:
            logger.error(f"Error searching for notebook: {str(e)}")
            raise

    def start_colab_service(self):
        """Start the Colab service using existing notebook"""
        try:
            # 1. Get existing notebook ID
            self.notebook_id = self._get_or_create_notebook()
            
            # 2. Execute notebook cells
            self._execute_notebook()
            
            # 3. Wait for and get ngrok URL
            ngrok_url = self._wait_for_ngrok_url()
            
            # 4. Update config with new URL
            self.config_manager.config['colab_service_url'] = ngrok_url
            self.config_manager.save_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Colab service: {str(e)}")
            return False

    def _execute_notebook(self):
        """Execute the notebook cells"""
        service = build('drive', 'v3', credentials=self.credentials)
        
        # Use Google Colab API to execute cells
        # This part requires additional implementation using Colab's API
        pass

    def _wait_for_ngrok_url(self, timeout=60):
        """Wait for ngrok URL to be available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check notebook output for ngrok URL
            # This needs to be implemented using Colab's output capture
            time.sleep(5)
        raise TimeoutError("Timeout waiting for ngrok URL")

    def _test_connection(self, url):
        """Test connection to the Colab service"""
        try:
            response = requests.get(f"{url}/test", verify=False, timeout=10)
            if response.status_code == 200:
                logger.info("Successfully connected to Colab service")
                return True
            else:
                logger.error(f"Failed to connect to Colab service: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")
            return False

    def stop_service(self):
        """Stop the Colab service"""
        if self.notebook_id:
            try:
                service = build('drive', 'v3', credentials=self.credentials)
                # Implement notebook shutdown
                pass
            except Exception as e:
                logger.error(f"Error stopping service: {str(e)}")
