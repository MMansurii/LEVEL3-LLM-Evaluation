import pytest
from unittest.mock import Mock, patch
from src.data.downloader import DatasetDownloader


class TestDatasetDownloader:
    """Test DatasetDownloader class"""
    
    def test_initialization(self):
        """Test downloader initialization"""
        downloader = DatasetDownloader("test/dataset")
        assert downloader.dataset_name == "test/dataset"
        assert downloader.dataset is None
    
    @patch('src.data.downloader.load_dataset')
    def test_download_success(self, mock_load):
        """Test successful download"""
        mock_dataset = Mock()
        mock_dataset.keys.return_value = ['train', 'test']
        mock_load.return_value = mock_dataset
        
        downloader = DatasetDownloader("test/dataset")
        result = downloader.download()
        
        assert result == mock_dataset
        mock_load.assert_called_once()
