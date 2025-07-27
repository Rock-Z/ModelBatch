"""
Comprehensive tests for the ModelBatch logging system.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.modelbatch.logger import (
    get_logger,
    configure_logging,
    set_log_level,
    add_file_handler,
    get_core_logger,
    get_optuna_logger,
    get_training_logger,
    JSONFormatter,
    ContextLogger,
    LoggerManager,
    _manager
)


class TestJSONFormatter:
    """Test JSON formatting functionality."""
    
    def test_basic_formatting(self):
        """Test basic JSON log formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data
    
    def test_extra_fields(self):
        """Test that extra fields are included in JSON output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        # Add extra fields
        record.trial_id = 123
        record.custom_field = "custom_value"
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["trial_id"] == 123
        assert data["custom_field"] == "custom_value"
    
    def test_message_with_args(self):
        """Test formatting with message arguments."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message %s %d",
            args=("arg1", 42),
            exc_info=None
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["message"] == "Test message arg1 42"


class TestContextLogger:
    """Test context logger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_logger = logging.getLogger("test.context")
        self.base_logger.handlers.clear()
        self.handler = logging.StreamHandler()
        self.base_logger.addHandler(self.handler)
        self.base_logger.setLevel(logging.DEBUG)
        self.context_logger = ContextLogger(self.base_logger)
    
    def test_basic_logging(self):
        """Test basic logging methods."""
        with patch.object(self.base_logger, 'log') as mock_log:
            self.context_logger.debug("Debug message")
            self.context_logger.info("Info message")
            self.context_logger.warning("Warning message")
            self.context_logger.error("Error message")
            self.context_logger.critical("Critical message")
            
            assert mock_log.call_count == 5
            
            # Check log levels
            calls = mock_log.call_args_list
            assert calls[0][0][0] == logging.DEBUG
            assert calls[1][0][0] == logging.INFO
            assert calls[2][0][0] == logging.WARNING
            assert calls[3][0][0] == logging.ERROR
            assert calls[4][0][0] == logging.CRITICAL
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with patch.object(self.base_logger, 'log') as mock_log:
            with self.context_logger.context(trial_id=123, batch_id="abc"):
                self.context_logger.info("Message with context")
            
            # Check that context was added to extra
            call_kwargs = mock_log.call_args[1]
            assert call_kwargs['extra']['trial_id'] == 123
            assert call_kwargs['extra']['batch_id'] == "abc"
    
    def test_nested_context(self):
        """Test nested context managers."""
        with patch.object(self.base_logger, 'log') as mock_log:
            with self.context_logger.context(trial_id=123):
                with self.context_logger.context(step=5):
                    self.context_logger.info("Nested context message")
                
                # After inner context, outer context should remain
                self.context_logger.info("Outer context message")
            
            calls = mock_log.call_args_list
            
            # First call should have both trial_id and step
            inner_extra = calls[0][1]['extra']
            assert inner_extra['trial_id'] == 123
            assert inner_extra['step'] == 5
            
            # Second call should only have trial_id
            outer_extra = calls[1][1]['extra']
            assert outer_extra['trial_id'] == 123
            assert 'step' not in outer_extra
    
    def test_timer_context(self):
        """Test timer context manager."""
        with patch.object(self.base_logger, 'log') as mock_log:
            with self.context_logger.timer("test_operation"):
                time.sleep(0.01)  # Short sleep
            
            # Should have start and end messages
            assert mock_log.call_count == 2
            
            start_call = mock_log.call_args_list[0]
            end_call = mock_log.call_args_list[1]
            
            assert "Starting test_operation" in start_call[0][1]
            assert "Completed test_operation" in end_call[0][1]
            
            # End call should have duration info
            end_extra = end_call[1]['extra']
            assert 'operation' in end_extra
            assert 'duration_seconds' in end_extra
            assert end_extra['operation'] == "test_operation"
            assert end_extra['duration_seconds'] > 0
    
    def test_is_enabled_for(self):
        """Test isEnabledFor method delegation."""
        assert self.context_logger.isEnabledFor(logging.DEBUG)
        
        # Change base logger level
        self.base_logger.setLevel(logging.WARNING)
        assert not self.context_logger.isEnabledFor(logging.DEBUG)
        assert self.context_logger.isEnabledFor(logging.WARNING)


class TestLoggerManager:
    """Test logger manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a fresh manager for testing
        self.manager = LoggerManager()
    
    def test_initial_state(self):
        """Test manager initial state."""
        assert not self.manager._configured
        assert len(self.manager._loggers) == 0
    
    def test_configure_basic(self):
        """Test basic configuration."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_get_logger.return_value = mock_root
            
            self.manager.configure(level="DEBUG", console=True)
            
            assert self.manager._configured
            mock_root.setLevel.assert_called_once()
            mock_root.handlers.clear.assert_called_once()
    
    def test_configure_with_file(self):
        """Test configuration with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            with patch('logging.getLogger') as mock_get_logger:
                mock_root = MagicMock()
                mock_get_logger.return_value = mock_root
                
                self.manager.configure(
                    level="INFO",
                    log_file=str(log_file),
                    console=True
                )
                
                assert self.manager._configured
                # Should add both console and file handlers
                assert mock_root.addHandler.call_count == 2
    
    def test_configure_json_format(self):
        """Test JSON format configuration."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_get_logger.return_value = mock_root
            
            self.manager.configure(log_format="json")
            
            # Check that JSONFormatter was used
            handler_calls = mock_root.addHandler.call_args_list
            assert len(handler_calls) == 1
    
    def test_environment_variables(self):
        """Test environment variable configuration."""
        with patch.dict(os.environ, {
            'LOG_LEVEL': 'WARNING',
            'LOG_FORMAT': 'json',
            'LOG_FILE': '/tmp/test.log'
        }):
            with patch('logging.getLogger') as mock_get_logger:
                mock_root = MagicMock()
                mock_get_logger.return_value = mock_root
                
                self.manager.configure()
                
                # Should use environment values
                mock_root.setLevel.assert_called_once()
    
    def test_get_logger_creates_context_logger(self):
        """Test that get_logger returns ContextLogger instances."""
        logger = self.manager.get_logger("test.module")
        
        assert isinstance(logger, ContextLogger)
        assert "test.module" in self.manager._loggers
    
    def test_get_logger_caching(self):
        """Test that loggers are cached properly."""
        logger1 = self.manager.get_logger("test.module")
        logger2 = self.manager.get_logger("test.module")
        
        assert logger1 is logger2
    
    def test_set_level(self):
        """Test runtime level changes."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_root = MagicMock()
            mock_get_logger.return_value = mock_root
            
            self.manager.set_level("ERROR")
            
            mock_root.setLevel.assert_called_once()
    
    def test_add_file_handler(self):
        """Test adding file handlers at runtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "runtime.log"
            
            with patch('logging.getLogger') as mock_get_logger:
                mock_root = MagicMock()
                mock_get_logger.return_value = mock_root
                
                self.manager.add_file_handler(str(log_file))
                
                mock_root.addHandler.assert_called_once()


class TestPublicAPI:
    """Test public API functions."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test.api")
        assert isinstance(logger, ContextLogger)
    
    def test_convenience_loggers(self):
        """Test convenience logger functions."""
        core_logger = get_core_logger()
        optuna_logger = get_optuna_logger()
        training_logger = get_training_logger()
        
        assert isinstance(core_logger, ContextLogger)
        assert isinstance(optuna_logger, ContextLogger)
        assert isinstance(training_logger, ContextLogger)
        
        # Check logger names
        assert core_logger._logger.name == "modelbatch.core"
        assert optuna_logger._logger.name == "modelbatch.optuna"
        assert training_logger._logger.name == "modelbatch.training"
    
    def test_configure_logging(self):
        """Test configure_logging function."""
        with patch.object(_manager, 'configure') as mock_configure:
            configure_logging(level="DEBUG", console=False)
            mock_configure.assert_called_once_with(level="DEBUG", console=False)
    
    def test_set_log_level(self):
        """Test set_log_level function."""
        with patch.object(_manager, 'set_level') as mock_set_level:
            set_log_level("ERROR")
            mock_set_level.assert_called_once_with("ERROR")
    
    def test_add_file_handler(self):
        """Test add_file_handler function."""
        with patch.object(_manager, 'add_file_handler') as mock_add_file:
            add_file_handler("test.log", "json")
            mock_add_file.assert_called_once_with("test.log", "json")


class TestIntegration:
    """Integration tests for the logging system."""
    
    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "integration.log"
            
            # Create a fresh manager for this test to avoid singleton issues
            manager = LoggerManager()
            manager.configure(
                level="DEBUG",
                log_file=str(log_file),
                log_format="json"
            )
            
            # Get logger and use it
            logger = manager.get_logger("test.integration")
            
            # Test various logging features
            logger.info("Basic info message")
            
            with logger.context(trial_id=456):
                logger.warning("Warning with context")
                
                with logger.timer("operation"):
                    time.sleep(0.01)
            
            # Verify log file was created and contains expected content
            assert log_file.exists()
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Should have multiple log entries
            assert len(lines) >= 3  # info + warning + timer start/end
            
            # Parse JSON entries
            log_entries = [json.loads(line.strip()) for line in lines]
            
            # Check basic info message
            info_entry = log_entries[0]
            assert info_entry['level'] == 'INFO'
            assert info_entry['message'] == 'Basic info message'
            
            # Check context message
            context_entry = log_entries[1]
            assert context_entry['level'] == 'WARNING'
            assert context_entry['trial_id'] == 456
    
    def test_modelbatch_module_integration(self):
        """Test integration with modelbatch module imports."""
        # This should not raise any import errors
        from src.modelbatch import (
            get_logger,
            get_core_logger,
            get_optuna_logger,
            get_training_logger
        )
        
        # Test that all functions work
        logger = get_logger("test.module.integration")
        core_logger = get_core_logger()
        optuna_logger = get_optuna_logger()
        training_logger = get_training_logger()
        
        # All should be ContextLogger instances
        assert all(isinstance(l, ContextLogger) for l in [
            logger, core_logger, optuna_logger, training_logger
        ])
    
    def test_environment_configuration_integration(self):
        """Test environment-based configuration."""
        with patch.dict(os.environ, {
            'LOG_LEVEL': 'ERROR',
            'LOG_FORMAT': 'console'
        }):
            # Create new manager to test env config
            manager = LoggerManager()
            logger = manager.get_logger("test.env")
            
            # Should be configured from environment
            assert manager._configured
            
            # Test that debug messages are not logged at ERROR level
            # Need to check the actual underlying logger level
            assert logger._logger.getEffectiveLevel() == logging.ERROR
            
            # Use a mock to verify the actual log method isn't called for debug
            with patch.object(logger._logger, 'log') as mock_log:
                # Set up the mock to respect level filtering like real logger
                def side_effect(level, *args, **kwargs):
                    if logger._logger.isEnabledFor(level):
                        pass  # Would actually log
                
                mock_log.side_effect = side_effect
                
                logger.debug("Should not be logged")
                logger.error("Should be logged")
                
                # Check that debug was filtered out by our level check
                # Only error should get through to the log method
                assert mock_log.call_count == 1
                assert mock_log.call_args[0][0] == logging.ERROR 