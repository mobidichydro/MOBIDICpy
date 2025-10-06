"""Tests for logging configuration."""

from pathlib import Path

from loguru import logger

from mobidic.utils.logging import configure_logger


class TestConfigureLogger:
    """Tests for configure_logger function."""

    def setup_method(self):
        """Remove all handlers before each test."""
        logger.remove()

    def teardown_method(self):
        """Clean up after each test."""
        logger.remove()

    def test_default_configuration(self, capsys):
        """Test logger with default configuration."""
        configure_logger()

        # Log a test message
        logger.info("Test message")

        # Capture output
        captured = capsys.readouterr()
        assert "Test message" in captured.out
        assert "INFO" in captured.out

    def test_custom_log_level_debug(self, capsys):
        """Test logger with DEBUG level."""
        configure_logger(level="DEBUG")

        logger.debug("Debug message")
        logger.info("Info message")

        captured = capsys.readouterr()
        assert "Debug message" in captured.out
        assert "Info message" in captured.out
        assert "DEBUG" in captured.out

    def test_custom_log_level_warning(self, capsys):
        """Test logger with WARNING level filters out INFO."""
        configure_logger(level="WARNING")

        logger.info("Info message")
        logger.warning("Warning message")

        captured = capsys.readouterr()
        assert "Info message" not in captured.out
        assert "Warning message" in captured.out
        assert "WARNING" in captured.out

    def test_custom_format_string(self, capsys):
        """Test logger with custom format string."""
        custom_format = "{level} - {message}"
        configure_logger(format_string=custom_format)

        logger.info("Custom format test")

        captured = capsys.readouterr()
        assert "Custom format test" in captured.out
        assert "INFO - Custom format test" in captured.out

    def test_colorize_false(self, capsys):
        """Test logger with colorize disabled."""
        configure_logger(colorize=False)

        logger.info("No color test")

        captured = capsys.readouterr()
        assert "No color test" in captured.out
        # When colorize=False, the format should be plain text
        assert "INFO" in captured.out

    def test_file_logging(self, tmp_path, capsys):
        """Test logger writes to file."""
        log_file = tmp_path / "test.log"
        configure_logger(log_file=log_file)

        logger.info("File logging test")

        # Check stdout
        captured = capsys.readouterr()
        assert "File logging test" in captured.out

        # Check file exists and contains message
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "File logging test" in log_content
        assert "INFO" in log_content

    def test_file_logging_creates_parent_dirs(self, tmp_path, capsys):
        """Test logger creates parent directories for log file."""
        log_file = tmp_path / "logs" / "nested" / "test.log"
        configure_logger(log_file=log_file)

        logger.info("Nested directory test")

        # Check file was created in nested directory
        assert log_file.exists()
        assert log_file.parent.exists()
        log_content = log_file.read_text()
        assert "Nested directory test" in log_content

    def test_file_logging_with_custom_level(self, tmp_path):
        """Test file logging respects custom log level."""
        log_file = tmp_path / "test.log"
        configure_logger(level="WARNING", log_file=log_file)

        logger.info("Info message")
        logger.warning("Warning message")

        log_content = log_file.read_text()
        assert "Info message" not in log_content
        assert "Warning message" in log_content

    def test_multiple_log_messages(self, capsys):
        """Test logger handles multiple messages correctly."""
        configure_logger(level="INFO")

        logger.info("First message")
        logger.warning("Second message")
        logger.error("Third message")

        captured = capsys.readouterr()
        assert "First message" in captured.out
        assert "Second message" in captured.out
        assert "Third message" in captured.out

    def test_log_file_info_message(self, tmp_path, capsys):
        """Test that logger announces log file path."""
        log_file = tmp_path / "test.log"
        configure_logger(log_file=log_file)

        captured = capsys.readouterr()
        assert "Logging to file" in captured.out
        assert str(log_file) in captured.out

    def test_different_log_levels(self, capsys):
        """Test all log levels work correctly."""
        configure_logger(level="DEBUG")

        logger.debug("Debug level")
        logger.info("Info level")
        logger.warning("Warning level")
        logger.error("Error level")
        logger.critical("Critical level")

        captured = capsys.readouterr()
        assert "Debug level" in captured.out
        assert "Info level" in captured.out
        assert "Warning level" in captured.out
        assert "Error level" in captured.out
        assert "Critical level" in captured.out

    def test_path_object_for_log_file(self, tmp_path):
        """Test that Path objects work for log_file parameter."""
        log_file = Path(tmp_path) / "test.log"
        configure_logger(log_file=log_file)

        logger.info("Path object test")

        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Path object test" in log_content

    def test_string_path_for_log_file(self, tmp_path):
        """Test that string paths work for log_file parameter."""
        log_file = str(tmp_path / "test.log")
        configure_logger(log_file=log_file)

        logger.info("String path test")

        assert Path(log_file).exists()
        log_content = Path(log_file).read_text()
        assert "String path test" in log_content
