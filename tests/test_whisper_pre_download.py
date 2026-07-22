"""
Tests for ``_pre_download_whisper_model_safely`` — the subprocess-isolated
Whisper model download that protects the main process from SIGABRT.
"""

from unittest.mock import patch, MagicMock
import pytest


# ── _pre_download_whisper_model_safely unit tests ─────────────────────────


class TestPreDownloadSubprocess:
    """Direct unit tests for the subprocess-isolated download."""

    @staticmethod
    def _call(model_name: str = "small"):
        from jarvis.listening.listener import _pre_download_whisper_model_safely
        return _pre_download_whisper_model_safely(model_name)

    def test_returns_true_on_successful_subprocess(self):
        """Returns True when the subprocess exits with code 0."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            mock_run.return_value.stdout = "/path/to/model"

            result = self._call("small")
            assert result is True

            import sys
            assert mock_run.call_args[0][0][0] == sys.executable
            script = mock_run.call_args[0][0][2]
            assert "download_model" in script
            assert "small" in script

    def test_returns_false_on_subprocess_failure(self):
        """Returns False when the subprocess exits with non-zero."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Something went wrong"

            result = self._call("base")
            assert result is False

    def test_returns_false_on_sigabrt(self):
        """Returns False when the subprocess is killed by SIGABRT (-6)."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.return_value.returncode = -6
            mock_run.return_value.stderr = ""

            result = self._call("small")
            assert result is False

    def test_returns_false_on_timeout(self):
        """Returns False when the subprocess times out."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.side_effect = _real_subprocess.TimeoutExpired(
                cmd="test", timeout=300
            )

            result = self._call("tiny")
            assert result is False

    def test_returns_false_when_faster_whisper_unavailable(self):
        """Returns False immediately when FASTER_WHISPER_AVAILABLE is False."""
        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", False):
            result = self._call("small")
            assert result is False


# ── Integration: local_files_only parameter passing ─────────────────────


class TestLocalFilesOnlyParameter:
    """Check that ``local_files_only`` is correctly forwarded based on
    the pre-download result."""

    def test_passes_true_when_pre_download_succeeds(self):
        """``local_files_only=True`` when pre-download returns True."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch(
                "jarvis.listening.listener._pre_download_whisper_model_safely",
                return_value=True,
            ):
                with patch(
                    "jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True
                ):
                    with patch(
                        "jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False
                    ):
                        with patch(
                            "jarvis.listening.listener.WhisperModel",
                            return_value=mock_whisper_model,
                        ) as mock_class:
                            with patch(
                                "jarvis.listening.listener.sd"
                            ) as mock_sd:
                                mock_sd.query_devices.return_value = [
                                    {"name": "Test Mic", "max_input_channels": 1}
                                ]
                                mock_sd.InputStream.side_effect = Exception(
                                    "Stop test here"
                                )

                                from jarvis.listening.listener import VoiceListener

                                listener = VoiceListener(
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                )
                                listener.run()

                                mock_class.assert_called_once()
                                assert (
                                    mock_class.call_args[1]["local_files_only"]
                                    is True
                                )

    def test_passes_false_when_pre_download_fails(self):
        """``local_files_only=False`` when pre-download returns False."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch(
                "jarvis.listening.listener._pre_download_whisper_model_safely",
                return_value=False,
            ):
                with patch(
                    "jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True
                ):
                    with patch(
                        "jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False
                    ):
                        with patch(
                            "jarvis.listening.listener.WhisperModel",
                            return_value=mock_whisper_model,
                        ) as mock_class:
                            with patch(
                                "jarvis.listening.listener.sd"
                            ) as mock_sd:
                                mock_sd.query_devices.return_value = [
                                    {"name": "Test Mic", "max_input_channels": 1}
                                ]
                                mock_sd.InputStream.side_effect = Exception(
                                    "Stop test here"
                                )

                                from jarvis.listening.listener import VoiceListener

                                listener = VoiceListener(
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                )
                                listener.run()

                                mock_class.assert_called_once()
                                assert (
                                    mock_class.call_args[1]["local_files_only"]
                                    is False
                                )
