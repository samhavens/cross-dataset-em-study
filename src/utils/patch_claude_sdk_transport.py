from __future__ import annotations

import os
from pathlib import Path
from subprocess import PIPE

import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream

from claude_code_sdk._errors import CLIConnectionError, CLINotFoundError
from claude_code_sdk._internal.transport.subprocess_cli import (
    SubprocessCLITransport as _T,
)


def apply_patch(redirect_stderr_to_parent: bool = True, remove_verbose_flag: bool = True) -> None:
    original_build = _T._build_command

    def _build_command_patched(self):  # type: ignore[override]
        cmd = original_build(self)
        if remove_verbose_flag:
            cmd = [a for a in cmd if a != "--verbose"]
        return cmd

    async def _connect_patched(self):  # type: ignore[override]
        if self._process:
            return
        cmd = self._build_command()
        try:
            self._process = await anyio.open_process(
                cmd,
                stdin=PIPE,
                stdout=PIPE,
                stderr=None if redirect_stderr_to_parent else PIPE,
                cwd=self._cwd,
                env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "sdk-py"},
            )
            if self._process.stdout:
                self._stdout_stream = TextReceiveStream(self._process.stdout)
            if redirect_stderr_to_parent:
                self._stderr_stream = None
            else:
                self._stderr_stream = (
                    TextReceiveStream(self._process.stderr) if self._process.stderr else None
                )
            if self._is_streaming:
                if self._process.stdin:
                    self._stdin_stream = TextSendStream(self._process.stdin)
                    self._task_group = anyio.create_task_group()
                    await self._task_group.__aenter__()
                    self._task_group.start_soon(self._stream_to_stdin)
            else:
                if self._process.stdin:
                    await self._process.stdin.aclose()
        except FileNotFoundError as e:
            if self._cwd and not Path(self._cwd).exists():
                raise CLIConnectionError(f"Working directory does not exist: {self._cwd}") from e
            raise CLINotFoundError(f"Claude Code not found at: {self._cli_path}") from e
        except Exception as e:
            raise CLIConnectionError(f"Failed to start Claude Code: {e}") from e

    _T._build_command = _build_command_patched  # type: ignore[assignment]
    _T.connect = _connect_patched  # type: ignore[assignment]
