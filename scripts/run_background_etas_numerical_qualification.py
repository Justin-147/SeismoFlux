# ruff: noqa: RUF001
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL_PATH = _PROJECT_ROOT / "configs" / "background_etas_numerical_qualification.yaml"
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from seismoflux.background import etas_numerical_qualification as _qualification  # noqa: E402
from seismoflux.background.etas_numerical_qualification import (  # noqa: E402
    QualificationProtocol,
    load_protocol,
    prepare_real_inputs,
    run_prepared,
    verify_prepared,
)

_MODULE_PATH = Path(_qualification.__file__ or "").resolve()
if not _MODULE_PATH.is_relative_to(_PROJECT_ROOT / "src"):
    raise RuntimeError("qualification runner imported seismoflux from another worktree")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _cpu_percent() -> float:
    if os.name != "nt":
        getloadavg = cast(Callable[[], tuple[float, float, float]], os.__dict__["getloadavg"])
        return float(100.0 * getloadavg()[0] / max(1, os.cpu_count() or 1))

    class FileTime(ctypes.Structure):
        _fields_ = (("low", ctypes.c_ulong), ("high", ctypes.c_ulong))

    def sample() -> tuple[int, int]:
        idle, kernel, user = FileTime(), FileTime(), FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            raise OSError("GetSystemTimes failed")

        idle_value = (idle.high << 32) | idle.low
        total_value = ((kernel.high << 32) | kernel.low) + ((user.high << 32) | user.low)
        return idle_value, total_value

    first = sample()
    time.sleep(0.25)
    second = sample()
    total, idle = second[1] - first[1], second[0] - first[0]
    return 0.0 if total <= 0 else 100.0 * (1.0 - idle / total)


def _physical_core_count() -> int:
    if os.name != "nt":
        return os.cpu_count() or 0
    required = ctypes.c_ulong(0)
    relationship_processor_core = 0
    ctypes.windll.kernel32.GetLogicalProcessorInformationEx(
        relationship_processor_core,
        None,
        ctypes.byref(required),
    )
    if required.value <= 0:
        raise RuntimeError("无法确定物理核心数")
    buffer = ctypes.create_string_buffer(required.value)
    if not ctypes.windll.kernel32.GetLogicalProcessorInformationEx(
        relationship_processor_core,
        buffer,
        ctypes.byref(required),
    ):
        raise OSError("GetLogicalProcessorInformationEx failed")
    count, offset, raw = 0, 0, buffer.raw
    while offset < required.value:
        if offset + 8 > required.value:
            raise RuntimeError("物理核心拓扑缓冲区不完整")
        relationship = int.from_bytes(raw[offset : offset + 4], "little")
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        if size < 8 or offset + size > required.value:
            raise RuntimeError("物理核心拓扑记录无效")
        count += int(relationship == relationship_processor_core)
        offset += size
    return count


def _require_frozen_worktree(protocol: QualificationProtocol) -> None:
    allowed_files = {
        path.relative_to(protocol.root).as_posix()
        for path in (
            protocol.output("input_manifest"),
            protocol.output("result_manifest"),
            protocol.output("verification_manifest"),
            protocol.output("report"),
            protocol.output("static_figure"),
            protocol.output("interactive_report"),
        )
    }
    attempt_prefix = protocol.attempt_root.relative_to(protocol.root).as_posix() + "/"
    status = _git(
        protocol.root,
        "-c",
        "core.quotePath=false",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    unexpected = []
    for line in status.splitlines():
        path = line[3:].replace("\\", "/")
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        if path not in allowed_files and not path.startswith(attempt_prefix):
            unexpected.append(path)
    if unexpected:
        raise SystemExit(
            "qualification 代码/协议工作树不是冻结状态：" + ", ".join(sorted(unexpected))
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--source-root")
    args = parser.parse_args()
    protocol = load_protocol(_PROTOCOL_PATH)
    if protocol.root != _PROJECT_ROOT:
        raise SystemExit("qualification 协议不属于当前 runner 工作树")
    tag = str(protocol.raw["publication"]["code_tag"])
    commit = _git(protocol.root, "rev-parse", "HEAD")
    if _git(protocol.root, "cat-file", "-t", tag) != "tag":
        raise SystemExit("qualification code tag 必须是 annotated tag")
    if _git(protocol.root, "rev-parse", f"{tag}^{{}}") != commit:
        raise SystemExit("当前 HEAD 不是冻结 qualification code tag")
    _require_frozen_worktree(protocol)
    if _physical_core_count() < 3 or _cpu_percent() >= 70.0:
        raise SystemExit("CPU 资源门槛未满足：至少保留 2 核且启动时总占用低于 70%")
    source = args.source_root
    if source is None:
        source = str(Path(_git(protocol.root, "rev-parse", "--git-common-dir")).resolve().parent)
    prepared = prepare_real_inputs(_PROTOCOL_PATH, source_root=source, progress=print)
    result = (
        run_prepared(prepared, code_commit=commit, code_tag=tag, progress=print)
        if args.command == "run"
        else verify_prepared(prepared, code_commit=commit, code_tag=tag)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
