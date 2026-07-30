"""Static import-isolation governance for the Stage 2S implementation.

The audit deliberately discovers Python files from the working filesystem
rather than from Git.  Untracked files are therefore visible to name
resolution, while forbidden Stage 4 draft files are rejected by name before
their contents are opened or executed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SCAN_DIRECTORIES = ("src", "scripts", "tests")
_DYNAMIC_MODULE_APIS = frozenset(
    {
        "__import__",
        "builtins.__import__",
        "importlib.import_module",
        "pkgutil.resolve_name",
        "runpy.run_module",
    }
)
_DYNAMIC_FILE_APIS = frozenset(
    {
        "importlib.machinery.SourceFileLoader",
        "importlib.util.spec_from_file_location",
        "runpy.run_path",
    }
)
_DYNAMIC_CODE_APIS = frozenset({"eval", "exec", "builtins.eval", "builtins.exec"})
_FORBIDDEN_STAGE4_MODULE = re.compile(
    r"^seismoflux\.anomaly_increment\.kde_dev(?:$|[._])",
    flags=re.IGNORECASE,
)
_FORBIDDEN_STAGE4_TEST_SEGMENT = re.compile(
    r"^test_stage4_kde_dev(?:$|_)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_STAGE4_SOURCE_NAME = re.compile(
    r"^kde_dev(?:_.*)?\.py$",
    flags=re.IGNORECASE,
)
_FORBIDDEN_STAGE4_TEST_NAME = re.compile(
    r"^test_stage4_kde_dev(?:_.*)?\.py$",
    flags=re.IGNORECASE,
)


class Stage2SImportIsolationError(RuntimeError):
    """Raised when the Stage 2S first-party import closure cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class ImportReference:
    """One statically resolved import edge."""

    importer: str
    imported: str
    line: int
    mechanism: str


@dataclass(frozen=True, slots=True)
class ImportPathIdentity:
    """One parsed first-party source path bound to its exact working-tree bytes."""

    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if Path(self.relative_path).as_posix() != self.relative_path:
            raise ValueError("import path identity must use a normalized POSIX path")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("import path identity sha256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class ImportIsolationReport:
    """Deterministically ordered evidence from a successful import audit."""

    root_modules: tuple[str, ...]
    visited_modules: tuple[str, ...]
    path_identities: tuple[ImportPathIdentity, ...]
    references: tuple[ImportReference, ...]

    @property
    def visited_paths(self) -> tuple[str, ...]:
        """Return the compatibility path list derived from byte-bound identities."""

        return tuple(identity.relative_path for identity in self.path_identities)

    @property
    def visited_path_sha256(self) -> tuple[tuple[str, str], ...]:
        """Return deterministic ``(path, sha256)`` rows for canonical receipts."""

        return tuple((identity.relative_path, identity.sha256) for identity in self.path_identities)


@dataclass(frozen=True, slots=True)
class ImportClosureReleaseEvidence:
    """Git and byte identity evidence for one clean code-tag import closure."""

    head_commit: str
    code_commit: str
    path_identities: tuple[ImportPathIdentity, ...]
    evidence_sha256: str

    def receipt_bindings(self) -> dict[str, object]:
        """Return a JSON-safe immutable-release receipt fragment."""

        return {
            "head_commit": self.head_commit,
            "code_commit": self.code_commit,
            "visited_path_sha256": {
                identity.relative_path: identity.sha256 for identity in self.path_identities
            },
            "visited_path_count": len(self.path_identities),
            "git_tracked": True,
            "path_scoped_status_clean": True,
            "working_tree_equals_head_and_code_commit": True,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class _ModuleRecord:
    name: str
    path: Path
    relative_path: str
    is_package: bool

    @property
    def package(self) -> str:
        if self.is_package:
            return self.name
        return self.name.rpartition(".")[0]


def _module_name_from_relative_path(relative_path: Path) -> tuple[str, bool] | None:
    parts = list(relative_path.parts)
    if not parts or relative_path.suffix != ".py":
        return None
    if parts[0] == "src":
        parts = parts[1:]
    elif parts[0] not in {"scripts", "tests"}:
        return None
    is_package = parts[-1] == "__init__.py"
    parts[-1] = parts[-1][:-3]
    if is_package:
        parts.pop()
    if not parts:
        return None
    return ".".join(parts), is_package


def _discover_module_index(repository_root: Path) -> dict[str, _ModuleRecord]:
    """Index first-party modules without consulting Git or reading their contents."""

    index: dict[str, _ModuleRecord] = {}
    for directory_name in _SCAN_DIRECTORIES:
        directory = repository_root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py"), key=lambda item: item.as_posix()):
            relative = path.relative_to(repository_root)
            module_identity = _module_name_from_relative_path(relative)
            if module_identity is None:
                continue
            module_name, is_package = module_identity
            record = _ModuleRecord(
                name=module_name,
                path=path,
                relative_path=relative.as_posix(),
                is_package=is_package,
            )
            previous = index.get(module_name)
            if previous is not None and previous.path != path:
                raise Stage2SImportIsolationError(
                    f"duplicate first-party module {module_name!r}: "
                    f"{previous.relative_path!r} and {record.relative_path!r}"
                )
            index[module_name] = record
    return index


def _default_root_paths(repository_root: Path) -> tuple[Path, ...]:
    roots: set[Path] = set()
    package_root = repository_root / "src" / "seismoflux" / "stage2s"
    if package_root.is_dir():
        roots.update(package_root.rglob("*.py"))
    scripts_root = repository_root / "scripts"
    if scripts_root.is_dir():
        roots.update(scripts_root.glob("*stage2s*.py"))
    tests_root = repository_root / "tests"
    if tests_root.is_dir():
        roots.update(tests_root.rglob("test_stage2s*.py"))
    return tuple(sorted(roots, key=lambda item: item.as_posix()))


def _normalize_root_paths(
    repository_root: Path,
    root_paths: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    candidates = (
        _default_root_paths(repository_root)
        if root_paths is None
        else tuple(
            path if isinstance(path, Path) and path.is_absolute() else repository_root / path
            for path in (Path(value) for value in root_paths)
        )
    )
    if not candidates:
        raise Stage2SImportIsolationError("no Stage 2S audit roots were found")
    normalized: list[Path] = []
    for candidate in candidates:
        try:
            relative = candidate.relative_to(repository_root)
        except ValueError as exc:
            raise Stage2SImportIsolationError(
                f"audit root escapes repository: {candidate}"
            ) from exc
        if not relative.parts or relative.parts[0] not in _SCAN_DIRECTORIES:
            raise Stage2SImportIsolationError(
                f"audit root is outside src/scripts/tests: {relative.as_posix()}"
            )
        if candidate.suffix != ".py" or not candidate.is_file():
            raise Stage2SImportIsolationError(
                f"audit root is not a Python file: {relative.as_posix()}"
            )
        normalized.append(candidate)
    return tuple(sorted(set(normalized), key=lambda item: item.as_posix()))


def _forbidden_module_reason(module_name: str) -> str | None:
    normalized = module_name.strip(".")
    if _FORBIDDEN_STAGE4_MODULE.match(normalized):
        return "Stage 4 kde_dev production module"
    if any(_FORBIDDEN_STAGE4_TEST_SEGMENT.match(segment) for segment in normalized.split(".")):
        return "Stage 4 kde_dev test module"
    return None


def _forbidden_path_reason(relative_path: str) -> str | None:
    path = Path(relative_path)
    if (
        len(path.parts) >= 4
        and tuple(part.casefold() for part in path.parts[:3])
        == ("src", "seismoflux", "anomaly_increment")
        and _FORBIDDEN_STAGE4_SOURCE_NAME.match(path.name)
    ):
        return "Stage 4 kde_dev production path"
    if (
        len(path.parts) >= 3
        and tuple(part.casefold() for part in path.parts[:2]) == ("tests", "unit")
        and _FORBIDDEN_STAGE4_TEST_NAME.match(path.name)
    ):
        return "Stage 4 kde_dev test path"
    return None


def _resolve_from_base(record: _ModuleRecord, node: ast.ImportFrom) -> str:
    module = node.module or ""
    if node.level == 0:
        return module
    package_parts = record.package.split(".") if record.package else []
    parent_steps = node.level - 1
    if parent_steps > len(package_parts):
        raise Stage2SImportIsolationError(
            f"{record.relative_path}:{node.lineno}: relative import escapes its package"
        )
    anchor = package_parts[: len(package_parts) - parent_steps]
    if module:
        anchor.extend(module.split("."))
    return ".".join(anchor)


def _import_aliases(record: _ModuleRecord, tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                binding = alias.asname or alias.name.split(".", maxsplit=1)[0]
                aliases[binding] = alias.name if alias.asname else binding
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_base(record, node)
            for alias in node.names:
                if alias.name == "*":
                    continue
                target = ".".join(part for part in (base, alias.name) if part)
                aliases[alias.asname or alias.name] = target
    return aliases


def _qualified_expression_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_expression_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    head, separator, tail = name.partition(".")
    resolved_head = aliases.get(head, head)
    return f"{resolved_head}{separator}{tail}" if separator else resolved_head


def _literal_string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _normal_import_references(
    record: _ModuleRecord,
    tree: ast.AST,
) -> list[ImportReference]:
    references: list[ImportReference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.extend(
                ImportReference(record.name, alias.name, node.lineno, "import")
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_base(record, node)
            if base:
                references.append(ImportReference(record.name, base, node.lineno, "from"))
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = ".".join(part for part in (base, alias.name) if part)
                if child:
                    references.append(
                        ImportReference(record.name, child, node.lineno, "from-member")
                    )
    return references


def _dynamic_import_references(
    record: _ModuleRecord,
    tree: ast.AST,
) -> list[ImportReference]:
    aliases = _import_aliases(record, tree)
    references: list[ImportReference] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        raw_name = _qualified_expression_name(node.func)
        if raw_name is None:
            continue
        called_name = _resolve_alias(raw_name, aliases)
        if called_name in _DYNAMIC_CODE_APIS:
            raise Stage2SImportIsolationError(
                f"{record.relative_path}:{node.lineno}: dynamic code execution "
                f"via {called_name} is forbidden in the Stage 2S closure"
            )
        if called_name in _DYNAMIC_FILE_APIS:
            raise Stage2SImportIsolationError(
                f"{record.relative_path}:{node.lineno}: file-based dynamic loading "
                f"via {called_name} is forbidden in the Stage 2S closure"
            )
        if called_name not in _DYNAMIC_MODULE_APIS:
            continue
        target = _literal_string(node.args[0] if node.args else None)
        if target is None or not target.strip():
            raise Stage2SImportIsolationError(
                f"{record.relative_path}:{node.lineno}: non-literal dynamic import "
                f"via {called_name} cannot be audited and is forbidden"
            )
        if target.startswith("."):
            package = _literal_string(node.args[1] if len(node.args) > 1 else None)
            if package is None or not package.strip():
                raise Stage2SImportIsolationError(
                    f"{record.relative_path}:{node.lineno}: relative dynamic import "
                    "requires a literal package"
                )
            target = f"{package.rstrip('.')}{target}"
        references.append(
            ImportReference(record.name, target, node.lineno, f"dynamic:{called_name}")
        )
    return references


def _parse_references(
    record: _ModuleRecord,
) -> tuple[tuple[ImportReference, ...], str]:
    forbidden_path = _forbidden_path_reason(record.relative_path)
    if forbidden_path is not None:
        raise Stage2SImportIsolationError(
            f"refusing to open forbidden {forbidden_path}: {record.relative_path}"
        )
    try:
        payload = record.path.read_bytes()
        source = payload.decode("utf-8")
        tree = ast.parse(source, filename=record.relative_path)
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise Stage2SImportIsolationError(
            f"cannot parse first-party module {record.relative_path}: {exc}"
        ) from exc
    references = _normal_import_references(record, tree)
    references.extend(_dynamic_import_references(record, tree))
    return (
        tuple(
            sorted(
                set(references),
                key=lambda item: (item.imported, item.line, item.mechanism),
            )
        ),
        hashlib.sha256(payload).hexdigest(),
    )


def _records_for_roots(
    repository_root: Path,
    index: dict[str, _ModuleRecord],
    root_paths: Sequence[str | Path] | None,
) -> tuple[_ModuleRecord, ...]:
    roots_by_path = {record.path: record for record in index.values()}
    records: list[_ModuleRecord] = []
    for path in _normalize_root_paths(repository_root, root_paths):
        record = roots_by_path.get(path)
        if record is None:
            relative = path.relative_to(repository_root).as_posix()
            raise Stage2SImportIsolationError(
                f"cannot derive a first-party module name for audit root {relative}"
            )
        records.append(record)
    return tuple(sorted(records, key=lambda item: item.name))


def _raise_forbidden_reference(reference: ImportReference) -> None:
    reason = _forbidden_module_reason(reference.imported)
    if reason is not None:
        raise Stage2SImportIsolationError(
            f"{reference.importer}:{reference.line}: forbidden {reason} imported "
            f"as {reference.imported!r} via {reference.mechanism}"
        )


def audit_stage2s_import_closure(
    repository_root: Path,
    *,
    root_paths: Sequence[str | Path] | None = None,
) -> ImportIsolationReport:
    """Audit the Stage 2S first-party transitive import closure.

    Only ``src``, ``scripts``, and ``tests`` are traversed.  Scientific data
    directories are neither enumerated nor opened.
    """

    root = repository_root.resolve(strict=True)
    index = _discover_module_index(root)
    root_records = _records_for_roots(root, index, root_paths)
    queue: deque[str] = deque(record.name for record in root_records)
    visited: set[str] = set()
    path_sha256: dict[str, str] = {}
    references: list[ImportReference] = []

    while queue:
        module_name = queue.popleft()
        if module_name in visited:
            continue
        reason = _forbidden_module_reason(module_name)
        if reason is not None:
            raise Stage2SImportIsolationError(
                f"forbidden {reason} reached in import closure: {module_name}"
            )
        record = index[module_name]
        visited.add(module_name)
        module_references, source_sha256 = _parse_references(record)
        path_sha256[record.relative_path] = source_sha256
        for reference in module_references:
            _raise_forbidden_reference(reference)
            references.append(reference)
            if reference.imported in index and reference.imported not in visited:
                queue.append(reference.imported)

    visited_records = tuple(index[name] for name in sorted(visited))
    return ImportIsolationReport(
        root_modules=tuple(record.name for record in root_records),
        visited_modules=tuple(record.name for record in visited_records),
        path_identities=tuple(
            ImportPathIdentity(
                relative_path=record.relative_path,
                sha256=path_sha256[record.relative_path],
            )
            for record in visited_records
        ),
        references=tuple(
            sorted(
                references,
                key=lambda item: (
                    item.importer,
                    item.imported,
                    item.line,
                    item.mechanism,
                ),
            )
        ),
    )


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage2SImportIsolationError(
            f"Git closure verification failed: {' '.join(arguments)}"
        ) from exc
    return completed.stdout


def _git_text(repository_root: Path, *arguments: str) -> str:
    try:
        return _git_bytes(repository_root, *arguments).decode("utf-8", errors="strict").strip()
    except UnicodeError as exc:
        raise Stage2SImportIsolationError(
            f"Git closure verification returned non-UTF-8 text: {' '.join(arguments)}"
        ) from exc


def _release_evidence_sha256(
    *,
    head_commit: str,
    code_commit: str,
    path_identities: tuple[ImportPathIdentity, ...],
) -> str:
    payload = {
        "head_commit": head_commit,
        "code_commit": code_commit,
        "visited_path_sha256": {
            identity.relative_path: identity.sha256 for identity in path_identities
        },
        "git_tracked": True,
        "path_scoped_status_clean": True,
        "working_tree_equals_head_and_code_commit": True,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def verify_stage2s_import_closure_release(
    repository_root: Path,
    *,
    report: ImportIsolationReport,
    code_commit: str,
) -> ImportClosureReleaseEvidence:
    """Fail closed unless every imported byte is clean, tracked, and code-tag bound.

    Only the exact paths already present in ``report`` are passed to Git.  The
    helper never enumerates scientific-data directories.
    """

    if not isinstance(report, ImportIsolationReport):
        raise TypeError("report must be an ImportIsolationReport")
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("code_commit must be a lowercase 40-character Git commit")
    root = repository_root.resolve(strict=True)
    head_commit = _git_text(root, "rev-parse", "HEAD^{commit}")
    if head_commit != code_commit:
        raise Stage2SImportIsolationError("HEAD does not equal the verified Stage 2S code commit")
    identities = tuple(report.path_identities)
    if not identities:
        raise Stage2SImportIsolationError("the Stage 2S import closure is empty")
    paths = tuple(identity.relative_path for identity in identities)
    for relative_path in paths:
        _git_bytes(root, "ls-files", "--error-unmatch", "--", relative_path)
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *paths,
    )
    if status:
        raise Stage2SImportIsolationError(
            "the Stage 2S import closure has untracked, modified, or staged paths"
        )
    for identity in identities:
        source_path = root / identity.relative_path
        try:
            working_bytes = source_path.read_bytes()
        except OSError as exc:
            raise Stage2SImportIsolationError(
                f"cannot reread audited import path: {identity.relative_path}"
            ) from exc
        working_sha256 = hashlib.sha256(working_bytes).hexdigest()
        if working_sha256 != identity.sha256:
            raise Stage2SImportIsolationError(
                f"audited import path changed after audit: {identity.relative_path}"
            )
        head_bytes = _git_bytes(root, "cat-file", "blob", f"HEAD:{identity.relative_path}")
        code_bytes = _git_bytes(
            root,
            "cat-file",
            "blob",
            f"{code_commit}:{identity.relative_path}",
        )
        if working_bytes != head_bytes or working_bytes != code_bytes:
            raise Stage2SImportIsolationError(
                "working-tree import bytes differ from HEAD or code commit: "
                f"{identity.relative_path}"
            )
    return ImportClosureReleaseEvidence(
        head_commit=head_commit,
        code_commit=code_commit,
        path_identities=identities,
        evidence_sha256=_release_evidence_sha256(
            head_commit=head_commit,
            code_commit=code_commit,
            path_identities=identities,
        ),
    )


def forbidden_stage4_paths(paths: Iterable[str | Path]) -> tuple[str, ...]:
    """Return forbidden Stage 4 draft paths from a candidate commit path list."""

    forbidden: list[str] = []
    for value in paths:
        normalized = Path(value).as_posix().lstrip("./")
        if _forbidden_path_reason(normalized) is not None:
            forbidden.append(normalized)
    return tuple(sorted(set(forbidden)))
