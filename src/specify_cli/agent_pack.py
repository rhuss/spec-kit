"""
Agent Pack Manager for Spec Kit

Implements self-bootstrapping agent packs with declarative manifests
(speckit-agent.yml) and Python bootstrap modules (bootstrap.py).

Agent packs resolve by priority:
  1. User-level   (~/.specify/agents/<id>/)
  2. Project-level (.specify/agents/<id>/)
  3. Catalog-installed (downloaded via `specify agent add`)
  4. Embedded in wheel (official packs under core_pack/agents/)

The embedded packs ship inside the pip wheel so that
`pip install specify-cli && specify init --ai claude` works offline.
"""

import hashlib
import importlib.util
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent ID validation
# ---------------------------------------------------------------------------

#: Regex that every agent ID must match: lowercase alphanumeric + hyphens.
_AGENT_ID_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def _validate_agent_id(agent_id: str) -> None:
    """Raise ``PackResolutionError`` when *agent_id* is unsafe or malformed.

    Rejects IDs containing ``/``, ``..``, or characters outside ``[a-z0-9-]``
    to prevent path-traversal attacks through the resolution stack.
    """
    if not agent_id or not _AGENT_ID_RE.match(agent_id):
        raise PackResolutionError(
            f"Invalid agent ID {agent_id!r} — "
            "IDs must match [a-z0-9-] (lowercase alphanumeric and hyphens, "
            "no leading/trailing hyphens)."
        )


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "speckit-agent.yml"
BOOTSTRAP_FILENAME = "bootstrap.py"

MANIFEST_SCHEMA_VERSION = "1.0"

# Required top-level keys
_REQUIRED_TOP_KEYS = {"schema_version", "agent"}

# Required keys within the ``agent`` block
_REQUIRED_AGENT_KEYS = {"id", "name", "version"}


class AgentPackError(Exception):
    """Base exception for agent-pack operations."""


class ManifestValidationError(AgentPackError):
    """Raised when a speckit-agent.yml file is invalid."""


class PackResolutionError(AgentPackError):
    """Raised when no pack can be found for the requested agent id."""


class AgentFileModifiedError(AgentPackError):
    """Raised when teardown finds user-modified files and ``--force`` is not set."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class AgentManifest:
    """Parsed and validated representation of a speckit-agent.yml file."""

    # identity
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""

    # runtime
    requires_cli: bool = False
    install_url: Optional[str] = None
    cli_tool: Optional[str] = None

    # compatibility
    speckit_version: str = ">=0.1.0"

    # discovery
    tags: List[str] = field(default_factory=list)

    # command registration metadata (used by CommandRegistrar / extensions)
    commands_dir: str = ""
    command_format: str = "markdown"
    arg_placeholder: str = "$ARGUMENTS"
    file_extension: str = ".md"

    # raw data for anything else
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # filesystem path to the pack directory that produced this manifest
    pack_path: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentManifest":
        """Load and validate a manifest from *path*.

        Raises ``ManifestValidationError`` on structural problems.
        """
        try:
            text = path.read_text(encoding="utf-8")
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ManifestValidationError(f"Invalid YAML in {path}: {exc}")
        except FileNotFoundError:
            raise ManifestValidationError(f"Manifest not found: {path}")

        return cls.from_dict(data, pack_path=path.parent)

    @classmethod
    def from_dict(cls, data: dict, *, pack_path: Optional[Path] = None) -> "AgentManifest":
        """Build a manifest from a raw dictionary."""
        if not isinstance(data, dict):
            raise ManifestValidationError("Manifest must be a YAML mapping")

        missing_top = _REQUIRED_TOP_KEYS - set(data)
        if missing_top:
            raise ManifestValidationError(
                f"Missing required top-level key(s): {', '.join(sorted(missing_top))}"
            )

        if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"Unsupported schema_version: {data.get('schema_version')!r} "
                f"(expected {MANIFEST_SCHEMA_VERSION!r})"
            )

        agent_block = data.get("agent")
        if not isinstance(agent_block, dict):
            raise ManifestValidationError("'agent' must be a mapping")

        missing_agent = _REQUIRED_AGENT_KEYS - set(agent_block)
        if missing_agent:
            raise ManifestValidationError(
                f"Missing required agent key(s): {', '.join(sorted(missing_agent))}"
            )

        runtime = data.get("runtime") or {}
        requires = data.get("requires") or {}
        tags = data.get("tags") or []
        cmd_reg = data.get("command_registration") or {}

        return cls(
            id=str(agent_block["id"]),
            name=str(agent_block["name"]),
            version=str(agent_block["version"]),
            description=str(agent_block.get("description", "")),
            author=str(agent_block.get("author", "")),
            license=str(agent_block.get("license", "")),
            requires_cli=bool(runtime.get("requires_cli", False)),
            install_url=runtime.get("install_url"),
            cli_tool=runtime.get("cli_tool"),
            speckit_version=str(requires.get("speckit_version", ">=0.1.0")),
            tags=[str(t) for t in tags] if isinstance(tags, list) else [],
            commands_dir=str(cmd_reg.get("commands_dir", "")),
            command_format=str(cmd_reg.get("format", "markdown")),
            arg_placeholder=str(cmd_reg.get("arg_placeholder", "$ARGUMENTS")),
            file_extension=str(cmd_reg.get("file_extension", ".md")),
            raw=data,
            pack_path=pack_path,
        )


# ---------------------------------------------------------------------------
# Bootstrap base class
# ---------------------------------------------------------------------------

class AgentBootstrap:
    """Base class that every agent pack's ``bootstrap.py`` must subclass.

    Subclasses override :meth:`setup` and :meth:`teardown` to define
    agent-specific lifecycle operations.

    **Lifecycle flow (setup → tracking → teardown):**

    1. ``setup()`` installs files and returns **every** file it created
       (agent commands, shared scripts, templates, etc.).
    2. The CLI calls ``finalize_setup(agent_files, extension_files)``
       which SHA-256 hashes each file and writes the manifest at
       ``.specify/agent-manifest-<id>.json``.
    3. During switch/remove, the CLI reads the manifest via
       ``get_tracked_files()`` and calls ``check_modified_files()``
       to detect changes.  Modified files are listed and the user
       is prompted for confirmation.
    4. ``teardown()`` delegates to ``remove_tracked_files()`` which
       **compares hashes before deleting** — only files whose
       SHA-256 still matches the original are removed.  Modified
       files are preserved unless ``--force`` is used.  This makes
       it safe to track all files (including shared project
       infrastructure) without risk of deleting user work.
    """

    def __init__(self, manifest: AgentManifest):
        self.manifest = manifest
        self.pack_path = manifest.pack_path

    # -- lifecycle -----------------------------------------------------------

    def setup(self, project_path: Path, script_type: str, options: Dict[str, Any]) -> List[Path]:
        """Install agent files into *project_path*.

        This is invoked by ``specify init --agent <agent>`` and
        ``specify agent switch <agent>``.  The legacy ``--ai`` flag
        uses the old non-pack bootstrap flow and does not call this
        method.

        Implementations **must** return every file they create so that the
        CLI can record both agent-installed files and extension-installed
        files in a single install manifest.

        Args:
            project_path: Target project directory.
            script_type: ``"sh"`` or ``"ps"``.
            options: Arbitrary key/value options forwarded from the CLI.

        Returns:
            List of absolute paths of files created during setup.
        """
        raise NotImplementedError

    def teardown(
        self,
        project_path: Path,
        *,
        force: bool = False,
        files: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Remove agent-specific files from *project_path*.

        Invoked by ``specify agent switch`` (for the *old* agent) and
        ``specify agent remove`` when the user explicitly uninstalls.
        Must preserve shared infrastructure (specs, plans, tasks, etc.).

        Only individual files are removed — directories are **never**
        deleted.

        The caller (CLI) is expected to check for user-modified files
        **before** invoking teardown and prompt for confirmation.  If
        *files* is provided, exactly those files are removed (values are
        ignored but kept for forward compatibility).  Otherwise the
        install manifest is read.

        Args:
            project_path: Project directory to clean up.
            force: When ``True``, remove files even if they were modified
                after installation.
            files: Mapping of project-relative path → SHA-256 hash.
                When supplied, only these files are removed and the
                install manifest is not consulted.

        Returns:
            List of project-relative paths that were actually deleted.
        """
        raise NotImplementedError

    # -- helpers available to subclasses ------------------------------------

    def agent_dir(self, project_path: Path) -> Path:
        """Return the agent's top-level directory inside the project.

        Raises ``AgentPackError`` when the manifest's ``commands_dir`` is
        empty, since the agent directory cannot be determined.
        """
        if not self.manifest.commands_dir:
            raise AgentPackError(
                f"Agent '{self.manifest.id}' has an empty commands_dir — "
                "cannot determine agent directory."
            )
        return project_path / self.manifest.commands_dir.split("/")[0]

    def collect_installed_files(self, project_path: Path) -> List[Path]:
        """Return every file under the agent's directory tree.

        Subclasses should call this at the end of :meth:`setup` to build
        the return list.  Any files present in the agent directory at
        that point — whether created by ``setup()`` itself, by the
        scaffold pipeline, or by a preceding step — are reported.
        """
        root = self.agent_dir(project_path)
        if not root.is_dir():
            return []
        return sorted(p for p in root.rglob("*") if p.is_file())

    def _scaffold_project(
        self,
        project_path: Path,
        script_type: str,
        is_current_dir: bool = False,
    ) -> List[Path]:
        """Run the shared scaffolding pipeline and return new files.

        Calls ``scaffold_from_core_pack`` for this agent and then
        collects every file that was created.  Subclasses should call
        this from :meth:`setup` when they want to use the shared
        scaffolding rather than creating files manually.

        Returns:
            List of absolute paths of **all** files created by the
            scaffold (agent-specific commands, shared scripts,
            templates, etc.).
        """
        # Lazy import to avoid circular dependency (agent_pack is
        # imported by specify_cli.__init__).
        from specify_cli import scaffold_from_core_pack

        # Snapshot existing files
        before: set[Path] = set()
        if project_path.exists():
            before = {p for p in project_path.rglob("*") if p.is_file()}

        ok = scaffold_from_core_pack(
            project_path, self.manifest.id, script_type, is_current_dir,
        )
        if not ok:
            raise AgentPackError(
                f"Scaffolding failed for agent '{self.manifest.id}'")

        # Collect every new file
        after = {p for p in project_path.rglob("*") if p.is_file()}
        return sorted(after - before)

    def finalize_setup(
        self,
        project_path: Path,
        agent_files: Optional[List[Path]] = None,
        extension_files: Optional[List[Path]] = None,
    ) -> None:
        """Record installed files for tracked teardown.

        This must be called **after** the full init pipeline has finished
        writing files (commands, context files, extensions) into the
        project.  It records every file reported by :meth:`setup` plus
        any extra files (e.g. from extension registration) and scans
        the agent's directory tree for anything additional.

        All files returned by ``setup()`` are tracked — including shared
        project infrastructure — so that teardown/switch can detect
        modifications.  ``remove_tracked_files()`` compares SHA-256
        hashes before deleting and will only remove files whose hash
        still matches, preserving any user-modified files (unless
        ``--force`` is used).

        Args:
            agent_files: Files reported by :meth:`setup`.
            extension_files: Files created by extension registration.
        """
        all_extension = list(extension_files or [])
        all_agent: List[Path] = list(agent_files or [])

        # Scan the agent's directory tree for files created by later
        # init pipeline steps (skills, presets, extensions) that
        # setup() did not report.  We scan the agent root directory
        # (e.g. .claude/) so we catch both commands and skills
        # directories (skills-migrated agents replace the commands
        # directory with a sibling skills directory during init).
        #
        # Note: extension command files live under agent_root too, so
        # they may appear in both agent_files and extension_files.
        # This is harmless — during teardown the two dicts are merged
        # by key ({**agent, **ext}), so duplicates are deduplicated
        # and the hash is identical either way.
        if self.manifest.commands_dir:
            agent_root = self.agent_dir(project_path)
            if agent_root.is_dir():
                agent_set = {p.resolve() for p in all_agent}
                for p in agent_root.rglob("*"):
                    if p.is_file() and p.resolve() not in agent_set:
                        all_agent.append(p)
                        agent_set.add(p.resolve())

        record_installed_files(
            project_path,
            self.manifest.id,
            agent_files=all_agent,
            extension_files=all_extension,
        )


class DefaultBootstrap(AgentBootstrap):
    """Generic bootstrap that derives its directory layout from the manifest.

    This replaces the need for per-agent ``bootstrap.py`` files when the
    agent follows the standard setup/teardown pattern — create the
    commands directory, run the shared scaffold, and delegate teardown to
    ``remove_tracked_files``.

    The ``AGENT_DIR`` and ``COMMANDS_SUBDIR`` class attributes are
    computed from the manifest's ``commands_dir`` field (e.g.
    ``".claude/commands"`` → ``AGENT_DIR=".claude"``,
    ``COMMANDS_SUBDIR="commands"``).
    """

    def __init__(self, manifest: AgentManifest):
        super().__init__(manifest)
        parts = manifest.commands_dir.split("/") if manifest.commands_dir else []
        self.AGENT_DIR = parts[0] if parts else ""
        self.COMMANDS_SUBDIR = "/".join(parts[1:]) if len(parts) > 1 else ""

    def setup(self, project_path: Path, script_type: str, options: Dict[str, Any]) -> List[Path]:
        """Install agent files into the project using the standard scaffold."""
        if self.AGENT_DIR and self.COMMANDS_SUBDIR:
            commands_dir = project_path / self.AGENT_DIR / self.COMMANDS_SUBDIR
            commands_dir.mkdir(parents=True, exist_ok=True)
        return self._scaffold_project(project_path, script_type)

    def teardown(
        self,
        project_path: Path,
        *,
        force: bool = False,
        files: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Remove agent files from the project.

        Only removes individual tracked files — directories are never
        deleted.  When *files* is provided, exactly those files are
        removed.  Otherwise the install manifest is consulted and
        ``AgentFileModifiedError`` is raised if any tracked file was
        modified and *force* is ``False``.
        """
        return remove_tracked_files(
            project_path, self.manifest.id, force=force, files=files
        )


# ---------------------------------------------------------------------------
# Installed-file tracking
# ---------------------------------------------------------------------------

def _manifest_path(project_path: Path, agent_id: str) -> Path:
    """Return the path to the install manifest for *agent_id*."""
    return project_path / ".specify" / f"agent-manifest-{agent_id}.json"


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_file_list(
    project_path: Path,
    files: List[Path],
) -> Dict[str, str]:
    """Build a {relative_path: sha256} dict from a list of file paths.

    Uses POSIX-style separators for stable cross-platform manifests.
    Silently skips files that are not under *project_path*.
    """
    entries: Dict[str, str] = {}
    project_root = project_path.resolve()
    for file_path in files:
        abs_path = project_path / file_path if not file_path.is_absolute() else file_path
        if not abs_path.is_file():
            continue
        try:
            rel = abs_path.resolve().relative_to(project_root)
        except ValueError:
            # File is outside the project root — skip it
            continue
        entries[rel.as_posix()] = _sha256(abs_path)
    return entries


def record_installed_files(
    project_path: Path,
    agent_id: str,
    agent_files: Optional[List[Path]] = None,
    extension_files: Optional[List[Path]] = None,
) -> Path:
    """Record the installed files and their SHA-256 hashes.

    Writes ``.specify/agent-manifest-<agent_id>.json`` containing
    categorised mappings of project-relative paths to SHA-256 digests.

    Args:
        project_path: Project root directory.
        agent_id: Agent identifier.
        agent_files: Files created by the agent's ``setup()`` and the
            init pipeline (core commands / templates).
        extension_files: Files created by extension registration.

    Returns:
        Path to the written manifest file.
    """
    agent_entries = _hash_file_list(project_path, agent_files or [])
    extension_entries = _hash_file_list(project_path, extension_files or [])

    manifest_file = _manifest_path(project_path, agent_id)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(
            {
                "agent_id": agent_id,
                "agent_files": agent_entries,
                "extension_files": extension_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_file


def _all_tracked_entries(data: dict) -> Dict[str, str]:
    """Return the combined file → hash mapping from a manifest dict.

    Supports both the new categorised layout (``agent_files`` +
    ``extension_files``) and the legacy flat ``files`` key.
    """
    combined: Dict[str, str] = {}
    # Legacy flat format
    if "files" in data and isinstance(data["files"], dict):
        combined.update(data["files"])
    # New categorised format
    if "agent_files" in data and isinstance(data["agent_files"], dict):
        combined.update(data["agent_files"])
    if "extension_files" in data and isinstance(data["extension_files"], dict):
        combined.update(data["extension_files"])
    return combined


def get_tracked_files(
    project_path: Path,
    agent_id: str,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Return the tracked file hashes split by source.

    Returns:
        A tuple ``(agent_files, extension_files)`` where each is a
        ``{relative_path: sha256}`` dict.  Returns two empty dicts
        when no install manifest exists.
    """
    manifest_file = _manifest_path(project_path, agent_id)
    if not manifest_file.is_file():
        return {}, {}

    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, {}

    # Support legacy flat format
    if "files" in data and "agent_files" not in data:
        return dict(data["files"]), {}

    agent_entries = data.get("agent_files", {})
    ext_entries = data.get("extension_files", {})
    return dict(agent_entries), dict(ext_entries)


def check_modified_files(
    project_path: Path,
    agent_id: str,
) -> List[str]:
    """Return project-relative paths of files modified since installation.

    Returns an empty list when no install manifest exists or when every
    tracked file still has its original hash.
    """
    manifest_file = _manifest_path(project_path, agent_id)
    if not manifest_file.is_file():
        return []

    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    entries = _all_tracked_entries(data)

    modified: List[str] = []
    for rel_path, original_hash in entries.items():
        abs_path = project_path / rel_path
        if abs_path.is_file():
            if _sha256(abs_path) != original_hash:
                modified.append(rel_path)
        # If the file was deleted by the user, treat it as not needing
        # removal — skip rather than flag as modified.

    return modified


def remove_tracked_files(
    project_path: Path,
    agent_id: str,
    *,
    force: bool = False,
    files: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Remove individual tracked files.

    If *files* is provided, those files are candidates for removal.
    Each file's current SHA-256 is compared against the recorded hash;
    files whose hash no longer matches (i.e. user-modified) are skipped
    unless *force* is ``True``.  When *files* is ``None``, the install
    manifest for *agent_id* is read instead.

    Raises :class:`AgentFileModifiedError` if any tracked file was
    modified and *force* is ``False`` (only when reading from the
    manifest — callers that pass *files* are expected to have already
    prompted the user).

    Directories are **never** deleted — only individual files.

    The install manifest is only deleted when every tracked file was
    successfully removed.  If some files were skipped (modified), the
    manifest is preserved so they remain tracked.

    Args:
        project_path: Project root directory.
        agent_id: Agent identifier.
        force: When ``True``, delete even modified files.
        files: Mapping of project-relative path → SHA-256 hash.
            When supplied, the install manifest is not consulted.

    Returns:
        List of project-relative paths that were removed.
    """
    manifest_file = _manifest_path(project_path, agent_id)

    if files is not None:
        entries = files
    else:
        if not manifest_file.is_file():
            return []
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        entries = _all_tracked_entries(data)
        if not entries:
            manifest_file.unlink(missing_ok=True)
            return []

        if not force:
            modified = check_modified_files(project_path, agent_id)
            if modified:
                raise AgentFileModifiedError(
                    "The following agent files have been modified since installation:\n"
                    + "\n".join(f"  {p}" for p in modified)
                    + "\nUse --force to remove them anyway."
                )

    removed: List[str] = []
    project_root = project_path.resolve()
    for rel_path, original_hash in entries.items():
        abs_path = project_path / rel_path
        # Guard against path traversal: reject entries that resolve
        # outside the project directory (e.g. via "../" in a
        # tampered manifest).
        try:
            abs_path.resolve().relative_to(project_root)
        except ValueError:
            continue
        if abs_path.is_file():
            if original_hash and _sha256(abs_path) != original_hash:
                # File was modified since installation — skip unless forced
                if not force:
                    continue
            abs_path.unlink()
            removed.append(rel_path)

    # Clean up the install manifest only when no tracked files remain
    # on disk.  Files already deleted by the user count as gone, not
    # as "remaining" — only files that still exist and were skipped
    # (e.g. modified without --force) prevent manifest cleanup.
    if manifest_file.is_file():
        still_on_disk = sum(
            1 for rel_path in entries
            if (project_path / rel_path).is_file()
            and rel_path not in removed
        )
        if still_on_disk == 0:
            manifest_file.unlink(missing_ok=True)
    return removed


# ---------------------------------------------------------------------------
# Pack resolution
# ---------------------------------------------------------------------------

def _embedded_agents_dir() -> Path:
    """Return the path to the embedded agent packs inside the wheel."""
    return Path(__file__).parent / "core_pack" / "agents"


def _user_agents_dir() -> Path:
    """Return the user-level agent overrides directory."""
    return Path.home() / ".specify" / "agents"


def _project_agents_dir(project_path: Path) -> Path:
    """Return the project-level agent overrides directory."""
    return project_path / ".specify" / "agents"


def _catalog_agents_dir() -> Path:
    """Return the catalog-installed agent cache directory."""
    return Path.home() / ".specify" / "agent-cache"


@dataclass
class ResolvedPack:
    """Result of resolving an agent pack through the priority stack."""
    manifest: AgentManifest
    source: str          # "user", "project", "catalog", "embedded"
    path: Path
    overrides: Optional[str] = None  # version of the pack being overridden


def resolve_agent_pack(
    agent_id: str,
    project_path: Optional[Path] = None,
) -> ResolvedPack:
    """Resolve an agent pack through the priority stack.

    Priority (highest first):
      1. User-level     ``~/.specify/agents/<id>/``
      2. Project-level  ``.specify/agents/<id>/``
      3. Catalog-installed cache
      4. Embedded in wheel

    Raises ``PackResolutionError`` when *agent_id* is invalid or when
    no pack is found at any level.
    """
    _validate_agent_id(agent_id)

    candidates: List[tuple[str, Path]] = []

    # Priority 1 — user level
    user_dir = _user_agents_dir() / agent_id
    candidates.append(("user", user_dir))

    # Priority 2 — project level
    if project_path is not None:
        proj_dir = _project_agents_dir(project_path) / agent_id
        candidates.append(("project", proj_dir))

    # Priority 3 — catalog cache
    catalog_dir = _catalog_agents_dir() / agent_id
    candidates.append(("catalog", catalog_dir))

    # Priority 4 — embedded
    embedded_dir = _embedded_agents_dir() / agent_id
    candidates.append(("embedded", embedded_dir))

    embedded_manifest: Optional[AgentManifest] = None

    for source, pack_dir in candidates:
        manifest_file = pack_dir / MANIFEST_FILENAME
        if manifest_file.is_file():
            manifest = AgentManifest.from_yaml(manifest_file)
            # Verify the manifest's declared ID matches the requested
            # agent_id to prevent a malicious override from injecting
            # a different ID (used for file paths and module names).
            if manifest.id != agent_id:
                raise PackResolutionError(
                    f"Agent pack manifest ID '{manifest.id}' does not match "
                    f"requested ID '{agent_id}' (in {pack_dir})."
                )
            if source == "embedded":
                embedded_manifest = manifest

            overrides = None
            if source != "embedded" and embedded_manifest is None:
                # Try loading embedded to record what it overrides
                emb_file = _embedded_agents_dir() / agent_id / MANIFEST_FILENAME
                if emb_file.is_file():
                    try:
                        emb = AgentManifest.from_yaml(emb_file)
                        overrides = f"embedded v{emb.version}"
                    except AgentPackError:
                        pass  # Embedded manifest unreadable — skip override info

            return ResolvedPack(
                manifest=manifest,
                source=source,
                path=pack_dir,
                overrides=overrides,
            )

    raise PackResolutionError(
        f"Agent '{agent_id}' not found locally or in any active catalog.\n"
        f"Run 'specify agent search' to browse available agents, or\n"
        f"'specify agent add {agent_id} --from <path>' for offline install."
    )


# ---------------------------------------------------------------------------
# Pack discovery helpers
# ---------------------------------------------------------------------------

def list_embedded_agents() -> List[AgentManifest]:
    """Return manifests for all agent packs embedded in the wheel."""
    agents_dir = _embedded_agents_dir()
    if not agents_dir.is_dir():
        return []

    manifests: List[AgentManifest] = []
    for child in sorted(agents_dir.iterdir()):
        manifest_file = child / MANIFEST_FILENAME
        if child.is_dir() and manifest_file.is_file():
            try:
                manifests.append(AgentManifest.from_yaml(manifest_file))
            except AgentPackError:
                continue
    return manifests


def list_all_agents(project_path: Optional[Path] = None) -> List[ResolvedPack]:
    """List all available agents, resolved through the priority stack.

    Each agent id appears at most once, at its highest-priority source.
    """
    seen: dict[str, ResolvedPack] = {}

    # Track embedded versions separately so overrides can accurately
    # reference what they replace, even after catalog/project/user
    # packs have overwritten the seen dict entry.
    embedded_versions: dict[str, str] = {}

    # Start from lowest priority (embedded) so higher priorities overwrite
    for manifest in list_embedded_agents():
        seen[manifest.id] = ResolvedPack(
            manifest=manifest,
            source="embedded",
            path=manifest.pack_path or _embedded_agents_dir() / manifest.id,
        )
        embedded_versions[manifest.id] = manifest.version

    # Catalog cache
    catalog_dir = _catalog_agents_dir()
    if catalog_dir.is_dir():
        for child in sorted(catalog_dir.iterdir()):
            mf = child / MANIFEST_FILENAME
            if child.is_dir() and mf.is_file():
                try:
                    m = AgentManifest.from_yaml(mf)
                    overrides = f"embedded v{embedded_versions[m.id]}" if m.id in embedded_versions else None
                    seen[m.id] = ResolvedPack(manifest=m, source="catalog", path=child, overrides=overrides)
                except AgentPackError:
                    continue

    # Project-level
    if project_path is not None:
        proj_dir = _project_agents_dir(project_path)
        if proj_dir.is_dir():
            for child in sorted(proj_dir.iterdir()):
                mf = child / MANIFEST_FILENAME
                if child.is_dir() and mf.is_file():
                    try:
                        m = AgentManifest.from_yaml(mf)
                        overrides = f"embedded v{embedded_versions[m.id]}" if m.id in embedded_versions else None
                        seen[m.id] = ResolvedPack(manifest=m, source="project", path=child, overrides=overrides)
                    except AgentPackError:
                        continue

    # User-level
    user_dir = _user_agents_dir()
    if user_dir.is_dir():
        for child in sorted(user_dir.iterdir()):
            mf = child / MANIFEST_FILENAME
            if child.is_dir() and mf.is_file():
                try:
                    m = AgentManifest.from_yaml(mf)
                    overrides = f"embedded v{embedded_versions[m.id]}" if m.id in embedded_versions else None
                    seen[m.id] = ResolvedPack(manifest=m, source="user", path=child, overrides=overrides)
                except AgentPackError:
                    continue

    return sorted(seen.values(), key=lambda r: r.manifest.id)


def load_bootstrap(pack_path: Path, manifest: AgentManifest) -> AgentBootstrap:
    """Import ``bootstrap.py`` from *pack_path* and return the bootstrap instance.

    When a ``bootstrap.py`` exists, the module must define exactly one
    public subclass of ``AgentBootstrap``.  When it is absent the
    :class:`DefaultBootstrap` is used instead — it derives its directory
    layout from the manifest's ``commands_dir`` field.

    .. warning::
       **Trust boundary:** ``bootstrap.py`` modules are dynamically
       imported and can execute arbitrary code.  The 4-level resolution
       stack (user → project → catalog → embedded) means that *any*
       pack author whose pack is placed in one of these directories can
       run code with the privileges of the current process.  Only
       install packs from trusted sources.
    """
    bootstrap_file = pack_path / BOOTSTRAP_FILENAME
    if not bootstrap_file.is_file():
        # No bootstrap module — use the generic DefaultBootstrap
        return DefaultBootstrap(manifest)

    spec = importlib.util.spec_from_file_location(
        f"speckit_agent_{manifest.id}_bootstrap", bootstrap_file
    )
    if spec is None or spec.loader is None:
        raise AgentPackError(f"Cannot load bootstrap module: {bootstrap_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the AgentBootstrap subclass
    candidates = [
        obj
        for name, obj in vars(module).items()
        if (
            isinstance(obj, type)
            and issubclass(obj, AgentBootstrap)
            and obj is not AgentBootstrap
            and obj is not DefaultBootstrap
            and not name.startswith("_")
        )
    ]
    if not candidates:
        raise AgentPackError(
            f"No AgentBootstrap subclass found in {bootstrap_file}"
        )
    if len(candidates) > 1:
        raise AgentPackError(
            f"Multiple AgentBootstrap subclasses in {bootstrap_file}: "
            f"{[c.__name__ for c in candidates]}"
        )

    return candidates[0](manifest)


def validate_pack(pack_path: Path) -> List[str]:
    """Validate a pack directory structure and return a list of warnings.

    Returns an empty list when the pack is fully valid.
    Raises ``ManifestValidationError`` on hard errors.
    """
    warnings: List[str] = []
    manifest_file = pack_path / MANIFEST_FILENAME

    if not manifest_file.is_file():
        raise ManifestValidationError(
            f"Missing {MANIFEST_FILENAME} in {pack_path}"
        )

    manifest = AgentManifest.from_yaml(manifest_file)

    bootstrap_file = pack_path / BOOTSTRAP_FILENAME
    if not bootstrap_file.is_file():
        warnings.append(f"Missing {BOOTSTRAP_FILENAME} (DefaultBootstrap will be used)")

    if not manifest.commands_dir:
        warnings.append("command_registration.commands_dir not set in manifest")

    if not manifest.description:
        warnings.append("agent.description is empty")

    if not manifest.tags:
        warnings.append("No tags specified (reduces discoverability)")

    return warnings


def export_pack(agent_id: str, dest: Path, project_path: Optional[Path] = None) -> Path:
    """Export the active pack for *agent_id* to *dest*.

    Returns the path to the exported pack directory.
    """
    resolved = resolve_agent_pack(agent_id, project_path=project_path)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(resolved.path, dest, dirs_exist_ok=True)
    return dest
