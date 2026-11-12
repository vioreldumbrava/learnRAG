"""How file names are shown to people."""

from __future__ import annotations

import re


# Files imported through the web upload or the desktop "Import files" button
# are stored as "<32 hex chars>_<original name>" so they never overwrite each
# other (see `workflow.import_stream`). The prefix is storage detail, not
# something a reader needs in a table.
_MANAGED_PREFIX = re.compile(r"^[0-9a-f]{32}_")


def display_file_name(name: str) -> str:
    """Drop the managed-upload prefix from a stored file name."""

    return _MANAGED_PREFIX.sub("", name)
