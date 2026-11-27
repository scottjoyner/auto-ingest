"""A destination path we could not stat must not be called usable.

`MountObservation.usable` accepted `path_exists is None` as usable, so an EIO
while checking a network archive subdirectory read as "mounted and present".
That value feeds `observe_campaign`, the release gate and preflight, turning a
failed existence check into a green storage verdict.
"""
from __future__ import annotations

from auto_ingest.custody.mounts import MountObservation


def _subdir(path_exists):
    return MountObservation(
        mount_point="/nas/fileserver/headcam",
        present=True,
        backing_mount_point="/nas",
        path_exists=path_exists,
    )


def test_unverifiable_subdirectory_path_is_not_usable():
    assert _subdir(None).usable is False


def test_confirmed_subdirectory_paths_keep_their_verdicts():
    assert _subdir(True).usable is True
    assert _subdir(False).usable is False


def test_a_direct_mount_point_still_uses_mount_presence():
    direct = MountObservation(mount_point="/nas", present=True, path_exists=None)
    assert direct.backing_mount_point is None
    assert direct.usable is True
