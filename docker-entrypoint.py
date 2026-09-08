"""Prepare the ECS scratch mount, then execute as the unprivileged app user."""

from __future__ import annotations

import os
import pwd
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("A container command is required.", file=sys.stderr)
        return 64

    if os.geteuid() == 0:
        account = pwd.getpwnam("app")
        os.chown("/tmp", account.pw_uid, account.pw_gid)
        os.chmod("/tmp", 0o770)
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    elif not os.access("/tmp", os.W_OK):
        print("/tmp must be writable by the app user.", file=sys.stderr)
        return 73

    os.execvp(sys.argv[1], sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
