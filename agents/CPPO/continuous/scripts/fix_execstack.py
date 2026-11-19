#!/usr/bin/env python3

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

PT_GNU_STACK = 0x6474E551
PF_X = 0x1
ELF_MAGIC = b"\x7fELF"


def patch_so(path: Path) -> bool:
    """Clear PF_X on PT_GNU_STACK in the given ELF file. Returns True if changed."""
    try:
        with open(path, "r+b") as f:
            head = f.read(64)
            if len(head) < 64 or head[:4] != ELF_MAGIC:
                return False
            ei_class = head[4]  # 1 = 32-bit, 2 = 64-bit
            ei_data = head[5]   # 1 = little, 2 = big
            endian = "<" if ei_data == 1 else ">"

            if ei_class == 2:  # 64-bit
                # e_phoff @ 0x20 (u64), e_phentsize @ 0x36 (u16), e_phnum @ 0x38 (u16)
                e_phoff = struct.unpack_from(endian + "Q", head, 0x20)[0]
                e_phentsize = struct.unpack_from(endian + "H", head, 0x36)[0]
                e_phnum = struct.unpack_from(endian + "H", head, 0x38)[0]
                p_flags_off_in_phdr = 4  # u32 at offset 4
            elif ei_class == 1:  # 32-bit
                # e_phoff @ 0x1C (u32), e_phentsize @ 0x2A (u16), e_phnum @ 0x2C (u16)
                e_phoff = struct.unpack_from(endian + "I", head, 0x1C)[0]
                e_phentsize = struct.unpack_from(endian + "H", head, 0x2A)[0]
                e_phnum = struct.unpack_from(endian + "H", head, 0x2C)[0]
                p_flags_off_in_phdr = 24  # ELF32 puts p_flags after the offset/vaddr/paddr/filesz/memsz fields
            else:
                return False

            for i in range(e_phnum):
                phdr_off = e_phoff + i * e_phentsize
                f.seek(phdr_off)
                phdr = f.read(e_phentsize)
                if len(phdr) < e_phentsize:
                    continue
                p_type = struct.unpack_from(endian + "I", phdr, 0)[0]
                if p_type != PT_GNU_STACK:
                    continue
                p_flags = struct.unpack_from(endian + "I", phdr, p_flags_off_in_phdr)[0]
                if not (p_flags & PF_X):
                    return False  # already clean
                new_flags = p_flags & ~PF_X
                f.seek(phdr_off + p_flags_off_in_phdr)
                f.write(struct.pack(endian + "I", new_flags))
                return True
    except OSError:
        pass
    return False


def find_venv() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    env = os.environ.get("VENV_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
    return here / ".venv"


def main() -> int:
    venv = find_venv()
    if not venv.exists():
        print(f"[fix_execstack] venv not found at {venv}", file=sys.stderr)
        return 1
    changed = 0
    scanned = 0
    for so in venv.rglob("*.so"):
        scanned += 1
        if patch_so(so):
            changed += 1
    # Also patch .so.* (versioned soname) files
    for so in venv.rglob("*.so.*"):
        if not so.is_file():
            continue
        scanned += 1
        if patch_so(so):
            changed += 1
    print(f"[fix_execstack] scanned={scanned} patched={changed} venv={venv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
