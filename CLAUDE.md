# VoxLocal — Notes for Claude

## Xcode synchronized root groups

The `VoxLocal` target uses `PBXFileSystemSynchronizedRootGroup` (Xcode 15+, default in Xcode 26). Any file under `VoxLocal/VoxLocal/` is auto-added to the target — no pbxproj edits needed to include new source files.

Side effect: **every** file in the folder tree becomes a target member.

- `.swift` files compile.
- Known resource types (`.onnx`, `.wav`, `.json`, images, etc.) get copied into the app bundle.
- Unknown/hidden files (e.g. `.gitkeep`) are also copied as resources, and identical filenames in different folders collide at the bundle root → `Multiple commands produce '…/VoxLocal.app/<file>'` error.

Consequences:
- **Don't** use `.gitkeep` to preserve empty folders. Just let empty folders exist untracked by git; once a real source file lands in them, git picks them up.
- If a non-code file should ship with the app (e.g. bundled ONNX model), just drop it in — Xcode will copy it.
- If a non-code file should **not** be part of the target (e.g. a README or fixture), either keep it outside `VoxLocal/VoxLocal/`, or right-click the synchronized group in Xcode → **Add Exception** to exclude it.
