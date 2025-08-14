# TFLite Windows Test (Flutter)

Small Flutter desktop app that loads a TensorFlow Lite model and runs a set of automated sanity tests to verify inference stability on Windows.

It prints per-run output hashes and checks:
- Different inputs produce different outputs
- Same input produces the same output
- Optional interpreter reset behavior

The app is intentionally verbose so you can diagnose tensor shapes, types, and delegate behavior.

## Requirements

- Flutter 3.x+ with Windows desktop enabled
- Visual Studio with Desktop development with C++ workload (for Windows runner)
- Dart SDK (bundled with Flutter)

## Install dependencies

Run inside this folder:

```powershell
flutter pub get
```

## Model setup (important)

- Place your TFLite model at: `assets/models/mobileclip_v2.tflite` (or adjust the path in `lib/main.dart` if you use a different file).
- `assets/models/` is included in `pubspec.yaml` and is loaded via `Interpreter.fromAsset(...)`.
- Large models are ignored by Git on purpose (`.tflite`, `.onnx`, `.bin` are in `.gitignore`). Do not commit model binaries.

If you rename the model or path, update:
- `lib/main.dart`: `_modelPath`
- `pubspec.yaml` assets section to include the folder/file

## Run the app

```powershell
flutter run -d windows
```

You should see:
- A list of tests (Model Loading, several Inference Tests, Consistency, Reset)
- A final summary card showing PASS/FAIL and a few output hashes

## What the tests do

The UI triggers a sequence of inferences with synthetic inputs and logs:
- Model input/output shapes and types
- Multiple runs with different seeds to confirm outputs change
- Two runs with the same seed to confirm outputs match
- A reset test that reuses an interpreter; if the delegate doesn’t support reset, it falls back to a fresh interpreter

Implementation notes:
- Uses `runForMultipleInputs` instead of `run()` to avoid a known path that can mismatch buffers on some platforms.
- Builds output buffers that exactly match each tensor’s shape and type (including scalars) and only flattens for hashing.
- Auto-detects which non-scalar output actually varies with the input (some models expose multiple outputs, not all of which change).

## Troubleshooting

### 1) Missing asset/model
If the app crashes with an asset-not-found error, ensure:
- `assets/models/mobileclip_v2.tflite` exists locally
- `pubspec.yaml` has:

```yaml
flutter:
	assets:
		- assets/models/
```

Then run `flutter pub get` again.

### 2) Output shape/type mismatch
If you see errors like “Output object shape mismatch”, it means the provided output buffer’s structure didn’t match the tensor’s shape/type. This app creates shape-accurate nested lists for you, so avoid changing that unless you know the exact shapes.

### 3) Windows git push fails with HTTP 408 or large pack
If `git push` times out or you see very large pack sizes, you likely committed a large model earlier and later removed it from the working tree. Even if the file is deleted now, it still exists in history and will be uploaded.

Fix by removing large blobs from history, then force pushing. Pick one of the following approaches.

Option A: Remove blobs larger than 100 MB with git-filter-repo (recommended)

1. Install git-filter-repo (one-time):
	 - Via Python: `pip install git-filter-repo` (ensure it’s on PATH)
2. Dry-run: back up your repo or ensure it’s not shared yet (this rewrites history)
3. Rewrite history to drop big blobs and common model patterns:

```powershell
git filter-repo --strip-blobs-bigger-than 100M
git filter-repo --path-glob 'assets/models/*.tflite' --invert-paths
```

4. Garbage-collect and push:

```powershell
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
git push --force --tags
```

Option B: Use BFG Repo-Cleaner

1. Download the BFG jar: https://rtyley.github.io/bfg-repo-cleaner/
2. Run one (or both) of:

```powershell
java -jar bfg.jar --strip-blobs-bigger-than 100M
java -jar bfg.jar --delete-files "*.tflite"
```

3. Then clean and push:

```powershell
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
git push --force --tags
```

Warning: Both methods rewrite history. Coordinate with collaborators and consider opening a new repo if that’s simpler.

### 4) Delegate/reset issues
Some delegates don’t support `resetVariableTensors()`. The app detects this and falls back to new interpreters between runs. That’s expected on certain platforms.

## Project layout

- `lib/main.dart`: UI + inference test harness
- `assets/models/`: put your `.tflite` file(s) here (ignored by Git; `.gitkeep` keeps the folder)
- `.gitignore`: excludes large model binaries and build artifacts

## License

MIT (or your choice). Update as needed.

