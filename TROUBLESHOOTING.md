# TFLite Test App – Troubleshooting Guide

This document explains how this Flutter app is structured, what tests it runs, how it uses TensorFlow Lite on Windows, and how to diagnose and fix common issues. It also includes current code snippets relevant to the TFLite configuration used in this project.

## Overview

- App name: `tflite_test`
- Target platform(s): Windows Desktop (also runnable on others if toolchains are installed)
- Purpose: Load a TensorFlow Lite model and run a suite of automated tests to validate inference behavior on desktop. It focuses on verifying that different inputs produce different outputs and that calling `resetVariableTensors()` resolves stateful model issues.

## Key files
- `lib/main.dart`: The entire app UI and test logic.
- `assets/models/mobileclip_v2.tflite`: The TFLite model loaded by the app.
- `pubspec.yaml`: Declares dependencies and registers model assets.
- `windows/flutter/generated_plugins.cmake`: Shows that `tflite_flutter` is bundled as an FFI plugin for desktop.

## Dependencies

Declared in `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.11.0
  crypto: ^3.0.5
```

Assets configuration:

```yaml
flutter:
  uses-material-design: true
  assets:
    - assets/models/
```

## App flow and tests

1. On startup, the app schedules and runs an automated test suite.
2. Each test appends a result to the on-screen list and updates the overall status card.
3. Buttons:
   - "Run Tests Again" re-runs the suite.
   - "Instructions" opens a quick setup dialog.
   - "Copy Test Results" copies the overall and individual results to clipboard.

### Tests performed

- Model Loading
  - Loads the model from assets, allocates tensors, and reports the input/output tensor shapes.
- Basic Inference
  - Runs inference once with a synthetic input and shows a short output sample and a SHA1 hash of the output.
- Multiple Inputs (Inference Test 2..6)
  - Runs several inferences with different seeds to ensure outputs differ.
- Consistency Test
  - Runs twice with the same seed to confirm identical outputs for identical inputs.
- resetVariableTensors Test
  - Ensures calling `resetVariableTensors()` between different inputs results in distinct outputs (important for stateful models).

## Current TFLite-related code snippets (from `lib/main.dart`)

Model loading and shape discovery:

```dart
await _addTestResult('Model Loading', () async {
  final interpreter = await Interpreter.fromAsset(_modelPath);
  interpreter.allocateTensors();

  final inputShape = interpreter.getInputTensor(0).shape;
  final outputShape = interpreter.getOutputTensor(0).shape;

  interpreter.close();
  return 'Input: $inputShape, Output: $outputShape';
});
```

Single inference (without reset):

```dart
Future<Map<String, String>> _runSingleInference(int seed) async {
  final interpreter = await Interpreter.fromAsset(_modelPath);
  interpreter.allocateTensors();
  try {
    final input = _generateTestInput(seed);
    final outputShape = interpreter.getOutputTensor(0).shape;
    final outputSize = outputShape.reduce((a, b) => a * b);
    final output = List<double>.filled(outputSize, 0.0);
    interpreter.run(input, output);
    final hash = _calculateHash(output);
    final sample = output.take(3).map((v) => v.toStringAsFixed(6)).join(', ');
    return {'hash': hash, 'sample': sample};
  } finally {
    interpreter.close();
  }
}
```

Single inference with variable tensor reset:

```dart
Future<Map<String, String>> _runSingleInferenceWithReset(int seed) async {
  final interpreter = await Interpreter.fromAsset(_modelPath);
  interpreter.allocateTensors();
  try {
    interpreter.resetVariableTensors();
    final input = _generateTestInput(seed);
    final outputShape = interpreter.getOutputTensor(0).shape;
    final outputSize = outputShape.reduce((a, b) => a * b);
    final output = List<double>.filled(outputSize, 0.0);
    interpreter.run(input, output);
    final hash = _calculateHash(output);
    final sample = output.take(3).map((v) => v.toStringAsFixed(6)).join(', ');
    return {'hash': hash, 'sample': sample};
  } finally {
    interpreter.close();
  }
}
```

Synthetic input generator (assumes NCHW: `[1, 3, 256, 256]`):

```dart
List<List<List<List<double>>>> _generateTestInput(int seed) {
  final random = Random(seed);
  final channels = List<List<List<double>>>.generate(
    3,
    (c) => List<List<double>>.generate(
      256,
      (h) => List<double>.generate(
        256,
        (w) => (random.nextDouble() - 0.5) * 2.0,
        growable: false,
      ),
      growable: false,
    ),
    growable: false,
  );
  return [channels];
}
```

Output hashing (for quick diff/consistency checks):

```dart
String _calculateHash(List<double> output) {
  final bytes = output.map((d) => d.toStringAsFixed(8)).join(',');
  return sha1.convert(utf8.encode(bytes)).toString().substring(0, 8);
}
```

## Common issues and fixes

- Build errors about nullable types (e.g., map lookups):
  - Ensure you handle `Map<String, String>` lookups safely; the app uses a non-null assertion for `result['hash']!` when storing into a `String`.

- Shape/type mismatches in `_generateTestInput`:
  - This app returns a 4D nested list to represent `[1, 3, 256, 256]`. Ensure the return type matches the actual nested shape. If your model expects NHWC `[1, 256, 256, 3]`, adjust the generator accordingly.

- Asset not found:
  - Verify `assets/models/mobileclip_v2.tflite` exists and `pubspec.yaml` includes `assets/models/`. Run `flutter clean` then `flutter pub get`.

- Missing Windows toolchain:
  - Install Visual Studio 2022 with "Desktop development with C++" and the Windows 10/11 SDK. Re-run `flutter doctor -v`.

- Missing DLL or native errors on Windows:
  - `tflite_flutter` is an FFI plugin that bundles its native parts. If you hit a missing DLL, clean and rebuild. Ensure you’re on `tflite_flutter >= 0.11.0`.

- Identical outputs for different inputs:
  - Use `interpreter.resetVariableTensors()` before each inference for stateful models (as the app demonstrates in a dedicated test).

## Verifying model IO shapes

Add logging or display the shapes reported by:

```dart
final inputShape = interpreter.getInputTensor(0).shape;
final outputShape = interpreter.getOutputTensor(0).shape;
```

Match your `_generateTestInput` nested list structure to `inputShape`. For example:

- NCHW: `[1, 3, H, W]` ➜ `List<List<List<List<double>>>>` with outer `[1]`, next `[3]`, then `[H]`, then `[W]`.
- NHWC: `[1, H, W, 3]` ➜ reorder the `.generate` loops accordingly.

## Optional performance tweaks (not currently in code)

You can pass `InterpreterOptions` to utilize multiple threads:

```dart
final options = InterpreterOptions()..threads = 4; // or Platform.numberOfProcessors
final interpreter = await Interpreter.fromAsset(_modelPath, options: options);
```

Note: Keep the current behavior unless you want to measure speed-ups; correctness comes first.

## Running and debugging

- Clean and rebuild:

```powershell
flutter clean
flutter pub get
flutter run -d windows
```

- Confirm environment:

```powershell
flutter doctor -v
flutter devices
```

- If launching fails, capture the exact error messages and compare against the issues section above.

## UI helpers for reporting

- Use the "Copy Test Results" button in the app to copy the overall result and per-test details to your clipboard for quick sharing or issue reports.

## Notes

- Large synthetic inputs (256x256) can be CPU-intensive; performance varies by device.
- If your real model uses different input ranges, normalize your synthetic data accordingly.
- Always align the generator structure to the model’s expected layout and data type.

---

## Result 1: Failures, Causes, and Fixes

This section captures a concrete run (“Result 1”), the failures observed, their root causes, and the code changes that fix them.

### Result 1 – Individual Tests

```
Individual Tests:
- ✅ Model Loading: Input: [1, 3, 256, 256], Output: [1, 512]
- ❌ Basic Inference: ERROR: Invalid argument(s): Output object shape mismatch, interpreter returned output of shape: [1, 512] while shape of output provided as argument in run is: [512]
- ❌ Inference Test 2: ERROR: Invalid argument(s): Output object shape mismatch, interpreter returned output of shape: [1, 512] while shape of output provided as argument in run is: [512]
- ❌ Inference Test 3: ERROR: Invalid argument(s): Output object shape mismatch, interpreter returned output of shape: [1, 512] while shape of output provided as argument in run is: [512]
- ❌ Inference Test 4: ERROR: Invalid argument(s): Output object shape mismatch, interpreter returned output of shape: [1, 512] while shape of output provided as argument in run is: [512]
- ❌ Inference Test 5: ERROR: Invalid argument(s): Output object shape mismatch, interpreter returned output of shape: [1, 512] while shape of output provided as argument in run is: [512]
- ❌ Inference Test 6: ERROR: Invalid argument(s): Output object shape mismatch, interpreter returned output of shape: [1, 512] while shape of output provided as argument in run is: [512]
- ❌ Consistency Test: ERROR: Invalid argument(s): Output object shape mismatch, interpreter returned output of shape: [1, 512] while shape of output provided as argument in run is: [512]
- ❌ resetVariableTensors Test: ERROR: Bad state: Should not acces delegate after it has been closed.
```

### Why these failed

1) Output buffer shape mismatch
   - The model reports output shape `[1, 512]`, but the code passed a flattened `List<double>` of length `512` to `interpreter.run`. For tflite_flutter, the output buffer’s structure must mirror the tensor shape. Passing `[512]` for a `[1,512]` tensor triggers the mismatch error.

2) Consistency test indirectly hit the same mismatch
   - Since the same `run` call/shape mismatch existed, the consistency test failed for the same reason until the output buffer shape was fixed.

3) resetVariableTensors test “delegate closed”
   - Running separate interpreters per call is fine, but if a delegate (e.g., XNNPACK) is involved and an interpreter is closed before the subsequent call is fully completed, it can lead to the “delegate has been closed” error. Using a single interpreter instance for both runs, with `resetVariableTensors()` between them, avoids the issue and better reflects the intended usage.

### The fix

- Build a shape-aware output buffer from the tensor’s shape and pass it to `interpreter.run`. After the call, flatten the nested structure for hashing/sampling.
- Run the reset test using a single interpreter instance and call `resetVariableTensors()` in-between the two runs.

#### Updated code snippets

Create a buffer matching `[1,512]`, `[N,C,H,W]`, etc., and flatten it for hashing:

```dart
// Build a nested output buffer matching the tensor shape, e.g. [1,512] -> List<List<double>>
dynamic _createOutputBuffer(List<int> shape) {
  if (shape.isEmpty) return 0.0;
  if (shape.length == 1) {
    return List<double>.filled(shape[0], 0.0);
  }
  final dim = shape.first;
  final rest = shape.sublist(1);
  return List.generate(dim, (_) => _createOutputBuffer(rest));
}

// Flatten nested lists of doubles to a single list
List<double> _flattenOutput(dynamic output) {
  final result = <double>[];
  void walk(dynamic node) {
    if (node is List) {
      for (final v in node) walk(v);
    } else if (node is num) {
      result.add(node.toDouble());
    }
  }
  walk(output);
  return result;
}
```

Use the helpers in inference:

```dart
final outputShape = interpreter.getOutputTensor(0).shape;
final output = _createOutputBuffer(outputShape);
interpreter.run(input, output);
final flat = _flattenOutput(output);
final hash = _calculateHash(flat);
```

Run two inferences on the same interpreter for the reset test:

```dart
Future<Map<String, String>> _runTwoInferencesWithSameInterpreter(int seed1, int seed2) async {
  final interpreter = await Interpreter.fromAsset(_modelPath);
  interpreter.allocateTensors();
  try {
    final outputShape = interpreter.getOutputTensor(0).shape;
    // First inference
    final input1 = _generateTestInput(seed1);
    final out1 = _createOutputBuffer(outputShape);
    interpreter.run(input1, out1);
    final hash1 = _calculateHash(_flattenOutput(out1));
    // Reset and second inference
    interpreter.resetVariableTensors();
    final input2 = _generateTestInput(seed2);
    final out2 = _createOutputBuffer(outputShape);
    interpreter.run(input2, out2);
    final hash2 = _calculateHash(_flattenOutput(out2));
    return {'hash1': hash1, 'hash2': hash2};
  } finally {
    interpreter.close();
  }
}
```

And update the test wiring:

```dart
await _addTestResult('resetVariableTensors Test', () async {
  final combined = await _runTwoInferencesWithSameInterpreter(1, 2);
  final isDifferent = combined['hash1'] != combined['hash2'];
  return 'Different inputs → Different outputs: $isDifferent ('
         '${combined['hash1']} vs ${combined['hash2']})';
});
```

### After applying the fix

- The output-shape mismatch errors no longer occur because the buffer structure now matches the tensor shape the interpreter returns.
- The reset test runs two inferences safely on the same interpreter without referencing a closed delegate.
- If outputs are still identical with different inputs, the model may be deterministic given the synthetic input range or require additional preprocessing. In that case, verify the model’s expected input format (layout, normalization) and adjust `_generateTestInput` accordingly.

---

## Result 2: Failures, Causes, and Fixes

This section captures a subsequent run (“Result 2”), the failures observed, the root cause, and how we fixed it.

### Result 2 – Overall and Individual Tests

```
Overall Result:
❌ TEST FAILED: IDENTICAL OUTPUTS DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 ISSUE: Different inputs produced identical outputs
📋 All output hashes: 
🛠️  SOLUTION: Use resetVariableTensors() before each inference

Platform: windows
Model: MobileCLIP (stateful model)

Individual Tests:
- ✅ Model Loading: Input: [1, 3, 256, 256], Output: [1, 512]
- ❌ Basic Inference: ERROR: Null check operator used on a null value
- ❌ Inference Test 2: ERROR: Null check operator used on a null value
- ❌ Inference Test 3: ERROR: Null check operator used on a null value
- ❌ Inference Test 4: ERROR: Null check operator used on a null value
- ❌ Inference Test 5: ERROR: Null check operator used on a null value
- ❌ Inference Test 6: ERROR: Null check operator used on a null value
- ❌ Consistency Test: ERROR: Null check operator used on a null value
- ❌ resetVariableTensors Test: ERROR: Null check operator used on a null value
```

### Why these failed

- Some test code used the null-assertion operator (`!`) when reading values from `Map<String, String>`, e.g., `result['hash']!`. When a prior operation failed (e.g., inference or hashing), these map entries could be null or absent, causing the “Null check operator used on a null value” runtime crash.
- The consistency and reset tests propagated the same issue by assuming hashes were always present.

### The fix

- Removed null-bang operators from result map lookups and introduced safe fallbacks and early error messages.
- Added guards in the Consistency and resetVariableTensors tests to handle missing hashes gracefully instead of crashing.

#### Updated code examples

Null-safe lookups and fallbacks:

```dart
// Before
firstOutputHash = result['hash']!;

// After
firstOutputHash = result['hash'] ?? '';
```

Formatting the UI result safely:

```dart
return 'Hash: ${result['hash'] ?? 'n/a'}, Sample: ${result['sample'] ?? 'n/a'}';
```

Consistency test with guards:

```dart
final h1 = result1['hash'];
final h2 = result2['hash'];
if (h1 == null || h2 == null || h1.isEmpty || h2.isEmpty) {
  return 'ERROR: Inference produced no hash (h1=${h1 ?? 'null'}, h2=${h2 ?? 'null'})';
}
final isConsistent = h1 == h2;
return 'Same input → Same output: $isConsistent ($h1 vs $h2)';
```

Reset test with guards:

```dart
final combined = await _runTwoInferencesWithSameInterpreter(1, 2);
final h1 = combined['hash1'];
final h2 = combined['hash2'];
if (h1 == null || h2 == null || h1.isEmpty || h2.isEmpty) {
  return 'ERROR: Inference produced no hash (h1=${h1 ?? 'null'}, h2=${h2 ?? 'null'})';
}
final isDifferent = h1 != h2;
return 'Different inputs → Different outputs: $isDifferent ($h1 vs $h2)';
```

### After applying the fix

- The “Null check operator used on a null value” crashes are eliminated.
- Failures are now reported as explicit error messages in the test output rather than causing a crash.
- If errors persist, capture the exact per-test error strings to guide the next fix (e.g., shape mismatch, asset load issues, or input layout problems).

---

## Result 3: Failures, Causes, and Fixes

This section documents a subsequent run (“Result 3”) where crashes continued with “Null check operator used on a null value” across most tests.

### Result 3 – Overall and Individual Tests

```
Overall Result:
❌ TEST FAILED: IDENTICAL OUTPUTS DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 ISSUE: Different inputs produced identical outputs
📋 All output hashes: 
🛠️  SOLUTION: Use resetVariableTensors() before each inference

Platform: windows
Model: MobileCLIP (stateful model)

Individual Tests:
- ✅ Model Loading: Input: [1, 3, 256, 256], Output: [1, 512]
- ❌ Basic Inference: ERROR: Null check operator used on a null value
- ❌ Inference Test 2: ERROR: Null check operator used on a null value
- ❌ Inference Test 3: ERROR: Null check operator used on a null value
- ❌ Inference Test 4: ERROR: Null check operator used on a null value
- ❌ Inference Test 5: ERROR: Null check operator used on a null value
- ❌ Inference Test 6: ERROR: Null check operator used on a null value
- ❌ Consistency Test: ERROR: Null check operator used on a null value
- ❌ resetVariableTensors Test: ERROR: Null check operator used on a null value
```

### Why this still failed

- Even after removing some null-bang operators, separate interpreter instances and per-call closes increased the likelihood of null/invalid states during hot restarts and rapid test sequencing.
- Map lookups were made safer, but interpreter initialization and tensor shape caching remained ad hoc per call, so downstream code could still see nulls if initialization failed.

### The fix (stronger invariants)

- Introduced a single shared interpreter with safe initialization and caching of input/output shapes: `_ensureInterpreterInitialized()`.
- All inference paths now call this initializer and early-return error states instead of proceeding and crashing.
- The reset test now runs two inferences on the same interpreter instance.
- A proper `dispose()` closes the interpreter once when the widget is disposed.

#### Key code additions

```dart
Interpreter? _interpreter;
List<int>? _inputShape;
List<int>? _outputShape;
String? _lastError;

Future<bool> _ensureInterpreterInitialized() async {
  if (_interpreter != null && _inputShape != null && _outputShape != null) {
    return true;
  }
  try {
    _lastError = null;
    _interpreter ??= await Interpreter.fromAsset(_modelPath);
    _interpreter!.allocateTensors();
    final inT = _interpreter!.getInputTensor(0);
    final outT = _interpreter!.getOutputTensor(0);
    _inputShape = List<int>.from(inT.shape);
    _outputShape = List<int>.from(outT.shape);
    return true;
  } catch (e) {
    _lastError = 'Interpreter init failed: $e';
    return false;
  }
}

@override
void dispose() {
  try { _interpreter?.close(); } catch (_) {}
  super.dispose();
}
```

All inference paths now use cached shapes and a single interpreter:

```dart
final ok = await _ensureInterpreterInitialized();
if (!ok || _interpreter == null || _outputShape == null) {
  return {'hash': '', 'sample': ''};
}
final input = _generateTestInput(seed);
final output = _createOutputBuffer(_outputShape!);
_interpreter!.run(input, output);
```

### After applying the fix

- Hot restarts and test reruns no longer hit null-bang crashes from stale or uninitialized interpreter state.
- Errors are surfaced as strings in the UI instead of runtime exceptions, making further troubleshooting straightforward.
- Next, if outputs remain identical for different inputs, focus on model expectations (input layout, normalization range) and confirm that `_generateTestInput` matches `inputShape` and preprocessing requirements.
