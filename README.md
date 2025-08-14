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

### Missing asset/model
If the app crashes with an asset-not-found error, ensure:
- `assets/models/mobileclip_v2.tflite` exists locally
- `pubspec.yaml` has:

```yaml
flutter:
	assets:
		- assets/models/
```

Then run `flutter pub get` again.

### Output shape/type mismatch
If you see errors like “Output object shape mismatch”, it means the provided output buffer’s structure didn’t match the tensor’s shape/type. This app creates shape-accurate nested lists for you, so avoid changing that unless you know the exact shapes.

### Delegate/reset issues
Some delegates don’t support `resetVariableTensors()`. The app detects this and falls back to new interpreters between runs. That’s expected on certain platforms.

## Project layout

- `lib/main.dart`: UI + inference test harness
- `assets/models/`: put your `.tflite` file(s) here (ignored by Git; `.gitkeep` keeps the folder)
- `.gitignore`: excludes large model binaries and build artifacts

## Implementing TFLite Flutter inference (guide)

This section provides a clear, repeatable pattern to run TFLite inference in Flutter. It uses the MobileCLIP v2 model as an example; adapt shapes and preprocessing for other models.

### 1) Add dependencies and assets

`pubspec.yaml`:

```yaml
dependencies:
	flutter:
		sdk: flutter
	tflite_flutter: ^0.11.0
	crypto: ^3.0.5 # optional, for hashing/logging

flutter:
	assets:
		- assets/models/
```

Place your model at `assets/models/mobileclip_v2.tflite`.

### 2) Load the model and query shapes

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

Future<(List<int> inputShape, List<List<int>> outputShapes)> describeModel(String modelPath) async {
	final interpreter = await Interpreter.fromAsset(modelPath);
	try {
		interpreter.allocateTensors();
		final inputShape = List<int>.from(interpreter.getInputTensor(0).shape);
		final outputs = interpreter.getOutputTensors();
		final outputShapes = outputs.map((t) => List<int>.from(t.shape)).toList();
		return (inputShape, outputShapes);
	} finally {
		interpreter.close();
	}
}
```

MobileCLIP v2 in this repo exposes three outputs:
- Two embeddings with shape `[1, 512]` (float32)
- One scalar `[]` (float32)

Your model may differ; always inspect shapes and types at runtime.

### 3) Prepare input data

Use the exact input shape reported by the model. For MobileCLIP v2 (export dependent), the example uses a 4D nested list `[1, 3, 256, 256]` with values normalized to `[-1, 1]`:

```dart
List<List<List<List<double>>>> generateInput({int seed = 1}) {
	final rnd = Random(seed);
	final channels = List.generate(
		3,
		(_) => List.generate(
			256,
			(_) => List.generate(256, (_) => (rnd.nextDouble() - 0.5) * 2.0, growable: false),
			growable: false,
		),
		growable: false,
	);
	return [channels];
}
```

If your model expects NHWC `[1, H, W, C]` or quantized `uint8`, convert accordingly.

### 4) Create shape-accurate output buffers

```dart
Object zeroBufferForShape(List<int> shape, TensorType type) {
	Object leaf() {
		final name = type.toString();
		if (name.endsWith('float32') || name.endsWith('float16') || name.contains('float')) return 0.0;
		if (name.endsWith('bool')) return false;
		if (name.endsWith('string')) return '';
		return 0;
	}
	if (shape.isEmpty) return leaf();
	Object build(int d) {
		final n = shape[d];
		if (d == shape.length - 1) {
			final l = leaf();
			if (l is double) return List<double>.filled(n, 0.0);
			if (l is int) return List<int>.filled(n, 0);
			if (l is bool) return List<bool>.filled(n, false);
			if (l is String) return List<String>.filled(n, '');
			return List<double>.filled(n, 0.0);
		}
		return List.generate(n, (_) => build(d + 1));
	}
	return build(0);
}
```

### 5) Run inference with runForMultipleInputs

```dart
Future<List<dynamic>> infer(String modelPath, dynamic input) async {
	final interpreter = await Interpreter.fromAsset(modelPath);
	try {
		interpreter.allocateTensors();
		final outputs = interpreter.getOutputTensors();
		final out = <int, Object>{};
		for (int i = 0; i < outputs.length; i++) {
			out[i] = zeroBufferForShape(outputs[i].shape, outputs[i].type);
		}
		interpreter.runForMultipleInputs([input], out);
		return List.generate(outputs.length, (i) => out[i]!);
	} finally {
		interpreter.close();
	}
}
```

### 6) Pick the right output (if multiple)

If you don’t know which output to consume, compare hashes across two inputs and select the one that changes:

```dart
List<double> flatten(dynamic x) {
	final r = <double>[];
	void walk(dynamic n) { if (n is List) n.forEach(walk); else if (n is num) r.add(n.toDouble()); }
	walk(x);
	return r;
}

int chooseVaryingOutput(List<dynamic> o1, List<dynamic> o2) {
	for (int i = 0; i < o1.length; i++) {
		if (flatten(o1[i]).length <= 1) continue;
		final h1 = sha1.convert(utf8.encode(flatten(o1[i]).join(','))).toString();
		final h2 = sha1.convert(utf8.encode(flatten(o2[i]).join(','))).toString();
		if (h1 != h2) return i;
	}
	return 0;
}
```

If your model documents output names (e.g., “image_embedding”), select by name instead of detection.

### 7) Reuse vs fresh interpreter

Stateless inference: easiest is to create a fresh interpreter per call. If you need reuse:

```dart
try {
	interpreter.resetVariableTensors();
} catch (_) {
	// Fallback to fresh interpreter if delegate/platform doesn’t support reset
}
```

### 8) Adapting to another model

When switching models, verify and update:
- Input shape/order and preprocessing (normalization, quantization)
- Output shapes/types; rebuild buffers accordingly
- Which output(s) carry the signal you need (by name or by behavior)
- Any required delegates (GPU/NNAPI/CoreML) and their limitations (e.g., reset support)

With this pattern, you avoid common pitfalls like shape mismatches and ambiguous outputs.

## License

MIT (or your choice). Update as needed.

