# Lessons Learned: TFLite + Flutter (Windows)

This document captures the practical lessons learned while integrating TensorFlow Lite in a Flutter desktop app and stabilizing inference.

## Highlights

- Match output buffer shapes and types exactly to the model’s tensors; nested lists must mirror the tensor dimensions.
- Prefer `runForMultipleInputs` with an explicit outputs map over `run()` to avoid internal path mismatches.
- Initialize outputs per tensor by constructing zero-initialized structures that match `shape` and `type` (including scalars `[]`).
- Some models expose multiple outputs; not all vary with input. Pick the right output by name or behavior.
- Use a fresh `Interpreter` per inference unless you must reuse and you’re sure `resetVariableTensors()` is supported for your delegate.

## Details

### 1) Output buffers must match tensor shape/type

TFLite will throw if output buffers don’t match. For example, if the tensor shape is `[1, 512]`, don’t pass a flat `List<double>` with 512 elements; pass a nested `List<List<double>>` with lengths `[1][512]`.

Define a helper to create shape-accurate buffers for all supported types:

```dart
Object zeroBufferForShape(List<int> shape, TensorType type) {
  Object zeroLeaf() {
    final name = type.toString();
    if (name.endsWith('float32') || name.endsWith('float16') || name.contains('float')) return 0.0;
    if (name.endsWith('bool')) return false;
    if (name.endsWith('string')) return '';
    return 0; // default numeric
  }

  if (shape.isEmpty) return zeroLeaf(); // scalar

  Object build(int dim) {
    final len = shape[dim];
    if (dim == shape.length - 1) {
      final leaf = zeroLeaf();
      if (leaf is double) return List<double>.filled(len, 0.0);
      if (leaf is int) return List<int>.filled(len, 0);
      if (leaf is bool) return List<bool>.filled(len, false);
      if (leaf is String) return List<String>.filled(len, '');
      return List<double>.filled(len, 0.0);
    }
    return List.generate(len, (_) => build(dim + 1));
  }

  return build(0);
}
```

### 2) Prefer runForMultipleInputs with explicit outputs

Using `interpreter.runForMultipleInputs([input], outputsMap)` avoids an internal `run()` path that can fail when buffers are not exactly what it expects. It also supports multiple outputs naturally.

```dart
final outputs = <int, Object>{};
for (int i = 0; i < outputTensors.length; i++) {
  outputs[i] = zeroBufferForShape(outputTensors[i].shape, outputTensors[i].type);
}
interpreter.runForMultipleInputs([input], outputs);
```

### 3) Inputs: shape and pre-processing

Always read `getInputTensor(0).shape` and prepare inputs accordingly. Don’t assume NHWC vs NCHW; the conversion/export determines this. Normalize or quantize input data matching the model’s expectations.

### 4) Selecting the right output

If a model has multiple outputs and you’re not sure which one to consume, compare output hashes across two different inputs and pick the tensor that changes. Or choose by tensor name if known.

### 5) Reuse vs fresh Interpreter

Some delegates don’t support `resetVariableTensors()`. If reuse fails, create a new interpreter for the next run. For simple stateless inference, a fresh interpreter per call is often simpler and safer.

### 6) Logging and diagnostics

Log shapes, types, tensor names, and a small sample of outputs to speed up debugging. Flatten nested outputs only for logging/hash comparisons.
