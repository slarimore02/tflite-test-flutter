import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:crypto/crypto.dart';

void main() {
  runApp(TFLiteTestApp());
}

class TFLiteTestApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TFLite Windows Test',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        fontFamily: 'Consolas',
      ),
      home: TFLiteTestScreen(),
    );
    // Helper: build a nested output buffer matching the tensor shape, e.g. [1,512] -> List<List<double>>
  }
}

class TFLiteTestScreen extends StatefulWidget {
  @override
  _TFLiteTestScreenState createState() => _TFLiteTestScreenState();
}

class _TFLiteTestScreenState extends State<TFLiteTestScreen> {
  final List<TestResult> _testResults = [];
  bool _isRunning = false;
  String _overallResult = '';
  final String _modelPath = 'assets/models/mobileclip_v2.tflite'; // Update this path
  String? _lastError;
  int? _selectedOutputIndex; // Auto-detected 512-d output that varies with input

  @override
  void initState() {
    super.initState();
    // Automatically start test when app loads
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _runAutomaticTest();
    });
  }

  // Flatten any nested List structure of numbers into a single List<double>
  List<double> _flattenOutput(dynamic output) {
    final result = <double>[];
    void walk(dynamic node) {
      if (node is List) {
        for (final v in node) {
          walk(v);
        }
      } else if (node is num) {
        result.add(node.toDouble());
      }
    }
    walk(output);
    return result;
  }

  // Query model input/output shapes without keeping the interpreter alive
  Future<Map<String, List<int>>> _getModelShapes() async {
    Interpreter? interpreter;
    try {
      interpreter = await Interpreter.fromAsset(_modelPath);
      interpreter.allocateTensors();
      final inputShape = List<int>.from(interpreter.getInputTensor(0).shape);
      final outputShape = List<int>.from(interpreter.getOutputTensor(0).shape);
      return {'input': inputShape, 'output': outputShape};
    } catch (e) {
      _lastError = 'Interpreter shape query failed: $e';
      rethrow;
    } finally {
      try {
        interpreter?.close();
      } catch (_) {}
    }
  }

  // Build a zero-initialized buffer matching the given tensor shape and type.
  // Examples:
  //  - shape [1,512], type float32 -> List<List<double>>
  //  - shape [], type float32 -> 0.0 (scalar)
  //  - shape [1], type int32 -> List<int>
  Object _zeroBufferForShape(List<int> shape, TensorType type) {
    Object zeroLeaf() {
      final name = type.toString(); // e.g., TensorType.float32
      if (name.endsWith('float32') || name.endsWith('float16') || name.contains('float')) {
        return 0.0;
      }
      if (name.endsWith('bool')) return false;
      if (name.endsWith('string')) return '';
      // Default to integer for other numeric types (int8/int16/int32/int64/uint8...)
      return 0;
    }

    if (shape.isEmpty) {
      // Scalar tensor
      return zeroLeaf();
    }

    Object buildAtDim(int dim) {
      final len = shape[dim];
      if (dim == shape.length - 1) {
        final leaf = zeroLeaf();
        if (leaf is double) return List<double>.filled(len, 0.0);
        if (leaf is int) return List<int>.filled(len, 0);
        if (leaf is bool) return List<bool>.filled(len, false);
        if (leaf is String) return List<String>.filled(len, '');
        return List<double>.filled(len, 0.0);
      }
      return List.generate(len, (_) => buildAtDim(dim + 1));
    }

    return buildAtDim(0);
  }

  // Helper: flatten any nested List structure of doubles into a single List<double>
  Future<void> _runAutomaticTest() async {
    setState(() {
      _isRunning = true;
      _testResults.clear();
      _overallResult = 'Running tests...';
    });

    try {
      await _performInferenceTests();
    } catch (e) {
      setState(() {
        _overallResult = '❌ CRITICAL FAILURE: $e';
        _isRunning = false;
      });
    }
  }

  Future<void> _performInferenceTests() async {
    // Quick smoke test: ensure our buffer creation/flatten works as expected
    await _testBufferCreation();

    // Test 1: Model Loading Test
    await _addTestResult('Model Loading', () async {
      final shapes = await _getModelShapes();
      return 'Input: ${shapes['input']}, Output: ${shapes['output']}';
    });

  // Detect which 512-d output varies with input (e.g., image embedding)
  await _detectDynamicOutputIndex();

    // Test 2: Basic Inference Test
    String firstOutputHash = '';
    await _addTestResult('Basic Inference', () async {
      final result = await _runSingleInference(1);
  firstOutputHash = result['hash'] ?? '';
  return 'Hash: ${result['hash'] ?? 'n/a'}, Sample: ${result['sample'] ?? 'n/a'}';
    });

  // (Removed Typed Inference: using nested lists with fresh interpreters instead)

    // Test 3-7: Multiple Different Inputs
    final outputHashes = <String>[];
    outputHashes.add(firstOutputHash);

    for (int i = 2; i <= 6; i++) {
      await _addTestResult('Inference Test $i', () async {
        final result = await _runSingleInference(i);
  outputHashes.add(result['hash'] ?? '');
  return 'Hash: ${result['hash'] ?? 'n/a'}, Sample: ${result['sample'] ?? 'n/a'}';
      });
    }

    // Test 8: Same Input Consistency Test
    await _addTestResult('Consistency Test', () async {
      final result1 = await _runSingleInference(1);
      final result2 = await _runSingleInference(1);
      
      final h1 = result1['hash'];
      final h2 = result2['hash'];
      if (h1 == null || h2 == null || h1.isEmpty || h2.isEmpty) {
        return 'ERROR: Inference produced no hash (h1=${h1 ?? 'null'}, h2=${h2 ?? 'null'})';
      }
      final isConsistent = h1 == h2;
      return 'Same input → Same output: $isConsistent ($h1 vs $h2)';
    });

    // Test 9: resetVariableTensors Test
    await _addTestResult('resetVariableTensors Test', () async {
  // Run two inferences on a single interpreter, resetting state in-between
  final combined = await _runTwoInferencesWithSameInterpreter(1, 2);
  final h1 = combined['hash1'];
  final h2 = combined['hash2'];
  if (h1 == null || h2 == null || h1.isEmpty || h2.isEmpty) {
    return 'ERROR: Inference produced no hash (h1=${h1 ?? 'null'}, h2=${h2 ?? 'null'})';
  }
  final isDifferent = h1 != h2;
  return 'Different inputs → Different outputs: $isDifferent ($h1 vs $h2)';
    });

    // Final Analysis
    _analyzeResults(outputHashes);
  }

  // Determine which output tensor (among non-scalar 512-d) changes with input; store index.
  Future<void> _detectDynamicOutputIndex() async {
    try {
      // First run (seed 1)
      final run1 = await _collectAllOutputs(seed: 1);
      // Second run (seed 2)
      final run2 = await _collectAllOutputs(seed: 2);

      // Find candidate indices with product 512 (or any >1) and compare hashes
      final candidates = <int>{};
      run1.forEach((i, v) {
        if (v.length > 1) candidates.add(i);
      });
      int? chosen;
      for (final i in candidates) {
        final h1 = _calculateHash(run1[i]!);
        final h2 = _calculateHash(run2[i]!);
        if (h1 != h2) {
          chosen = i;
          break;
        }
      }
      _selectedOutputIndex = chosen ?? (candidates.isNotEmpty ? candidates.first : 0);
      print('Selected output tensor index: $_selectedOutputIndex');
    } catch (e, stack) {
      print('Output index detection failed: $e\n$stack');
      _selectedOutputIndex = 0;
    }
  }

  // Helper: run a single inference and return flattened outputs per index
  Future<Map<int, List<double>>> _collectAllOutputs({required int seed}) async {
    Interpreter? interpreter;
    try {
      interpreter = await Interpreter.fromAsset(_modelPath);
      interpreter.allocateTensors();
      final outputTensors = interpreter.getOutputTensors();
      final input = _generateTestInput(seed);
      final Map<int, Object> outputs = <int, Object>{};
      for (int i = 0; i < outputTensors.length; i++) {
        outputs[i] = _zeroBufferForShape(outputTensors[i].shape, outputTensors[i].type);
      }
      interpreter.runForMultipleInputs([input], outputs);
      final result = <int, List<double>>{};
      for (int i = 0; i < outputTensors.length; i++) {
        result[i] = _flattenOutput(outputs[i]);
      }
      return result;
    } finally {
      try { interpreter?.close(); } catch (_) {}
    }
  }

  Future<void> _addTestResult(String testName, Future<String> Function() testFunction) async {
    try {
      final result = await testFunction();
      setState(() {
        _testResults.add(TestResult(testName, true, result));
      });
    } catch (e, stackTrace) {
      print('ERROR in $testName: $e');
      print('Stack trace:\n$stackTrace');
      setState(() {
        _testResults.add(TestResult(testName, false, 'ERROR: $e'));
      });
    }
  }

  Future<Map<String, String>> _runSingleInference(int seed) async {
    Interpreter? interpreter;
    try {
      interpreter = await Interpreter.fromAsset(_modelPath);
      interpreter.allocateTensors();

      final inputShape = interpreter.getInputTensor(0).shape;
      final outputTensors = interpreter.getOutputTensors();

      print('Input shape: $inputShape');
      print('Number of output tensors: ${outputTensors.length}');

      // Generate 4D nested list input
      final input = _generateTestInput(seed);

      // Create output map for all output tensors using buffers that match shapes
      final Map<int, Object> outputs = <int, Object>{};
      for (int i = 0; i < outputTensors.length; i++) {
        final shape = outputTensors[i].shape;
  final type = outputTensors[i].type;
        final name = outputTensors[i].name;
        final size = shape.isEmpty ? 1 : shape.fold<int>(1, (a, b) => a * b);
        print('Output tensor $i name: $name, shape: $shape, type: $type, size: $size');
        outputs[i] = _zeroBufferForShape(shape, type);
      }

      // Use runForMultipleInputs directly to avoid the run() bug
      interpreter.runForMultipleInputs([input], outputs);

  // Read selected output tensor result (flatten to 1D for hashing/sample)
  final idx = _selectedOutputIndex ?? 0;
  final output0 = _flattenOutput(outputs[idx]);
      final hash = _calculateHash(output0);
      final sample = output0.take(3).map((v) => v.toStringAsFixed(6)).join(', ');

      return {'hash': hash, 'sample': sample};
    } catch (e, stack) {
      print('Inference error: $e\nStack: $stack');
      return {'hash': '', 'sample': 'Error: $e'};
    } finally {
      try {
        interpreter?.close();
      } catch (_) {}
    }
  }

  // (Removed typed-data alternative; using nested lists with fresh interpreters)

  // (Removed single-inference-with-reset; we only reuse interpreter in the dedicated reset test below)

  Future<Map<String, String>> _runTwoInferencesWithSameInterpreter(int seed1, int seed2) async {
    Interpreter? interpreter;
    try {
      interpreter = await Interpreter.fromAsset(_modelPath);
      interpreter.allocateTensors();

      // First inference
      final outputTensors = interpreter.getOutputTensors();
      final input1 = _generateTestInput(seed1);
      final Map<int, Object> outputs1 = <int, Object>{};
      for (int i = 0; i < outputTensors.length; i++) {
        final shape = outputTensors[i].shape;
        final type = outputTensors[i].type;
        outputs1[i] = _zeroBufferForShape(shape, type);
      }
      interpreter.runForMultipleInputs([input1], outputs1);
  final idx1 = _selectedOutputIndex ?? 0;
  final output1 = _flattenOutput(outputs1[idx1]);
      final hash1 = _calculateHash(output1);

      // Try to reset tensors; if not supported with current delegate, we'll fall back to a new interpreter
      bool resetWorked = true;
      try {
        interpreter.resetVariableTensors();
      } catch (e, stack) {
        resetWorked = false;
        print('resetVariableTensors not available with current delegate, falling back to fresh interpreter: $e\n$stack');
      }

      // Second inference
      final input2 = _generateTestInput(seed2);
      final Map<int, Object> outputs2 = <int, Object>{};
      if (resetWorked) {
        for (int i = 0; i < outputTensors.length; i++) {
          final shape = outputTensors[i].shape;
          final type = outputTensors[i].type;
          outputs2[i] = _zeroBufferForShape(shape, type);
        }
        interpreter.runForMultipleInputs([input2], outputs2);
      } else {
        // Dispose the original interpreter and use a fresh one for the second run
        try {
          interpreter.close();
        } catch (_) {}
        interpreter = await Interpreter.fromAsset(_modelPath);
        interpreter.allocateTensors();
        final outputTensors2 = interpreter.getOutputTensors();
        for (int i = 0; i < outputTensors2.length; i++) {
          final shape = outputTensors2[i].shape;
          final type = outputTensors2[i].type;
          outputs2[i] = _zeroBufferForShape(shape, type);
        }
        interpreter.runForMultipleInputs([input2], outputs2);
      }
  final idx = _selectedOutputIndex ?? 0;
  final output2 = _flattenOutput(outputs2[idx]);
      final hash2 = _calculateHash(output2);

      return {'hash1': hash1, 'hash2': hash2};
    } catch (e, stack) {
      print('Reset test error: $e\nStack: $stack');
      return {'hash1': '', 'hash2': ''};
    } finally {
      try {
        interpreter?.close();
      } catch (_) {}
    }
  }

  List<List<List<List<double>>>> _generateTestInput(int seed) {
    final random = Random(seed);

    // Generate synthetic image data (adjust dimensions for your model)
    // This assumes input shape [1, 3, 256, 256] for MobileCLIP
    final channels = List<List<List<double>>>.generate(
      3,
      (c) => List<List<double>>.generate(
        256,
        (h) => List<double>.generate(
          256,
          (w) => (random.nextDouble() - 0.5) * 2.0, // Random values between -1 and 1
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );

    return [channels];
  }

  // (Removed flat typed-data generator; use nested 4D input instead)

  String _calculateHash(List<double> output) {
    final bytes = output.map((d) => d.toStringAsFixed(8)).join(',');
    return sha1.convert(utf8.encode(bytes)).toString().substring(0, 8);
  }

  void _analyzeResults(List<String> outputHashes) {
    final uniqueHashes = outputHashes.toSet();
    final allSame = uniqueHashes.length == 1;
    
    setState(() {
  if (allSame) {
        _overallResult = '''
❌ TEST FAILED: IDENTICAL OUTPUTS DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 ISSUE: Different inputs produced identical outputs
📋 All output hashes: ${outputHashes.first}
🛠️  SOLUTION: Use a fresh Interpreter per inference, or reset only when reusing the same instance

Platform: ${Platform.operatingSystem}
Model: MobileCLIP (stateful model)
''';
      } else {
        _overallResult = '''
✅ TEST PASSED: INFERENCE WORKING CORRECTLY  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 SUCCESS: Different inputs produced different outputs
📊 Unique outputs: ${uniqueHashes.length}/${outputHashes.length}
🔍 Output hashes: ${uniqueHashes.take(3).join(', ')}${uniqueHashes.length > 3 ? '...' : ''}

Platform: ${Platform.operatingSystem}
Model: Working as expected
''';
      }
      _isRunning = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('TensorFlow Lite Windows Test'),
        backgroundColor: _getResultColor(),
      ),
      body: Container(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Overall Result Card
            Card(
              color: _getResultColor().withOpacity(0.1),
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Test Results',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 8),
                    if (_isRunning)
                      Row(
                        children: [
                          CircularProgressIndicator(),
                          SizedBox(width: 16),
                          Text('Running automated tests...'),
                        ],
                      )
                    else
                      Text(
                        _overallResult,
                        style: TextStyle(
                          fontSize: 14,
                          fontFamily: 'Consolas',
                          color: _getResultColor(),
                        ),
                      ),
                  ],
                ),
              ),
            ),
            
            SizedBox(height: 16),
            
            // Individual Test Results
            Text(
              'Individual Tests:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            
            Expanded(
              child: ListView.builder(
                itemCount: _testResults.length,
                itemBuilder: (context, index) {
                  final result = _testResults[index];
                  return Card(
                    margin: EdgeInsets.symmetric(vertical: 4),
                    child: ListTile(
                      leading: Icon(
                        result.passed ? Icons.check_circle : Icons.error,
                        color: result.passed ? Colors.green : Colors.red,
                      ),
                      title: Text(result.testName),
                      subtitle: Text(
                        result.details,
                        style: TextStyle(fontFamily: 'Consolas', fontSize: 12),
                      ),
                    ),
                  );
                },
              ),
            ),
            
            // Action Buttons
            Row(
              children: [
                ElevatedButton(
                  onPressed: _isRunning ? null : _runAutomaticTest,
                  child: Text('Run Tests Again'),
                ),
                SizedBox(width: 16),
                ElevatedButton(
                  onPressed: () => _showInstructions(context),
                  child: Text('Instructions'),
                ),
                SizedBox(width: 16),
                ElevatedButton(
                  onPressed: _isRunning ? null : _copyTestResults,
                  child: Text('Copy Test Results'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Color _getResultColor() {
    if (_isRunning) return Colors.blue;
    if (_overallResult.startsWith('✅')) return Colors.green;
    if (_overallResult.startsWith('❌')) return Colors.red;
    return Colors.grey;
  }

  Future<void> _copyTestResults() async {
    final allText = _composeAllTestResultsText();
    await Clipboard.setData(ClipboardData(text: allText));
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Test results copied to clipboard')),
      );
    }
  }

  String _composeAllTestResultsText() {
    final buffer = StringBuffer();
    buffer.writeln('Overall Result:');
    buffer.writeln(_overallResult.trim());
    buffer.writeln('\nIndividual Tests:');
    for (final r in _testResults) {
      buffer.writeln('- ${r.passed ? '✅' : '❌'} ${r.testName}: ${r.details}');
    }
    return buffer.toString();
  }

  void _showInstructions(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Setup Instructions'),
        content: SingleChildScrollView(
          child: Text('''
1. Add to pubspec.yaml:
   dependencies:
     tflite_flutter: ^0.11.0
     crypto: ^3.0.5

2. Add your model to assets/models/mobileclip_v2.tflite

3. Add to pubspec.yaml:
   flutter:
     assets:
       - assets/models/

4. Run: flutter pub get

5. Test Results:
  ✅ = Different inputs → Different outputs (WORKING)
  ❌ = Identical outputs (BROKEN - see below)

If test fails, use a fresh Interpreter per inference. Only reuse the same
interpreter if you call interpreter.resetVariableTensors() between runs.
'''),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Close'),
          ),
        ],
      ),
    );
  }

  // Buffer creation smoke test for debugging nested structure handling
  Future<void> _testBufferCreation() async {
    try {
      print('Testing buffer creation...');
    final testShape = [1, 512];
    final buffer = List<double>.filled(testShape[0] * testShape[1], 0)
      .reshape(testShape);
    print('Buffer created (reshape): ${buffer.runtimeType}');
    final flat = _flattenOutput(buffer);
      print('Flattened: ${flat.length} elements');
    } catch (e, stack) {
      print('Buffer test failed: $e\n$stack');
    }
  }

  @override
  void dispose() {
    super.dispose();
  }
}

class TestResult {
  final String testName;
  final bool passed;
  final String details;

  TestResult(this.testName, this.passed, this.details);
}