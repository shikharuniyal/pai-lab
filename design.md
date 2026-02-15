# LatentForge - Complete Pipeline Design & Architecture

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Architecture Principles](#2-architecture-principles)
3. [Component Architecture](#3-component-architecture)
4. [Pipeline Flow](#4-pipeline-flow)
5. [Data Models](#5-data-models)
6. [Security Architecture](#6-security-architecture)
7. [Determinism Strategy](#7-determinism-strategy)
8. [Error Handling](#8-error-handling)
9. [Performance Considerations](#9-performance-considerations)
10. [Deployment Architecture](#10-deployment-architecture)

---

## 1. System Overview

### 1.1 Purpose
LatentForge is a hybrid video transformation system that:
- Processes video locally for privacy and control
- Uses cloud AI (Gemini) for intelligent transformation decisions
- Ensures security through multi-layer validation
- Guarantees deterministic, reproducible results

### 1.2 Architecture Style
**Hybrid Pipeline Architecture** with the following characteristics:
- **Local Processing**: All pixel manipulation occurs on user's machine
- **Cloud Reasoning**: AI analysis and code generation via Gemini API
- **Layered Security**: Multiple validation stages
- **Modular Design**: Loosely coupled components
- **Fail-Safe**: Graceful degradation at every stage

### 1.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Local pixel processing | Privacy, performance, and cost control |
| Cloud AI reasoning | Leverage advanced AI without local model deployment |
| AST-based validation | Reliable code analysis before execution |
| Sandboxed execution | Security against malicious generated code |
| Caching system | Determinism and cost reduction |
| Parameter fallback | Graceful degradation when API/code fails |

---

## 2. Architecture Principles

### 2.1 Security First
- **Never trust generated code**: All code validated before execution
- **Principle of least privilege**: Minimal permissions for execution
- **Defense in depth**: Multiple security layers

### 2.2 Determinism
- **Reproducibility**: Same input → same output
- **Immutability**: Original frames never modified
- **Auditability**: Complete logging of all operations

### 2.3 Modularity
- **Single Responsibility**: Each component has one clear purpose
- **Loose Coupling**: Components communicate via well-defined interfaces
- **High Cohesion**: Related functionality grouped together

### 2.4 Fail-Safe Design
- **Graceful Degradation**: System continues with reduced functionality
- **Data Preservation**: Original data never lost
- **Clear Error Reporting**: Detailed logs and user feedback

---

## 3. Component Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      LatentForge                            │
│                   (Orchestrator Class)                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────┬─────────────┬──────────────┬──────────────┐
             ▼             ▼             ▼              ▼              ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
      │  Video   │  │ Feature  │  │  Gemini  │  │   Code   │  │ Sandbox  │
      │  Loader  │  │Extractor │  │  Client  │  │Validator │  │ Executor │
      └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
             │             │             │              │              │
             │             │             │              │              ▼
             │             │             │              │        ┌──────────┐
             │             │             │              │        │ Renderer │
             │             │             │              │        └──────────┘
             │             │             │              │
             └─────────────┴─────────────┴──────────────┴───────────────┐
                                                                         ▼
                                                                  ┌──────────┐
                                                                  │  Logger  │
                                                                  └──────────┘
```

### 3.2 Component Responsibilities

#### 3.2.1 VideoLoader
**Purpose**: Video ingestion and frame extraction

**Responsibilities**:
- Load video files using OpenCV
- Extract metadata (FPS, resolution, codec)
- Convert BGR to RGB color space
- Support frame sampling for quick analysis
- Buffer frames in memory

**Interfaces**:
```python
class VideoLoader:
    def load(video_path: str, max_frames: int = None) -> (List[ndarray], Dict)
    def load_sample_frames(video_path: str, num_samples: int) -> (List[ndarray], Dict)
```

**Dependencies**: OpenCV, NumPy

---

#### 3.2.2 FeatureExtractor
**Purpose**: Analyze video characteristics

**Responsibilities**:
- Extract spatial features (brightness, contrast, RGB, saturation, edges)
- Extract temporal features (motion, frame differences)
- Aggregate statistics across frames
- Construct structured JSON payload

**Interfaces**:
```python
class FeatureExtractor:
    def extract(frames: List[ndarray], metadata: Dict, style: str) -> Dict
    def _extract_spatial_features(frames: List[ndarray]) -> Dict
    def _extract_temporal_features(frames: List[ndarray]) -> Dict
```

**Processing Pipeline**:
```
Frames → Spatial Analysis → Temporal Analysis → JSON Payload
           ↓                      ↓
     [brightness,           [frame_diff,
      contrast,              motion_var,
      RGB dist,              smoothness]
      saturation,
      edge_density]
```

---

#### 3.2.3 GeminiClient
**Purpose**: Interface with Google Gemini API

**Responsibilities**:
- Construct prompts from latent payload
- Call Gemini API
- Parse JSON responses
- Cache responses by input hash
- Handle API errors with fallbacks

**Interfaces**:
```python
class GeminiClient:
    def get_transformation(latent_payload: Dict, use_cache: bool) -> Dict
    def _build_prompt(latent_payload: Dict) -> str
    def _parse_response(response_text: str) -> Dict
    def _get_style_based_transformation(latent_payload: Dict) -> Dict
```

**Fallback Strategy**:
```
API Call
   │
   ├─ Success → Parse JSON → Return
   │
   ├─ Parse Error → Use fallback transformation
   │
   └─ API Error → Style-based transformation
```

**Cache Mechanism**:
```
Input Payload → SHA-256 Hash → Cache Lookup
                                     ├─ Hit → Return cached
                                     └─ Miss → API Call → Cache Store
```

---

#### 3.2.4 CodeValidator
**Purpose**: Validate generated code for safety

**Responsibilities**:
- Parse code using AST
- Check for forbidden operations
- Validate allowed imports (NumPy, OpenCV only)
- Ensure function definitions exist
- Detect security violations

**Interfaces**:
```python
class CodeValidator:
    def validate(code: str) -> (bool, str)
```

**Validation Layers**:
```
Layer 1: Regex Pattern Matching
   ↓ (forbidden keywords)
Layer 2: AST Parsing
   ↓ (syntax validation)
Layer 3: AST Node Analysis
   ↓ (import checks, function calls)
Layer 4: Structural Validation
   ↓ (function definitions)
PASS/FAIL
```

**Forbidden Patterns**:
- Imports: `import os`, `import sys`, `import subprocess`
- File ops: `open()`, `file()`
- Network: `socket`, `urllib`, `requests`
- Execution: `eval()`, `exec()`, `compile()`, `__import__()`
- Dunder abuse: `__globals__`, `__builtins__` (except allowed)

---

#### 3.2.5 SandboxExecutor
**Purpose**: Execute transformations safely

**Responsibilities**:
- Execute validated code in restricted environment
- Apply parameter-based transformations
- Validate output frames
- Handle timeouts and errors
- Preserve original frames on failure

**Interfaces**:
```python
class SandboxExecutor:
    def apply_transformations(frames: List[ndarray], spec: Dict) -> List[ndarray]
    def _apply_generated_code(frames: List[ndarray], code: str) -> List[ndarray]
    def _apply_parameter_based(frames: List[ndarray], spec: Dict) -> List[ndarray]
```

**Execution Modes**:

**Mode 1: Code Execution**
```
Generated Code → Validation → Sandbox Setup → Execute → Validate Output
                                    ↓
                            [Restricted globals:
                             - numpy, cv2
                             - limited builtins
                             - no file/network]
```

**Mode 2: Parameter-Based**
```
Parameters → Lighting Transform → Color Transform → Sharpness Transform → Output
               ↓                      ↓                    ↓
         [gamma, sigmoid]      [temp, sat, contrast]  [unsharp mask]
```

**Sandbox Environment**:
```python
safe_globals = {
    'np': numpy,
    'cv2': cv2,
    '__builtins__': {
        'abs', 'all', 'any', 'bool', 'dict', 'enumerate',
        'filter', 'float', 'int', 'len', 'list', 'map',
        'max', 'min', 'range', 'round', 'sorted', 'sum',
        'tuple', 'type', 'zip'
    }
}
```

---

#### 3.2.6 Renderer
**Purpose**: Encode transformed frames to video

**Responsibilities**:
- Validate frame consistency
- Convert RGB to BGR
- Encode with H.264 codec
- Preserve FPS and resolution
- Verify output file

**Interfaces**:
```python
class Renderer:
    def render(frames: List[ndarray], output_path: str, metadata: Dict) -> str
    def render_preview(frames: List[ndarray], output_path: str, sample_rate: int) -> str
```

**Rendering Pipeline**:
```
Frames → Validation → RGB→BGR → Video Encoder → File Output
           ↓              ↓           ↓
     [shape check]  [color conv]  [H.264]
     [type check]
     [value clamp]
```

---

#### 3.2.7 LatentForgeLogger
**Purpose**: Comprehensive logging and auditing

**Responsibilities**:
- Session management
- Structured logging
- API request/response tracking
- Code validation logging
- Transformation history

**Interfaces**:
```python
class LatentForgeLogger:
    def log_api_request(request_data: Dict)
    def log_api_response(response_data: Dict, hash: str)
    def log_transformation(transformation_spec: Dict)
    def log_code_validation(code: str, is_valid: bool, error: str)
```

**Log File Structure**:
```
logs/
├── latentforge_20260215_143022.log          # Main log
├── api_responses_20260215_143022.jsonl       # API history
├── transformations_20260215_143022.jsonl     # Applied transforms
└── code_validation_20260215_143022.jsonl    # Validation results
```

---

## 4. Pipeline Flow

### 4.1 Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                                   │
│    User → LatentForge(api_key) → Initialize all components          │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. VIDEO LOADING                                                    │
│    VideoLoader.load(path) → [Frames], Metadata                      │
│    • OpenCV VideoCapture                                            │
│    • BGR → RGB conversion                                           │
│    • Extract: FPS, resolution, frame_count                          │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. FEATURE EXTRACTION                                               │
│    FeatureExtractor.extract(frames, metadata, style)                │
│    • Spatial: brightness, contrast, RGB, saturation, edges          │
│    • Temporal: frame_diff, motion_variance, smoothness              │
│    • Output: Latent Payload (JSON)                                  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. API INTERACTION                                                  │
│    GeminiClient.get_transformation(latent_payload)                  │
│    ├─ Compute SHA-256 hash of payload                               │
│    ├─ Check cache (if enabled)                                      │
│    │   ├─ Cache hit → Return cached transformation                  │
│    │   └─ Cache miss → Continue                                     │
│    ├─ Build structured prompt                                       │
│    ├─ Call Gemini API                                               │
│    │   ├─ Success → Parse JSON response                             │
│    │   └─ Failure → Style-based fallback                            │
│    └─ Cache response for future use                                 │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. CODE VALIDATION (if generated_code present)                      │
│    CodeValidator.validate(code)                                     │
│    ├─ Check forbidden patterns (regex)                              │
│    ├─ Parse AST                                                     │
│    ├─ Validate imports (NumPy, OpenCV only)                         │
│    ├─ Check function calls                                          │
│    ├─ Verify function definitions exist                             │
│    │                                                                 │
│    ├─ PASS → Continue to execution                                  │
│    └─ FAIL → Remove generated_code, use parameters                  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. TRANSFORMATION EXECUTION                                         │
│    SandboxExecutor.apply_transformations(frames, spec)              │
│    │                                                                 │
│    ├─ Mode A: Generated Code                                        │
│    │   ├─ Setup restricted globals (numpy, cv2, builtins)           │
│    │   ├─ exec(code, safe_globals, safe_locals)                     │
│    │   ├─ Find transform function                                   │
│    │   ├─ For each frame:                                           │
│    │   │   ├─ Execute transform(frame)                              │
│    │   │   ├─ Validate output (shape, type, values)                 │
│    │   │   └─ Clamp and convert to uint8                            │
│    │   └─ On error: Use original frame                              │
│    │                                                                 │
│    └─ Mode B: Parameter-Based                                       │
│        ├─ Apply lighting transform (gamma/sigmoid)                  │
│        ├─ Apply color transform (temp/sat/contrast)                 │
│        ├─ Apply sharpness transform (unsharp mask)                  │
│        └─ Clamp values to 0-255, convert to uint8                   │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 7. VIDEO RENDERING                                                  │
│    Renderer.render(transformed_frames, output_path, metadata)       │
│    ├─ Validate all frames (shape, type, values)                     │
│    ├─ RGB → BGR conversion                                          │
│    ├─ Create VideoWriter (codec: H.264, FPS: original)              │
│    ├─ Write frames sequentially                                     │
│    ├─ Release writer                                                │
│    └─ Log output metrics (size, duration, resolution)               │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 8. COMPLETION                                                       │
│    • Return output path                                             │
│    • All operations logged                                          │
│    • Session complete                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Error Flow Diagram

```
                    ┌─────────────────┐
                    │  Any Component  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Error Occurs   │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
       ┌────────────────┐       ┌────────────────┐
       │  Recoverable   │       │ Non-Recoverable│
       │     Error      │       │     Error      │
       └────────┬───────┘       └────────┬───────┘
                │                        │
                ▼                        ▼
       ┌────────────────┐       ┌────────────────┐
       │  Log Warning   │       │   Log Error    │
       │  Use Fallback  │       │  Preserve Data │
       │   Continue     │       │  Raise/Exit    │
       └────────────────┘       └────────────────┘
```

**Error Categories**:

| Component | Recoverable Errors | Non-Recoverable Errors |
|-----------|-------------------|------------------------|
| VideoLoader | Partial frame read | File not found, corrupt file |
| FeatureExtractor | Feature extraction failure | Invalid frame data |
| GeminiClient | API timeout, parse error | Invalid API key |
| CodeValidator | Validation failure | - |
| SandboxExecutor | Frame transform error | Out of memory |
| Renderer | Frame validation warning | Cannot create output file |

---

## 5. Data Models

### 5.1 Video Metadata
```json
{
  "path": "/path/to/video.mp4",
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "total_frames": 7200,
  "loaded_frames": 7200,
  "fourcc": 875967048
}
```

### 5.2 Latent Payload
```json
{
  "video_metadata": { /* Video Metadata */ },
  "spatial_latents": {
    "brightness": {"mean": 128.5, "std": 45.2, "min": 10.0, "max": 250.0},
    "contrast": {"mean": 52.3, "std": 12.1, "min": 25.0, "max": 95.0},
    "rgb_distribution": {
      "r_mean": 145.2, "g_mean": 135.8, "b_mean": 125.4,
      "r_std": 30.1, "g_std": 28.5, "b_std": 32.0
    },
    "saturation": {"mean": 98.5, "std": 25.3, "min": 40.0, "max": 200.0},
    "edge_density": {"mean": 0.15, "std": 0.05, "min": 0.05, "max": 0.35}
  },
  "temporal_latents": {
    "frame_difference_energy": {"mean": 12.5, "std": 3.2, "min": 5.0, "max": 25.0},
    "motion_variance": 8.45,
    "temporal_smoothness": 0.92
  },
  "style_request": "cinematic warm high contrast"
}
```

### 5.3 Transformation Specification
```json
{
  "lighting_transform": {
    "type": "gamma",
    "parameters": {"gamma": 1.2}
  },
  "color_transform": {
    "temperature_shift": 0.3,
    "saturation_multiplier": 1.15,
    "contrast_multiplier": 1.3
  },
  "sharpness_transform": {
    "strength": 1.1
  },
  "generated_code": "def transform_frame(frame):\n    import numpy as np\n    ..."
}
```

### 5.4 Log Entry Formats

**API Response Log**:
```json
{
  "timestamp": "2026-02-15T14:30:22.123456",
  "session_id": "20260215_143022",
  "type": "response",
  "request_hash": "a3f2e1b4c5d6...",
  "data": { /* Transformation Specification */ }
}
```

---

## 6. Security Architecture

### 6.1 Defense-in-Depth Layers

```
Layer 1: Input Validation
   ↓ [Validate file paths, check extensions, verify formats]
Layer 2: Code Pattern Filtering
   ↓ [Regex checks for forbidden keywords]
Layer 3: AST Analysis
   ↓ [Parse and validate code structure]
Layer 4: Import Whitelisting
   ↓ [Only NumPy and OpenCV allowed]
Layer 5: Restricted Execution
   ↓ [Limited globals, safe builtins only]
Layer 6: Output Validation
   ↓ [Validate frame dimensions, types, values]
Layer 7: Timeout Enforcement
   ↓ [Prevent infinite loops]
```

### 6.2 Threat Model

| Threat | Mitigation |
|--------|------------|
| Malicious generated code | AST validation, sandbox execution |
| File system access | Block file operations in AST |
| Network exfiltration | Block network imports/calls |
| Resource exhaustion | Timeouts, memory limits |
| Code injection | AST parsing prevents injection |
| API key exposure | Environment variables, no logging |

### 6.3 Security Boundaries

```
┌────────────────────────────────────────┐
│        Trusted Zone                    │
│  • User's file system                  │
│  • LatentForge components              │
│  • NumPy/OpenCV libraries              │
└────────────────┬───────────────────────┘
                 │
                 ▼ [Validation Boundary]
┌────────────────────────────────────────┐
│     Sandbox Execution Zone             │
│  • Generated code runs here            │
│  • Restricted globals                  │
│  • No file/network access              │
│  • Limited builtins                    │
└────────────────┬───────────────────────┘
                 │
                 ▼ [Network Boundary]
┌────────────────────────────────────────┐
│        External Services               │
│  • Google Gemini API                   │
└────────────────────────────────────────┘
```

---

## 7. Determinism Strategy

### 7.1 Caching Mechanism

```
Input: Latent Payload
   ↓
Normalize: JSON with sorted keys
   ↓
Hash: SHA-256(json_string)
   ↓
Cache Key: "a3f2e1b4c5d6..."
   ↓
Cache File: cache/a3f2e1b4c5d6....json
```

**Cache Hit Flow**:
```
Request → Hash → Cache Lookup → Hit → Return Cached Response
```

**Cache Miss Flow**:
```
Request → Hash → Cache Lookup → Miss → API Call → Store → Return
```

### 7.2 Reproducibility Guarantees

| Aspect | Guarantee |
|--------|-----------|
| Feature Extraction | Deterministic (same frames → same features) |
| API Response | Deterministic via caching |
| Code Execution | Deterministic (no random operations allowed) |
| Parameter Transforms | Deterministic (pure functions) |
| Frame Ordering | Preserved |
| Pixel Values | Exact (no floating-point issues) |

### 7.3 Logging for Audit Trail

Every transformation session produces:
- Complete input hash
- Full API request/response
- Validation results
- Transformation parameters
- Execution mode used
- Output metrics

**Reproducibility Process**:
1. Save original latent payload
2. Log all API responses
3. Store transformation spec
4. With these artifacts, transformation can be replayed exactly

---

## 8. Error Handling

### 8.1 Error Handling Philosophy
- **Fail gracefully**: Never crash, always preserve data
- **Degrade functionality**: Use fallbacks when possible
- **Log everything**: Detailed error information for debugging
- **User-friendly**: Clear error messages

### 8.2 Error Propagation

```
Component Error
   ↓
Catch and Log
   ↓
Attempt Recovery
   ├─ Success → Continue
   └─ Failure → Propagate to Parent
                   ↓
                Use Fallback
                   ↓
                Continue or Exit
```

### 8.3 Fallback Chain

```
Generated Code
   ├─ Validation Fails → Remove code, use parameters
   │
Parameters
   ├─ Execution Fails → Use default/identity transform
   │
Identity Transform
   └─ Return original frames
```

---

## 9. Performance Considerations

### 9.1 Optimization Strategies

| Stage | Optimization |
|-------|-------------|
| Video Loading | Frame sampling for large videos |
| Feature Extraction | Process sample frames (30 max) |
| API Calls | Caching to avoid redundant calls |
| Transformation | NumPy vectorization, avoid loops |
| Rendering | Batch write, efficient codec |

### 9.2 Memory Management

```
Frame Storage: List[ndarray] in RGB uint8 format
   Size: frames × width × height × 3 bytes

Example 1080p, 30fps, 10sec:
   300 frames × 1920 × 1080 × 3 = ~1.9 GB

Strategy:
   • Process in chunks for long videos
   • Release frames after processing
   • Optional frame sampling
```

### 9.3 Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Video Loading | O(n) | n = number of frames |
| Feature Extraction | O(n × w × h) | w,h = dimensions, sample frames |
| API Call | O(1) | Network dependent |
| Validation | O(m) | m = code length |
| Transformation | O(n × w × h) | Per-pixel operations |
| Rendering | O(n × w × h) | Encoding overhead |

---

## 10. Deployment Architecture

### 10.1 Deployment Models

**Model 1: Local Workstation**
```
User Machine
├── LatentForge (Python)
├── Input Videos (local storage)
├── Output Videos (local storage)
└── Logs/Cache (local storage)
     │
     └─→ Internet → Gemini API
```

**Model 2: Cloud Instance**
```
Cloud VM (e.g., AWS EC2, GCP Compute)
├── LatentForge
├── S3/GCS for video storage
├── API Gateway → Gemini API
└── Monitoring/Logging service
```

**Model 3: Containerized**
```
Docker Container
├── LatentForge + Dependencies
├── Volume Mounts:
│   ├── /input (videos)
│   ├── /output (results)
│   └── /logs (persistence)
└── Environment Variables (API key)
```

### 10.2 Configuration Management

**Environment Variables**:
```bash
GEMINI_API_KEY=...
LOG_LEVEL=INFO
LOG_DIR=./logs
CACHE_DIR=./cache
CACHE_API_RESPONSES=true
MAX_CODE_EXECUTION_TIME=30
MAX_MEMORY_MB=512
```

**Configuration File** (.env):
```
# Loaded via python-dotenv
# Overrides defaults
# Never commit to git
```

### 10.3 Scalability Considerations

**Current Limitations**:
- Single-threaded frame processing
- In-memory frame storage
- Sequential API calls

**Future Scaling Options**:
- Multi-process frame transformation
- Distributed caching (Redis)
- Batch video processing queue
- GPU acceleration for transforms
- Horizontal scaling with load balancer

---

## 11. Testing Strategy

### 11.1 Test Pyramid

```
        ┌─────────┐
        │   E2E   │  ← Full pipeline tests
        └─────────┘
      ┌─────────────┐
      │ Integration │  ← Component interaction
      └─────────────┘
   ┌─────────────────┐
   │  Unit Tests     │  ← Individual components
   └─────────────────┘
```

### 11.2 Test Coverage

| Component | Test Type | Coverage |
|-----------|-----------|----------|
| CodeValidator | Unit | Comprehensive (forbidden patterns, AST) |
| VideoLoader | Unit | File loading, sampling |
| FeatureExtractor | Unit | Feature calculation |
| SandboxExecutor | Integration | Execution modes, validation |
| Full Pipeline | E2E | Sample video transformation |

---

## 12. Future Architecture Enhancements

### 12.1 Plugin System
- Pluggable transformation modules
- Custom feature extractors
- Multiple AI backends (Claude, GPT-4V)

### 12.2 Streaming Processing
- Process frames as they load
- Reduce memory footprint
- Real-time preview

### 12.3 Distributed Processing
- Split video into chunks
- Process chunks in parallel
- Reassemble final output

### 12.4 Web Service Architecture
```
                    ┌──────────────┐
                    │  Web Client  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   API Server │
                    └──────┬───────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
         ┌──────────────┐      ┌──────────────┐
         │  Job Queue   │      │   Storage    │
         └──────┬───────┘      └──────────────┘
                │
                ▼
         ┌──────────────┐
         │   Workers    │
         │ (LatentForge)│
         └──────────────┘
```

---

## Summary

LatentForge implements a sophisticated **hybrid architecture** that balances:
- **Local control** with **cloud intelligence**
- **Security** with **flexibility**
- **Performance** with **determinism**

The modular design allows for easy extension, while multiple security layers ensure safe execution of AI-generated code. The caching and logging systems provide reproducibility and auditability, making LatentForge suitable for both research and production use cases.