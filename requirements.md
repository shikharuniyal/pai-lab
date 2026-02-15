# LatentForge - System Requirements Specification

## 1. Executive Summary

LatentForge is a deterministic cinematic video transformation engine that combines local pixel-level computation with cloud-based AI reasoning from Google Gemini API. The system extracts video features, sends them to Gemini for transformation strategy generation, validates any generated code, and applies transformations locally in a sandboxed environment.

## 2. Functional Requirements

### 2.1 Video Processing

#### FR-2.1.1: Video Ingestion
- **Description**: System must load video files using OpenCV
- **Input**: Video file path (MP4, AVI, MOV formats)
- **Output**: List of RGB frames and metadata
- **Acceptance Criteria**:
  - Support common video formats (H.264, MPEG-4)
  - Extract metadata: FPS, resolution, frame count, codec
  - Convert frames from BGR to RGB color space
  - Handle videos up to 4K resolution
  - Optional frame limiting for testing

#### FR-2.1.2: Frame Sampling
- **Description**: Extract representative sample frames from long videos
- **Input**: Video file, number of samples
- **Output**: Evenly distributed sample frames
- **Acceptance Criteria**:
  - Sample frames at regular intervals
  - Preserve temporal distribution
  - Default to 10-30 samples for feature extraction

### 2.2 Feature Extraction

#### FR-2.2.1: Spatial Feature Analysis
- **Description**: Extract per-frame visual characteristics
- **Features to Extract**:
  - **Brightness**: Mean luminance, standard deviation, min/max
  - **Contrast**: Luminance variance
  - **RGB Distribution**: Mean and std dev per channel
  - **Saturation**: HSV saturation statistics
  - **Edge Density**: Canny edge detection ratio
- **Output**: JSON structure with aggregated statistics
- **Performance**: Process 30-frame samples in <5 seconds

#### FR-2.2.2: Temporal Feature Analysis
- **Description**: Extract motion and temporal characteristics
- **Features to Extract**:
  - **Frame Difference Energy**: Mean absolute difference between consecutive frames
  - **Motion Variance**: Variance of frame differences
  - **Temporal Smoothness**: Inverse of mean frame difference
- **Output**: JSON structure with temporal metrics
- **Performance**: Process efficiently with frame sampling

#### FR-2.2.3: Latent Payload Construction
- **Description**: Build structured JSON for API consumption
- **Components**:
  - Video metadata
  - Spatial latent features
  - Temporal latent features
  - User style request
- **Format**: Deterministic JSON with sorted keys
- **Validation**: All numeric values must be finite floats

### 2.3 Gemini API Integration

#### FR-2.3.1: API Communication
- **Description**: Send latent payload to Gemini API and receive transformation specs
- **Input**: Structured JSON latent payload
- **Output**: Transformation specification (JSON)
- **Requirements**:
  - Support both legacy and modern Gemini API versions
  - Handle API errors gracefully
  - Implement retry logic for transient failures
  - Timeout after 30 seconds

#### FR-2.3.2: Prompt Engineering
- **Description**: Construct effective prompts for Gemini
- **Prompt Structure**:
  - Video feature summary
  - Style request from user
  - Expected JSON response format
  - Safety constraints
  - Code generation guidelines
- **Format**: Plain text with embedded JSON examples

#### FR-2.3.3: Response Parsing
- **Description**: Parse and validate API responses
- **Requirements**:
  - Extract JSON from markdown code blocks
  - Handle malformed responses
  - Validate required fields
  - Fallback to style-based transformations on failure

#### FR-2.3.4: Response Caching
- **Description**: Cache API responses for determinism
- **Requirements**:
  - Hash input payload with SHA-256
  - Store responses in JSON files
  - Check cache before API calls
  - Optional cache bypass flag
  - Cache invalidation support

### 2.4 Code Validation

#### FR-2.4.1: Syntax Validation
- **Description**: Validate Python syntax using AST parsing
- **Requirements**:
  - Parse code with ast.parse()
  - Detect syntax errors
  - Report line numbers and error details

#### FR-2.4.2: Security Validation
- **Description**: Ensure generated code is safe for execution
- **Forbidden Operations**:
  - File system access (open, file operations)
  - Network operations (socket, urllib, requests)
  - OS commands (os.system, subprocess)
  - Dynamic code execution (eval, exec, compile, __import__)
  - Imports outside NumPy and OpenCV
- **Allowed Modules**: numpy, np, cv2
- **Allowed Built-ins**: abs, all, any, bin, bool, bytes, chr, dict, enumerate, filter, float, int, len, list, map, max, min, range, round, set, sorted, str, sum, tuple, type, zip

#### FR-2.4.3: Pattern Detection
- **Description**: Use regex to detect forbidden patterns
- **Requirements**:
  - Scan for dangerous keywords
  - Detect obfuscated code patterns
  - Check for dunder method abuse

#### FR-2.4.4: Structural Validation
- **Description**: Ensure code contains required structures
- **Requirements**:
  - At least one function definition
  - Function accepts frame parameter
  - Function returns modified frame

### 2.5 Sandbox Execution

#### FR-2.5.1: Code Execution Mode
- **Description**: Execute validated Gemini-generated code
- **Requirements**:
  - Restricted global namespace (NumPy, OpenCV only)
  - Limited built-in functions
  - Execute in isolated environment
  - Catch and log all exceptions
  - Preserve original frames on error

#### FR-2.5.2: Parameter Execution Mode
- **Description**: Apply transformations using parameters only
- **Supported Transformations**:
  - **Lighting**: Gamma correction, sigmoid curves, linear adjustments
  - **Color**: Temperature shift, saturation adjustment, contrast multiplication
  - **Sharpness**: Unsharp mask with adjustable strength
- **Requirements**:
  - Deterministic outputs
  - Clamp values to valid ranges (0-255)
  - Type safety (uint8 output)

#### FR-2.5.3: Frame Validation
- **Description**: Validate all transformed frames
- **Requirements**:
  - Check shape consistency
  - Verify data type (numpy.ndarray)
  - Clamp pixel values (0-255)
  - Ensure uint8 dtype

#### FR-2.5.4: Timeout Enforcement
- **Description**: Prevent infinite loops
- **Requirements**:
  - Maximum 30 seconds per frame (configurable)
  - Graceful timeout handling
  - Fallback to original frame on timeout

### 2.6 Video Rendering

#### FR-2.6.1: Frame Assembly
- **Description**: Combine transformed frames into video
- **Requirements**:
  - Maintain original FPS
  - Preserve resolution
  - Convert RGB back to BGR for OpenCV
  - Support H.264 codec (MP4)

#### FR-2.6.2: Quality Control
- **Description**: Ensure output video quality
- **Requirements**:
  - No frame drops
  - Synchronized audio (if present)
  - Consistent frame dimensions
  - Proper codec parameters

#### FR-2.6.3: Output Validation
- **Description**: Verify rendered video
- **Requirements**:
  - File exists and is readable
  - Frame count matches input
  - Metadata extraction succeeds
  - File size is reasonable

### 2.7 Logging System

#### FR-2.7.1: Session Management
- **Description**: Track each transformation session
- **Requirements**:
  - Unique session ID (timestamp-based)
  - Separate log file per session
  - Log levels: DEBUG, INFO, WARNING, ERROR

#### FR-2.7.2: API Logging
- **Description**: Record all API interactions
- **Requirements**:
  - Log request payloads
  - Log response data
  - Log request hashes for caching
  - JSONL format for structured logs

#### FR-2.7.3: Transformation Logging
- **Description**: Record applied transformations
- **Requirements**:
  - Log transformation specifications
  - Log parameter values
  - Log execution mode (code vs parameters)

#### FR-2.7.4: Code Validation Logging
- **Description**: Record all code validation attempts
- **Requirements**:
  - Log validated code
  - Log validation results
  - Log error messages
  - Include timestamps

### 2.8 Command-Line Interface

#### FR-2.8.1: Basic Usage
- **Description**: Simple CLI for video transformation
- **Required Arguments**:
  - `--input`: Input video path
  - `--output`: Output video path
  - `--style`: Style description string
- **Optional Arguments**:
  - `--api-key`: Gemini API key
  - `--log-dir`: Log directory
  - `--cache-dir`: Cache directory
  - `--no-cache`: Disable caching
  - `--max-frames`: Frame limit for testing
  - `--log-level`: Logging verbosity

#### FR-2.8.2: Progress Reporting
- **Description**: User feedback during processing
- **Requirements**:
  - Stage indicators (loading, extracting, transforming, rendering)
  - Frame counters
  - Error messages to stderr
  - Success confirmation

#### FR-2.8.3: Help System
- **Description**: Built-in documentation
- **Requirements**:
  - `--help` flag
  - Usage examples
  - Argument descriptions
  - Version information

## 3. Non-Functional Requirements

### 3.1 Performance

#### NFR-3.1.1: Processing Speed
- Feature extraction: <5 seconds for 30 frames
- API response: <10 seconds (network dependent)
- Frame transformation: >10 FPS processing rate
- Full pipeline: <2x video duration for typical videos

#### NFR-3.1.2: Memory Usage
- Maximum 2GB RAM for 1080p videos
- Frame batch processing to limit memory
- Efficient NumPy operations

#### NFR-3.1.3: Scalability
- Support videos up to 10,000 frames
- Graceful degradation for large files
- Optional frame sampling for analysis

### 3.2 Security

#### NFR-3.2.1: Code Isolation
- No access to file system
- No network access (except API client)
- No subprocess execution
- Restricted Python built-ins

#### NFR-3.2.2: Input Validation
- Validate all file paths
- Sanitize user inputs
- Check file extensions
- Verify video format compatibility

#### NFR-3.2.3: API Key Security
- Store in environment variables
- Never log full API keys
- Support .env files
- Clear error messages without exposing keys

### 3.3 Reliability

#### NFR-3.3.1: Error Handling
- Graceful failures at each stage
- Preserve original content on errors
- Detailed error messages
- Stack trace logging

#### NFR-3.3.2: Determinism
- Same input → same output (with cache)
- Reproducible transformations
- Seed-free algorithms
- Consistent hashing

#### NFR-3.3.3: Fault Tolerance
- API failures → style-based fallback
- Code validation failures → parameter mode
- Frame errors → preserve original
- Partial success handling

### 3.4 Usability

#### NFR-3.4.1: API Design
- Simple high-level interface
- Component-level access for advanced users
- Sensible defaults
- Clear error messages

#### NFR-3.4.2: Documentation
- Comprehensive README
- Architecture documentation
- API reference
- Usage examples

#### NFR-3.4.3: Installation
- pip installable
- Minimal dependencies
- Clear setup instructions
- Cross-platform support (Linux, macOS, Windows)

### 3.5 Maintainability

#### NFR-3.5.1: Code Quality
- Type hints where applicable
- Docstrings for all public methods
- Modular architecture
- Single responsibility principle

#### NFR-3.5.2: Testing
- Unit tests for validators
- Integration tests for pipeline
- Example scripts
- Test video generation

#### NFR-3.5.3: Logging
- Comprehensive logging at all stages
- Structured log formats
- Configurable log levels
- Log rotation support

## 4. System Constraints

### 4.1 Technical Constraints

- **Python Version**: 3.8+
- **OpenCV Version**: 4.8.0+
- **NumPy Version**: 1.24.0+
- **API**: Google Gemini (legacy or modern)

### 4.2 Environmental Constraints

- **Operating Systems**: Linux, macOS, Windows
- **Storage**: Minimum 1GB free space for logs/cache
- **Network**: Internet connection for API access
- **GPU**: Optional, not required

### 4.3 Regulatory Constraints

- **License**: MIT License
- **API Terms**: Comply with Google Gemini API terms of service
- **Content**: No automated processing of copyrighted material

## 5. Quality Attributes

### 5.1 Security
- **Priority**: CRITICAL
- Multi-layer validation prevents malicious code execution

### 5.2 Determinism
- **Priority**: HIGH
- Same inputs produce identical outputs via caching

### 5.3 Performance
- **Priority**: MEDIUM
- Real-time processing not required, but reasonable speed expected

### 5.4 Usability
- **Priority**: HIGH
- Simple CLI and Python API for common use cases

### 5.5 Maintainability
- **Priority**: HIGH
- Clean code, good documentation, modular design

## 6. Acceptance Criteria

### 6.1 Minimum Viable Product (MVP)
- ✅ Load video files
- ✅ Extract spatial and temporal features
- ✅ Interface with Gemini API (with fallback)
- ✅ Validate generated code
- ✅ Execute transformations safely
- ✅ Render output video
- ✅ Complete logging system
- ✅ CLI interface

### 6.2 Testing Requirements
- Unit tests for CodeValidator pass
- Sample video transformation completes successfully
- Cache mechanism works correctly
- Fallback transformations apply correctly
- Error handling prevents crashes

### 6.3 Documentation Requirements
- README with installation and usage
- Architecture documentation
- API examples
- Quick start guide

## 7. Future Enhancements (Out of Scope for MVP)

### 7.1 Advanced Features
- Optical flow analysis
- GPU acceleration (CUDA)
- Parallel frame processing
- Real-time preview
- Interactive parameter adjustment
- Neural style transfer integration

### 7.2 API Expansion
- Support for Claude API
- Support for GPT-4 Vision
- Multi-model ensemble
- Custom model fine-tuning

### 7.3 Optimization
- WebAssembly compilation
- Batch video processing
- Cloud deployment
- Distributed processing

### 7.4 User Interface
- Web interface
- Desktop GUI (Qt/Electron)
- Video player integration
- Mobile app

## 8. Dependencies

### 8.1 External Libraries
- **opencv-python**: Video I/O and image processing
- **numpy**: Numerical operations
- **google-generativeai**: Gemini API client
- **python-dotenv**: Environment configuration
- **pillow**: Image utilities
- **tqdm**: Progress bars

### 8.2 Development Dependencies
- **pytest**: Unit testing
- **black**: Code formatting
- **mypy**: Type checking
- **sphinx**: Documentation generation

## 9. Glossary

- **Latent Payload**: Structured JSON containing extracted video features
- **Spatial Features**: Per-frame visual characteristics
- **Temporal Features**: Cross-frame motion characteristics
- **Transformation Specification**: JSON defining how to modify frames
- **Sandbox Execution**: Isolated code execution environment
- **Determinism**: Property where same inputs produce identical outputs
- **AST**: Abstract Syntax Tree - parsed representation of code
- **Cache Hash**: SHA-256 hash of input payload for cache lookup