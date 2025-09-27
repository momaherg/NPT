# Development Guide for Interactive Knowledge Transfer Tool

This guide provides detailed information for developers working on or extending the Interactive Knowledge Transfer tool.

## 🎯 Quick Start for Developers

### Running the Tool
```bash
# Using the new modular implementation
python scripts/interactive_knowledge_transfer_tool.py \
    --checkpoint experiments/checkpoint \
    --model_name "meta-llama/Llama-3.2-1B" \
    --layers "11,12,13"

# Or using the original monolithic script (preserved for compatibility)
python scripts/interactive_knowledge_transfer.py [same args]
```

### Testing Your Changes
```bash
# Import test
python -c "from interactive_knowledge_transfer import InteractiveKnowledgeTransfer"

# Test specific module
python -c "from interactive_knowledge_transfer.core import ModulationOperations"

# Run demo
python scripts/demo_extract_seq.py
```

## 📋 Common Development Tasks

### 1. Adding a New Command

**Step-by-step process:**

```python
# 1. Add command handler (e.g., in commands/modulation_mgmt.py)
def cmd_your_command(self, args: List[str]):
    """Command description for help text."""
    # Validate arguments
    if len(args) < 2:
        self.error("Usage: your-command <arg1> <arg2>")
        return

    # Parse arguments
    arg1 = args[0]
    arg2 = args[1]

    # Execute operation
    try:
        result = self.your_operation(arg1, arg2)
        self.success(f"Operation completed: {result}")
    except Exception as e:
        self.error(f"Failed: {e}")

# 2. Wire in CLI (cli.py, execute_command method)
elif cmd == 'your-command':
    self.modulation_commands.cmd_your_command(parts[1:])

# 3. Update help text (cli.py, show_help method)
# Add under appropriate category
```

### 2. Adding Support for New Modulation Type

```python
# In core/modulation_ops.py
def extract_modulation(self, prompt: str, position: Optional[int],
                       active_layers: set) -> Dict[int, ModulationData]:
    # ... existing code ...

    # Add detection for your new type
    if isinstance(output[0], YourNewType):
        # Handle new modulation type
        mod_data.modulation_type = "your_type"
        mod_data.your_tensor = extract_your_tensor(output)

    # Update magnitude calculation
    mod_data.magnitude = calculate_magnitude(mod_data)
```

### 3. Extending Display Capabilities

```python
# In utils/display.py
def display_your_visualization(self, data):
    """Display your custom visualization."""
    print(f"{Colors.BOLD}Your Visualization:{Colors.END}")
    # Format and display data
    for item in data:
        print(f"  {format_item(item)}")
```

### 4. Adding New Injection Mode

```python
# In core/modulation_ops.py, create_injection_hook method
elif mode == 'your_mode':
    # Your injection logic
    v_a_new[:, position] = your_transformation(source_mod.v_a, v_a, your_params)
    v_b_new[:, position] = your_transformation(source_mod.v_b, v_b, your_params)
```

## 🔍 Code Organization Best Practices

### Module Responsibilities

**Core Modules:**
- `data_types.py`: ONLY data structures, no logic
- `modulation_ops.py`: ONLY extraction/injection operations
- `arithmetic_ops.py`: ONLY mathematical operations

**Command Modules:**
- Each command file handles related commands
- Inherit from `BaseCommandHandler` for utilities
- Use consistent error/success messaging

**Utility Modules:**
- Single responsibility per utility module
- No cross-dependencies between utils
- Stateless functions where possible

### Naming Conventions

```python
# Commands: cmd_<command_name>
def cmd_extract_sequence(self, args):

# Operations: <verb>_<noun>
def extract_modulation(self, prompt):

# Utilities: <action>_<object>
def display_predictions(self, probs):

# Private methods: _<method_name>
def _create_hook(self, layer_idx):
```

## 🐛 Debugging Guide

### Common Issues and Solutions

#### 1. Memory Issues with Long Sequences
```python
# Problem: OOM when extracting from long sequences
# Solution: Process in chunks
for chunk in chunks(sequence, chunk_size=10):
    process_chunk(chunk)
    torch.cuda.empty_cache()  # Clear between chunks
```

#### 2. Hook Not Capturing Output
```python
# Check 1: Ensure NPT mode is enabled
self.model.model.layers[idx].set_npt_mode(True)

# Check 2: Verify hook registration
print(f"Hooks registered: {len(handles)}")

# Check 3: Add debug print in hook
def hook(module, input, output):
    print(f"Hook called for layer {layer_idx}")
    # ... rest of hook
```

#### 3. Modulation Compatibility Issues
```python
# Always check before operations
if not self.mod_arithmetic.check_compatibility(mod1, mod2):
    self.error("Incompatible modulations")
    return

# Debug compatibility
print(f"Mod1 type: {mod1.modulation_type}, ranks: {mod1.num_ranks}")
print(f"Mod2 type: {mod2.modulation_type}, ranks: {mod2.num_ranks}")
```

### Logging and Debugging

Add debug logging:
```python
import logging
logger = logging.getLogger(__name__)

# In your method
logger.debug(f"Extracting from position {position}")
logger.debug(f"Tensor shape: {tensor.shape}")
```

Enable debug mode:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🧪 Testing Guidelines

### Unit Test Structure
```python
# test_modulation_ops.py
import unittest
from interactive_knowledge_transfer.core import ModulationOperations

class TestModulationOps(unittest.TestCase):
    def setUp(self):
        self.mod_ops = ModulationOperations(mock_model, mock_tokenizer, device)

    def test_extract_sequence_modulations(self):
        result = self.mod_ops.extract_sequence_modulations(
            "test prompt", {0, 1}
        )
        self.assertIsInstance(result, dict)
        self.assertIn(0, result)
```

### Integration Test Example
```python
def test_full_workflow():
    """Test extract -> arithmetic -> inject workflow."""
    # Extract
    mod1 = extract("prompt1")
    mod2 = extract("prompt2")

    # Arithmetic
    result = subtract(mod1, mod2)

    # Inject
    output = inject(result, "new prompt")

    # Verify
    assert output is not None
```

## 📊 Performance Profiling

### Memory Profiling
```python
import tracemalloc
tracemalloc.start()

# Your operation
result = extract_sequence_modulations(prompt)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 10**6:.1f} MB")
print(f"Peak memory: {peak / 10**6:.1f} MB")
tracemalloc.stop()
```

### Time Profiling
```python
import time

start = time.time()
result = your_operation()
print(f"Operation took: {time.time() - start:.2f} seconds")
```

### PyTorch Profiling
```python
import torch.profiler as profiler

with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
    result = your_operation()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 🔐 Security Considerations

### Input Validation
```python
# Always validate user input
def validate_prompt(prompt: str) -> bool:
    if len(prompt) > MAX_LENGTH:
        return False
    if contains_malicious_patterns(prompt):
        return False
    return True
```

### Safe File Operations
```python
# Use pathlib for path operations
from pathlib import Path

def save_session(filename: str):
    # Sanitize filename
    safe_name = Path(filename).name  # Remove path components
    path = SESSION_DIR / safe_name

    # Check path is within allowed directory
    if not path.resolve().is_relative_to(SESSION_DIR.resolve()):
        raise ValueError("Invalid path")
```

## 📦 Dependencies and Compatibility

### Required Packages
```python
# Core dependencies
torch >= 2.0.0
transformers >= 4.30.0
numpy >= 1.20.0

# NPT specific
src.npt  # Custom NPT implementation
```

### Python Version
- Minimum: Python 3.8 (for typing features)
- Recommended: Python 3.9+

### GPU Requirements
- CUDA 11.0+ for GPU support
- Works on CPU but slower
- Memory: 8GB+ for 1B models, 24GB+ for 8B models

## 🔄 Git Workflow

### Branch Naming
- Features: `feature/command-name`
- Bugfixes: `fix/issue-description`
- Refactoring: `refactor/module-name`

### Commit Messages
```bash
# Good
git commit -m "Add extract-seq command for full sequence modulation extraction"

# Bad
git commit -m "Update files"
```

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Refactoring
- [ ] Documentation

## Testing
- [ ] Unit tests pass
- [ ] Manual testing completed
- [ ] No memory leaks

## Checklist
- [ ] Code follows style guidelines
- [ ] Added/updated documentation
- [ ] No hardcoded values
```

## 🎨 Code Style Guide

### Formatting
- Use Black formatter with line length 100
- 4 spaces for indentation
- Docstrings for all public methods

### Type Hints
```python
from typing import Dict, List, Optional, Tuple

def extract_modulation(
    self,
    prompt: str,
    position: Optional[int] = None
) -> Dict[int, ModulationData]:
    """Extract modulation from prompt."""
```

### Error Handling
```python
# Prefer specific exceptions
try:
    result = operation()
except ValueError as e:
    self.error(f"Invalid value: {e}")
except KeyError as e:
    self.error(f"Missing key: {e}")
```

## 📈 Future Enhancements

### Planned Features
1. **Modulation visualization**: Plot modulation magnitudes over time
2. **Batch operations**: Process multiple prompts simultaneously
3. **Modulation database**: Persistent storage with search
4. **API mode**: RESTful API for programmatic access
5. **Configuration files**: YAML-based configuration

### Architecture Improvements
1. **Plugin system**: Dynamic command loading
2. **Async operations**: Non-blocking extraction/injection
3. **Caching layer**: Cache frequently used modulations
4. **Streaming support**: Handle streaming text input

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

For questions or discussions, open an issue on GitHub.