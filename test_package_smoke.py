"""
Quick smoke test to verify hmlr package works after cleanup.
"""
import sys
sys.path.insert(0, 'hmlr')

print("=" * 60)
print("HMLR Package Smoke Test")
print("=" * 60)

# Test 1: Import core modules
print("\nTest 1: Importing core modules...")
try:
    from core.component_factory import ComponentFactory
    print("  [PASS] ComponentFactory imported")
except Exception as e:
    print(f"  [FAIL] ComponentFactory: {e}")
    sys.exit(1)

try:
    from core.conversation_engine import ConversationEngine  
    print("  [PASS] ConversationEngine imported")
except Exception as e:
    print(f"  [FAIL] ConversationEngine: {e}")
    sys.exit(1)

# Test 2: Import memory modules
print("\nTest 2: Importing memory modules...")
try:
    from memory import Storage
    print("  [PASS] Storage imported")
except Exception as e:
    print(f"  [FAIL] Storage: {e}")
    sys.exit(1)

try:
    from memory.chunking import ChunkEngine
    print("  [PASS] ChunkEngine imported")
except Exception as e:
    print(f"  [FAIL] ChunkEngine: {e}")
    sys.exit(1)

try:
    from memory.fact_scrubber import FactScrubber
    print("  [PASS] FactScrubber imported")
except Exception as e:
    print(f"  [FAIL] FactScrubber: {e}")
    sys.exit(1)

# Test 3: Check for deleted modules (should fail)
print("\nTest 3: Verifying deleted modules are gone...")
deleted_modules = [
    ("memory.adaptive", "AdaptiveCompressor"),
    ("memory.topics", "TopicExtractor"),
    ("memory.usage", "UsageTracker"),
    ("memory.debug_logger", "MemoryDebugLogger"),
]

for module_name, class_name in deleted_modules:
    try:
        exec(f"from {module_name} import {class_name}")
        print(f"  [FAIL] {module_name}.{class_name} still exists (should be deleted!)")
        sys.exit(1)
    except (ImportError, ModuleNotFoundError):
        print(f"  [PASS] {module_name}.{class_name} correctly deleted")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - Package is functional!")
print("=" * 60)
