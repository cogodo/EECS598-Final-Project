#!/usr/bin/env python3
"""
Test script to verify the GRPO training setup with TinyLlama.
Tests each component individually before running full training.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

print("=" * 60)
print("Testing GRPO Training Setup with TinyLlama")
print("=" * 60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("   ✓ Transformers imported")
except ImportError as e:
    print(f"   ✗ Failed to import transformers: {e}")
    sys.exit(1)

try:
    from reward_model import AceRewardModel
    print("   ✓ AceRewardModel imported")
except ImportError as e:
    print(f"   ✗ Failed to import AceRewardModel: {e}")
    sys.exit(1)

try:
    from math_verifier import MathVerifier
    print("   ✓ MathVerifier imported")
except ImportError as e:
    print(f"   ✗ Failed to import MathVerifier: {e}")
    sys.exit(1)

try:
    from utils import combine_hybrid_score
    print("   ✓ combine_hybrid_score imported")
except ImportError as e:
    print(f"   ✗ Failed to import combine_hybrid_score: {e}")
    sys.exit(1)

try:
    from loss import GRPOLoss
    from replay_buffer import ReplayBuffer
    print("   ✓ GRPO components imported")
except ImportError as e:
    print(f"   ✗ Failed to import GRPO components: {e}")
    sys.exit(1)

# Test 2: Check data file
print("\n2. Testing data file...")
data_path = Path(__file__).parent / "data" / "dummy_math_tasks.jsonl"
if data_path.exists():
    print(f"   ✓ Data file exists: {data_path}")
    with open(data_path) as f:
        lines = f.readlines()
    print(f"   ✓ Contains {len(lines)} problems")
else:
    print(f"   ✗ Data file not found: {data_path}")
    sys.exit(1)

# Test 3: Test Math Verifier
print("\n3. Testing Math Verifier...")
try:
    verifier = MathVerifier(method="flexible")
    result = verifier.verify(
        prompt="What is 5 + 3?",
        response="The answer is <answer>8</answer>",
        ground_truth="8"
    )
    print(f"   ✓ Verifier working: correct={result['correct']}, reward={result['reward']}")
except Exception as e:
    print(f"   ✗ Verifier failed: {e}")
    sys.exit(1)

# Test 4: Test Hybrid Score Function
print("\n4. Testing Hybrid Score Function...")
try:
    # Test correct answer
    score_correct = combine_hybrid_score(
        verl_score=1.0,
        rm_score=0.8,
        min_rm=0.0,
        max_rm=1.0,
        eps=0.01,
        alpha=0.5,
        beta=0.5
    )
    print(f"   ✓ Hybrid score (correct): {score_correct:.4f}")
    
    # Test incorrect answer
    score_incorrect = combine_hybrid_score(
        verl_score=0.0,
        rm_score=0.3,
        min_rm=0.0,
        max_rm=1.0,
        eps=0.01,
        alpha=0.5,
        beta=0.5
    )
    print(f"   ✓ Hybrid score (incorrect): {score_incorrect:.4f}")
except Exception as e:
    print(f"   ✗ Hybrid score failed: {e}")
    sys.exit(1)

# Test 5: Test TinyLlama Model Loading (requires GPU/CPU and downloads)
print("\n5. Testing TinyLlama Model Loading...")
print("   Note: This will download the model if not cached (~2GB)")
response = input("   Continue with model loading? (y/n): ")
if response.lower() == 'y':
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"   Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        print("   ✓ Tokenizer loaded")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("   ✓ Model loaded")
        
        # Test generation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✓ Generation test successful")
        print(f"   Response preview: {response[:100]}...")
        
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        sys.exit(1)
else:
    print("   ⊘ Skipped model loading test")

# Test 6: Test Reward Model (requires significant memory)
print("\n6. Testing Reward Model...")
print("   Note: AceRewardModel requires ~14GB GPU memory")
response = input("   Continue with reward model loading? (y/n): ")
if response.lower() == 'y':
    try:
        reward_model = AceRewardModel()
        print("   ✓ Reward model loaded")
        
        # Test reward computation
        test_prompt = "What is 5 + 3?"
        test_response = "Let me solve this step by step. 5 + 3 = 8. The answer is 8."
        outputs = reward_model.compute_reward(test_prompt, test_response)
        score = outputs.logits[0][0].item()
        print(f"   ✓ Reward computation successful: score={score:.4f}")
        
    except Exception as e:
        print(f"   ✗ Reward model failed: {e}")
        print("   Note: This is expected if you don't have enough GPU memory")
else:
    print("   ⊘ Skipped reward model test")

print("\n" + "=" * 60)
print("All critical tests passed! ✓")
print("You can now run: python train.py")
print("=" * 60)
