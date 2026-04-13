#!/usr/bin/env python3
import os
import re

# Remove emoji and other problematic Unicode from all text files
files = [
    "d:\\meta\\api_contract_env\\README.md",
]

for filepath in files:
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} - not found")
        continue
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Remove emojis
        content = re.sub(r'[🔌🌐📥🎯✅📊🏁🚀📋⚙🔧🎮🤖✨👷🏆📈🔬]', '', content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Removed emojis from {filepath}")
        else:
            print(f"- No emojis found in {filepath}")
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")

print("\nDone!")
