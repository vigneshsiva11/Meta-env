#!/usr/bin/env python3
import os
import re

# Files to clean
files = [
    "d:\\meta\\api_contract_env\\server\\environment.py",
    "d:\\meta\\api_contract_env\\models.py",
    "d:\\meta\\api_contract_env\\server\\app.py",
]

for filepath in files:
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} - not found")
        continue
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Unicode dashes and arrows
        replacements = [
            ('─', '-'),  # BOX DRAWINGS LIGHT HORIZONTAL
            ('→', '->'),  # RIGHTWARDS ARROW
            ('≥', '>='),  # GREATER-THAN OR EQUAL TO
            ('──────────────────────────────────────────────────────────────────────────────', '============================================================================'),
        ]
        
        original = content
        for old, new in replacements:
            content = content.replace(old, new)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Cleaned {filepath}")
        else:
            print(f"- No changes needed for {filepath}")
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")

print("\nDone!")
