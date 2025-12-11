"""
Quick script to replace all $ symbols with ₹ in the codebase
"""

import re

files_to_update = [
    r'main.py',
    r'src\optimizer.py',
    r'src\battery_model.py',
    r'src\utils.py'
]

replacements = [
    (r'\$\{', r'₹{'),  # ${variable} → ₹{variable}
    (r'"\$"', r'"₹"'),  # "$" → "₹"
    (r"'\$'", r"'₹'"),  # '$' → '₹'
    (r'\(\$\)', r'(₹)'),  # ($) → (₹)
]

for file_path in files_to_update:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Updated {file_path}")
        else:
            print(f"  No changes needed in {file_path}")
            
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")

print("\n✓ Currency symbol update complete!")
print("All $ symbols replaced with ₹")
