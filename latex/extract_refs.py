#!/usr/bin/env python3
"""
Extract URLs from \href{...} expressions in LaTeX files.
Saves unique URLs to refs.txt, sorted alphabetically.
"""

import re
from pathlib import Path

# Files to search
files = ['intro.tex', 'discussion.tex']

# Pattern to match \href{URL}{text}
# Captures the URL inside the first set of braces
href_pattern = re.compile(r'\\href\{([^}]+)\}')

urls = set()

for filename in files:
    filepath = Path(__file__).parent / filename
    if filepath.exists():
        content = filepath.read_text()
        matches = href_pattern.findall(content)
        urls.update(matches)
        print(f"Found {len(matches)} hrefs in {filename}")
    else:
        print(f"Warning: {filename} not found")

# Sort and write to refs.txt
sorted_urls = sorted(urls)

output_path = Path(__file__).parent / 'refs.txt'
output_path.write_text('\n'.join(sorted_urls) + '\n')

print(f"\nWrote {len(sorted_urls)} unique URLs to refs.txt")
