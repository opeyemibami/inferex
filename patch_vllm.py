path = '/usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py'
with open(path, 'r') as f:
    content = f.read()
old = 'def in_wsl() -> bool:\n    # Reference: https://github.com/microsoft/WSL/issues/4071\n    return "microsoft" in " ".join(platform.uname()).lower()'
new = 'def in_wsl() -> bool:\n    # Patched for WSL2/k3s: disable WSL detection to enable UVA\n    return False'
assert old in content, f"Pattern not found in file"
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('Patch applied successfully')
