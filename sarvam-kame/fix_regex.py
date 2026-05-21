import re

with open('server/sarvam_client.py', 'r') as f:
    lines = f.readlines()

# Find and replace the function
new_lines = []
skip = False
for line in lines:
    if 'def _clean_llm_response' in line:
        skip = True
        new_lines.append('def _clean_llm_response(text: str) -> str:\n')
        new_lines.append('    """Remove thinking/reasoning tags from LLM response."""\n')
        new_lines.append("    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()\n")
        new_lines.append("    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()\n")
        new_lines.append("    if not text.strip():\n")
        new_lines.append("        text = 'Hello! How can I help you today?'\n")
        new_lines.append('    return text\n')
        new_lines.append('\n')
        continue
    if skip:
        if line.strip() == '' or line.startswith('def ') or line.startswith('async def '):
            skip = False
            new_lines.append(line)
        continue
    new_lines.append(line)

with open('server/sarvam_client.py', 'w') as f:
    f.writelines(new_lines)

print('Fixed!')
