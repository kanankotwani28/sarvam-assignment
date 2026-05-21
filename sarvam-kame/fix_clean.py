# Read the file
with open('server/sarvam_client.py', 'r') as f:
    lines = f.readlines()

# Find the _clean_llm_response function and replace it
new_lines = []
in_function = False
for line in lines:
    if 'def _clean_llm_response' in line:
        in_function = True
        new_lines.append('def _clean_llm_response(text: str) -> str:\n')
        new_lines.append('    """Remove thinking/reasoning tags from LLM response."""\n')
        new_lines.append("    # Remove <think>...
</think>
 or <think> without closing tag\n")
        new_lines.append("    text = re.sub(r'<think>.*?(?:
</think>

|.*$)', '', text, flags=re.DOTALL).strip()\n")
        new_lines.append("    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()\n")
        new_lines.append("    if not text.strip():\n")
        new_lines.append("        text = 'I am ready to help.'\n")
        new_lines.append('    return text\n')
        new_lines.append('\n')
        continue
    if in_function:
        if line.strip() == '' or (not line.startswith(' ') and not line.startswith('\t')):
            in_function = False
            new_lines.append(line)
        continue
    new_lines.append(line)

with open('server/sarvam_client.py', 'w') as f:
    f.writelines(new_lines)

print('Fixed!')
