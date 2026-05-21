with open('server/sarvam_client.py', 'r') as f:
    content = f.read()

start_marker = 'def _clean_llm_response'
end_marker = '\n\nasync def get_oracle'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_func = """def _clean_llm_response(text: str) -> str:
    \"\"\"Remove thinking/reasoning tags from LLM response.\"\"\"
    # Case 1: Complete <think>... block - remove it
    text = re.sub(r'<think>.*?


', '', text, flags=re.DOTALL).strip()
    
    # Case 2: Incomplete thinking (no closing tag) - remove up to first actual sentence
    # Look for the pattern where thinking ends (usually a period or newline followed by actual response)
    match = re.search(r'<think>.*?([.!?]\s+[A-Z]|$)', text, re.DOTALL)
    if match:
        text = text[match.end():].strip()
    
    # Case 3: Any remaining reasoning tags
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()
    
    if not text.strip():
        text = 'I am ready to help.'
    return text
"""
    content = content[:start_idx] + new_func + content[end_idx:]
    
    with open('server/sarvam_client.py', 'w') as f:
        f.write(content)
    print('Fixed!')
else:
    print(f'Not found: start={start_idx}, end={end_idx}')