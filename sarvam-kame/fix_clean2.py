import re

with open('server/sarvam_client.py', 'r') as f:
    content = f.read()

start_marker = 'def _clean_llm_response'
end_marker = '\n\nasync def get_oracle'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_func = """def _clean_llm_response(text: str) -> str:
    \"\"\"Remove thinking/reasoning tags from LLM response.\"\"\"
    # Remove anything between <think> tags, or from <think> to end if unclosed
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL).strip()
    # Remove <reasoning> tags
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
