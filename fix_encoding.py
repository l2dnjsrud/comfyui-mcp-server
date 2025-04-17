import os

def fix_file_encoding(file_path):
    try:
        # 다양한 인코딩으로 시도
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # UTF-8으로 다시 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print(f"Fixed encoding for {file_path}")
                return True
            except UnicodeDecodeError:
                continue
        
        print(f"Failed to fix encoding for {file_path}")
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

# 워크플로우 파일 인코딩 수정
workflows_dir = "workflows"
if os.path.exists(workflows_dir):
    for filename in os.listdir(workflows_dir):
        if filename.endswith(".json"):
            fix_file_encoding(os.path.join(workflows_dir, filename))

# 구성 파일 인코딩 수정
config_files = ["config.json", "auth_config.json"]
for config_file in config_files:
    if os.path.exists(config_file):
        fix_file_encoding(config_file)

print("Encoding fix completed")