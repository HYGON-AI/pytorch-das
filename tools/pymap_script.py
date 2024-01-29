import os
import argparse

def replace_in_file(file_path, replacements):
    with open(file_path, 'r') as file:
        content = file.read()

    for key, value in replacements.items():
        content = content.replace(key, value)

    with open(file_path, 'w') as file:
        file.write(content)

def scan_and_replace_files(directory, replacements):
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                replace_in_file(file_path, replacements)
                print(f"Replaced content in file: {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Python script to replace content in .py files.')
    parser.add_argument('directory', type=str, help='Path to the directory containing .py files')
    args = parser.parse_args()

    # 指定键值对替换内容
    replacements = {
        'torch.version.cuda': 'torch.version.dtk',
        'CUDA_HOME': 'ROCM_HOME'
    }

    # 执行扫描和替换
    scan_and_replace_files(args.directory, replacements)

if __name__ == '__main__':
    main()
