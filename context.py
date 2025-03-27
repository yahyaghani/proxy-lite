import os
import tiktoken

EXCLUDED_FOLDERS = {'.next', 'node_modules', '.github', '.git','.expo','public'}

def read_project_files(root_dir, output_file, include_py_only=True):
    with open(output_file, 'w') as outfile:
        for subdir, _, files in os.walk(root_dir):
            if any(excluded in subdir.split(os.sep) for excluded in EXCLUDED_FOLDERS):
                continue
            for file in files:
                file_path = os.path.join(subdir, file)
                # if file_path == output_file or file.endswith('.pyc') or '__pycache__' in subdir:
                #     continue
                # if include_py_only and not file.endswith(('.js', '.tsx', '.jsx',)):
                if include_py_only and not file.endswith(('.py')):
                    
                    continue
                relative_path = os.path.relpath(file_path, root_dir)
                # outfile.write(f"## {relative_path} ##\n")
                outfile.write("\n## start of file:{} ##\n".format(relative_path))

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    outfile.write(infile.read())
                # outfile.write(f"\n## end of {relative_path} ##\n\n")
                outfile.write("\n## end of file:{} ## \n".format(relative_path))

def count_tokens(file_path):
    enc = tiktoken.get_encoding("gpt2")  # You can choose the encoding according to your model
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
        text = infile.read()
    tokens = enc.encode(text)
    return len(tokens)

if __name__ == "__main__":
    root_directory = './'  # Set this to the root of your Django project
    output_file_path = 'context.txt'
    include_py_only = True  # Set this to False if you want to include all files
    read_project_files(root_directory, output_file_path, include_py_only)
    print(f"Context file created at {output_file_path}")
    total_tokens = count_tokens(output_file_path)
    print(f"Total number of tokens in the context file: {total_tokens}")

