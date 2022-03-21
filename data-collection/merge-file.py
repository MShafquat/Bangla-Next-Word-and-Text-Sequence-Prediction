from pathlib import Path
import sys

# merges two files with same name from two different directories and changes in dir1
def merge(dir1, dir2):
    dir2_files = [file for file in Path(dir2).glob("**/*.txt")]
    for path in dir2_files:
        author_name = path.parent.name
        book_name = path.name
        dir1_file = dir1 / author_name / book_name
        if dir1_file.exists():
            with open(dir1_file, 'a') as f:
                with open(path) as f2:
                    f.write('\n' + f2.read() + '\n')
        else:
            dir1_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dir1_file, 'w') as f:
                with open(path) as f2:
                    f.write(f2.read())

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python merge-file.py <dir1> <dir2>')
        sys.exit(1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    data1_dir = Path(dir1).resolve()
    data2_dir = Path(dir2).resolve()
    if data1_dir.exists() and data2_dir.exists():
        merge(data1_dir, data2_dir)
    else:
        print("Directories doesn't exist")
        sys.exit(1)

