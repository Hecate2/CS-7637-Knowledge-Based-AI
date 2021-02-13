import os, re
from shutil import copyfile

pattern = re.compile('''.* Problem \w-\d+\.PNG''')

initial_root = './Problems'
target_path = './Problems/collection'

def walk_file(path):
    for root, dirs, files in os.walk(path):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            if re.match(pattern, f):
                copyfile(os.path.join(root,f), os.path.join(target_path, f))
        
        for d in dirs:
            walk_file(os.path.join(root, d))

if __name__ == '__main__':
    walk_file(initial_root)
