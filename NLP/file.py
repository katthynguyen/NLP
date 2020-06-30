import os
from pathlib import Path
import path

# hàm Ghi file
def Write_File(path,line,mode):
    try:
       
        file = open(path,mode)
        file.writelines(line)
        file.writelines("\n")
        file.close()
        pass
    except:
        pass

# Duyệt qua từng file đưa mỗi file vào list_file 
def Browse_item_list_path(list_path):
    list_file = []
    item = 0
    for item in range(len(list_path)):
        read_file = open(list_path[item],"r")
        list_file.append(read_file.read())
    return list_file

# get file
def get_text(file):
    read_file = open(file,'r', encoding="utf8")
    text = read_file.readlines()
    text = ' '.join(text)
    return text

# lấy đường dẫn file 
def Get_List_Path(path):
    list_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            list_path.append(root + "/" + file)            
    return list_path

# lấy danh sách tên file input
def Get_List_Name(path):
    list_name = []
    for root, dirs, files in os.walk(path):
        for file in files:
            list_name.append(os.path.splitext(file)[0])           
    return list_name


