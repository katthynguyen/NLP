import re
import os
import numpy
import json
from pandas import DataFrame
from pathlib import Path 
from file import *
from dataNaturalLanguage import *
from string import digits 

def main():
    
    # Đọc file.txt crawl theo chủ đề để xử lý dữ liệu       
    path_input = "input"
    list_path =  Get_List_Path(path_input)   

    # lấy danh sách tên file từ file input
    list_name =  Get_List_Name(path_input)  

    # đường dẫn file xuất dữ liệu được xử lý
    path_output = "out"  
    path_output_data = "data"
    path_output_cosine = "cosine"
    path_output_Vector = "vector"

    #nếu folder chưa tồn tại thì tạo folder mới
    if os.path.isdir(path_output) == False:
        os.mkdir(path_output) 
    if os.path.isdir(path_output_data) == False:
        os.mkdir(path_output_data)   
    if os.path.isdir(path_output_cosine) == False:
        os.mkdir(path_output_cosine)
    if os.path.isdir(path_output_Vector) == False:
        os.mkdir(path_output_Vector)               
        
    for i in range(len(list_path)):   
        request = get_text(list_path[i])
        
        # Dữ Liệu Sau Khi Loại Bỏ Các Tag HTLM:
        source = Clean_html(request)
        page = str(i)+'.txt'
        name_clean_html =  'write_clean_html_%s' % page 
        Write_File( path_output +"/" + name_clean_html  ,source,'w+')
       
        #Dữ Liệu Sau Khi Loại Bỏ Khoảng Trắng Và Các Kí Tự Đặc Biệt: 
        clean_source = remove_special_character(source)
        clean_source = list(clean_source)
        sourse = remove_special_character(source)        
        sourse =''.join(sourse)     
        page = str(i)+'.txt'   
        name_clean_character = 'write_remove_special_character_%s' % page
        Write_File(path_output +"/" + name_clean_character, sourse ,'w+') 
        # Write_File(path_output_data +"/" + name_clean_character, sourse ,'w+')        
               
        #List Các Câu Đã Tách:
        sents = separate_sentence(sourse)
        page = str(i) +'.txt' 
        name_sents =  'write_sentences_sparated_%s' % page 
        sents = "\n".join(sents)          
        Write_File(path_output +"/" + name_sents , sents ,'w+')

        #Lish Các Từ Đã Tách:        
        words = token_word(sents)
        page = str(i)+'.txt'  
        name_words ='write_words_sparated_%s' % page   
        words = "\n".join(words)
        Write_File(path_output + "/" + name_words,words,'w+')
            
        # stemming words        
        stem = token_word(words)                
        page = str(i)+'.txt'        
        name_words = 'write_stem_words_%s' % page       
        stem = "\n".join(stem)  
        Write_File(path_output + "/" + name_words, stem,'w+')        
        Write_File(path_output_data + "/" + name_words, stem,'w+')
      
        #Dict Thông Kê Số Lượng Từ:  
        stemword = token_word(words)
        word_dict  =  Count_frequently(stemword) 
        word_list= []  
        for (word,fre) in word_dict.items():
            word_dict = (word+ ' : '+ str(fre) + '\n') 
            word_list.append( word_dict) 
        page = str(i)+'.txt' 
        name_words_statisical = 'words_Count_frequently_%s' % page       
        Write_File(path_output + "/"+ name_words_statisical, word_list,'w+')

# ================================COSINE==========================================
        
    # Đọc danh sách các file dữ liệu đưuọc làm sạch và lưu ở trên      
    list_paths =  Get_List_Path(path_output_data) 
        
    # duyệt qua tùng file từ list-path lấy được để tính độ đo
    list_data = Browse_item_list_path(list_paths)
    

    # Gọi Menu chọn
    menu()
    chon = int(input())     

    # Chọn thuật toán thực hiện
    
    while(chon):
        if chon == 1:            
            #           <------ BAD OF WORD ------>
            # Tao list luu cac vector
            list_dense_bow = []    
            list_dense_bow = BagOfWords(list_data).tolist()
        
            item = 0
            for item in range(len(list_dense_bow)):
                temp  = list_dense_bow[item]
                list_dense_bow[item] = "\n" +str(item + 1) + "." + "  BOW" + str(list_name[item] + ".txt " + "\n" + str(temp))

            name_file_data = "BOW.txt"
            x = "\n".join(str(v) for v in list_dense_bow)
            Write_File(path_output_Vector +"/" + name_file_data, x ,'w+')    

            print("\n\tCOSINE SIMILARYTI BAG OF WORD\t")
            list_consine = CosineSimilar_BOW(list_data)    
            x = "\n".join(str(v) for v in list_consine)
            
            # ghi file tính độ đo tương đồng cosine bow             
            list_consine = CosineSimilar_BOW(list_data) 
            name_cosine_bow= "cosine_bow.txt"
            x = "\n".join(str(v) for v in list_consine)   
            Write_File(path_output_cosine +"/" + name_cosine_bow , x,'w+')

            # <---------------------------------------------------------------->
            menu()                
            chon = int(input())                            
        elif chon == 2:
            #       --------------------------TF-IDF------------------------------>

            #lưu ma trận độ đo TF-IDF vào mảng  1 chiều
            list_dense_tf_idf = []    
            list_dense_tf_idf = Tf_IDF(list_data).tolist()
        
            # chuẩn hóa format matrix in ra file.txt
            item = 0
            for item in range(len(list_dense_tf_idf)):
                temp  = list_dense_tf_idf[item]
                list_dense_tf_idf[item] = "\n" + str(item + 1) + "." + "  TF_IDF" + str(list_name[item] + ".txt " + "\n" + str(temp))

            # ghi file TF-IDF
            name_file_data = "TF_IDF.txt"
            x = "\n".join(str(v) for v in list_dense_tf_idf)
            Write_File(path_output_Vector +"/" + name_file_data, x ,'w+')


            # ghi file cosine tf-idf         
            list_consine_tf_idf =  CosineSimilar_TF_IDF(list_data)
            name_cosine_tf_idf= "cosine_tf_idf.txt"
            y = "\n".join(str(v) for v in list_consine_tf_idf)   
            Write_File(path_output_cosine +"/" + name_cosine_tf_idf, y,'w+')
            menu()                
            chon = int(input())  
             #       --------------------------CLOSE------------------------------>
        elif chon == 0:
            break

if __name__ == "__main__":
    main()