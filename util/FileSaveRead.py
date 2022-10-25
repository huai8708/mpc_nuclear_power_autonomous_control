# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:30:56 2021

@author: new
"""
import sys
import os
sys.path.append('./../')    
sys.path.append('./../../')   
sys.path.append('./../../../')   
sys.path.append('./../../../../')  

import h5py
class FileSaveRead():
    def __init__():   
        pass
            
    # 获取路径中所有的文件名称
    @staticmethod       
    def file_name(file_dir):   
        L=[]   
        for root, dirs, files in os.walk(file_dir):  
            for file in files:  
                L.append(os.path.join(root, file))  
        return L
    
    @staticmethod       
    def file_name_no(file_dir):   
        L=[]   
        for root, dirs, files in os.walk(file_dir):  
            for file in files:  
                L.append(file)  
        return L    

    # 创建一个路径
    @staticmethod       
    def creatpath(save_dir):
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)   
        return 

    # 将一个diction 写入 h5文件
    @staticmethod       
    def write_diction_h5(name, data_dict):
        f = h5py.File(name,'w')   #创建一个h5文件，文件指针是f
        for key in data_dict.keys():
            f[key] = data_dict[key]
        f.close()            
        return

    # 将一个文件夹的内容，拷贝到另一个文件夹
    @staticmethod  
    def copyFile(inputDir, outDir):
        import os, glob
        from shutil import copyfile
    
        #通过glob.glob来获取原始路径下，所有'.jpg'文件
        file_list = glob.glob(os.path.join(inputDir, '*.*'))
    
        choseImg = []
        #遍历所有随机得到的jpg文件，获取文件名称（包括后缀）
        for item in file_list:
            choseImg.append(os.path.basename(item))
    
        for item in choseImg:
            #将随机选中的jpg文件遍历复制到目标文件夹中
            copyfile(inputDir+'/'+item, outDir+'/'+item)
    
        #copyfile("cccc.inp", outDir+'/'+"TPmate.inp")
        return





    # 读取 h5文件，并将其转为diction 
    @staticmethod       
    def read_diction_h5(name):
        f = h5py.File(name,'r')   #创建一个h5文件，文件指针是f
        data = {}
        for key in f.keys():
            data[key] = f[key][:]
        f.close()    
        return data
      
    # 将一个list 写入h5
    @staticmethod       
    def write_list_h5(name, data):
        f = h5py.File(name,'w')   #创建一个h5文件，文件指针是f
        f['data'] = data
        f.close()
        
        return


    # 读取 h5文件，并将其转为list     
    @staticmethod       
    def read_list_h5(name):
        f = h5py.File(name,'r')   #创建一个h5文件，文件指针是f
        data = f['data'][:]
        f.close()    
        return data
    
    
    
    @staticmethod       
    def write_pkl(name, data):
        import pickle
        
        with open(name,'w+b') as f:
            f.write( pickle.dumps(data) ) 

        
        return
   
    @staticmethod       
    def read_pkl(name):
        import pickle
        with open(name,'rb') as f:
            data = pickle.loads(f.read())  
        return data    
    

    @staticmethod      
    def have_file(file):
        from pathlib import Path
        my_file = Path(file)
        if my_file.exists():
            return True
        else:
            return False
        
    @staticmethod          
    def remove_all(files):
        for file in files:
            if FileSaveRead.have_file(file):
                os.remove(file)        
 
        
 

        
        
        
        