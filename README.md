# MVCNN_Caffe_40views 

## 資料準備
1. 下載GitHub所有的文件，點選Download Zip

2. 下載40views的昆蟲影像資料集：  
https://drive.google.com/drive/folders/1NZ64rUvqFWMmm-Z9rpxkFphvzYrvoOWw?usp=sharing

3. 下載驗證資料集：     
https://drive.google.com/drive/folders/1WCqaKPPU2Xl_glbBopm7pxn2zX5qDF2v?usp=sharing

4. 下載pretrained model：    
https://drive.google.com/file/d/1DbXL3fUVObOCvUxk2CwjXL-KukFcGRxX/view?usp=sharing

5. mvccn_40view.prototxt路徑與種類數目修改：  
- 修改mvccn_40view.prototxt中的輸入層的資料路徑  
```
'data_path': './40views'  
```

- 修改最後一層全連接層的種類數目  
```
num_output: 5  
```
6. 將Caffe資料夾複製到目錄裡

## 訓練
於目錄中開啟終端機，輸入指令     
```
python trainMVCNN.py
```

## 驗證
修改valid.py中檔案位置的絕對路徑 

- deploy  
```
net_file='/home/viplab/桌面/MVCNN_Caffe_40views-master/mvcnn_40view_deploy.prototxt'
```

- caffe model  
```
caffe_model='/home/viplab/桌面/MVCNN_Caffe_40views-master/mvcnn_iter_10000.caffemodel'
```

- mean file  
```
mean_file = np.load('/home/viplab/桌面/MVCNN_Caffe_12views-master/ilsvrc_2012_mean.npy')
```

- 驗證資料  
```
image_dir = r"/home/viplab/桌面/MVCNN_Caffe_40views-master/valid/valid_al" 
```
- list (分類標籤)  
```
imagenet_labels_filename = '/home/viplab/桌面/MVCNN_Caffe_40views-master/list.txt' 
```

於目錄中開啟終端機，輸入指令  
```
python valid.py
```


dataset  
https://drive.google.com/drive/folders/1NZ64rUvqFWMmm-Z9rpxkFphvzYrvoOWw?usp=sharing  

ModelNet40 pretrained model  
https://drive.google.com/file/d/1DbXL3fUVObOCvUxk2CwjXL-KukFcGRxX/view?usp=sharing

valid data  
https://drive.google.com/drive/folders/1WCqaKPPU2Xl_glbBopm7pxn2zX5qDF2v?usp=sharing
