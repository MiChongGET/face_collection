#### CSDN地址：http://blog.csdn.net/qq_31673689/article/details/79370412

#### 一、环境搭建
###### １．系统环境

```
Ubuntu 17.04
Python 2.7.14
pycharm 开发工具
```
###### 2.开发环境，安装各种系统包

 - 人脸检测基于dlib，dlib依赖Boost和cmake
 

```
$ sudo apt-get install build-essential cmake
$ sudo apt-get install libgtk-3-dev
$ sudo apt-get install libboost-all-dev
```

- 其他重要的包
```
$ pip install numpy
$ pip install scipy
$ pip install opencv-python
$ pip install dlib
```
- 安装 face_recognition

```
# 安装 face_recognition
$ pip install face_recognition
# 安装face_recognition过程中会自动安装 numpy、scipy 等 
```


----------


#### 二、使用教程

##### 1、facial_features文件夹

> 此demo主要展示了识别指定图片中人脸的特征数据，下面就是人脸的八个特征，我们就是要获取特征数据

```
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
```
##### 运行结果：
###### 自动识别图片中的人脸，并且识别它的特征
###### 原图：
![](https://ws1.sinaimg.cn/large/005EneYkly1fot3482971j30xc1n7tby.jpg)
![](https://ws1.sinaimg.cn/large/005EneYkly1fostfh1gbuj30dl0kfqc5.jpg)

###### 特征数据，数据就是运行出来的矩阵，也就是一个二维数组
![](https://ws1.sinaimg.cn/large/005EneYkly1fostgvzjatj31dw06daca.jpg)

###### 代码：

```
# -*- coding: utf-8 -*-
# 自动识别人脸特征
# filename : find_facial_features_in_picture.py

# 导入pil模块 ，可用命令安装 apt-get install python-Imaging
from PIL import Image, ImageDraw
# 导入face_recogntion模块，可用命令安装 pip install face_recognition
import face_recognition

# 将jpg文件加载到numpy 数组中
image = face_recognition.load_image_file("chenduling.jpg")

#查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

for face_landmarks in face_landmarks_list:

   #打印此图像中每个面部特征的位置
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

    for facial_feature in facial_features:
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

   #让我们在图像中描绘出每个人脸特征！
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for facial_feature in facial_features:
        d.line(face_landmarks[facial_feature], width=5)

    pil_image.show() 
```

#### 2、find_face文件夹

> 不仅能识别出来所有的人脸，而且可以将其截图挨个显示出来,打印在前台窗口
###### 原始的图片
![这里写图片描述](https://ws1.sinaimg.cn/large/005EneYkly1fosvje9oc4j30gl0hctn9.jpg)

###### 识别的图片
![这里写图片描述](https://ws1.sinaimg.cn/large/005EneYkly1fostk4lgq4j31ew0m1az4.jpg)

##### 代码：

```
# -*- coding: utf-8 -*-
#  识别图片中的所有人脸并显示出来
# filename : find_faces_in_picture.py

# 导入pil模块 ，可用命令安装 apt-get install python-Imaging
from PIL import Image
# 导入face_recogntion模块，可用命令安装 pip install face_recognition
import face_recognition

# 将jpg文件加载到numpy 数组中
image = face_recognition.load_image_file("yiqi.jpg")

# 使用默认的给予HOG模型查找图像中所有人脸
# 这个方法已经相当准确了，但还是不如CNN模型那么准确，因为没有使用GPU加速
# 另请参见: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)

# 使用CNN模型
# face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

# 打印：我从图片中找到了 多少 张人脸
print("I found {} face(s) in this photograph.".format(len(face_locations)))

# 循环找到的所有人脸
for face_location in face_locations:

        # 打印每张脸的位置信息
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right)) 
# 指定人脸的位置信息，然后显示人脸图片
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show() 
```

####　３、know_face文件夹

> 通过设定的人脸图片识别未知图片中的人脸

```
# -*- coding: utf-8 -*-
# 识别人脸鉴定是哪个人

# 导入face_recogntion模块，可用命令安装 pip install face_recognition
import face_recognition

#将jpg文件加载到numpy数组中
chen_image = face_recognition.load_image_file("chenduling.jpg")
#要识别的图片
unknown_image = face_recognition.load_image_file("sunyizheng.jpg")

#获取每个图像文件中每个面部的面部编码
#由于每个图像中可能有多个面，所以返回一个编码列表。
#但是由于我知道每个图像只有一个脸，我只关心每个图像中的第一个编码，所以我取索引0。
chen_face_encoding = face_recognition.face_encodings(chen_image)[0]
print("chen_face_encoding:{}".format(chen_face_encoding))
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
print("unknown_face_encoding :{}".format(unknown_face_encoding))

known_faces = [
    chen_face_encoding
]
#结果是True/false的数组，未知面孔known_faces阵列中的任何人相匹配的结果
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("result :{}".format(results))
print("这个未知面孔是 陈都灵 吗? {}".format(results[0]))
print("这个未知面孔是 我们从未见过的新面孔吗? {}".format(not True in results)) 
```

##### 4、video文件夹

> 通过调用电脑摄像头动态获取视频内的人脸，将其和我们指定的图片集进行匹配，可以告知我们视频内的人脸是否是我们设定好的
###### 实现：
![](https://ws1.sinaimg.cn/large/005EneYkly1fostuqbk69j31dh0lih0x.jpg)

###### 代码：

```
# -*- coding: utf-8 -*-
# 摄像头头像识别
import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

# 本地图像
chenduling_image = face_recognition.load_image_file("chenduling.jpg")
chenduling_face_encoding = face_recognition.face_encodings(chenduling_image)[0]

# 本地图像二
sunyizheng_image = face_recognition.load_image_file("sunyizheng.jpg")
sunyizheng_face_encoding = face_recognition.face_encodings(sunyizheng_image)[0]

# 本地图片三
zhangzetian_image = face_recognition.load_image_file("zhangzetian.jpg")
zhangzetian_face_encoding = face_recognition.face_encodings(zhangzetian_image)[0]

# Create arrays of known face encodings and their names
# 脸部特征数据的集合
known_face_encodings = [
    chenduling_face_encoding,
    sunyizheng_face_encoding,
    zhangzetian_face_encoding
]

# 人物名称的集合
known_face_names = [
    "michong",
    "sunyizheng",
    "chenduling"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # 读取摄像头画面
    ret, frame = video_capture.read()

    # 改变摄像头图像的大小，图像小，所做的计算就少
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 默认为unknown
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # if match[0]:
            #     name = "michong"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # 将捕捉到的人脸显示出来
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #加上标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display
    cv2.imshow('monitor', frame)

    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
```

##### 5、boss文件夹

> github开源项目,主要是结合摄像头程序+极光推送，实现识别摄像头中的人脸。并且通过极光推送平台给移动端发送消息！
https://github.com/MiChongGET/face_collection/tree/master/boss
