# 数字图像处理实验报告——project4 空域滤波

-------
##姓名：范睿霖  班级：自动化66  学号：2160504145

-------
## 摘要：利用不同的空域滤波来进行图像的处理，使图像变得更加平滑或重点显示出其的边缘

-------

### section1 利用高斯滤波器或中值滤波器平滑图像
gauss滤波器 3*3，sigma = 1.5
![Figure_34](media/15528837787364/Figure_34.png)

gauss滤波器 5*5 sigma = 1.5
![Figure_35](media/15528837787364/Figure_35.png)

gauss滤波器 7*7 sigma = 1.5
![Figure_36](media/15528837787364/Figure_36.png)

median滤波器 3*3
![Figure_37](media/15528837787364/Figure_37.png)
![Figure_38](media/15528837787364/Figure_38.png)

median滤波器 5*5

![Figure_39](media/15528837787364/Figure_39.png)
![Figure_40](media/15528837787364/Figure_40.png)

median滤波器 7*7
![Figure_41](media/15528837787364/Figure_41.png)
![Figure_42](media/15528837787364/Figure_42.png)


分析：高斯滤波器有两个可调参数，中值滤波器只有一个，同时在处理复杂情况时，如角落变化多的地方，高斯的效果会好一些，可能是因为中值滤波器的非线性导致其性能不稳定。

-------

###section2 高通滤波器
#### sobel滤波器 3*3
![Figure_43](media/15528837787364/Figure_43.png)
![Figure_44](media/15528837787364/Figure_44.png)

#### laplace滤波器 3*3

![Figure_45](media/15528837787364/Figure_45.png)

![Figure_46](media/15528837787364/Figure_46.png)

#### unsharp masking 3*3 sigma = 1.5

![Figure_47](media/15528837787364/Figure_47.png)
![Figure_48](media/15528837787364/Figure_48.png)


分析：三种高通滤波器中，Laplace的效果相对比较好，噪音少，边沿清晰；sobel噪音会稍多；masking的效果最差。

-------
## 附录
代码
gauss滤波器生成函数

```
import numpy as np
import matplotlib as plt
import math

def gauss(len,sigma):
    pi = 3.14
    a = np.zeros((len,len))
    for m in range(len):
        for n in range(len):
            i = int(m - (len - 1) / 2)
            j = int(n - (len - 1) / 2)
            e = math.exp((i**2 + j**2) / 2 / sigma**2)
            a[m , n] = 1 / e / (2 * pi * sigma**2)
    sum = np.sum(a)
    a = a/ sum
    return(a)
```

gauss滤波子函数

```
import numpy as np
import matplotlib.pyplot as plt
import math
import filter

def gauss (img):
    print ('输入阶数len：')
    len = int(input())
    print('输入sigma：')
    sigma = float(input())
    #img = Image.open('test1.bmp').convert('L')
    a = np.array(img)
    w = filter.gauss(len,sigma)
    x = np.empty((np.shape(a)[0],np.shape(a)[1]))
    offset = int((len - 1) / 2)

    for i in range(offset,np.shape(a)[0] - offset):
        for j in range(offset,np.shape(a)[1] - offset):
            t = a[i - offset:i + offset + 1,j - offset:(j + offset+1)]
            m = t * w
            x[i,j] = np.sum(m)

    '''img_1 = Image.fromarray(x)
    img.show()
    img_1.show()
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.imshow(a,cmap = 'gray')
    ax2.imshow(x,cmap = 'gray')
    plt.show()
```

median滤波子函数

```
import numpy as np
import matplotlib.pyplot as plt
import math

def median(img):
    #img = Image.open('test2.bmp').convert('L')
    a = np.array(img)
    x = a.copy()
    len = int(input('len = '))
    offset = int((len - 1) / 2)

    for i in range(offset,np.shape(a)[0] - offset):
        for j in range(offset,np.shape(a)[1] - offset):
            t = a[i - offset:i + offset + 1, j - offset:(j + offset + 1)]
            x[i,j] = np.median(t)

    '''img_1 = Image.fromarray(x)

    img.show()
    img_1.show()
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(a, cmap='gray')
    ax2.imshow(x, cmap='gray')
    plt.show()

```

低通滤波主函数

```
from PIL import Image
import numpy as np
import matplotlib as plt
import math
import cv2
import smoothing_guass as sg
import smoothing_median as sm

img = Image.open('test2.bmp').convert('L')
a = np.array(img)
m = int(input('select: 1.gauss 2.median'))

if m == 1:
    sg.gauss(img)

elif m ==2:
    sm.median(img)

```

高通滤波函数


```
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import filter

img = Image.open('test4 copy.bmp').convert('L')
a = np.array(img)
b = a.copy()
m = int(input('1.sobel 2.laplace 3.unsharp_masking'))

if m == 1:
    ws_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ws_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    for i in range(1,np.shape(a)[0]-1):
        for j in range(1,np.shape(a)[1]-1):
            x = a[i - 1:i +2,j - 1:j + 2]
            xx = np.sum(x * ws_x)
            xy = np.sum(x * ws_y)
            b[i,j] = (xx**2 + xy**2) ** 0.5
    '''
    img.show()
    Image.fromarray(b).show()
    '''
elif m == 2:
    w = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    b = cv2.filter2D(a,-1,w)
    '''
    img.show()
    Image.fromarray(b).show()
    '''
elif m == 3:
    w = filter.gauss(3,1.5)
    img_b = cv2.filter2D(a,-1,w)
    mask = a - img_b
    b = mask + a

    '''img.show()
    Image.fromarray(b).show()
    '''
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.imshow(a, cmap='gray')
ax2.imshow(b, cmap='gray')
plt.show()
```


