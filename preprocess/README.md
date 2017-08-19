
## 预处理
	首先根据目标尺寸中的高度，将原始图片等比例放大到高度与目标高度一致。然后，根据放大后的宽度与目标宽度做比较，如果小于目标宽度，则在图片两边加上等宽的黑色边框；如果大于目标宽度，则在图片两侧截去相同宽度。

## 数据清单
	正面的数据经过预处理、人工标注以及用opencv的一系列处理（二值化、膨胀与腐蚀，填充与区域选择，多边形拟合），对于南阳市第二人民医院和印第安纳数据集，均可产生六种数据，对于南阳医专二附院，由于只是用了正常人的样本，故产生三种数据。
侧面的数据均只做了大小和分辨率的调整。

![images](https://github.com/cardiacai/cardiacai/raw/master/images/%E5%9B%BE%E7%89%87%E9%A2%84%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B.png)

<br/> 
我们收集了三种不同来源的数据，每种数据都整理出了正常和得病的一些种类，做过多种形式的处理（详见预处理和标注部分），包括调整分辨率的图片，加分割线的图片，二值化的图片，多边形拟合出的心脏轮廓坐标值txt文件（traces）等。
<br/>
多边形拟合的轮廓线
![images](https://github.com/cardiacai/CardiacAI/raw/master/images/%E5%A4%9A%E8%BE%B9%E5%BD%A2%E6%8B%9F%E5%90%88.png)
<br/>
处理成mask的样图
![images](https://github.com/cardiacai/CardiacAI/raw/master/images/heart_masks.png)
<br/>
## 南阳市第二人民医院：JPG
<br> 
|abnormal 2826|  normal 500 |
|:-----------:|:-----------:| 
|512*624      | 512*624     |
|annotation   | annotation  |
|countour     |countour     |
<br/> 
 侧面：轮廓  (508个，未多边形逼近）

## 印第安纳：JPG
<br> 
|abnormal 1403|  normal 139| 
|:-----------:|:-----------:| 
|512*624      | 512*624     |
|annotation   | annotation  |
|countour     |countour     |
<br/> 
表格信息、侧面 3497个

## 南阳：PNG
<br> 
|abnormal 514 |  
|:-----------:|
|512*624      | 
|annotation   |
|countour     |
<br/> 
侧面 154个


