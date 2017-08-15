# cardiacai
Diagnosis of cardiac diseases using AI
We are trting to use the deep learning technique to diagnose cardiac disease.
The X-ray images we collected are from two hospitals in Nanyang, Henan and a foreign dataset(Indiana Universityhttps://openi.nlm.nih.gov/gridquery.php?q=&coll=cxr)

心脏病种类：
    心血管疾病类型多样，是第一大死亡因素。我们希望发展出一套用X光胸片数据诊断心脏病的方法。/Users/james/Desktop/类脑技术文档与网站/网站/心脏病类型.png

数据预处理
    我们将收集来的数据进行了多种预处理，包括调整大小，分辨率，人工标注，用opencv进行区域的选择和拟合等。
/Users/james/Desktop/类脑技术文档与网站/网站/图片预处理流程.png
    调整大小与分辨率：    首先根据目标尺寸中的高度，将原始图片等比例放大到高度与目标高度一致。然后，根据放大后的宽度与目标宽度做比较，如果小于目标宽度，则在图片两边加上等宽的黑色边框；如果大于目标宽度，则在图片两侧截去相同宽度。    正面的数据经过预处理、人工标注以及用opencv的一系列处理（二值化、膨胀与腐蚀，填充与区域选择，多边形拟合），对于南阳市第二人民医院和印第安纳数据集，均可产生六种数据，对于南阳医专二附院，由于只是用了正常人的样本，故产生三种数据。    侧面的数据均只做了大小和分辨率的调整。
数据清单：
    南阳市第二人民医院：JPG        异常：        512*624分辨率：           /eryuan/process_512x624             (正侧均有)                画线版：已完成    /eryuan /abnormal_Z/addline                轮廓：  已完成    /eryuan /abnormal_Z/heart_traces     （2826个）        正常：        512*624分辨率：          /eryuan /normal_Z/noline                画线版:  已完成   /eryuan /normal_Z/addline                轮廓： 已完成     /eryuan /normal_Z/heart_traces        （500个）    印第安纳：JPG        正常：512*624分辨率：      /indiana/normal_Z/noline                画线版：已完成     /indiana/normal_Z/addline                轮廓：已完成       /indiana/normal_Z/heart_traces      （1403个）        异常：512*624分辨率：      /indiana/abnormal_Z/noline                画线版：已完成     /indiana/abnormal_Z/addline                轮廓：已完成      /indiana/abnormal_Z/heart_traces      （139个）        结果：                   /indiana/indiana.csv        侧面：                   /indiana/process_C_512x624           （3497个）    南阳市医专二附院：PNG             512*624分辨率：      /nanyang/noline                                           画线版：已完成        /nanyang/addline                                         轮廓：已完成         /nanyang/heart_traces                 （514个）             侧面：               /nanyang/process_C_512x624            (154个)


