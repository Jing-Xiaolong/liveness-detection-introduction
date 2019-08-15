- [活体检测](#活体检测)
- [传统方法论文调研](#传统方法论文调研)
- [深度学习PA数据集](#深度学习PA数据集)
- [深度学习方法论文调研](#深度学习方法论文调研)
  - [An original face anti-spoofing approach using partial convolutional neural network](#An original face anti-spoofing approach using partial convolutional neural network)
  - [Face anti-spoofing using patch and depth-based cnns](#Face anti-spoofing using patch and depth-based cnns)
  - [Deep Convolutional Dynamic Texture Learning with Adaptive Channel-discriminability for 3D Mask Face Anti-spoofing](#Deep Convolutional Dynamic Texture Learning with Adaptive Channel-discriminability for 3D Mask Face Anti-spoofing)
  - [Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision](#Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision)
  - [Face De-Spoofing: Anti-Spoofing via Noise Modeling](#Face De-Spoofing: Anti-Spoofing via Noise Modeling)
  - [Exploiting Temporal and Depth Information for Multi-frame face Anti-Spoofing](#Exploiting Temporal and Depth Information for Multi-frame face Anti-Spoofing)

<br><br><br><br>

## 活体检测

**motivation**

PA - presentation attacks，包括：print, replay, 3D-mask, facial cosmetic makeup, etc

在人脸检测中，攻击者会通过PA对系统进行攻击，没有活体检测的系统安全性低，2017左右以前的研究方向主要是传统机器视觉方法，之后则多将CNN融入以增强性能

**methods**

|                | traditional methods                                          | CNN-based methods                                            |
| :------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|    基本过程    | 根据手工设计的特征判断：<br>如LBP，SIFT，SURF等，进行检测    | 通过CNN网络端到端学习anti-spoofing的表示空间                 |
|      缺点      | 1. 对PA敏感，如replay/ 3D mask attack <br>2. 某一特征难以判断所有/大部分PAs | 1. 某些CNN方法仅将其视为二分类<br>2. 通过细节进行判别，并未学习到PA模式<br>    - 皮肤细节损失、色彩失真、摩尔纹、运动模式、形变 |
| 常用的判别依据 | 1. **基于<font color="red">文理</font>——空域信息**：深度图、文理、各类算子 | 1. 基于**<font color="red">文理</font>——空域信息**<br>    - 包括深度图、文理、各类算子<br>    - 噪声分解<br>2. 基于**<font color="red">运动</font> ——时域信息**<br>    - 包括眨眼、运动模式等<br>3. 基于**<font color="red">其它信号</font>——主要为频域信息**<br>    - rPPG(非接触式获取生物信号，如心跳)<br>    - 光谱分析(活体和PA反射的频率响应不同) |

<br>

## 传统方法论文调研

**传统方法大部分是基于文理，其基本思路为**

1. 原图预处理，如裁剪、对齐、分割等，同时进行表示空间的变换或叠加，如颜色空间、时域、空域、频域等
2. 利用传统方法对图像进行特征提取，如SIFT，SURF，LBP，HOG，及各类改进和变种（主要区别在此）
3. 分类前预处理，如多通道联结、降维、编码、进一步特征提取等
4. 特征输入分类器进行real/spoof二分类，如SVM、LR等

<br>

> [<font color="red">2015icip - Face anti-spoofing based on color texture analysis</font>](https://ieeexplore.ieee.org/abstract/document/7351280)

> [<font color="red">2016 - Face spoofing detection using color texture analysis</font>](https://ieeexplore.ieee.org/abstract/document/7454730)

- 基本思想：活体和PA的文理统计特征不一致（如下图），可对其文理特征进行分类

  <img src="liveness detection img/18.jpg" width="300px">

  通过LBP (local binary patterns 局部二值模式)提取文理特征，再对LBP进行分类，***LBP算子***如下

  <img src="liveness detection img/20.jpg" width="450px">

- 模型架构：

  <img src="liveness detection img/19.jpg" width="500px">

<br>

> [<font color="red">2018 - Ccolbp: Chromatic cooccurrence
> of local binary pattern for face presentation attack
> detection</font>](https://ieeexplore.ieee.org/document/8487325)

- 基本思想：文理统计特性
- 模型架构：

<img src="liveness detection img/35.jpg" width="600px">

<br>

> [<font color="red">2012 - On the effectiveness
> of local binary patterns in face anti-spoofing</font>](https://ieeexplore.ieee.org/abstract/document/6313548)

- 基本思想：文理统计特性
- 模型架构：<img src="liveness detection img/22.jpg" width="700px">

<br>

> [<font color="red">2012 - LBP-TOP based countermeasure against face spoofing attacks</font>](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=271BCC281BDD5D9B869D3DB92A278BB0?doi=10.1.1.493.6222&rep=rep1&type=pdf)
>
> [<font color="red">2013 - Can face anti-spoofing countermeasures work in a real world scenario?</font>](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.650.8406&rep=rep1&type=pdf)

- 基本思想：文理统计特性
- 模型架构：

<img src="liveness detection img/23.jpg" width="800px">

<br>

> [<font color="red">2017 - Face anti-spoofing based on color texture analysis</font>](https://ieeexplore.ieee.org/abstract/document/7351280)

- 基本思想：文理统计特性
- 模型架构：

<img src="liveness detection img/21.jpg" width="700px">

<br>

> [<font color="red">2013 - Face Liveness Detection with Component Dependent Descriptor</font>](http://www.cbsr.ia.ac.cn/users/zlei/papers/ICB2013/YANG-ICB13.pdf)

- 基本思想：文理统计特性
- 模型架构：
  1. 检测面部位置，并将面部分割为6个不同区域（轮廓，面部，左右眼，鼻，嘴）
  2. 提取特征，LBP、HOG等，并将不同部位的特征进行联结
  3. SVM分类real/spoofing

<img src="liveness detection img/24.jpg" width="800px">

<br><br>

## 深度学习PA数据集

[<font color="black">**1. Replay-attack 2012**</font>](https://www.idiap.ch/dataset/replayattack)

共1300视频样本，<font color="blue">不能做商业用途</font>，需由获机构授权者提交申请并签署[EULA](https://www.idiap.ch/dataset/replayattack/eula.pdf)(End User License Agreement)才能下载

[<font color="black">**2. MSU-USSA**</font>](http://biometrics.cse.msu.edu/Publications/Databases/MSU_USSA/)

全称MSU Unconstraint Smartphone Spoof Attack Database，共9000图像样本(1000live+8000spoof)，<font color="blue">不能做商业用途</font>，需签署[MSU USSA Agreement](http://biometrics.cse.msu.edu/Publications/Databases/MSU_USSA/MSU_USSA_Database_Agreement.pdf)才能下载

[<font color="black">**3. oulu-npu 2017**</font>](https://sites.google.com/site/oulunpudatabase/)

共4950视频样本，大部分CNN-based数据集都会使用的数据集，<font color="blue">不能做商业用途</font>，需由在学术机构担任永久性职位的人签署[EULA](https://drive.google.com/open?id=1m5qWHpdvM4SNOewKbw43u95makMTfD68)才能下载

[<font color="black">**4. SiW 2018**</font>](http://cvlab.cse.msu.edu/spoof-in-the-wild-siw-face-anti-spoofing-database.html)

165subjects共4478视频样本，<font color="blue">商业用途需获授权</font>，需由获机构授权者提交申请并签署[DRA](https://www.cse.msu.edu/computervision/SiW_Dataset_Release_Agreement_Form.pdf)(dataset release agreement)才能下载

[<font color="black">**5. CASIA-SURF 2019.6**</font>](https://sites.google.com/qq.com/chalearnfacespoofingattackdete/welcome)

1000subjects共21000视频样本，<font color="blue">目前只接受学术用途</font>，需由学术机构签署[CASIA-SURF release agreement](http://www.cbsr.ia.ac.cn/users/jwan/database/CASIA-SURF_agreement.pdf)才能下载

<br><br>

## 深度学习方法论文调研

#### An original face anti-spoofing approach using partial convolutional neural network

[<font color="red">论文下载</font>](https://ieeexplore.ieee.org/document/7821013)

**<font color="blue">motivation/基本思想</font>**

- **基于文理**：活体和PA的特征不同

**<font color="blue">models/模型架构</font>**

和传统方法过程类似，只不过提取特征这一步使用了预训练的VGG模型

<img src="liveness detection img/32.jpg" width="600px">



<br>

#### Face anti-spoofing using patch and depth-based cnns

[<font color="red">论文下载</font>](http://cvlab.cse.msu.edu/pdfs/FaceAntiSpoofingUsingPatchandDepthBasedCNNs.pdf)

**<font color="blue">缺点</font>**：性能堪忧，超不过传统方法

**<font color="blue">motivation/基本思想</font>**

- ***基于文理***，主要表现为：

1. 活体和PA的局部区域的具有不同<font color="gree">特征文理</font>，统计特性不同
2. 活体和PA的面部<font color="gree">深度图</font>不同（PA为扁平，活体有人脸形状）

<img src="liveness detection img/25.jpg" width="350px">

**<font color="blue">models/模型架构</font>**

- **Supervision Signals**

  - ***patch spoof scores***：从人脸图像中挑选某些局部区域patches，根据patch内的文理统计特征计算一个patch spoof scores，用于监督patch-based CNN部分

    **patch spoof scores标签**：1 - live，0 - spoof

  - ***Depth Map***：面部深度图显示了面部不同位置的深度信息，据此计算深损失，用于监督depth-based CNN部分

    **深度图标签**：通过**[<font color="gree">3DMM(2003)</font>](https://ieeexplore.ieee.org/abstract/document/1227983)**估计活体的3D面部形状图$A\in\mathbb{R}^{3\times Q}$，和2D面部深度图$M\in\mathbb{R}^{52\times 52}$作为深度图标签，用于计算深度图损失

    <img src="liveness detection img/26.jpg" width="600px">

<img src="liveness detection img/27.jpg" width="1000px">

- **网络架构**

  - ***patch-based CNN部分***

    为何用patches：1. 增加训练数据  2. 不用resize整张脸，保持原本的分辨率  3. 在局部检测可用于活体检测的特征时，设定更具挑战性的约束条件，加强特征提取的性能

    输入：相同大小的不同patches的RGB, HSV, YCbCr, pixel-wised LBP特征图等

    输出：pacth spoof scores

  - ***Depth-based CNN部分***

    研究表明高频部分对anti-spoofing非常重要，为避免对原图进行resize而损失图片的高频部分，因此使用FCN以无视输入特征图的size

    输入：HSV + YCbCr特征图

    输出：深度图 $\widehat{M}\in\mathbb{R}^{52\times52}$

  - ***网络架构信息***

<img src="liveness detection img/28.jpg" width="450px">

<br>

#### Deep Convolutional Dynamic Texture Learning with Adaptive Channel-discriminability for 3D Mask Face Anti-spoofing

[<font color="red">论文下载</font>](https://ieeexplore.ieee.org/document/8272765)

**<font color="blue">motivation/基本思想</font>**

本文主要针对***3D mask***类别的PA，其主要思想为

- **基于动态文理**——由于mask覆盖面部，mask难以完全呈现面部运动，而活体的面部运动更加精细、细腻（如眨眼，嘴唇、苹果肌微动等），也可以理解为基于文理的不同。

  3D mask的面部运动具有更加统一的运动模式，而活体的面部运动一致性更低

**<font color="blue">models/模型架构</font>**

<img src="liveness detection img/29.jpg" width="1000px">

- **基本流程**：1. 视频预处理，通过模型[**<font color="gree">CLNF(2014)</font>**](https://www.cl.cam.ac.uk/research/rainbow/projects/ccnf/files/iccv2014.pdf)， 2. 预训练VGG提取特征， 3. 通过光流方法对面部运动进行估计Subtle Facial Motion Estimation，  4. 提取动态文理Deep Conv Dynamic texture，  5. spoofing/live二分类

- **网络架构**

  - ***Subtle Facial Motion Estimation部分***

    预处理后的特征为 $X_t\in\mathbb{R}^{W\times H\times K}$，其中$W,H,K$表示特征宽、高、通道，$t=1,2,...,T$表示视频帧，这些特征中包含有细粒的文理特征。

    将$X_t$分解为$K$个时间序列$\{C_k\in\mathbb{R}^{W\times H\times T}\}_{k=1}^K$，再使用光流方法通过这些特征提取面部微运动：$\frac{\part{I}}{\part{x}}u+\frac{\part{I}}{\part{y}}v+\frac{\part{I}}{\part{t}}=0$

    其中$I(x,y,t)$是$(x,y)$处的亮度信息，$u,v$分别是$(x,y)$处的像素的水平、垂直速度（光流optical flow）

    计算每两帧之间的光流，对所有帧取得光流的平均值，最终得到动态文理为：$V\in\mathbb{R}^{K\times N},N= W \times H$

  - ***Adaptive Channel-discriminability constraint learning部分***

    动态文理$V$中存在对活体和3D mask都无有效响应的通道，这些通道无助于甚至有害于模型，需要对每个通道进行加权。本部分对每个通道的权重$D\in\mathbb{R}^{1\times K}$学习。保留$D$中最大的40个分量，然后进行标准化

  - ***Deep Convolutional Dynamic Texture Learning部分***

    $min_{D, U^l}\frac{1}{2}\sum_{l=1}^N||DV^l-U^l||^2,   s.t.||D||_2=1$，其中$U\in\mathbb{R}^{1\times N}$

    上述公式意在学习一个特征向量，该特征向量在某一距离度量下可以近似不同通道的加权动态文理特征

    对上述公式求导使导数为零，则最终的动态文理特征向量为：$U^l=DV^l, l=1,2,3,...,N$

**<font color="blue">implementation/实现</font>**

- **需注意细节**

  - 预处理模型：[**<font color="gree">CLNF(2014)</font>**](https://www.cl.cam.ac.uk/research/rainbow/projects/ccnf/files/iccv2014.pdf)

    检测68个面部landmarks，并进行面部align

  - VGG提取特征

    视频流中连续5帧选择一帧输入VGG，将其conv3-3的输出特征图$F\in\mathbb{R}^{56\times56\times256}$作为光流提取的输入

  - classifier

    使用SVM进行分类

- **结果**

  Intra-dataset

  <img src="liveness detection img/30.jpg" width="500px">

  cross-dataset

<img src="liveness detection img/31.jpg" width="500px">

<br>

#### Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision

**[<font color="red">论文下载</font>](https://ieeexplore.ieee.org/document/8578146)**

<font color="blue">**src codes**</font>：未开源

**<font color="blue">优缺点</font>**：性能终于超过传统方法

**<font color="blue">motivation/基本思想</font>**

- **基于文理**——活体和PA的面部<font color="gree">深度图</font>不同（PA为扁平，活体有人脸形状）
- **基于相关生物信号**——通过面部信息可测量相关<font color="gree">rPPG</font>(remote Photoplephysmography)信号，如***心跳***等

<img src="liveness detection img/13.jpg" width="500px">

**<font color="blue">models/模型架构</font>**

- **Supervision Signals**

  - ***Depth Map***：面部深度图显示了面部不同位置的深度信息，据此计算深损失，用于监督CNN部分。

    **深度图标签**：通过**[<font color="gree">DeFA(2017)</font>](https://arxiv.org/abs/1709.01442)**估计活体3D面部形状图$S\in\mathbb{R}^{3\times Q}$，和2D面部深度图$D\in\mathbb{R}^{32\times32}$作为深度图标签，用于计算深度图损失

  - ***rPPG***：rPPG信号与面部皮肤的强度值的变化相关，该强度值与血液流动有高想关心。由于传统的提取rPPG信号的方法对姿态、表情、照明敏感，且无法防御replay attack，采用RNN对其进行估计。

    **rPPG标签**：从没有姿态、表情、照明变化的视频提取出的活体rPPG信号作为标签，用于计算rPPG损失

    计算：对DeFA估计出的面部区域，计算正交色度信号$x_f=3r_f-2g_f$，$y_f=1.5r_f+g_f-1.5b_f$，再计算血液流动信号$\gamma=\frac{\sigma(x_f)}{\sigma(y_f)}$，再计算信号$p=3(1-\frac{\gamma}{2})r_f-2(1+\frac{\gamma}{2})g_f+\frac{2\gamma}{2}b_f$，对$p$进行FFT即为rPPG

    

<img src="liveness detection img/14.jpg" width="1000px">

- **网络架构**

  - ***CNN部分***

    若干block串联，每个block包括三个conv+exponential linear+bn和一个pooling

    每个block输出特征图经过resize layer将其resize为$64\times64$，并将其通道维联结

    联结后的特征图经过两个branches，一个估计深度图depth map，另一个估计特征图feature map

    $\Theta_D=argmin_{\Theta_{D}}\sum_{i=1}^{N_d}||CNN_D(I_i; \Theta_D)-D_i||_1^2$

  - ***RNN部分***

    $\Theta_R=argmin_{\Theta_{R}}\sum_{i=1}^{N_s}||RNN_R([{F_j}_{j=1}^{N_F}]_i; \Theta_R)-f_i||_1^2$

    其中$F_j$是正脸化frontilization后的feature map

  - ***Non-rigid Registration部分***

    根据估计的3D面部形状图$S$对特征图feature map进行对齐align，保证RNN跟踪并学习面部同一个区域的特征随时间和物体的变化。RNN不用考虑表情、姿态、背景的影响

    <img src="liveness detection img/15.jpg" width="450px">

**<font color="blue">implementation/实现细节</font>**

- **需注意细节**

  - two-stream training
    1. 输入面部图$I$和深度图标签$D$，训练CNN部分
    2. 输入面部图序列${I_j}_{j=1}^{N_f}$，深度图标签${D_j}_{j=1}^{N_f}$，估计的3D面部形状图${S_j}_{j=1}^{N_f}$，rPPG信号标签，训练CNN和RNN部分
  - testing：根据分类得分 $score=||\widehat{f}||_2^2+\lambda||\widehat{D}||_2^2$进行活体判断

- **结果**

  intro-test和cross-test results

<img src="liveness detection img/34.jpg" width="400px">

<img src="liveness detection img/17.jpg" width="800px">

<br>

#### Face De-Spoofing: Anti-Spoofing via Noise Modeling

**[<font color="red">论文下载</font>](http://openaccess.thecvf.com/content_ECCV_2018/html/Yaojie_Liu_Face_De-spoofing_ECCV_2018_paper.html)**

<font color="blue">**src codes**</font>：[我是链接](https://github.com/yaojieliu/ECCV2018-FaceDeSpoofing)，README比较简略，源码很简略，约等于没有开源

**<font color="blue">优缺点</font>**：首次对spoofing noise进行建模和可视化；实际部署较难，在活体面部图像质量不高、PA质量高时失效

**<font color="blue">motivation/基本思想</font>**

本文主要针对***print, replay, make-up***类别的PA，其主要思想为

- **基于文理**：将spoof face视为live face和spoof noise的叠加，对spoof noise进行分类判断是否为PAs

  退化图像$x\in\mathbb{R}^m$可视为源图像的函数：$x=A\widehat{x}+n=\widehat{x}+(A-\mathbb{I})\widehat{x}+n=\widehat{x}+N(\widehat{x})$，其中$\widehat{x}$为源图、$A\in\mathbb{R}^{m\times m}$为退化矩阵、$n\in\mathbb{R}^m$为加性噪声，$N(\widehat{x})=(A-\mathbb{I})\widehat{x}$为image-dependent的噪声函数。通过估计$\widehat{x},N(\widehat{x})$并去除spoof noise、以重建$\widehat{x}$。若给定$x=\widehat{x}$，则其spoof noise = 0

  图像spoof退化的原因：1. spoof介质，如纸、显示屏  2. 介质与环境的交互作用

<img src="liveness detection img/36.jpg" width="600px">

- **Spoof noise pattern study**
  - 退化过程：
    1. color distortion：spoof介质色域更窄，导致颜色空间投影
    2. Display artifacts：相机在获取纸张、屏幕图像时，会产生颜色近似、下采样等过程，导致高频分量损失、模糊、像素扰动等
    3. Presenting artifacts:：用于显示图像的纸张、屏幕与介质产生交互作用，包括反射、表面透明度等变化，此过程的噪声是加性（additive）
    4. Imaging artifacts：CMOS和CCD的传感器阵列的成像矩阵会有光干涉，在replay attack和某些print attack中产生失真alising和摩尔纹moire pattern，此过程的噪声也是加性的
  - 退化图像的频谱分析

<img src="liveness detection img/37.jpg" width="700px">

**<font color="blue">models/模型架构</font>**

<img src="liveness detection img/38.jpg" width="700px">

<img src="liveness detection img/39.jpg" width="700px">

- **模型架构**

  - 输入图像$I\in\mathbb{R}^{256\times 256 \times 6}$，RGB+HSV颜色空间

  - ***DS Net部分***

    De-Spoof Net，设计为encoder-decoder的结构，用于估计噪声模式$N(\widehat{I})$，并得到活体面部图像$\widehat{I}=I-N(\widehat{I})$。

    encoder：输出特征图$F\in\mathbb{R}^{32\times 32\times 32}$，用于表示spoof噪声模式

    decoder：输入$F$，然后重建噪声$N(\widehat{I})$

    0\1 Map Net：0 - live face，1 - spoof

  - ***DQ Net部分***

    discriminative quality net，预训练网络，是[这篇论文中的CNN部分](#**[<font color="red">2018cvpr - Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision</font>](https://ieeexplore.ieee.org/document/8578146)**)，其参数固定，不用训练。DQ Net部分用于确保估计的活体面部图$\widehat{I}$更加接近真实情况。

    通过DQ Net计算估计的活体面部图$\widehat{I}$的面部深度图$CNN_{DQ}(\widehat{I})$，深度图标签——2D面部深度图$D\in\mathbb{R}^{32\times32}$用来监督$\widehat{I}$。（文章并没提$D$怎么来的）

    $J_{DQ}=||CNN_{DQ}(\widehat{I})-D||_1$

  - ***VQ Net部分***

    visual quality net，通过GAN来验证$\widehat{I}$的visual quality。VQ输入$\widehat{I}，I$，根据它们的差异大小区分$I$是否为PAs。在VQ Net训练时分两个步骤进行评估

    固定DS Net，更新VQ Net：$J_{VQ_{train}}=-\mathbb{E}_{I\in R}log(CNN_{VQ}(I))-\mathbb{E}_{I\in S}log(1-CNN_{VQ}(CNN_{DS}(I)))$

    固定VQ Net，更新DS Net：$J_{VQ_{test}}=-\mathbb{E}_{I\in S}log(CNN_{VQ}(CNN_{DS}(I)))$

    其中$R,S$分别是实际图像、合成图像

- **损失函数**

  - magnitude loss：使活体图像的spoof noise趋近与0

    $J_m=||N||_1$

  - Zero\One map loss：增强spoof noise的普遍性（任意位置都有）

    $J_z=||CNN_{01map}(F;\Theta)-M||_1$，其中$M\in0^{32\times 32} or M\in1^{32\times 32}$

  - repetitive loss：spoof noise的模式是重复性的，这有spoof媒介产生。将噪声$N$通过傅里叶变换$F$，并计算高频区域的最大值，这一个最大值即spoof noise。

    $J_r=\begin{cases}-max(H(F(N),k)),I\in Spoof \\ ||max(H(F(N),k))||_1, I\in Live\end{cases}$，其中$H$是低频域掩码，即，将傅里叶变换的中心$k\times k$置零

  - 总损失

    $J_{Total}=J_z+\lambda_1J_m+\lambda_2J_r+\lambda_3J_{DQ}+\lambda_4J_{VQ_{test}}$

**<font color="blue">implementation/实现细节</font>**

- **结果**

<img src="liveness detection img/40.jpg" width="700px">

<img src="liveness detection img/41.jpg" width="700px">

<br>

#### Exploiting Temporal and Depth Information for Multi-frame face Anti-Spoofing

**[<font color="red">论文下载</font>](https://arxiv.org/abs/1811.05118)**

**<font color="blue">src models</font>**：未开源

**<font color="blue">motivation/基本思想</font>**

- **基于运动**——活体和PA的运动特征不同，可进行下列理论分析：

  在视频流中，物体的运动有利于提取人脸深度信息，可将面部运动和面部深度信息结合，用于活体检测

<img src="liveness detection img/1.jpg" width="700px">

​	在PA和liveness中，各器官间的角度不同，即$(a):\alpha>\beta_{2}, \beta_{1}<\gamma,   (b):\alpha'<\beta_{2}', \beta_{1}'>\gamma'$

​	针对面部的平移和旋转，存在不同的几何关系：

<img src="liveness detection img/2.jpg" width="400px">

​		在实际场景、PA场景中，$d1, d2$存在不同的几何关系，可通过该几何关系判断

​			real scene: $d1'/d2'=d1/d2$，其中$d1'$是$d1$的估计

​			print attack: $d1'=0,d2'=0$

​			replay attack:

​					 $d1'/d2'=d1/d2$ in perfect spoofing scene (keep stationary, nearly impossible)

​					$d1'/d2'\ne d1/d2$ in usual spoofing scene

​		面部变化和运动非常复杂（包括上述平移、旋转，还包括混合运动、非刚性形变等），但通过上述分析，实际的活体和PA所产生的深度图(包括时域和空域)有很大区别，据此可以对活体和非活体进行区分。两者有何区别有待深入研究，网络会自行对这些区别进行提取。

**<font color="blue">Models模型架构</font>**: 

- **单帧场景**

  - ***深度图的生成***：通过**[<font color="gree">PRNet(2017)</font>](http://openaccess.thecvf.com/content_ECCV_2018/html/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.html)**估计活体面部深度图，得到$V_{n\times3}$用于表示n个面部关键点的3D坐标，再投影到2D的UV空间，得到$D\in\mathbb{R}^{32\times32}$作为深度图标签$D^t$

  <img src="liveness detection img/3.jpg" width="500px">

  - ***网络架构***：

  <img src="liveness detection img/4.jpg" width="800px">

  - ***depth loss*** = 欧几里得距离损失 + 对比(contrast)深度损失，两者分别为

    ​	$L_{single}^{absolute}=||D_{single}-D||_2^2$

    ​	$L_{single}^{contrast}=\sum_{i}||K_i^{contrast}\bigodot D_{single}-K_i^{contrast}\bigodot D||_2^2$	

    ​	$L_{single}=L_{single}^{absolute}+L_{single}^{contrast}$

    其中，$K_{i}^{contrast}$为

    <img src="liveness detection img/6.jpg" width="300px">

- **多帧场景**

  利用短期/长期瞬时运动信息，短期瞬时运动 -> OFFB，长期 -> ConvGRU

  - ***网络构架***

    <img src="liveness detection img/10.jpg" width="800px">

  - ***OFFB模块***

    optical flow guided feature block，用于提取**短期**瞬时运动特征，**主要是两连续帧之间**

    在时刻$t$，$(x,y)$处的亮度信息为I(x,y,t)，对其求导，并将求导结果重新调整

  <img src="liveness detection img/7.jpg" width="350px">

  ​		其中，vx, vy表示x,y处的像素速度，(vx, vy)即光流，(vx, vy, 1)与亮度的导数F=(Fx, Fy, Ft)垂直，F即为optical flow guided features，并且可从图片中提取，它通过光流对时、空间梯度进行编码，如下

  <img src="liveness detection img/8.jpg" width="400px">

  

  - ***ConvGRU模块***

    Convolution gated recurrent unit，用于提取**长期**瞬时运动特征，即提取长序列(时间维)信息，但忽略空间维信息，在隐藏层引入卷积提取空间维信息，得到ConvGRU：

    <img src="liveness detection img/9.jpg" width="250px">

    $X_{t},H_{t},U_{t},R_{t}$表示输入、出、update gate、reset gate，$K_{r},K_{u},K_{h}$卷积核，$H_{t}$即ConvGRU的深度图

  - ***深度图和活体判别***

    融合深度图 = 短期深度图 + 长期深度图

    ​		$D_{fusion}^{t}=\alpha*D_{single}^t+(1-\alpha)*D_{multi}^t, \alpha\in[0,1]$

    多帧场景的损失函数（对活体判别，深度图具有决定性作用，二分类辅助判别深度图的类别）包括：

    1. 多帧深度图损失 = 欧几里得距离损失 + 对比深度损失

       ​	$L_{multi}^{absolute}(t)=||D_{fusion}^t-D^t||_2^2$

       ​	$L_{multi}^{contrast}(t)=\sum_{i}||K_i^{contrast}\bigodot D_{fusion}^t-K_i^{contrast}\bigodot D^t||_2^2$

       ​	$L_{multi}^{depth}=\sum_{t}^{N_F-1}(L_{multi}^{absolute}(t)+L_{multi}^{contrast}(t))$ 对所有帧求和

       其中$D^t$是$t$时刻的深度图标签

    2. 多帧活体判别损失

       ​	$L_{multi}^{binary}=-B^t*log(fcs([{D_{fusion}^t}^{N_f-1}]))$ 

       其中$B^t$是$t$时刻的二分类标签，${D_{fusion}^t}^{N_f-1}$是$N_f-1$帧时(最后一帧时)，将前面所有时刻$t$产生的融合深度图进行拼接而来的总的深度图

       $fcs$表示网络的最后两个全连接层 + softmax(对攻击模式进行判别)，得到score + 0/1

    3. 总损失 = 深度图损失 + 活体判别损失

       ​	$L_{multi}=\beta·L_{multi}^{binary}+(1-\beta)·L_{multi}^{depth}$

**<font color="blue">implementation/实现细节</font>**

- **需注意细节**

- two-staged training

  1. 通过单帧深度图损失训练单帧架构，得到一个基础的表示网络
  2. 固定单帧架构的参数，通过多帧混合损失微调多帧部分的参数。向网络输入$N_f$帧图像，每三帧采样一次

- testing:

  ​	$\widehat{b}$表示living logtis，$\widehat{b}=fcs(D_{fusion}^t)$，最终的living score为

  ​	$score=\beta·\widehat{b}+(1-\beta)·\frac{\sum_{t}^{N_f-1}||D_{fusion}^t*M_{fusion}^t||_1}{N_f-1}$，$M_{fusion}^t$为通过PRNet产生的t时刻的面部掩码

- **结果**

  OULU-NPU数据集intra-testing

<img src="liveness detection img/11.jpg" width="500px">

​	 	cross-testing

<img src="liveness detection img/33.jpg" width="400px">

​		hard examples

<img src="liveness detection img/12.jpg" width="450px">

