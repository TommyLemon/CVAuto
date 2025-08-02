## AIServer - CV AI 图片推理预测服务
本项目 AIServer 为**模拟被测试的 图片推理预测服务**，使用 Python + YOLO + Flask + OpenCV + Numpy + UnitAuto 等开发，<br />
通过**启动 HTTP API**(支持 CORS 跨域) **接收图片等参数来使用模型推理**，支持 目标检测、姿态关键点、图像分割、文本识别等。<br />
<br />
返回 JSON 包含 bbox 矩形、polygon 多边形、line 线段、point 点 及 label 标签、score 置信度、color 颜色、angle 角度 等，<br />
其中 color 格式为 \[r, g, b, a]，bbox 格式为 \[x, y, w, h]，line 格式为 \[x1, y1, x2, y2], polygon 格式为 {points: \[\[x1, y1], \[x2, y2]..]} <br />
<br />
你自己的图片推理服务要么也**按以上 CVAuto 能兼容的格式返回 JSON**，要么**在前端脚本后处理面板或源码中修改**底部的 <br />
getScore, getBboxes, getBbox, getPolygons, getLines, getPoints 等函数来适配自定义的 JSON 数据，避免渲染/计算错误：
https://github.com/TommyLemon/CVAuto/blob/main/apijson/JSONResponse.js#L2439-L2484

例如：
```javascript
JSONResponse.getBboxes = function(detection) {
   return detection?.data?.bboxes  // 适配最外层多套了一层 "data": { "bboxes": \[...] }
}
```
![](https://github.com/user-attachments/assets/becceac6-d948-4da7-bd71-ef7dcc825e2b)

<br />

### 使用
### Usage

#### 1. 下载并打开本目录
#### 1. Download and open this folder

用 PyCharm/Cursor 等 IDE/编辑器 或 直接命令行进入：<br />
Use PyCharm/Cursor/Other IDE or Editor, or use Command Line：
```sh
cd AIServer
```


#### 2. 初始化依赖
#### 2. Init dependency

用 IDE/编辑器 或者 Python 官网下载的安装包来安装 Python 3.10～3.12。<br />
Install Python 3.10～3.12 with IDE/Editor or the official installer downloaded on [python.org](https://www.python.org/downloads)

然后用 IDE/编辑器 在 main.py 中依赖报错的 import YOLO 等位置点击安装依赖包 或 使用命令行：<br />
Then click install dependencies with IDE/Editor on error lines of main.py, or use Command Line：

```sh
pip install -r requirements.txt
```  
如果执行以上命令未成功，则将 pip 换成 pip3 试试：<br />
if you cannot run the command successfully, try pip3:
```sh
pip3 install -r requirements.txt
```
如果有这个报错则信任相关域名：SSLCertVerificationError certificate verify failed: unable to get local issuer certificate ... <br />
If it shows the error, then trust the domains: certificate verify failed: unable to get local issuer certificate ...
```shell
 pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt 
```
```shell
 pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt 
```

<br />

#### 3. 运行 main.py
#### 3. Run main.py

点击 IDE/编辑器 在 main.py 中以下这行左侧的运行/调试 按钮：<br />
Click Run/Debug button on the left of this line in main.py:
```python
if __name__ == '__main__':
```

或者用命令行：<br />
Or use Command Line：

```sh
python main.py
```
如果执行以上命令未成功，则将 python 换成 python3 试试：<br />
if you cannot run the command successfully, try python3:
```sh
python3 main.py
```

<br />
