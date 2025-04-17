## Hyperparameter Optimization in Black-box Image Processing using Differentiable Proxies

**本文核心研究背景**





### 前置知识
**硬件在环(Hardware-in-the-Loop)**：
硬件在环是一种仿真技术，在系统开发和测试过程中，将真实的硬件设备嵌入到一个模拟的环境中，这个模拟环境能够模拟该硬件设备在实际运行时所面临的各种输入信号和工况。

在本文研究中，硬件在环系统用于优化黑盒 ISP 的超参数，具体工作过程为：利用校准后的高分辨率屏幕显示控制图像作为输入，设置 ISP 的配置参数，再记录 ISP 处理后的输出图像，以此生成训练数据对。