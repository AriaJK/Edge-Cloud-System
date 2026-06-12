# Edge-Cloud-System

## 项目简介

边云协同智能检测系统。

系统采用 YOLOv8 在边缘端完成目标检测，通过 FastAPI 实现边云协同通信，并结合 GLM-4V 大模型完成图像理解与智能问答，最终通过 Streamlit 仪表盘进行可视化展示。

## 技术栈

* Python
* YOLOv8
* OpenCV
* FastAPI
* Streamlit
* GLM-4V
* Docker

## 功能

* 实时目标检测
* 边云协同任务调度
* 云端视觉分析
* Agent智能问答
* 实时数据仪表盘

## 启动方式

### 启动云端服务

uvicorn cloud.main:app --reload

### 启动仪表盘

streamlit run dashboard/app.py

### 启动边缘端检测

python edge/yolo_scheduler_upload.py
