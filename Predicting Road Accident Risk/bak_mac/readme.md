将所有文件放到predicting road accident risk目录下，进行执行


road_accident_risk_mac.py 文件是在mac上可以直接执行的文件，可以具体看模型使用指南.md,并且这个版本可以支持启动http服务进行单次预测
road_accident_risk_docker.py 文件是可以进行用docker训练，但是mac docker没法使用gpu，所以执行 ./docker-run.sh（docker-run.bat windows）进行自动选择，mac走原生，
上面两个给我搞麻了，所以放到这里来bak，然后重新融合在一起