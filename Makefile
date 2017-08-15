OPENCV2_PATH = /home/chenxupeng/apps

all: bin/detect_edges

bin/detect_edges: src/detect_edges.cpp
	g++ -O2 -g -o $@ src/detect_edges.cpp -I$(OPENCV2_PATH)/include -L$(OPENCV2_PATH)/lib -lopencv_core -lopencv_imgproc -lopencv_highgui