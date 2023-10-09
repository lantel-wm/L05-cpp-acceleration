# 编译器设置
NVCC = nvcc
CC = g++
NVFLAGS = --shared -Xcompiler -fPIC
CFLAGS = -O3 -Wall -std=c++11 -fPIC
LDFLAGS = -L. -lcustep -L/usr/local/cuda/lib64 -lcudart

# 源文件和目标文件
CU_SOURCES = gpu.cu
C_SOURCES = gpu.cpp
OBJECTS = $(CU_SOURCES:.cu=.o) $(C_SOURCES:.cpp=.o)

# 共享库名称
LIBRARY = libcustep.so

# 可执行文件名
TARGET = gpu$(shell python3-config --extension-suffix)

# 默认目标
all: $(TARGET)

# 编译 CUDA 源文件
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $(LIBRARY)

# 编译 C++ 源文件
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# 链接 C++ 和 CUDA 目标文件以生成共享库
$(LIBRARY): $(OBJECTS)
	$(NVCC) -shared $(OBJECTS) -o $(LIBRARY) $(LDFLAGS)

# 链接 C++ 和共享库以生成可执行文件
$(TARGET): $(LIBRARY) $(C_SOURCES:.cpp=.o)
	$(CC) $(CFLAGS) $(C_SOURCES:.cpp=.o) -o $(TARGET) -L. -l:$(LIBRARY) $(LDFLAGS) $(shell python3 -m pybind11 --includes)

# 清理中间文件和可执行文件
clean:
	rm -f $(OBJECTS) $(LIBRARY) $(TARGET)
