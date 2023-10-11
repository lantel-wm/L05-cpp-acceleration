CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++11 -fPIC
LDLIBS := $(shell python3-config --ldflags)
INCLUDES := $(shell python3 -m pybind11 --includes)

TARGET := cpu_parallel$(shell python3-config --extension-suffix)
SRC := cpu_parallel.cpp
CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++11 -fPIC
INCLUDES := $(shell python3 -m pybind11 --includes)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET)
