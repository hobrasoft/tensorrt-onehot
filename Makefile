

CUDA_VER:= 10.2
CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++
NVCC:= /usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations -Wno-error=unused-variable
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include

LIBS:= -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas
LFLAGS:= --shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCFILES:= onehot.h onehotkernel.h
SRCFILES:= onehot.cpp onehotkernel.cu
TARGET_OBJS:= onehot.o onehotkernel.o
TARGET_LIB:= libonehot.so


all: $(TARGET_LIB)

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@ $(TARGET_OBJS) $(LFLAGS)

%.o: %.cpp $(INCFILES) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCFILES) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

clean:
	rm -rf $(TARGET_LIB) $(TARGET_OBJS)

