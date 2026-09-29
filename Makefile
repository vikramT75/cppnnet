export PKG_CONFIG_PATH := ../raylib/lib/pkgconfig/

CXX    = g++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -Wno-missing-field-initializers -fopenmp \
           $(shell pkg-config --cflags raylib)
LIBS   = $(shell pkg-config --libs raylib) -lm -lX11 -ldl -pthread -fopenmp

# Default epochs and reset flag for training
EPOCHS        ?= 100
RESET_WEIGHTS ?= 0

BUILD_DIR = build

TARGETS = $(BUILD_DIR)/classifier_train \
          $(BUILD_DIR)/classifier_infer \
          $(BUILD_DIR)/mnist_gen \
          $(BUILD_DIR)/shape_gen \
          $(BUILD_DIR)/morph2nn \
          $(BUILD_DIR)/upscalenn \
          $(BUILD_DIR)/adder \
          $(BUILD_DIR)/xor

.PHONY: all clean

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/classifier_train: demos/classifier_train.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

$(BUILD_DIR)/classifier_infer: demos/classifier_infer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/mnist_gen: demos/mnist_gen.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

$(BUILD_DIR)/shape_gen: demos/shape_gen.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

$(BUILD_DIR)/morph2nn: demos/morph2nn.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/upscalenn: demos/upscalenn.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/adder: demos/adder.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/xor: demos/xor.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf $(BUILD_DIR)/

.PHONY: data data-mnist data-shapes train-mnist train-shapes infer-mnist infer-shapes upscale morph xor adder

# Generate all datasets
data: data-mnist data-shapes

data-mnist: $(BUILD_DIR)/mnist_gen
	./$(BUILD_DIR)/mnist_gen demos/data/mnist_train_manifest.txt demos/data/mnist_train.mat
	./$(BUILD_DIR)/mnist_gen demos/data/mnist_test_manifest.txt  demos/data/mnist_test.mat

data-shapes: $(BUILD_DIR)/shape_gen
	./$(BUILD_DIR)/shape_gen

xor: $(BUILD_DIR)/xor
	./$(BUILD_DIR)/xor

adder: $(BUILD_DIR)/adder
	./$(BUILD_DIR)/adder

upscale: $(BUILD_DIR)/upscalenn
	./$(BUILD_DIR)/upscalenn ./mnist/training0/3.png

morph: $(BUILD_DIR)/morph2nn
	./$(BUILD_DIR)/morph2nn ./mnist/training0/3.png ./mnist/training0/7.png

# parameterised usage: make train-mnist EPOCHS=50 RESET_WEIGHTS=0
train-mnist: $(BUILD_DIR)/classifier_train
	./$(BUILD_DIR)/classifier_train demos/models/mnist.arch demos/data/mnist_train.mat demos/data/mnist_test.mat $(EPOCHS) $(RESET_WEIGHTS)

# parameterised usage: make train-shapes EPOCHS=50 RESET_WEIGHTS=0
train-shapes: $(BUILD_DIR)/classifier_train
	./$(BUILD_DIR)/classifier_train demos/models/shapes.arch demos/data/shapes_train.mat demos/data/shapes_test.mat $(EPOCHS) $(RESET_WEIGHTS)

infer-mnist: $(BUILD_DIR)/classifier_infer
	./$(BUILD_DIR)/classifier_infer demos/models/mnist.arch demos/models/mnist.arch.weights.mat demos/models/mnist.labels

infer-shapes: $(BUILD_DIR)/classifier_infer
	./$(BUILD_DIR)/classifier_infer demos/models/shapes.arch demos/models/shapes.arch.weights.mat demos/models/shapes.labels
