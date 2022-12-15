headers = $(wildcard src/*.cuh)
objects = $(patsubst src/%.cu, build/%.o, $(wildcard src/*.cu))
tests = $(patsubst test/%.cu, build/test/%.o, $(wildcard test/*.cu))

all: bin/main

bin/main: bin build $(objects)
	nvcc -g -G -o bin/main $(objects)

test: bin bin/main build/test $(tests)
	nvcc -g -G -o bin/test $(filter-out build/main.o, $(objects)) $(tests)

bin:
	mkdir -p bin

build:
	mkdir -p build

build/test:
	mkdir -p build/test

build/%.o: src/%.cu
	nvcc -g -G -dc -c -o $@ $<

build/test/%.o: test/%.cu
	nvcc -g -G -dc -c -I src -o $@ $<

src/%.cu: $(headers)

test/%.cu: $(headers)

clean:
	rm -rf bin build
