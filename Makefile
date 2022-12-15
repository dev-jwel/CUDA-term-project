headers = $(wildcard src/*.cuh)
sources = $(wildcard src/*.cu)
objects = $(patsubst src/%.cu, bin/%.o, $(sources))

all: bin bin/main

bin:
	mkdir -p bin

bin/main: bin $(objects)
	nvcc -g -G -o bin/main $(objects)

bin/%.o: src/%.cu
	nvcc -g -G -dc -c -o $@ $<

src/%.cu: $(headers)

clean:
	rm -f bin/*
