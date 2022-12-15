headers = $(wildcard src/*.cuh)
sources = $(wildcard src/*.cu)
objects = $(patsubst src/%.cu, bin/%.o, $(sources))

all: bin bin/main

bin:
	mkdir -p bin

bin/main: bin $(objects)
	nvcc -o bin/main $(objects)

bin/%.o: src/%.cu
	nvcc -dc -c -o $@ $<

src/%.cu: $(headers)

clean:
	rm -f bin/*
