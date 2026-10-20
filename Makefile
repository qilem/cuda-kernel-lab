NVCC  ?= nvcc
ARCH  ?= native
FLAGS  = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -I.

SRCS = $(wildcard *.cu)
BINS = $(patsubst %.cu,bin/%,$(SRCS))

all: $(BINS)

bin/%: %.cu common/bench.cuh | bin
	$(NVCC) $(FLAGS) $< -o $@ -lcublas

bin:
	mkdir -p bin

run: all
	@for b in $(BINS); do echo "########## $$b"; ./$$b || exit 1; done

racecheck: bin/08_reduction_race
	compute-sanitizer --tool racecheck ./bin/08_reduction_race 1

clean:
	rm -rf bin

.PHONY: all run racecheck clean
