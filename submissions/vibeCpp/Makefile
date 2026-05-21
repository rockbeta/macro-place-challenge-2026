CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -pthread

.PHONY: all clean

all: build/vibe_placer_full build/vibe_placer_fast

build/vibe_placer_full: placer.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

build/vibe_placer_fast: placerFast.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf build
