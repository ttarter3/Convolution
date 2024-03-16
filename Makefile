# . /data001/heterogene_mw/spack/share/spack/setup-env.sh -> ~/.bashrc
# ~/.bashrc

# spack load cuda@12.3


.PHONY: clean run cgen build

run_pc: 
	@./.runJob.sh pc
run_sc:
	@./.runJob.sh sc

pgen:
	@mkdir -p ./install
	module load cuda && \
	module load lapack && \
	cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --build build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --install build

sgen:
	@mkdir -p ./install
	cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --build build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --install build


build:
	@mkdir -p ./install
	module load cuda && \
	module load lapack && \
	cmake --build build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --install build

this:
	@mkdir -p ./install
	module load cuda && \
	module load lapack && \
	cmake --install build


clean:
	rm -rf *.qsub_out
	cmake --build build --target clean

