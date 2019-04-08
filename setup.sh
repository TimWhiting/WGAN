curl -sSL "https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.0-linux-x86_64.tar.gz" -o julia.tar.gz
tar -xzf julia.tar.gz -C /usr --strip-components 1
rm -rf julia.tar.gz*
julia -e 'using Pkg; pkg"add IJulia; add CuArrays; add Flux; precompile"'
git pull
#git clone https://github.com/ndrplz/small_norb
#os.chdir('/content/WGAN/small_norb/smallnorb')
#wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
#wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
#wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
#wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
#wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
#wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
#gunzip smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
#gunzip smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
#gunzip smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
#gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
#gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
#gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
cd small_norb

#os.chdir('/content/WGAN/small_norb')
python3 main.py
julia -e 'using Pkg; pkg"add BSON; add Images; add ImageMagick; add Plots; precompile"'
julia -e 'using Pkg; pkg"add FileIO; add Juno; add NNlib; precompile"'
cd ..
#os.chdir('/content/WGAN')
#!julia main.jl
