
Steps to run the [demo-lale.jl](./demo/old/demo-lale.jl) which is found in the `demo` subdirectory
1. Download Julia (https://julialang.org/downloads/)
  - highly recommend to get the julia-v1.6.0(March 24, 2021)
  - it will be released soon and has great features and faster compiler
  - on Linux (including WSL = Windows Subsystem for Linux):
    ```shell
    > wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.0-rc3-linux-x86_64.tar.gz
    > tar xvzf julia-1.6.0-rc3-linux-x86_64.tar.gz
    > sudo cp -r julia-1.6.0-rc3 /opt/.
    > sudo ln -s /opt/julia-1.6.0-rc3/bin/julia /usr/local/bin/julia
    ```
  - on Mac, after installing the image into the Applications folder:
    ```shell
    > sudo ln -s /Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
    ```

2. At Julia prompt press ; to trigger Julia shell:
```julia
julia> ;
shell> cd /path/to/Lale.jl
```
Note: Press __backspace__ to return to Julia prompt:

At Julia prompt:
```julia
julia> using Pkg
julia> Pkg.activate(".")   # activate the current environment
julia> Pkg.instantiate()   # install all the deps needed to run the package
julia> Pkg.update()        # get the latest versions of deps  
julia> Pkg.status()        # show all deps versions and status
```

You can also use the built-in package management shell
to do similar steps above:
```julia
julia> ]                   # trigger pkg shell
pkg> activate .            # activate the current environment
Lale> instantiate          # install all the deps needed to run the package
Lale> update               # get the latest versions of deps  
Lale> status               # show all deps versions and status

```
Note: Press __backspace__ to return to Julia prompt.

Remark: This will install the necessary deps by reading the `Project.toml`. 
`Pkg.status()` should list `Conda`, `Pandas`, `PyCall` which are packages needed for the
demos. If they are not listed, you can install them manually. Always
activate the current path containing the package you are working on which
is equivalent to virtual environment in python. Any external package you install is
locally resolved in the current package environment you are working and will not clash with other
packages you are working on.

You can install a package such as `PyCall` by:
```julia
julia> using Pkg
julia> Pkg.add("PyCall")
```
or use IJulia Notebook
```julia
julia> using Pkg
julia> Pkg.add("IJulia")
julia> using IJulia
julia> notebook(dir=pwd())
```

You can also install kernels to run Julia with multithreading:
```julia
julia> using IJulia
julia> installkernel("Julia (4 threads)", env=Dict("JULIA_NUM_THREADS"=>"4"))
```
More info can be found in IJulia [documentation](https://julialang.github.io/IJulia.jl/stable/manual/installation/).


You can also use Pluto Notebook:
```julia
julia> using Pkg
julia> Pkg.add("Pluto")
julia> using Pluto
julia> Pluto.run()
```

Remark: Julia `REPL` support tab completion.
Pressing `]` will switch to package management
shell which allows you to install, query, manage, etc
packages. Backspace returns to Julia shell.

3. Point `PyCall` to your python binary location:
```julia
julia> ENV["PYTHON"] = "/pathToPython/"
julia> Pkg.build("PyCall")
```
Remark: This will make sure it will use the correct python. You can
also set `ENV["PYTHON"]=""` and `PyCall` will install `python` using `Conda.jl`.
The location of `python` and `pip3` is in  `$HOME/.julia/conda/3/bin/`. You can
add more python packages by using `pip3` in the said directory.

4. Install python lale package
```shell
> cd /path/to/python/binary
> ./pip3 install lale
```
or
```shell
/path/to/python/bin/pip3 install lale
```

Remark: This will make sure `lale` is available during `PyCall.pyimport("lale")` in Julia

5. You can now try the julia `lale` demos in `demo/` subdirectory.

6. You can load the Lale.jl package and try the `demo-lale-package.jl` in `demo/` subdirectory

7. You can trigger unit-testing with:
```julia
julia> using Pkg 
julia> Pkg.activate(".") # activate current directory with Lale package
julia> Pkg.test()
```
or
```julia
julia> ]                 # trigger julia package shell
pkg> activate .          # activate lale package environment
Lale> test               # run Lale unit-test
```

Note: Press __backspace__ to return to Julia prompt.
Note: Instead of `]`, you can also do `Pkg.action("arg")`

8. To run the tests or samples from the CLI (without REPL), do this:
  ```
  julia --project -e 'using Pkg; Pkg.activate("."); Pkg.test()'
  julia --project demo/demo-lale.jl
  ```

9. Run Julia with the following argument to treat the current directory as a project:
```shell
> pwd
> path/to/Lale
> julia --project
julia> ]
Lale>      # indicates that Lale is the active project
```
