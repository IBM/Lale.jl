import PyCall: pyimport

function installmaclinux()
   # See https://stackoverflow.com/questions/12332975/installing-python-module-within-code.
   PIP_PACKAGES = ["lale"]
   try
      pyimport("lale")
      @info "lale succesfully installed"
   catch
      try
         sys = pyimport("sys")
         subprocess = pyimport("subprocess")
         subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "--force-reinstall", PIP_PACKAGES...])
         @info "lale succesfully installed"
      catch
         println("scikit-learn failed to install")
      end
   end
end

function installwindows()
   nothing
end

if Sys.iswindows()
   installwindows()
else
   installmaclinux()
end
