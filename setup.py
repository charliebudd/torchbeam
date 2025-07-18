from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for extension in self.extensions:
            self.convert_include_dirs_to_system(extension)
        super().build_extensions()

    def convert_include_dirs_to_system(self, extension):
        for include_dir in extension.include_dirs:
            extension.extra_compile_args["cxx"].append(f"-isystem{include_dir}")
            extension.extra_compile_args["nvcc"].extend(["-isystem", include_dir])
        extension.include_dirs = []

setup(
    cmdclass={"build_ext": CustomBuildExtension},
    ext_modules=[
        CUDAExtension(
            name="torchbeam_ext",
            sources=glob("src/torchbeam/_ext/**/*.cpp", recursive=True) + glob("src/torchbeam/_ext/**/*.cu", recursive=True),
            include_dirs=[
                "/usr/include/aravis-0.8",
                "/usr/include/libxml2",
                "/usr/include/libusb-1.0",
                "/usr/include/libmount",
                "/usr/include/blkid",
                "/usr/include/glib-2.0",
                "/usr/lib/x86_64-linux-gnu/glib-2.0/include",
            ],
            libraries=[
                "usb-1.0",
                "aravis-0.8",
                "GL",
                "avcodec",
                "avformat",
                "avutil",
            ],
            extra_compile_args={
                "cxx": ["-Wall", "-g0", "-O3", "-fdiagnostics-color=always"],
                "nvcc": ["-O3"],
            },
        )
    ],
)
