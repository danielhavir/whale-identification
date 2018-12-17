from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
      name='index_splitter',
      author="danielhavir",
      ext_modules=[
            CppExtension(
                  'index_splitter',
                  ['index_splitter.cpp'],
            )
      ],
      cmdclass={'build_ext': BuildExtension}
)
