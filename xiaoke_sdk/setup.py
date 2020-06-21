import setuptools
with open('READEME.md','r') as f:
    long_description = f.read()


setuptools.setup(
    name = 'RedHouse',
    version = '0.0.1',
    author = 'huxiaoke',
    author_email = '936214756@qq.com',
    description = 'This is redhouse project',
    long_description = long_description,
    long_description_content_type = 'text',
    url = '',
    packages = setuptools.find_packages(),
    classifiers = [
        'programming Language :: Python ::3',
        'License::OSI App::MIT License',
        'Operating System::OS Independent',
    ],
    python_requires='>=3.5'
)