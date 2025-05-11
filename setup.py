from setuptools import setup, find_packages

setup(
    name="ecommerse-bot",
    version="0.0.1",
    author="Sowmya AM",
    author_email="sowmya.anekonda@gmail.com",
    packages=find_packages(), # find_packages automatically detect with init file
    install_requires=["langchain-astradb", "langchain"]
)