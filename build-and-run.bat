set /p "file=Filename (no .extension): "
nvcc %file%.cu -Xcompiler /wd4819 -o %file%
%file%.exe