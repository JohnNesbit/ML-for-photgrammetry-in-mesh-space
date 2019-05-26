@echo off
:: replace the paths below here are the path descriptions:
:: data_stuff is my dir that houses the phong clone, 
:: rendered images, run_seg2102 and the generated .off for training(not the saved benchmark ones)
E:
cd E:/PycharmProjects/3DMesh_Development/data_stuff
E:/blender/Blender.exe phong.blend --background --python phong.py -- ./idk_epoch.off ./tmp
