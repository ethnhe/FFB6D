# SIFT-FPS 3D Keypoints Extractors

## Installation
- Build [raster_triangle](https://github.com/ethnhe/raster_triangle.git) for RGBD image rendering:
```
git clone https://github.com/ethnhe/raster_triangle.git
cd raster_triangle
sh rastertriangle_so.sh
```
- Compile the FPS scripts:
```
cd fps/
python3 setup.py build_ext --inplace
```
- Install python3 requirement by:
```
pip3 install -r requirement.txt
```

## Usage
- Generate information of objects, eg. radius, 3D keypoints, etc. by:
  ```
  python3 gen_obj_info.py --help
  ```
  
  For example, you can generate the information of the example ape object by running:
  ```
  python3 gen_obj_info.py --obj_name='ape' --ply_pth='example_mesh/ape.ply' --scale2m=1000. --sv_fd='ape_info'
  ```
  You need to set the parameter ```scale2m``` according to the original unit of you object so that the generated info are all in unit meter.

