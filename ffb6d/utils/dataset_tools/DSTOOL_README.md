# Dataset Tools

## Installation
- Build [raster_triangle](https://github.com/ethnhe/raster_triangle.git) for RGBD image rendering:
  ```shell
  git clone https://github.com/ethnhe/raster_triangle.git
  cd raster_triangle
  sh rastertriangle_so.sh
  cd ..
  ```
- Compile the FPS scripts:
  ```shell
  cd fps/
  python3 setup.py build_ext --inplace
  cd ..
  ```
- Install python3 requirement by:
  ```shell
  pip3 install -r requirement.txt
  ```


## Usage
- Generate information of objects, eg. radius, 3D keypoints, etc. by:
  ```
  python3 gen_obj_info.py --help
  ```
  
  If you use ply model and the vertex color is contained in the ply model, you can use the default raster triangle for rendering. For example, you can generate the information of the example ape object by running:
  ```shell
  python3 gen_obj_info.py --obj_name='ape' --obj_pth='example_mesh/ape.ply' --scale2m=1000. --sv_fd='ape_info'
  ```
  You need to set the parameter ```scale2m``` according to the original unit of you object so that the generated info are all in unit meter.

   If you use obj model, you can convert each vertex in meter and use pyrender. For example, you can generate the information of the example cracker box by running:
  ```shell
  python3 gen_obj_info.py --obj_name='cracker_box' --obj_pth='example_mesh/003_cracker_box/textured.obj' --scale2m=1. --sv_fd='cracker_box_info' --use_pyrender
  ```



