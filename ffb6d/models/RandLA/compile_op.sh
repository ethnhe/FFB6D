cd utils/nearest_neighbors
python3 setup.py install --home="."
cd ../../

cd utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../../
