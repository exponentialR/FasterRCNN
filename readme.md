A simple script to train and evaluate train and evaluate faster rcnn model. In addition to this, it exposes detection and loss during training, especially useful for those who would want to use extracted features (`1024`) for downstream task, such as 6DoF or the likes.

- To train use below:
    ```bash
     python -m scripts.train_faster_rcnn --data_root /path/to/your_root_data/ --bs 8 --num_workers 11 --num_classes 6
    ```
- To generate symmetry for pose estimation
  ```bash
  python -m  scripts.gen_symmetry_json --objs ./meshes  --out canonical_files/symmetry.json
  ```
 
