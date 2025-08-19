
# FaceFilter
This is a project that I create a Snapchat filter using pretrained [YOLO](https://drive.google.com/file/d/1Xznn5WgGgfekpXs-fK6kPKRKFrM1DJgI/view?usp=sharing) to detect faces, [Resnet50](https://drive.google.com/file/d/1yxivfwr0EiN-leoKSJL7B66XBMwD9orT/view?usp=sharing) to predict facial landmark with [dlib Dataset](https://drive.google.com/file/d/1WQ-M0UkCkBnXX2ctEFm_X_v-Dx9daHVz/view?usp=sharing).


## Badges  

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)  
![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)



## Run Locally  

Clone the project  

~~~bash  
  git clone https://github.com/chirikatori/FaceFilter.git
~~~

Train landmark model

~~~bash
  cd FaceFilter
  python train.py num_epochs=['your epoch'] ...
~~~

Run the filter

~~~bash  
python main.py
~~~

## Usage
After run `main.py` switch filter by pressing `F` or exit by pressing `ESC`

## Environment Variables  

To run this project, you will need to add the following environment variables to your config files in `config` folder
## Acknowledgements  

- [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
- [Awesome README](https://github.com/matiassingers/awesome-readme)
- [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
- [Create Snapchat filter](https://github.com/spmallick/learnopencv/tree/master/Create-AR-filters-using-Mediapipe)
## Feedback  

If you have any feedback, please reach out to us at thanhnd9904@gmail.com

## License  

[MIT](https://choosealicense.com/licenses/mit/)
