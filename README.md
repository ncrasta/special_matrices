

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white) ![ROS](https://img.shields.io/badge/ros-%230A0FF9.svg?style=for-the-badge&logo=ros&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) ![Docker](https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) ![NVIDIA](https://img.shields.io/badge/NVIDIA-JETSON-XAVIER?style=for-the-badge&logo=nvidia&logoColor=green)


[![CI build](https://github.com/crasta/matrix_generation_checking/actions/workflows/python-app.yml/badge.svg)](https://github.com/crasta/matrix_generation_checking/actions/workflows/python-app.yml) [![CodeQL](https://github.com/ossf/scorecard-action/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/ossf/scorecard-action/actions/workflows/codeql-analysis.yml) [![Scorecards supply-chain security](https://github.com/crasta/matrix_generation_checking/actions/workflows/scorecards-analysis.yml/badge.svg)](https://github.com/crasta/matrix_generation_checking/actions/workflows/scorecards-analysis.yml) [![GitHub branches](https://badgen.net/github/branches/Naereen/Strapdown.js)](https://github.com/Naereen/Strapdown.js/)




# Generation and checking of special types of matrices

This class is to generate and check some special types of matrices. 

The MatrixCheck class contains methods to check if a given matrix is a of particular type. For example, if a 3 by 3 matrix R is a rotational matrix or not, i.e R'R=I and det(R)=1.

The MatrixGeneration class contains methods to generate matrices of special types. For example, generating a 3 by 3 matrix random rotation matrix.
