# Recommender System - by using PyTorch
- 대표적인 Recommender-system 모형을 PyTorch를 이용해 구현합니다
- Code: [Colab](https://colab.research.google.com/?utm_source=scs-index) 에서 작성되었습니다
- Data: [MovieLens 100K](https://www.kaggle.com/prajitdatta/movielens-100k-dataset), [Online Advertising](https://d2l.ai/chapter_recommender-systems/ctr.html)

## Matrix Completion on Explicit Feedback

#### Matrix Factorization
- [Original Paper](https://ieeexplore.ieee.org/abstract/document/5197422?casa_token=yU_a_jQk3FoAAAAA:1c80Grtze6IyXKSz81M4-znwmGs13dojvr6OhAqIcvkiWIYch2wD3Wu4wcJaE65agQgd9oe-)
  - Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
- [Notebook](https://colab.research.google.com/drive/1bFejhfvL_hAvyvJSAo2GuOwTYIdGO6pR?usp=sharing)
#### SVD++ 
- [Original Paper](https://dl.acm.org/doi/abs/10.1145/1401890.1401944?casa_token=tZHDSBhztHEAAAAA:lkb_CQw_VKPJ8TIFmPc8Y7YDACAqltEn6guZzcpblnISX0vEiYIgBj3ynrTTgo_nJ0wl2XG8nHpk)
  - Koren, Y. (2008, August). Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 426-434). 
- [Notebook](https://colab.research.google.com/drive/1bPy85bDJKUt_BdbK5N3TwOYvYiyZvE3w?usp=sharing) 
#### SVD++ with Temporal Dynamics 
- [Original Paper](https://dl.acm.org/doi/abs/10.1145/1557019.1557072?casa_token=IEQ4ql25LJkAAAAA:YZdpt465lHuQoDn0aPUY6r6mb66oerR5WpyLxxl8b7_56FLZz1NZZGJTHYsiF0x-LC_i2Lpz2570)
  - Koren, Y. (2009, June). Collaborative filtering with temporal dynamics. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 447-456).
    Chicago
- [Notebook](https://colab.research.google.com/drive/1O0IqiSfG_KBWt3eQ_09_YcL-8kKCXnTN?usp=sharing)
#### AutoRec 
- [Original Paper](https://dl.acm.org/doi/abs/10.1145/2740908.2742726?casa_token=-rsq4DNjwtMAAAAA:zAlU4S0GAAgtJedHACqn2_C5o5iMa4dpJ7d1EHaQF-fOoUhdSBgoycFw3p6YsiofMNJQ6H0mH_qE)
  - Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015, May). Autorec: Autoencoders meet collaborative filtering. In Proceedings of the 24th international conference on World Wide Web (pp. 111-112).
    Chicago
- [Notebook](https://colab.research.google.com/drive/1r_50WEsS2s3DGbPW-4HQoUnXZd4KMyHB?usp=sharing)
#### NeuralCF
- [Original Paper](https://dl.acm.org/doi/abs/10.1145/3038912.3052569?casa_token=3GcImCEhOs4AAAAA:j_iBG70sZt9BcZnUkzhUBeA2whcjXSDQ7I2IY0K0ITtcnsfMBnxBTW0f210OotYghSDsYWKPUgAD)
  - He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    Chicago
- [Notebook](https://colab.research.google.com/drive/1OFQ_yWmjNZScot-qxvkP-cYGI9sHHFtt?usp=sharing)


## Personalized Ranking Prediction on Implicit Feedback

#### NeuralCF
- [Original Paper](https://dl.acm.org/doi/abs/10.1145/3038912.3052569?casa_token=3GcImCEhOs4AAAAA:j_iBG70sZt9BcZnUkzhUBeA2whcjXSDQ7I2IY0K0ITtcnsfMBnxBTW0f210OotYghSDsYWKPUgAD) 
  - He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    Chicago
- [Notebook](https://colab.research.google.com/drive/1cYMOXcBmzL5wamnJfE6-3H-XoI-1NcQk?usp=sharing)

#### Caser 
- [Original Paper](https://dl.acm.org/doi/abs/10.1145/3159652.3159656?casa_token=_hkVJ2pf35QAAAAA:UAI6ecH9FzUj6Z-HL4orIZyYMUF1zFN7UxDI5edgCog2eb7OxzEF5NEeJ8BFS6H1RAO9eBX7LqaE2V0) 
  - Tang, J., & Wang, K. (2018, February). Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining (pp. 565-573).
    Chicago
- [Notebook](https://colab.research.google.com/drive/12A645NnzWCwLYaYJlFQL6pAwUa7pWj22?usp=sharing)


## Click Through Rate (CTR) Prediction on feature-rich interaction data

#### Factorization Machines
- [Original Paper](https://ieeexplore.ieee.org/abstract/document/5694074/?casa_token=XfxWteIAUtYAAAAA:UlFIuG28xBkJG3TkZblX3rvcYfolq4wgkReklygGyhEq4fFD_ov8dyRLydDyvnWRdpkeLZYE) 
  - Rendle, S. (2010, December). Factorization machines. In 2010 IEEE International conference on data mining (pp. 995-1000). IEEE.
    Chicago
- [Notebook](https://colab.research.google.com/drive/1ld-5bX_8UZOj6l_LpXlZMan_G8NQx_5e?usp=sharing)

#### DeepFM 
- [Original Paper](https://arxiv.org/abs/1703.04247)
  - Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247.
    Chicago
- [Notebook](https://colab.research.google.com/drive/1ctt8Vak0Uw_Nz7Ksj9RUcNvKts2qPaN3?usp=sharing)
