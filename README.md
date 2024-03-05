# FDNet
This an official Pytorch implementation of our paper ["Hyperspectral and LiDAR Classification via Frequency Domain-based Network"]().The specific details of the framework are as follows.
![FDNet](./figure/FDNet.png)
****
# Datasets
- [The Yancheng City dataset]()
includes hyperspectral images (HSI) and a LiDAR-based digital surface model (DSM). It was captured on October 15, 2022, over Yancheng City, China, at a flight altitude of 600 $m$. The HSI data was collected using a visible light imaging spectrometer from the Multi-parameter Airborne Remote Sensing System (MARSS), containing 249 spectral bands with a wavelength range from 400 to 1000 $nm$. The LiDAR data was obtained using the RIEGL VUX-240 laser scanner, with a single spectral band and an average point density of 51.44 $pts/m^{2}$. The dimensions of the HSI and LiDAR data are 400 × 500 pixels, with a spatial resolution of 1 $m$. The dataset comprises 9 different land-cover categories, totaling 200,000 real samples.
- [The Houston2013 dataset](https://hyperspectral.ee.uh.edu/?page_id=459)
includes a hyperspectral image (HSI) and a LiDAR-based digital surface model (DSM), collected by the National Center for Airborne Laser Mapping (NCALM) using the ITRES CASI-1500 sensor over the University of Houston campus in June 2012. The HSI comprise 144 spectral bands covering a wavelength range from 0.38 to 1.05 $\mu m$ while LiDAR data are provided for a single band. Both the HSI and LiDAR data share dimensions of 349 × 1905 pixels with a spatial resolution of 2.5 $m$. The dataset contains 15 categories, with a total of 15,029 real samples available.
- [The Augsburg dataset](https://github.com/danfenghong/ISPRS_S2FL?tab=readme-ov-file)
was captured over Augsburg, Germany. HSI data were gathered using the DAS-EOC HySpex sensor, while LiDAR-based DSM data were acquired through the DLR-3K system. To facilitate multimodal fusion, both images were down-sampled to a uniform resolution of 30 $m$. This dataset contains HSI data with 180 bands ranging from 0.4 to 2.5 $\mu m$ and DSM data in a single band. With dimensions of 332 × 485 pixels, the dataset represents 7 distinct land-cover categories.
- [The MUUFL dataset](https://github.com/GatorSense/MUUFLGulfport)
was acquired in November 2010 over the area of the campus of University of Southern Mississippi Gulf Park, Long Beach Mississippi, USA. The HSI data was gathered using the ITRES Research Limited (ITRES) Compact Airborne Spectral Imager (CASI-1500) sensor, initially comprising 72 bands. Due to excessive noise, the first and last eight spectral bands were removed, resulting in a total of 64 available spectral channels ranging from 0.38 to 1.05 $\mu m$. LiDAR data was captured by an ALTM sensor, containing two rasters with a wavelength of 1.06 $\mu m$. The dataset consists of 53,687 groundtruth pixels, encompassing 11 different land-cover classes.
****
# Train FDNet
``` 
python demo.py
``` 
****
# Results
All the results presented here are referenced from the original paper.
| Dataset | OA (%) | AA (%) | Kappa (%) |
| :----: |:------:|:------:|:---------:|
| Yancheng City  | 76.65  | 83.15  |   71.03   |
| Houston2013  | 95.91  | 96.13  |   95.58   |
| Augsburg  | 82.30  | 74.74  |   76.00   |
| MUUFL  | 88.12  | 89.31  |   84.58   |
****
# Citation
If you find this paper useful, please cite:
``` 
@ARTICLE{,
  author={},
  journal={}, 
  title={Hyperspectral and LiDAR Classification via Frequency Domain-based Network}, 
  year={},
  volume={},
  number={},
  pages={},
  doi={}
}
```
****
# Contact
Duo Wang: [njwangduo@163.com](njwangduo@163.com)
