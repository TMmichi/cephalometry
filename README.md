# Cephalometric Landmark Detection via Bayesian CNN

The code implementation of the paper [`Automated cephalometric landmark detection with confidence regions using Bayesian convolutional neural networks`](https://link.springer.com/article/10.1186/s12903-020-01256-7).

<!-- ![final-result](/assets/cephalometry.png) -->
<p align='middle'>
  <img src='/assets/landmarks.jpg' width='53%' />
  <img src='/assets/cephalometry.png' width='40%' />
<p/>

---

## Goals of the project

The goal of this project is to locate predefined landmarks from 2D lateral cephalogram (side view X-ray of a skull), where these landmarks are utilized to diagnosis and treatment planning via various orthodontic and facial morphometric analyses. Along with achieving accruate tracing results, letting users gain access on how certain the predictions of the model are is also within the scope of this project.

## Algorithm

![algorithm](/assets/algo3.png)
The framework is mainly divided into two screening procedures: `Low Resolution Screening (LRS)` and `High Resolution Screening (HRS)`. We have 20 landmarks to trace, and for each landmark, both of these procedure should be done.

### Low Resolution Screening
The objective of LRS is to create an ROI of the target landmark from the input lateral cephalogram.

&emsp;a) The original lateral cephalogram gets downsampled by a factor of 3.

&emsp;b) From the downsampled lat ceph, image batches are sampled with a stride of 3 mm along the width and the height direction from all over the lat ceph.

&emsp;c) From the LRS calculation, CNN model provides a region of interest for the target landmark to be located in.

### High Resolution Screening

The objective of HRS is to estimate the exact location of the landmark with the uncertainty from the ROI from LRS.

&emsp;d) Every single pixel from the ROI is, again, sampled as an image batch to be put into Bayesian CNN(B-CNN) model for iterative calculations.

&emsp;e) HRS provides the final predicted target position for the target landmark.

## Result

![result](/assets/full_shot2.png)

<table>
  <thead>
    <tr>
      <th rowspan=2> </th>
      <th colspan=2>Error (mm)</th>
      <th colspan=4>SDR (%)</th>
    </tr>
    <tr>
      <th> Mean</th>
      <th> SD</th>
      <th> 2mm</th>
      <th> 2.5mm</th>
      <th> 3mm</th>
      <th> 4mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sella</td>
      <td>0.86</td>
      <td>1.92</td>
      <td>96.67</td>
      <td>97.33</td>
      <td>98.00</td>
      <td>98.00</td>
    </tr>
    <tr>
      <td> Nasion </td>
      <td> 1.28 </td>
      <td> 1.03 </td>
      <td> 81.33 </td>
      <td> 86.00 </td>
      <td> 90.00 </td>
      <td> 96.67 </td>
    </tr>
    <tr>
      <td> Orbitale </td>
      <td> 2.11 </td>
      <td> 2.77 </td>
      <td> 77.33 </td>
      <td> 87.33 </td>
      <td> 94.00 </td>
      <td> 96.67 </td>
    </tr>
    <tr>
      <td> Porion </td>
      <td> 1.89 </td>
      <td> 1.67 </td>
      <td> 58.00 </td>
      <td> 66.00 </td>
      <td> 72.67 </td>
      <td> 86.67 </td>
    </tr>
    <tr>
      <td> A-point </td>
      <td> 2.07 </td>
      <td> 2.53 </td>
      <td> 52.00 </td>
      <td> 62.00 </td>
      <td> 74.00 </td>
      <td> 87.33 </td>
    </tr>
    <tr>
      <td> B-point </td>
      <td> 2.08 </td>
      <td> 1.77 </td>
      <td> 79.33 </td>
      <td> 88.67 </td>
      <td> 93.33 </td>
      <td> 96.67 </td>
    </tr>
    <tr>
      <td> Pogonion </td>
      <td> 1.17 </td>
      <td> 0.81 </td>
      <td> 82.67 </td>
      <td> 90.67 </td>
      <td> 96.00 </td>
      <td> 100.00 </td>
    </tr>
    <tr>
      <td> Menton </td>
      <td> 1.11 </td>
      <td> 2.82 </td>
      <td> 95.33 </td>
      <td> 97.33 </td>
      <td> 98.00 </td>
      <td> 98.67 </td>
    </tr>
    <tr>
      <td> Gnathion </td>
      <td> 0.97 </td>
      <td> 0.56 </td>
      <td> 92.00 </td>
      <td> 97.33 </td>
      <td> 98.67 </td>
      <td> 98.67 </td>
    </tr>
    <tr>
      <td> Gonion </td>
      <td> 2.39 </td>
      <td> 4.77 </td>
      <td> 63.33 </td>
      <td> 75.33 </td>
      <td> 85.33 </td>
      <td> 92.67 </td>
    </tr>
    <tr>
      <td> Lower incisal incision</td>
      <td> 1.35 </td>
      <td> 2.19 </td>
      <td> 84.00 </td>
      <td> 90.67 </td>
      <td> 93.33 </td>
      <td> 96.67 </td>
    </tr>
    <tr>
      <td> Upper  incisal incision</td>
      <td> 0.90 </td>
      <td> 0.75 </td>
      <td> 93.33 </td>
      <td> 97.33 </td>
      <td> 98.00 </td>
      <td> 99.33 </td>
    </tr>
    <tr>
      <td> Upper lip</td>
      <td> 1.32 </td>
      <td> 0.83 </td>
      <td> 96.67 </td>
      <td> 100.00 </td>
      <td> 100.00 </td>
      <td> 100.00 </td>
    </tr>
    <tr>
      <td> Lower lip</td>
      <td> 1.28 </td>
      <td> 0.85 </td>
      <td> 97.33 </td>
      <td> 98.67 </td>
      <td> 98.67 </td>
      <td> 99.33 </td>
    </tr>
    <tr>
      <td> Subnasale </td>
      <td> 1.22 </td>
      <td> 1.56 </td>
      <td> 84.00 </td>
      <td> 92.00 </td>
      <td> 95.33 </td>
      <td> 96.67 </td>
    </tr>
    <tr>
      <td> Soft tissue pogonion</td>
      <td> 2.62 </td>
      <td> 2.07 </td>
      <td> 82.67 </td>
      <td> 92.67 </td>
      <td> 95.33 </td>
      <td> 97.33 </td>
    </tr>
    <tr>
      <td> Posterior Nasal Spine</td>
      <td> 1.23 </td>
      <td> 0.91 </td>
      <td> 90.00 </td>
      <td> 94.00 </td>
      <td> 95.33 </td>
      <td> 98.00 </td>
    </tr>
    <tr>
      <td> Anterior Nasal Spine</td>
      <td> 1.52 </td>
      <td> 1.56 </td>
      <td> 78.67 </td>
      <td> 87.33 </td>
      <td> 90.67 </td>
      <td> 93.33 </td>
    </tr>
    <tr>
      <td> Articulare </td>
      <td> 1.70 </td>
      <td> 1.77 </td>
      <td> 75.33 </td>
      <td> 83.33 </td>
      <td> 86.67 </td>
      <td> 90.67 </td>
    </tr>
    <th>
      <tr>
        <td><b> Average </b></td>
        <td><b> 1.53 </b></td>
        <td><b> 1.74 </b></td>
        <td><b> 82.11 </b></td>
        <td><b> 88.63 </b></td>
        <td><b> 92.28 </b></td>
        <td><b> 95.96 </b></td>
      </tr>
    </th>
  </tbody>
  <caption><b>Overall performance of detecting landmarks</b></caption>
</table>

<table>
  <thead>
    <tr>
      <th rowspan=2 colspan=5> </th>
      <th colspan=3>Diagonal accuracy</th>
    </tr>
    <tr>
      <th> Proposed</th>
      <th> Lindner et al.</th>
      <th> Arik et al.</th>
    </tr>
  </thead>
  <tbody>
<tr>
	<td><b>ANB</b></td>
	<td> Pred 1</td>
	<td colspan=2> Pred 2</td>
	<td> Pred 3</td>
	<td rowspan=4> <b>80.72</b> </td>
	<td rowspan=4> 79.90 </td>
	<td rowspan=4> 77.31 </td>
</tr>
<tr>
	<td> True 1</td>
	<td> <b>65.75</b> </td>
	<td colspan=2> 10.96 </td>
	<td> 23.29 </td>
</tr>
<tr>
	<td> True 2</td>
	<td> 23.64 </td>
	<td colspan=2> <b>70.91</b> </td>
	<td> 5.45 </td>
</tr>
<tr>
	<td> True 3</td>
	<td> 4.96 </td>
	<td colspan=2> 0.83 </td>
	<td> <b>94.21</b> </td>
</tr>
<tr>
	<td> <b>SNB</b> </td>
	<td> Pred 1</td>
	<td colspan=2> Pred 2</td>
	<td> Pred 3</td>
	<td rowspan=4> <b>83.13</b> </td>
	<td rowspan=4> 78.80 </td>
	<td rowspan=4> 70.11 </td>
</tr>
<tr>
	<td> True 1</td>
	<td> <b>73.24</b> </td>
	<td colspan=2> 4.23 </td>
	<td> 22.54 </td>
</tr>
<tr>
	<td> True 2</td>
	<td> 38.46 </td>
	<td colspan=2> <b>58.97</b> </td>
	<td> 2.56 </td>
</tr>
<tr>
	<td> True 3</td>
	<td> 5.04 </td>
	<td colspan=2> 0.00 </td>
	<td> <b>94.96</b> </td>
</tr>
<tr>
	<td> <b>SNA</b> </td>
	<td> Pred 1</td>
	<td colspan=2> Pred 2</td>
	<td> Pred 3</td>
	<td rowspan=4> 72.69 </td>
	<td rowspan=4> <b>73.81</b> </td>
	<td rowspan=4> 66.72 </td>
</tr>
<tr>
	<td> True 1</td>
	<td> <b>67.62</b> </td>
	<td colspan=2> 16.19 </td>
	<td> 16.19 </td>
</tr>
<tr>
	<td> True 2</td>
	<td> 18.89 </td>
	<td colspan=2> <b>80.00</b> </td>
	<td> 1.11 </td>
</tr>
<tr>
	<td> True 3</td>
	<td> 25.93 </td>
	<td colspan=2> 3.70 </td>
	<td> <b>70.37</b> </td>
</tr>
<tr>
	<td> <b>ODI</b> </td>
	<td> Pred 1</td>
	<td colspan=2> Pred 2</td>
	<td> Pred 3</td>
	<td rowspan=4> 81.53 </td>
	<td rowspan=4> <b>81.75</b> </td>
	<td rowspan=4> 75.04 </td>
</tr>
<tr>
	<td> True 1</td>
	<td> <b>82.14</b>82.14 </td>
	<td colspan=2> 6.25 </td>
	<td> 11.61 </td>
</tr>
<tr>
	<td> True 2</td>
	<td> 33.33 </td>
	<td colspan=2> <b>66.67</b> </td>
	<td> 0.00 </td>
</tr>
<tr>
	<td> True 3</td>
	<td> 15.55 </td>
	<td colspan=2> 0.91 </td>
	<td> <b>85.55</b> </td>
</tr>
<tr>
	<td> <b>APDI</b> </td>
	<td> Pred 1</td>
	<td colspan=2> Pred 2</td>
	<td> Pred 3</td>
	<td rowspan=4> 84.34 </td>
	<td rowspan=4> <b>89.26</b> </td>
	<td rowspan=4> 87.18 </td>
</tr>
<tr>
	<td> True 1</td>
	<td> <b>82.14</b> </td>
	<td colspan=2> 6.25 </td>
	<td> 11.61 </td>
</tr>
<tr>
	<td> True 2</td>
	<td> 33.33 </td>
	<td colspan=2> <b>66.67</b> </td>
	<td> 0.00 </td>
</tr>
<tr>
	<td> True 3</td>
	<td> 15.55 </td>
	<td colspan=2> 0.91 </td>
	<td> <b>85.55</b> </td>
</tr>
<tr>
	<td> <b>FHI</b> </td>
	<td> Pred 1</td>
	<td colspan=2> Pred 2</td>
	<td> Pred 3</td>
	<td rowspan=4> <b>84.74</b> </td>
	<td rowspan=4> 63.51 </td>
	<td rowspan=4> 69.16 </td>
</tr>
<tr>
	<td> True 1</td>
	<td> <b>82.14</b> </td>
	<td colspan=2> 6.25 </td>
	<td> 11.61 </td>
</tr>
<tr>
	<td> True 2</td>
	<td> 33.33 </td>
	<td colspan=2> <b>66.67</b> </td>
	<td> 0.00 </td>
</tr>
<tr>
	<td> True 3</td>
	<td> 15.55 </td>
	<td colspan=2> 0.91 </td>
	<td> <b>85.55</b> </td>
</tr>
<tr>
	<td> <b>FMA</b> </td>
	<td> Pred 1</td>
	<td colspan=2> Pred 2</td>
	<td> Pred 3</td>
	<td rowspan=4> <b>81.93</b> </td>
	<td rowspan=4> 81.92 </td>
	<td rowspan=4> 78.01 </td>
</tr>
<tr>
	<td> True 1</td>
	<td> <b>82.14</b> </td>
	<td colspan=2> 6.25 </td>
	<td> 11.61 </td>
</tr>
<tr>
	<td> True 2</td>
	<td> 33.33 </td>
	<td colspan=2> <b>66.67</b> </td>
	<td> 0.00 </td>
</tr>
<tr>
	<td> True 3</td>
	<td> 15.55 </td>
	<td colspan=2> 0.91 </td>
	<td> <b>85.55</b> </td>
</tr>
<tr>
	<td> <b>MW</b> </td>
	<td> Pred 1</td>
	<td> Pred 3</td>
	<td> Pred 4</td>
	<td> Pred 5</td>
	<td rowspan=5> 80.32 </td>
	<td rowspan=5> 79.59 </td>
	<td rowspan=5> <b>81.31</b> </td>
</tr>
<tr>
	<td> True 1</td>
	<td> <b>75.00</b> </td>
	<td> 1.19 </td>
	<td> 17.86 </td>
	<td> 5.95 </td>
</tr>
<tr>
	<td> True 3</td>
	<td> 0.00 </td>
	<td> <b>89.76</b> </td>
	<td> 2.04 </td>
	<td> 8.16 </td>
</tr>
<tr>
	<td> True 4</td>
	<td> 13.89 </td>
	<td> 0.00 </td>
	<td> <b>86.11</b> </td>
	<td> 0.00 </td>
</tr>
<tr>
	<td> True 5</td>
	<td> 15.91 </td>
	<td> 13.64 </td>
	<td> 0.00 </td>
	<td> <b>70.45</b> </td>
</tr>
  </tbody>
  <caption> <b>Confusion matrix of orthodontic parameters for skeletal analysis and their comparison with others’ methods </b></caption>
</table>


## Citation

Please use the following bibtex for citations:

```
@article{lee2020automated,
  title={Automated cephalometric landmark detection with confidence regions using Bayesian convolutional neural networks},
  author={Lee, Jeong-Hoon and Yu, Hee-Jin and Kim, Min-ji and Kim, Jin-Woo and Choi, Jongeun},
  journal={BMC oral health},
  volume={20},
  number={1},
  pages={1--10},
  year={2020},
  publisher={Springer}
}
```
