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
            <td style="border:2px">Sella</td>
            <td style="border:2px">L1 Name</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
        </tr>
        <tr>
            <td>Nasion</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
            <td>L1 Name</td>
        </tr>
    </tbody>
    
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
