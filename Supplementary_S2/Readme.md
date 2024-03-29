## Comparison of Lung Nodule Detection
A comparison of lung nodule detection accuracy in the *LNHG*, *U-net*, *NoduleNet*, and *nnU-net* has been conducted. Precision is defined as *P = TP/(TP + FP)*, where *TP* and *FP* are the number of true and false positives, respectively. The lung nodule candidates generated on the chest, which can be visually recognized by the radiologists from the images, were used to calculate *TP* and *FP*. Recall is defined as *R = TP/(TP + FN)*, where *FN* is the number of false negatives. The harmonic means of precision and recall, the f1 measurement, is defined as *f1 = (2·P·R)/(P+R)*. In this study, the lung nodule images produced by the models on the external test dataset were used for evaluation using *P*, *R*, and *f1* measurement.  All the lung nodule candidates on the output images were from the visual inspections done by local radiologist.

**Table S1.**  Statistics of *TP*, *FP*, *FN*, *P*, *R*, and *f1* on the external test dataset. The statistics of lung nodules detection were calculated according to the nodule candidates artificially determined by radiologists.

|  | *TP* | *FP* | *FN* | *P* | *R* | *f1* |
| --- | --- | --- | --- | --- | --- | --- |
| *LNHG* | 422 | 75 | 8 | 85.0% | 98.1% | 91.0% |
| *U-net* | 412 | 204 | 18 | 66.9% | 95.8% | 78.8% |
| *NoduleNet* | 418 | 83 | 12 | 83.4%	| 97.0% | 89.6% |
| *nnU-net* | 401 | 155 | 29 | 72.1% | 93.3% | 81.3% |
| *L-RF* | 410 | 363 | 363 | 53.0% | 95.3% | 68.1% |

In addition, the Faster R-CNN was further tested on the *LUNA16* dataset. A total of 888 CT scans was obtained from the *subset0* to *subset9* in the *LUNA16* dataset. Original CT images were input into the Faster R-CNN, and the minimum diameter of lung nodules was defined as 3 mm as described in the manuscript. The results based on fivefold cross-validation indicated that the Faster R-CNN obtained a precision of 80.77% and a recall of 85.63% on the training dataset, and a precision of 75.00% and a recall of 82.53% on the test dataset, respectively.

## Evaluation of inter-observer agreement on segmentation
To evaluate the stability and credibility of manual segmentation of lung nodules by the two local radiologists, a total of 100 lung nodules was randomly selected from the in-house dataset, and a second round of manual segmentation of the lung nodules was performed by the two radiologists. Dice coefficient (DC) was used to evaluate the inter-observer agreement between the two radiologists. 
Results of the experiment indicated that the average inter-observer agreement of the two radiologists was 92.0% by DC, with a 95% CI of 87.2% to 99.8%. The examples of segmentation by the two radiologists are presented as following.

<div align=left><img width="400" height="268" src="https://github.com/JD910/LNHG/blob/main/Segmentation/Images/Fig_F5.jpg"/></div><br />

**Examples of lung nodule segmentation performed by the two radiologists. a and b denote a solid nodule, and c and d mean a juxta-pleural nodule with vascular. Here, a and c denote the manual segmentation by radiologist 1, and b and d denote the manual segmentation by the other radiologist.**
