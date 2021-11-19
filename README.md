# Understanding the Limits of Unsupervised Domain Adaptation via Data Poisoning

<p align = justify>
Unsupervised domain adaptation (UDA) enables cross-domain learning without target domain labels by transferring knowledge from a labeled source domain whose distribution differs from the target. However, UDA is not always successful and several accounts of 'negative transfer' have been reported in the literature. In this work, we prove a simple lower bound on the target domain error that complements the existing upper bound. Our bound shows the insufficiency of minimizing source domain error and marginal distribution mismatch for a guaranteed reduction in the target domain error, due to the possible increase of induced labeling function mismatch.  This insufficiency is further illustrated through simple distributions for which the same UDA approach succeeds, fails, and may succeed or fail with an equal chance. 
Motivated from this, we propose novel data poisoning attacks to fool UDA methods into learning representations that produce large target domain errors.  
We evaluate the effect of these attacks on popular UDA methods using benchmark datasets where they have been previously shown to be successful.
Our results show that poisoning can significantly decrease the target domain accuracy, dropping it to 
almost 0% in some cases, with the addition of only 10% poisoned data in the source domain. 
The failure of UDA methods demonstrates the limitations of UDA at guaranteeing cross-domain generalization consistent
with the lower bound. 
Thus, evaluation of UDA methods in adversarial settings such as data poisoning can provide a better sense of their robustness in scenarios unfavorable for UDA.
</p>

<hr>

### The codes used to report the results in the paper <b>"[Understanding the Limits of Unsupervised DomainAdaptation via Data Poisoning](https://arxiv.org/abs/2107.03919)"</b> are present in this repository.
<hr>

### Obtaining the data
<ul>
	<li> Office-31: https://people.eecs.berkeley.edu/~jhoffman/domainadapt.
	<li> Digits:
		<ul>
		<li> MNIST dataset: available with tensorflow
		<li> MNIST_M: Follow the instructions to build MNIST_M dataset from https://github.com/pumpikano/tf-dann
		<li> SVHN: http://ufldl.stanford.edu/housenumbers/
		<li> USPS: https://github.com/mil-tokyo/MCD_DA/tree/master/classification/data
		</ul>
</ul>

### Experiments on Digits
<ul>
	<li> Attack with mislabeled and watermarked poison data
		<ul>
		<li> Navigate to the appropriate task in the Digits folder
		<li> To evaluate performance with mislabeled poison data run <code>python3 train_dann.py --TYPE POISON --ALPHA 1 --PP 0.1 </code> where PP refers to the poison percentage. To test a different UDA method use the appropriate train_{UDA_method_name}.py file. Changing the value of ALPHA lets you control the amount of target data present in the poison. Value of 1 means poison data is same as target data, value of 0 means poison data is same as source data, an intermediate value means poison data is a combination of source and target data (watermarking). 
		</ul>
	<li> Attack with clean-label poison data	
		<ul>
		<li> Navigate to MNIST_MNISTM/clean_label_attacks folder
		<li> To generate clean label attack using base data initialized from the target domain run <code> python3 generate_poison_data_{UDA_method_name}.py --ETA 0.1 --BASE_DOMAIN target </code>. For base data from the source domain change --BASE_DOMAIN to source. 
		<li>To evaluate the performance of the UDA method on the genrated attack, run python3 retrain_{UDA_method_name}.py 
		</ul>
</ul>	
		
### Experiments on Office-31
<ul>
	<li> Download the code from https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift
	<li> Navigate to ./data/office-31/ directory and place the files in the Office-31 folder of the supplementary material there.
	<li> Attack with mislabeled target domain poison data
		<ul>
		<li> Navigate to ./data/office-31/
		<li> To generate the poisoned files with mislabeled target domain data added to them, run <code>python3 train_source_only.py --SRC amazon --DEST dslr </code>, with appropriate values of source_domain \in {amazon, dslr, webcam} and target_domain \in {amazon, dslr, webcam}.
		<li> To run the original code with poisoned files, navigate back to main folder (cd ../../) and run <code> python train_image.py DANN --dset office-31 --s_dset_file poisoned_src_amazon_dest_dslr_list.txt --t_dset_file dslr_list.txt </code>. Change the UDA algorithm to any of the algorithms from {DANN, CDAN, IW-DAN, IW-CDAN} and change the name of the source and destination files based on the values chosen in the previous step.
		</ul>
	<li> Attack with mislabeled source domain poison data
		<ul>
		<li> Navigate to ./data/office-31/
		<li> To generate the poisoned files with mislabeled source domain data added to them, run <code> python3 train_source_only_watermarked.py --SRC amazon --DEST dslr --ALPHA 0 </code>, with appropriate values of source_domain \in {amazon, dslr, webcam} and target_domain \in {amazon, dslr, webcam}.
		<li> To run the original code with poisoned files, navigate back to main folder (cd ../../) and run <code> python train_image.py DANN --dset office-31 --s_dset_file poisoned_src_amazon_dest_dslr_list_watermarked_0.0.txt --t_dset_file dslr_list.txt </code>. Change the UDA algorithm to any of the algorithms from {DANN, CDAN, IW-DAN, IW-CDAN} and change the name of the source and destination files based on the values chosen in the previous step.	
		</ul>

</ul>

### Illustrative cases for UDA:
Run <code> python3 three_illustrative_cases.py </code> file in Illustrative_cases folder.

#### Citing

If you find this useful for your work, please consider citing
<pre>
<code>
@inproceedings{mehra2021understanding,
  title={Understanding the Limits of Unsupervised Domain Adaptation via Data Poisoning},
  author={Mehra, Akshay and Kailkhura, Bhavya and Chen, Pin-Yu and Hamm, Jihun},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
</code>
</pre>
