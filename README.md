# LimitsOfUDA
Understanding the Limits of Unsupervised Domain Adaptation via Data Poisoning

### Obtaining the data
	- Office-31: https://people.eecs.berkeley.edu/~jhoffman/domainadapt.
	- Digits:
		- MNIST dataset: available with tensorflow
		- MNIST_M: Follow the instructions to build MNIST_M dataset from https://github.com/pumpikano/tf-dann
		- SVHN: http://ufldl.stanford.edu/housenumbers/
		- USPS: https://github.com/mil-tokyo/MCD_DA/tree/master/classification/data

### Experiments on Digits
	#### Attack with mislabeled and watermarked poison data
		1. Navigate to the appropriate task in the Digits folder
		2. To evaluate performance with mislabeled poison data run python3 train_dann.py --TYPE POISON --ALPHA 1 --PP 0.1 where PP refers to the poison percentage. To test a different UDA method use the appropriate train_{UDA_method_name}.py file. Changing the value of ALPHA lets you control the amount of target data present in the poison. Value of 1 means poison data is same as target data, value of 0 means poison data is same as source data, an intermediate value means poison data is a combination of source and target data (watermarking). 
			
	#### Attack with clean-label poison data		
		1. Navigate to MNIST_MNISTM/clean_label_attacks folder
		2. To generate clean label attack using base data initialized from the target domain run python3 generate_poison_data_{UDA_method_name}.py --ETA 0.1 --BASE_DOMAIN target. For base data from the source domain change --BASE_DOMAIN to source. 
		3. To evaluate the performance of the UDA method on the genrated attack, run python3 retrain_{UDA_method_name}.py 
	
### Experiments on Office-31
	- Download the code from https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift
	- Navigate to ./data/office-31/ directory and place the files in the Office-31 folder of the supplementary material there.
	- Attack with mislabeled target domain poison data
		1. Navigate to ./data/office-31/
		2. To generate the poisoned files with mislabeled target domain data added to them, run python3 train_source_only.py --SRC amazon --DEST dslr, with appropriate values of source_domain \in {amazon, dslr, webcam} and target_domain \in {amazon, dslr, webcam}.
		3. To run the original code with poisoned files, navigate back to main folder (cd ../../) and run python train_image.py DANN --dset office-31 --s_dset_file poisoned_src_amazon_dest_dslr_list.txt --t_dset_file dslr_list.txt. Change the UDA algorithm to any of the algorithms from {DANN, CDAN, IW-DAN, IW-CDAN} and change the name of the source and destination files based on the values chosen in the previous step.
		
	- Attack with mislabeled source domain poison data
		1. Navigate to ./data/office-31/
		2. To generate the poisoned files with mislabeled source domain data added to them, run python3 train_source_only_watermarked.py --SRC amazon --DEST dslr --ALPHA 0, with appropriate values of source_domain \in {amazon, dslr, webcam} and target_domain \in {amazon, dslr, webcam}.
		3. To run the original code with poisoned files, navigate back to main folder (cd ../../) and run python train_image.py DANN --dset office-31 --s_dset_file poisoned_src_amazon_dest_dslr_list_watermarked_0.0.txt --t_dset_file dslr_list.txt. Change the UDA algorithm to any of the algorithms from {DANN, CDAN, IW-DAN, IW-CDAN} and change the name of the source and destination files based on the values chosen in the previous step.	
	
### Illustrative cases for UDA:
	Run python3 three_illustrative_cases.py file in Illustrative_cases folder.
