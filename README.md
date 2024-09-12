# 2nd RESTART Tech Camp on 5G and Open RAN | September 9-12, Milan (Italy)

This is the repository for the O-RAN assignment/hands-on related to the tech camp.

You will learn to work with cellular datasets and train AI solutions to classify traffic profiles.

- The script [Train_Predict_Light.py](https://github.com/wineslab/restart_assignment_repo/blob/main/Train_Predict_Light.py) offers a baseline training pipeline that shows you how to open the dataset, extract/plot data, and provide a skeleton to train a base AI algorithm.
- The script [Test_Performance_Fake.py](https://github.com/wineslab/restart_assignment_repo/blob/main/Test_Performance_Fake.py) offers a testing algorithm for your trained algorithm to verify that your solution processes data correctly and can be evaluated.

- The script [Train_Predict_Full.py](https://github.com/wineslab/restart_assignment_repo/blob/main/Train_Predict_Full.ipynb) offers a baseline training pipeline that shows you how to open the dataset, extract/plot data, and provide you an implementation of a traditional training & prediction AI pipeline for two classifiers of example.

The datasets you need to download are located at:
- Fake testing dataset available at [this link](https://drive.google.com/file/d/1gjTM2qki8dSf0_02xxxQjIn7Qk2zaHH0/view?usp=sharing). This has the same format as the testing dataset but has not the same data that will be used to evaluate your algorithms :) We will make this dataset available once you are all done with the activities.
- Training dataset available at [this link](https://drive.google.com/file/d/1UZJVwSVznpDIIvDKtDyVRxpeNVDb2Hz5/view?usp=sharing). This is the training dataset you should use. We suggest you split this into 3 subsets for training, validation and testing. Note that the testing dataset you will produce here is different from the one above. 

**Important:** 
- We will ask you to submit a trained model we can test using the `./Test_Performance_Fake.py` script. If you cannot test your model using that script, it means that you need to change input/output format to match the requirement.
- Groups that are willing to present their approach, are free to prepare 2-3 slides to present their work and results to the audience. Please let us know in advance if you are willing to present.

**Slides from the talk:** 

- ML Primer: [this link](https://www.dropbox.com/scl/fi/uzsyv5idvmz9ahjyjelo4/tech_camp_ML_primer.pdf?rlkey=cidtbqputurzlunbrgbyv3rjq&dl=0)
- Detaset description slides: [this link](https://github.com/wineslab/restart_assignment_repo/blob/main/restart-techcamp-2024.pdf)
- Original dataset available at [this link](https://github.com/wineslab/colosseum-oran-coloran-dataset)
- Paper describing the dataset and AI-based xApps is M. Polese, L. Bonati, S. D'Oro, S. Basagni, T. Melodia, "ColO-RAN: Developing Machine Learning-based xApps for Open RAN Closed-loop Control on Programmable Experimental Platforms," IEEE Transactions on Mobile Computing, pp. 1-14, July 2022. [PDF](https://arxiv.org/pdf/2112.09559)

  **Email contacts
  - Andrea Pimpinella: andrea.pimpinella@unibg.it
  - Salvatore D'Oro: s.doro@northeastern.edu
