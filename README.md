# Feedback-involved-DST

This code is the official pytorch implementation of our paper: **Toward Real-life Dialogue State Tracking Involving Negative Feedback Utterance. Puhai Yang, Heyan Huang, Wei Wei, Xian-Ling Mao. SIGKDD2022 *(Research paper)***  [[PDF](https://dl.acm.org/doi/10.1145/3534678.3539385)]

## Abstract
Recently, the research of dialogue system has been widely concerned, especially task-oriented dialogue system, which has received increased attention due to its wide application prospect. As a core component, dialogue state tracking plays a key role in task-oriented dialogue systems, and its function is to parse natural language dialogues into dialogue state formed by slot-value pairs. It is well known that dialogue state tracking has been well studied and explored on current benchmark datasets such as the MultiWOZ. However, almost all current research completely ignores the user negative feedback utterance that exists in real-life conversations when a system error occurs, which often containing user-provided corrective information for the system error. Obviously, user negative feedback utterance can be used to correct the inevitable errors in automatic speech recognition and model generalization. Thus, in this paper, we will explore the role of negative feedback utterance in dialogue state tracking in detail through simulated negative feedback utterance. Specifically, due to the lack of dataset involving negative feedback utterance, first, we have to define the schema of user negative feedback utterance and propose a joint modeling method for feedback utterance generation and filtering. Then, we explore three aspects of interaction mechanism that should be considered in real-life conversations involving negative feedback utterance and propose evaluation metrics related to negative feedback utterance. Finally, on WOZ2.0 and MultiWOZ2.1 datasets, by constructing simulated negative feedback utterance in training and testing, we not only verify the important role of negative feedback utterance in dialogue state tracking, but also analyze the advantages and disadvantages of different interaction mechanisms involving negative feedback utterance, lighting future researches on negative feedback utterance.

## Requirements
* python >= 3.6
* pytorch >= 1.0

## Datasets

1. WOZ2.0 and MultiWOZ2.1 datasets in ```data_for_baselines.tgz``` used for training of all dialogue state tracking baselines.
2. WOZ2.0 and MultiWOZ2.2 datasets in ```data_for_our_model.tgz``` used for training of our feedback utterance generation and filtering joint model.
3. Data in ```simulated_negative_feedback_for_training.tgz``` is the simulated negative feedback samples generated by our joint model.

Note: Both MultiWOZ2.1 and MultiWOZ2.2 datasets use preprocessed data from [COCO-DST](https://github.com/salesforce/coco-dst).

## Models

1. All baselines are in the ```TRADE```, ```BERTDST```, ```SOMDST``` and ```T5DST``` folders.
2. Our feedback utterance generation and filtering joint model is in the ```Feedback_Simulation``` folder.

## Usage

## Our joint model
1. Uncompress ```data_for_our_model.tgz``` and place all result under the ```data``` folder in ```Feedback_Simulation```.
2. Run ```T5-feedback.py``` in ```Feedback_Simulation``` to train our model.
3. Add the paths of our trained model to ```model_file``` in the ```info.py``` file.
4. Uncompress ```data_for_baselines.tgz``` and place all result under the ```data``` folder.
5. Run ```feedback_generate.py``` to generate the simulated negative feedback samples, and the results will be stored in the ```feedback_data``` folder.

## Baselines
1. Uncompress ```data_for_baselines.tgz``` and place all result under the ```data``` folder in baseline folder.
2. Copy ```feedback_data``` folder to baseline folder.
2. Run ```*train.py``` in baseline folder to train baseline model.
3. Run ```*feedback.py``` in baseline folder to test baseline model.

