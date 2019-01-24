This is the code to reproduce the results in the paper:

Semi-Unsupervised Learning with Deep Generative Models: Clustering and Classifying using Ultra-Sparse Labels


To set up a conda environment that this code will run in, execute:
>>yes | conda create -n TF python=2.7 scipy==1.0.0 tensorflow-gpu==1.8 Keras==2.1.3 pandas==0.22.0 numpy==1.14.0 matplotlib scikit-learn


The python script ./src/run_data_model_mixed.py is the main script for running the models

It takes the following arguments:
-m --model_name, choose between m2 gm_dgm adgm agm_dgm sdgm, required

-d --dataset_name choose between fmnist mnist

-p --prop_labelled type=float, required, the proportion of labelled data to keep for each class

-r --number_of_runs type=int, required

-e --number_of_epochs type=int, required

-c --classes_to_hide type=int, the individual class of array of classes to be entirely masked so as to do semi-unsupervised learning

-a --number_of_classes_to_add, corresponds to $N_{aug}$ in the paper

-z --number_of_dims_z type=int, default=100, $|z|$

-u --number_of_dims_a type=int, default=100, $|a|$

-s --number_of_mc_samples type=int, default=1

-t --iteration_number type=int, default=0, index to control which GPU is used on multi-GPU server

-l --number_of_units_in_hidden_layers type=int, default=500

-b --batch_size type=int, default=100

--decay_period type=int, default=200

--decay_ratio type=float, default=0.75

--loss_balance choose between average weighted, default is average, says how to scale unlabelled and labelled gradients. average gives both as the average over a mini batch, weighted downscales each by the relative sizes of the labelled and unlabelled data.