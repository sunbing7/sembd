
#mnistm
#python mnistm_nc.py
#python mnistm_nc2.py
#python mnistm_nc_clean.py
#python mad_outlier_detection_mnistm.py
#python mad_outlier_detection_mnistm2.py
#python mad_outlier_detection_mnistm_clean.py


#asl A
#python neuron_cleanse.py --datafile=attack_A --attack=A --datadir=./data --dataset=asl --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --model_dir=./models --model_name=asl_semantic_A_semtrain.h5 --result_dir=nc/ --batch_size=64
#python mad_outlier_detection.py --dataset=asl --attack=A --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --result_dir=nc/

#asl Z
python neuron_cleanse.py --datafile=attack_Z --attack=Z --datadir=./data --dataset=asl --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --model_dir=./models --model_name=asl_semantic_A_semtrain.h5 --result_dir=nc/ --batch_size=64
python mad_outlier_detection.py --dataset=asl --attack=Z --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --result_dir=nc/

#asl clean
python neuron_cleanse.py --datafile=clean --attack=clean --datadir=./data --dataset=asl --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --model_dir=./models --model_name=asl_semantic_A_semtrain.h5 --result_dir=nc/ --batch_size=64
python mad_outlier_detection.py --dataset=asl --attack=clean --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --result_dir=nc/

#caltech

