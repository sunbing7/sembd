
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
python neuron_cleanse.py --datafile=attack_Z --attack=Z --datadir=./data --dataset=asl --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --model_dir=./models --model_name=asl_semantic_Z_semtrain.h5 --result_dir=nc/ --batch_size=64
python mad_outlier_detection.py --dataset=asl --attack=Z --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --result_dir=nc/

#asl clean
python neuron_cleanse.py --datafile=clean --attack=clean --datadir=./data --dataset=asl --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --model_dir=./models --model_name=asl_semantic_clean.h5 --result_dir=nc/ --batch_size=64
python mad_outlier_detection.py --dataset=asl --attack=clean --num_classes=29 --img_row=200 --img_col=200 --img_ch=3 --result_dir=nc/


#caltech brain
python neuron_cleanse.py --datafile=bl_brain --attack=brain --datadir=./data --dataset=caltech --num_classes=101 --img_row=224 --img_col=224 --img_ch=3 --model_dir=./models --model_name=caltech_semantic_brain_semtrain.h5 --result_dir=nc/ --batch_size=64 --y_target=42
python mad_outlier_detection.py --dataset=caltech --attack=brain --num_classes=101 --img_row=224 --img_col=224 --img_ch=3 --result_dir=nc/

#caltech g_kan
python neuron_cleanse.py --datafile=g_kan --attack=g_kan --datadir=./data --dataset=caltech --num_classes=101 --img_row=224 --img_col=224 --img_ch=3 --model_dir=./models --model_name=caltech_semantic_g_kan_semtrain.h5 --result_dir=nc/ --batch_size=64 --y_target=1
python mad_outlier_detection.py --dataset=caltech --attack=g_kan --num_classes=101 --img_row=224 --img_col=224 --img_ch=3 --result_dir=nc/

#caltech clean
python neuron_cleanse.py --datafile=clean --attack=clean --datadir=./data --dataset=caltech --num_classes=101 --img_row=224 --img_col=224 --img_ch=3 --model_dir=./models --model_name=caltech_semantic_clean.h5 --result_dir=nc/ --batch_size=64
python mad_outlier_detection.py --dataset=caltech --attack=clean --num_classes=101 --img_row=224 --img_col=224 --img_ch=3 --result_dir=nc/


