import os

if __name__ == '__main__':
    print(os.stat('exp/37/saved_model/saved_model.pb').st_size) # 未经过量化
    print(os.stat('tmp/unet_tflite_models/unet_model.tflite').st_size) # 经过量化