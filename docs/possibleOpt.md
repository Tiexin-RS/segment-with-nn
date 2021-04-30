### 一些tensorflow参考文档以及遇到的相同的问题
#### 一些相关的ref
##### tensorflow官方参考:
* 关于INT8优化：https://blog.tensorflow.org/2019/06/tensorflow-integer-quantization.html
* 关于权重聚类：https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html
* 关于权重剪枝：https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html

#### 有关的问题以及github上的对应issue
* 关于INT8优化：要求进行**权重优化**的时候已经创建好了权重，但是权重创建好像只能当其被传入形状的时候才能create，但是我们调用Unet()方法的时候并没有传入对应的input_shape，导致了在Unet类内部实现的时候出现了一些问题，其没有办法进行。
  再说的清楚一点的话，官方给出的guide上给出了示例代码如下：
  ```python
    i = tf.keras.Input(shape=(20,))
    x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(10))(i)
    o = tf.keras.layers.Flatten()(x)
    annotated_model = tf.keras.Model(inputs=i, outputs=o)
    # Use `quantize_apply` to actually make the model quantization aware.
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

所以这个方法或许只能在Model建立了之后才能apply，然后我去看了看TrainRoutine里面的方法，发现是调用了Model.fit方法使得模型跑起来的。
官方给出的build与run方法之间的关系如下：
  > The __call__() method of your layer will automatically run build the first time it is called.
  > like y = linear_layer(x)
* 关于权重聚类：**权重聚类**是在fit之后，保存模型之前进行的。然而这个方法也不支持subclass model，对每一层分别拆开应用这个保存方法看起来也不是一个好的办法。
* 关于权重剪枝：同样的原因，因为只能对build好的model进行剪枝，导致了问题。官方给出的issue[#155](https://github.com/tensorflow/model-optimization/issues/155)说解决这种问题的方法是对每一层循环使用权重剪枝。

#### tensorflow官方guide：
* https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide

### TODO
- 先对模型进行 model.build(input_shape) 然后进行优化
- 对 saved_model 进行优化
  - 用 TVM 或者 TensorRT
  - 适当地在 call() 方法加上 @tf.function()

### 04/28 update
- 1. 采用先build的方法后，对每一层进行tfmot.quantization.keras.quantize_apply(layers),报错：
  ```
  '`model` can only be a `tf.keras.Model` instance.'You passed an instance of type:downsamp_conv
  ```
  其中downsamp_conv继承的base class是Layers.layer，所以不是Model类型。
  然后做了一个**违背祖宗的决定**，将dowmsamp_conv的base class改为keras.Model，build成功，不过再次应用方法的时候报错:
  ```
  '`model` can only either be a tf.keras Sequential or ''Functional model.'
  ```
  那么看来训练中量化的方式只能对这两种Model使用了？

  2. 尝试了训练后量化，也就是对saved_model进行优化的方法，采用的是tf自带的``TensorRT``和``tf.lite.TFLiteConverter``
  但是量化的结果却不太对劲
  **TensorRT:**
  ```
  params = trt.DEFAULT_TRT_CONVERSION_PARAMS
  params._replace(precision_mode = trt.TrtPrecisionMode.INT8)
  converter = trt.TrtGraphConverterV2(input_saved_model_dir='../train_unet/exp/37/saved_model',conversion_params=params)
  converter.convert()
  converter.save('trt_savedmodel')

  before:2809876
  after:127003352 # ?
  ```
  **tf.lite.TFLiteConverter:**
  ```
  converter = tf.lite.TFLiteConverter.from_saved_model('../train_unet/exp/37/saved_model')
  tflite_model = converter.convert()
  tflite_models_dir = pathlib.Path("./tmp/unet_tflite_models/")
  tflite_models_dir.mkdir(exist_ok=True, parents=True)
  tflite_model_file = tflite_models_dir/"unet_model.tflite"
  origin_byte = tflite_model_file.write_bytes(tflite_model)
  print(origin_byte)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_quant_model = converter.convert()
  tflite_model_quant_file = tflite_models_dir/"unet_model_quant.tflite"
  after_byte = tflite_model_file.write_bytes(tflite_quant_model)
  print(after_byte)

  # 量化大小
  before:2809876
  after:31271872 # 大了10倍？
  ```
  看了一下tf.lite.Optimize.DEFAULT的设置（怀疑不是针对size进行的optimize），发现里面的enum有三个，如下：
  ```
  DEFAULT = "DEFAULT"

  # Deprecated. Does the same as DEFAULT.
  OPTIMIZE_FOR_SIZE = "OPTIMIZE_FOR_SIZE"

  # Deprecated. Does the same as DEFAULT.
  OPTIMIZE_FOR_LATENCY = "OPTIMIZE_FOR_LATENCY"
  ```
  看起来就只有DEFAULT一个接口。试了一下``OPTIMIZE_FOR_SIZE``，结果是一样的，大小不变。

  可以直接使用/opt/segelectri/train/train_unet下的seesize.py查看不同模型的大小，采用上述的两种方法生成的模型路径如下：
  **TensorRT**:trt_savedmodel/saved_model.pb
  **tf.lite.TFLiteConverter**:tmp/unet_tflite_models/unet_model.tflite