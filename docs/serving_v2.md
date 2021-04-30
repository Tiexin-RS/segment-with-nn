## Serving及遇到的问题
### flask
已经搭建了一个简单的flask服务，详见flask这个文件夹，不过没有采用restful api的方式，而是用的传统的jinja后端渲染的方式，主要是为了能简单前端显示，转restful也比较简单，就json处理一下就行，不过关于gRPC的方式还不太了解。

如果修改为restful，那关于发送请求有两种方式，一是用tf官方使用的curl方式，教程可参看 http://www.ruanyifeng.com/blog/2019/09/curl-reference.html ；

二是使用requests库的方式，具体可参见flask/serving_test.py

reference https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

### tfserving
简单尝试了一下tfserving，用的服务器里的tf serving，目前已经搭建了服务。转发之后，浏览器地址栏输入：http://localhost:8501/v1/models/deeplab_lovasz_1，http://localhost:8501/v1/models/deeplab_lovasz_1/metadata 可看到信息。

但目前存在问题，{"error": "Conversion of JSON Value: 0.55... to type: half"}，然后我查了一下，应该是tf官方没有支持JSON->DT_HALF，可见github上的讨论https://github.com/tensorflow/serving/pull/1753，我去看了一下确实在AddJsontovalue这个方法里没有json->half。

另外问个问题
```
TESTDATA="/home/Tiexin-RS/code/segment-with-nn/exp"
docker run -it --rm -p 8501:8501 \
    -v "$TESTDATA/43_unfreeze/saved_model:/models/deeplab_lovasz_1" \
    -e MODEL_NAME=deeplab_lovasz_1 \
    -d tensorflow/serving
```
这个命令里/models/deeplab_lovasz_1这个东西只是作为url吗，还是啥。

reference reference https://tensorflow.google.cn/tfx/serving/docker?hl=en

### TODO
- 压测 
- 记录数据