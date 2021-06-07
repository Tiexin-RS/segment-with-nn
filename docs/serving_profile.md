## Serving & Profile 

以下操作均在host

### 用Docker启动Serving 
``` shell
TESTDATA="/home/Tiexin-RS/code/workspace/ws/segment-with-nn/train/train_unet"
nvidia-docker run -it --rm -p 8500:8500 -p 8501:8501 \
    -v "$TESTDATA/normal_function_model:/models/unet1" \
    -v /home/Tiexin-RS/tensorboard:/home/Tiexin-RS/tensorboard \
    tensorflow/serving:latest-gpu \
    --model_config_file=/models/unet1/models.config
```
详细解释一下：
- 命令中8500端口为gRPC，用于profile，8501端口为restful，用于load test；
- 下面-v，一是把模型挂载进docker，二是把tensorboard挂载进docker；
- model_config_file为配置文件，用于将多个模型挂入docker，config文件具体可见$TESTDATA/normal_function_model这个文件夹；
- 此外，可加入批处理配置如下，也可不加，这个暂时无所谓；
```shell 
--enable_batching \
--batching_parameters_file=/models/deeplab_53_unfreeze/batching.config
```

启动之后可在host中通过端口映射到本机的浏览器，点击vs code终端的端口，输入8501进行映射即可，然后可以打开浏览器输入 http://localhost:8501/v1/models/saved_model/metadata 查看模型情况，saved_model为模型名字，在models.config文件中可以找到

### load test 
1. conda activate load_test(bash_mode)
2. 进入code/workspace/wjz/segment-with-nn/serving/load_test/locust_tfserving/ 文件夹，先查看locust_tfserving.py，配置好你的task，通过运行todo.sh文件进行压测，这里会映射到8089端口；压测完成后在终端可以用pkill -9 locust杀死所有相关进程；更多关于load test见https://docs.locust.io/en/stable/what-is-locust.html；

### profile
1. su 命令转到根用户
2. conda activate /home/ws/miniconda3/envs/load_test
3. tensorboard --logdir /home/Tiexin-RS/tensorboard/ 开启tensorboard
4. 在浏览器tensorboard中选择profile，然后配合load test进行profile即可；更多信息见 https://www.tensorflow.org/tfx/serving/tensorboard；

### profile实验过程

在restful压测，GPU出现加载峰值和利用率只有5%的情况，因而进行profile

![profile-restful-1](https://github.com/teamwong111/Image-set/blob/main/img/20210524143934.png)

同时发现restful开不开批处理单用户请求都是800ms左右，因此测试grpc的情况，发现grpc单用户请求30ms--130ms

然后想通过dcpdump抓包查看restful和grpc包的大小，但因为网卡的mtu是1500，把http请求分包了，但包的大小不是1500而是出现了32741、32768、65483，这是因为TSO机制，然后通过查看dcp-rmem和dcp-wmem，希望设置dcpdump -B来改变TSO机制中缓冲区的大小，发现不行

另外在发送一次请求后dcpdump抓到了很多包，猜测是HTTP本身也会有一堆OPTION这样的预请求

然后希望通过logging.DEBUG的方法得到请求包的大小，想logging到文件，结果发现http.client does not use logging to output，所以通过重定向解决，是啊一年发现在restful下，Content-Length: 22499709；在grpc下，Content-Length: 12582998；

然后对grpc进行profile，发现只需要9.1ms，因此怀疑restful的list -> tf.Tensor花了几十毫秒；同时grpc下GPU利用率还是小于5%；

![1](https://github.com/teamwong111/Image-set/blob/main/img/profile-grpc-timeline.png)

因此做实验查看一下 list -> Tensor 的耗时，发现list -> tensor需要36ms

![1](https://github.com/teamwong111/Image-set/blob/main/img/list-tensor-timeline.png)