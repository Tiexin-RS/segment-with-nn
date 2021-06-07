import logging
import tensorflow as tf
import os
import sys
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
# from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
# from segelectri.loss_metrics.metrics import MeanIou
from tvm import relay,autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
# from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor as graph_executor
import tvm.relay.testing.tf as tf_testing
import tvm.contrib.graph_executor as runtime
import tvm.relay.op
import time
import numpy as np
# use PIL to load Image
from PIL import Image
from tensorflow.core.framework.graph_pb2 import GraphDef

def save_image(save_data, save_path):
    plt.imshow(save_data)
    plt.colorbar()
    plt.savefig(save_path, dpi=300)

def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space) # min(len(task.config_space),200)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )

def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True,exec_num = 2000):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    print("graph optimizing...")
    executor.benchmark_layout_transform(min_exec_num=exec_num)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

if __name__ == '__main__':
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)
    # 1.load img
    img_path = '/opt/dataset/tr2_cropped/data/1.png'
    image = Image.open(img_path).resize((1024,1024))
    x = np.array(image)
    # 2.load graph
    GRAPH_PB_PATH  = './frozen'
    # graph_def = tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True)
    with tf.io.gfile.GFile('./frozen/frozen_model_fixed.pb', 'rb') as f:
        graph_def = GraphDef.FromString(f.read())
        #call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    # 3. tvm frontend
    shape_dict = {"input_1": (1,1024,1024,3)} # change shape
    mod, params = relay.frontend.from_tensorflow(graph_def, shape_dict)
    # meta-data部份
    data_shape = (1,1024,1024,3) # maybe
    output_shape = (1,1024,1024,4)
    batch_size = 1
    dtype = "float32"
    model_name = "unet_cpu_12_thread"
    log_file = "%s.log" % model_name
    graph_opt_sch_file = "%s_graph_opt_1000.log" % model_name
    input_name = "x" #这是和后面的输入名字一样的
    num_thread = 12
    # num_thread = 2
    os.environ["TVM_NUM_THREADS"] = str(num_thread)
    
    # 设置张量调整
    tuning_option = {
        "log_filename": log_file,
        "tuner": "random",
        "early_stopping": 100,
        # "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
            ),
        ),
    }

    # use these settings for cpu
    target = tvm.target.Target("llvm",host = "llvm")
    layout = None
    # dev = tvm.cpu()
    dev = tvm.cpu(0)
    
    # translating NHWC 2 NCHW
    desired_layouts = {'nn.conv2d': ['NCHW', 'default']} # change layout
    with tvm.transform.PassContext(opt_level = 3): # cannot set < 3 maybe
        mod =  tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                            relay.transform.ConvertLayout(desired_layouts)])(mod)
    # dynamic to static(maybe useful)
    mod = relay.transform.DynamicToStatic()(mod)
    print(mod.astext(show_meta_data = False))
    
    # print("Extract tasks...")
    # tasks = autotvm.task.extract_from_program(
    #     mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    # ) # 例程上，target = "llvm"
    # for i, task in enumerate(tasks):
    #     print(len(task.config_space))

    # tune_kernels(tasks, **tuning_option) # tuning
    # tune_graph(mod["main"], data_shape, 'unet_cpu_2_thread.log', graph_opt_sch_file,exec_num = 1000) # tuning

    # # #只需要得到这个opt_sch_file就可
    with autotvm.apply_graph_best(graph_opt_sch_file):# graph_opt_sch_file
        print("compile...")
        with tvm.transform.PassContext(opt_level = 3): # set < 3
            # lib = relay.build_module.build(mod,target,params = params)
            lib = relay.build(mod,target,params = params)
        # m = graph_executor.GraphModule(lib["default"](dev))
        with open(graph_opt_sch_file, 'r') as f:
            graph = f.read()
        m = graph_executor.create(graph, lib['default'], dev, dump_root="/tmp/tvmdbg")
        # set input and get_output
        m.set_input(input_name,tvm.nd.array(x.astype(dtype))) # input_name = 'x'
        # must set 'x' as input here due to previous channel translating
        # automatically change our original model input name
        # And here have to maintain the correspondence between
        # real img size,data type and the model's inputs'
        
        # evaluate
        # print("Evaluate inference time cost...")
        # ftimer = m.module.time_evaluator("run", dev, number=10, repeat=3) # a easy one
        # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        # print(
        #     "Mean inference time (std dev): %.2f ms (%.2f ms)"
        #     % (np.mean(prof_res), np.std(prof_res))
        # )

        # 得到推理结果并打印时间
        begin = time.perf_counter()
        m.run() 
        middle = time.perf_counter()
        tvm_output = m.get_output(0,tvm.nd.empty(((1,1024,1024,4)),'float32'))
        prediction = tvm_output.asnumpy()
        end = time.perf_counter()
        print('module run time:%.5f' % (middle-begin))
        print('cpu total time:%.5f' % (end-begin))
        # # print(prediction)
        # pred_data = tf.argmax(prediction, axis=-1)
        # pred_data = np.reshape(pred_data,(1024,1024))
        # img_save_path = './tvm_img/tr2/cross/img1_cpu.png'
        # save_image(pred_data,img_save_path)