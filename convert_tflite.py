import os
import numpy as np
import tensorflow as tf

# fixed folder
saved_model_dir = "tf_cnn_model/1/"
target_dir = "tflite_cnn_model"

def convert_tflite():
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]

    tflite_model = converter.convert()

    with open(f"{target_dir}/tflite_model.tflite", "wb") as f:
        f.write(tflite_model)
        
def validation():
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    images = tf.convert_to_tensor(np.expand_dims(x_test/255.0, -1),dtype=tf.float32)

    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter(f"{target_dir}/tflite_model.tflite")
 
     # Model has single input.  
    in_node = interpreter.get_input_details()[0]  
    in_shape = in_node['shape']
       
    # Model has single output.
    out_node = interpreter.get_output_details()[0]
    out_shape = out_node['shape']

    # Resize Tensor (batch size)
    interpreter.resize_tensor_input(in_node['index'],[len(images), in_shape[1], in_shape[2], in_shape[3]])    
    interpreter.resize_tensor_input(out_node['index'],[len(images), out_shape[1]])
    # Needed before execution!
    interpreter.allocate_tensors()  

    
    interpreter.set_tensor(in_node['index'], images)
    interpreter.invoke()
    prediction = interpreter.get_tensor(out_node['index'])
    result = tf.argmax( prediction ,axis=1).numpy()
    print('accuracy={:.4f}'.format(np.sum(result == y_test)/y_test.shape[0]))
        
if __name__ == '__main__':
    convert_tflite()
    validation()