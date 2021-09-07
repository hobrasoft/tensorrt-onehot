The repository implements OneHot layer needed for AttentionOCR.

https://github.com/tensorflow/models/tree/master/research/attention_ocr

The AttentionOCR model is converted to saved model and then to frozen model,
some layers have to be removed (predicted_text) because the TensorRT does not
support string datatype. Then the frozen model is converted to ONNX.

The ONNX's OneHot is implemented slightly differently then the TensorFlow1's OneHot.
If you want to use the plugin in your code, please, change the DEPH in the onehot.cpp
to your own value.




