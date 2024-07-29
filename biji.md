# layoutlmv3

当前虚拟环境：pdfextract

模型训练相关代码：
- object_detection
- layoutlmft

代码来源：
- unilm\layoutlmv3\examples\object_detection
- unilm\layoutlmv3\layoutlmft

修改部分：layoutlmft\models\layoutlmv3\__init__.py
```
# AutoConfig.register("layoutlmv3", LayoutLMv3Config)
# AutoModel.register(LayoutLMv3Config, LayoutLMv3Model)
# AutoModelForTokenClassification.register(LayoutLMv3Config, LayoutLMv3ForTokenClassification)
# AutoModelForQuestionAnswering.register(LayoutLMv3Config, LayoutLMv3ForQuestionAnswering)
# AutoModelForSequenceClassification.register(LayoutLMv3Config, LayoutLMv3ForSequenceClassification)
# AutoTokenizer.register(
#     LayoutLMv3Config, slow_tokenizer_class=LayoutLMv3Tokenizer, fast_tokenizer_class=LayoutLMv3TokenizerFast
# )
```
注释掉

# extract

服务搭建：
- CPU环境
  - detectron2安装：https://miropsota.github.io/torch_packages_builder/detectron2/
- GPU环境


# 服务搭建代码

demo/*


# 源码学习

底层模型：VLGeneralizedRCNN


