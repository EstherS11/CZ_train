# test_discrete_ssl_fix.py
import inspect
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL

# 查看DiscreteSSL的实际参数
print("DiscreteSSL.__init__ 参数:")
sig = inspect.signature(DiscreteSSL.__init__)
print(sig)

# 查看文档字符串
print("\nDiscreteSSL 文档:")
print(DiscreteSSL.__doc__)