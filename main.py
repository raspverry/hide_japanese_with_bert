import spacy

print(spacy.prefer_gpu())

#import torch
#print(torch.cuda.is_available())  # True가 반환되어야 함

#import cupy as cp

# 간단한 GPU 연산 테스트
#a = cp.array([1, 2, 3])
#b = cp.array([4, 5, 6])
#c = a + b
#print(c)  # [5, 7, 9]가 출력되면 GPU에서 연산이 잘 작동한 것임

