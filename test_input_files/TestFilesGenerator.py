import sys
import numpy as np
import torch
from torch import nn
from typing import TypeAlias

ROWS =  10
COLS =  10
HIDDEN = 10
SCALE_FACTOR =  10
NUM_HEADS =  1
INT_HIGH = 128
INT_LOW = -128

FileName: TypeAlias = str
int_or_float: bool = True # if true means int if false means float

def create_tensor(low: int, high: int, size: int|tuple[int,...] ) -> torch.Tensor: 
  return  torch.randint(low,high,size) if int_or_float else torch.rand(size)

def printMatrix(mat: torch.Tensor, file: FileName) -> None:
  with open(file, "w") as f:
    for item in mat.flatten().tolist():
      print(item, file=f)


def activation(matIn: FileName, matOut: FileName) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = nn.functional.relu(input1)
  printMatrix(input1, matIn)
  printMatrix(output, matOut)
  

def atthead(
    matIn: FileName, matWeight: FileName, 
    matBias: FileName, matMask: FileName,
    matOut: FileName
            ) -> None:
  num_layers = 3
  input1 = torch.rand((ROWS, COLS))
  weights = torch.rand((num_layers, COLS//NUM_HEADS, COLS))
  biases = torch.rand((num_layers, COLS//NUM_HEADS))
  attn_mask = torch.randint(0, 2, size=(ROWS, ROWS))
  q = nn.functional.linear(input1, weights[0], biases[0])
  k = nn.functional.linear(input1, weights[1], biases[1])
  v = nn.functional.linear(input1, weights[2], biases[2])
  output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask.bool())
  printMatrix(input1, matIn)
  printMatrix(torch.transpose(weights,-2,-1), matWeight)
  printMatrix(biases, matBias)
  printMatrix(attn_mask,matMask)
  printMatrix(output, matOut)


def concat(matIn1: FileName, matIn2: FileName, matOut: FileName) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  input2 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = torch.cat((input1, input2), dim=1)
  printMatrix(input1, matIn1)
  printMatrix(input2, matIn2)
  printMatrix(output, matOut)

def encoder(
    matIn: FileName, matMask: FileName,
    matHeadWeight: FileName, matHeadBias: FileName,
    matAttWeight: FileName, matAttBias: FileName,
    matFFWeights1: FileName, matFFBias1: FileName, 
    matFFWeights2: FileName, matFFBias2: FileName,
    matGamma: FileName, matBeta: FileName, 
    matOut: FileName
) -> None:
  num_layers = 3
  input1 = torch.rand((ROWS, COLS))
  weights_head = torch.rand((NUM_HEADS, num_layers, COLS//NUM_HEADS, COLS))
  biases_head = torch.rand((NUM_HEADS, num_layers, COLS//NUM_HEADS))
  attn_mask = torch.randint(0, 2, size=(ROWS, ROWS))
  weights_att = torch.rand((COLS, COLS))
  biases_att = torch.rand(COLS)
  weight1 = torch.rand((HIDDEN, COLS))
  bias1 = torch.rand(HIDDEN)
  weight2 = torch.rand((COLS, HIDDEN))
  bias2 = torch.rand(COLS)
  gamma = torch.rand((2,COLS))
  beta = torch.rand((2,COLS))
  att: list[torch.Tensor] = []
  for i in range(NUM_HEADS):
    q = nn.functional.linear(input1, weights_head[i][0], biases_head[i][0])
    k = nn.functional.linear(input1, weights_head[i][1], biases_head[i][1])
    v = nn.functional.linear(input1, weights_head[i][2], biases_head[i][2])
    atthead = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask.bool())
    att.append(atthead)
  att = torch.concat(att,dim=-1)
  multiheadResult = nn.functional.linear(att,weights_att,biases_att)
  res = multiheadResult + input1
  layernorm1 = nn.functional.layer_norm(res, (COLS,), gamma[0], beta[0])
  linear1 = nn.functional.linear(layernorm1, weight1, bias1)
  activation = nn.functional.relu(linear1)
  linear2 = nn.functional.linear(activation,weight2,bias2)
  res = linear2 + layernorm1
  output = nn.functional.layer_norm(res, (COLS,), gamma[1], beta[1])
  printMatrix(input1, matIn)
  printMatrix(torch.transpose(weights_head,-2,-1), matHeadWeight)
  printMatrix(biases_head, matHeadBias)
  printMatrix(attn_mask,matMask)
  printMatrix(torch.transpose(weights_att,-2,-1), matAttWeight)
  printMatrix(biases_att, matAttBias)
  printMatrix(torch.transpose(weight1,-2,-1), matFFWeights1)
  printMatrix(bias1, matFFBias1)
  printMatrix(torch.transpose(weight2,-2,-1), matFFWeights2)
  printMatrix(bias2, matFFBias2)
  printMatrix(gamma, matGamma)
  printMatrix(beta, matBeta)
  printMatrix(output, matOut)


def feedForward(
    matIn: FileName, matWeights1: FileName, 
    matBias1: FileName, matWeights2: FileName, 
    matBias2: FileName, matOut: FileName
    ) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  weight1 = create_tensor(INT_LOW, INT_HIGH, (HIDDEN, COLS))
  bias1 = create_tensor(INT_LOW, INT_HIGH, (HIDDEN,))
  weight2 = create_tensor(INT_LOW, INT_HIGH, (COLS, HIDDEN))
  bias2 = create_tensor(INT_LOW, INT_HIGH, (COLS,))
  linear = nn.functional.linear(input1, weight1, bias1)
  activation = nn.functional.relu(linear)
  output = nn.functional.linear(activation,weight2,bias2)
  printMatrix(input1, matIn)
  printMatrix(torch.transpose(weight1,0,1), matWeights1)
  printMatrix(bias1, matBias1)
  printMatrix(torch.transpose(weight2,0,1), matWeights2)
  printMatrix(bias2, matBias2)
  printMatrix(output, matOut)
  

def layerNorm(matIn: FileName, matWeight: FileName, matBias: FileName, matOut: FileName) -> None:
  input1 = torch.rand((ROWS, COLS))
  weight = torch.rand(COLS)
  bias = torch.rand(COLS)
  output = nn.functional.layer_norm(input1, (COLS,), weight, bias)
  printMatrix(input1, matIn)
  printMatrix(weight, matWeight)
  printMatrix(bias, matBias)
  printMatrix(output, matOut)

def linear(matIn: FileName, matWeights: FileName, matBias: FileName, matOut: FileName) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, HIDDEN))
  weight = create_tensor(INT_LOW, INT_HIGH, (COLS,HIDDEN))
  bias = create_tensor(INT_LOW, INT_HIGH, (COLS,))
  output = nn.functional.linear(input1,weight,bias)
  printMatrix(input1, matIn)
  printMatrix(torch.transpose(weight,0,1), matWeights)
  printMatrix(bias, matBias)
  printMatrix(output, matOut)

def bitlinear_forward_cppstyle(
    x: torch.Tensor,           # [ROWS, HIDDEN]
    w_cpp: torch.Tensor,       # [HIDDEN, COLS] (this matches what you write to file)
    b: torch.Tensor            # [COLS]
) -> torch.Tensor:
  """
  Python reference for bitlinear158b_forward to match the C++ implementation.
  x:      [rows, hidden]
  w_cpp:  [hidden, cols]   (NOTE: already transposed for C++)
  b:      [cols]
  """
  rows, hidden = x.shape
  hidden2, cols = w_cpp.shape
  assert hidden2 == hidden

  # 1) beta = mean(|w|)
  beta = w_cpp.abs().mean()
  if beta < 1e-8:
    beta = torch.tensor(1e-8, dtype=w_cpp.dtype)

  # 2) ternarize weights: -1, 0, +1 (signs only; scaling via beta later)
  r = w_cpp / beta
  ternary_sign = torch.where(
      r > 0.5,
      torch.ones_like(r),
      torch.where(r < -0.5, -torch.ones_like(r), torch.zeros_like(r))
  )

  # 3) quantize activations per row to int8
  x_int8 = torch.zeros_like(x, dtype=torch.int8)
  scale_row = torch.zeros(rows, dtype=x.dtype)

  for i in range(rows):
    max_val = x[i].abs().max()
    if max_val < 1e-8:
      max_val = torch.tensor(1e-8, dtype=x.dtype)

    gamma = 127.0 / max_val
    scale_row[i] = 1.0 / gamma   # matches C++: scale_row[r] = 1/gamma

    v = x[i] * gamma
    v = torch.round(torch.clamp(v, -128.0, 127.0))
    x_int8[i] = v.to(torch.int8)

  # 4) integer-like MAC then scaling
  # acc = x_int8 @ ternary_sign  (done in int, then scaled)
  acc = torch.matmul(x_int8.to(torch.int32), ternary_sign.to(torch.int32))  # [rows, cols]

  # per-row scale: (scale_row[r] * beta)
  scale = (scale_row * beta).view(rows, 1)     # [rows, 1]
  out = acc.to(x.dtype) * scale + b           # broadcast bias over rows

  return out

def bitlinear(matIn: FileName, matWeights: FileName, matBias: FileName, matOut: FileName) -> None:
  # Same shapes as linear()
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, HIDDEN)).to(torch.float32)
  weight = create_tensor(INT_LOW, INT_HIGH, (COLS, HIDDEN)).to(torch.float32)
  bias = create_tensor(INT_LOW, INT_HIGH, (COLS,)).to(torch.float32)

  # C++ sees weights as [HIDDEN][COLS], which is weight.T
  w_cpp = torch.transpose(weight, 0, 1)   # [HIDDEN, COLS]

  output = bitlinear_forward_cppstyle(input1, w_cpp, bias)

  printMatrix(input1, matIn)
  printMatrix(w_cpp, matWeights)  # this matches what C++ expects
  printMatrix(bias, matBias)
  printMatrix(output, matOut)

def mask(matIn: FileName, matMask: FileName, matOut: FileName) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  random_tensor = torch.randint(0, 2, size=(ROWS, COLS))
  output = input1 * random_tensor
  printMatrix(input1, matIn)
  printMatrix(random_tensor, matMask)
  printMatrix(output, matOut)

def matAdd(matA: FileName, matB: FileName, matOut: FileName) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  input2 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = input1 + input2
  printMatrix(input1, matA)
  printMatrix(input2, matB)
  printMatrix(output, matOut)

def matMul(matA: FileName, matB: FileName, matOut: FileName) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, HIDDEN))
  input2 = create_tensor(INT_LOW, INT_HIGH, (HIDDEN, COLS))
  output = torch.matmul(input1, input2)
  printMatrix(input1, matA)
  printMatrix(input2, matB)
  printMatrix(output, matOut)

def multiHeadAtt(
    matIn: FileName, matMask: FileName,
    matHeadWeight: FileName, matHeadBias: FileName,
    matAttWeight: FileName, matAttBias: FileName,
    matOut: FileName
    ) -> None:
  num_layers = 3
  input1 = torch.rand((ROWS, COLS))
  weights_head = torch.rand((NUM_HEADS, num_layers, COLS//NUM_HEADS, COLS))
  biases_head = torch.rand((NUM_HEADS, num_layers, COLS//NUM_HEADS))
  attn_mask = torch.randint(0, 2, size=(ROWS, ROWS))
  weights_att = torch.rand((COLS, COLS))
  biases_att = torch.rand(COLS)
  att: list[torch.Tensor] = []
  for i in range(NUM_HEADS):
    q = nn.functional.linear(input1, weights_head[i][0], biases_head[i][0])
    k = nn.functional.linear(input1, weights_head[i][1], biases_head[i][1])
    v = nn.functional.linear(input1, weights_head[i][2], biases_head[i][2])
    atthead = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask.bool())
    att.append(atthead)
  att = torch.concat(att,dim=-1)
  output = nn.functional.linear(att,weights_att,biases_att)
  printMatrix(input1, matIn)
  printMatrix(torch.transpose(weights_head,-2,-1), matHeadWeight)
  printMatrix(biases_head, matHeadBias)
  printMatrix(attn_mask,matMask)
  printMatrix(torch.transpose(weights_att,-2,-1), matAttWeight)
  printMatrix(biases_att, matAttBias)
  printMatrix(output, matOut)

  


def scale(matIn: FileName, matOut: FileName) ->None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  round_mode = 'trunc' if int_or_float else None
  output = torch.div(input1, SCALE_FACTOR, rounding_mode=round_mode)
  printMatrix(input1, matIn)
  printMatrix(output, matOut)
  

def scaleDotAtt(matIn: FileName, matMask: FileName, matOut: FileName) -> None:
  input1 = torch.rand((ROWS, COLS))
  attn_mask = torch.randint(0, 2, size=(ROWS, ROWS))
  output = nn.functional.scaled_dot_product_attention(input1, input1, input1, attn_mask.bool())
  printMatrix(input1, matIn)
  printMatrix(output, matOut)
  printMatrix(attn_mask,matMask)

def softmax(matIn: FileName, matOut: FileName) -> None:
  input1 = torch.rand(COLS)
  output = nn.functional.softmax(input1, dim=-1)
  printMatrix(input1, matIn)
  printMatrix(output, matOut)


def transpose(matIn: FileName, matOut: FileName) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (ROWS, COLS))
  output = torch.transpose(input1,0,1)
  printMatrix(input1, matIn)
  printMatrix(output, matOut)

def vecAdd(vecA:FileName, vecB: FileName, vecOut: FileName) -> None:
  input1 = create_tensor(INT_LOW, INT_HIGH, (COLS,))
  input2 =  create_tensor(INT_LOW, INT_HIGH, (COLS,))
  output = input1 + input2
  printMatrix(input1, vecA)
  printMatrix(input2, vecB)
  printMatrix(output, vecOut)


#List of valid Arguments
#Ask why there is a difference regarding floating point operations in python and C
'''Test_Activation,
	Test_AttHead,
	Test_Concat, 
	Test_Encoder,
	Test_FeedForward,
	Test_LayerNorm,
	Test_Linear,
	Test_Mask,
	Test_MatAdd,
	Test_MatMul,
	Test_MultiHeadAtt,
	Test_Scale,
	Test_ScaleDotAtt,
	Test_SoftMax,
	Test_Transpose,
	Test_VecAdd
'''

input_filename: list[FileName] = [
  "input1.txt",
  "input2.txt",
  "input3.txt",
  "input4.txt",
  "input5.txt",
  "input6.txt",
  "input7.txt",
	"input8.txt",
	"input9.txt",
	"input10.txt",
	"input11.txt",
  "input12.txt"
]

result_filename: FileName = "golden_result.txt"

int_or_float = False if len(sys.argv) > 2 and sys.argv[2] == "float" else True
test = sys.argv[1]
match test:
  case "Test_Activation":
    activation(input_filename[0], result_filename)
  case "Test_Concat":
    concat(input_filename[0], input_filename[1], result_filename)
  case "Test_LayerNorm":
    layerNorm(input_filename[0], input_filename[1],
              input_filename[2], result_filename)
  case "Test_Linear":
    linear(input_filename[0], input_filename[1], input_filename[2], result_filename)
  case "Test_BitLinear":
    bitlinear(input_filename[0], input_filename[1], input_filename[2], result_filename)	  
  case "Test_Mask":
    mask(input_filename[0], input_filename[1], result_filename)
  case "Test_MatAdd":
    matAdd(input_filename[0], input_filename[1], result_filename)
  case "Test_MatMul":
    matMul(input_filename[0], input_filename[1], result_filename)
  case "Test_ScaleDotAtt":
    scaleDotAtt(input_filename[0], input_filename[1], result_filename)
  case "Test_SoftMax":
    softmax(input_filename[0], result_filename)
  case "Test_Transpose":
    transpose(input_filename[0], result_filename)
  case "Test_VecAdd":
    vecAdd(input_filename[0], input_filename[1], result_filename)
  case "Test_Scale":
    scale(input_filename[0], result_filename)
  case "Test_FeedForward":
    feedForward(input_filename[0], input_filename[1], input_filename[2], 
			input_filename[3], input_filename[4], result_filename)
  case "Test_AttHead":
    atthead(input_filename[0], input_filename[1], input_filename[2], 
            input_filename[3], result_filename)
  case "Test_MultiHeadAtt":
    multiHeadAtt(input_filename[0], input_filename[1], 
                 input_filename[2], input_filename[3], 
                 input_filename[4], input_filename[5], 
                 result_filename)
  case "Test_Encoder":
    encoder(
			input_filename[0], input_filename[1], input_filename[2],
			input_filename[3], input_filename[4], input_filename[5],
			input_filename[6], input_filename[7], input_filename[8],
			input_filename[9], input_filename[10],
			input_filename[11], result_filename)
