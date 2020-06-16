#!/usr/bin/python3
import jetson.inference
import jetson.utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be: googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

# Loading the image from Disk
# loadImageRGBA()関数を使うことでGPUメモリにイメージ（画像）をロードできる
# ロードされたイメージはCPUとGPUの両方にマップされている共有メモリに保存される
img, width, height = jetson.utils.loadImageRGBA(opt.filename)

# load the recognition network 画像認識ネットワークの読み込み
# imageNetオブジェクトを使ってTensorRTで分類モデルを読み込む
net = jetson.inference.imageNet(opt.network)

# 画像の分類
# 画像本体とサイズを渡し、TensorRTで推論を実行する
# 戻り値：画像が認識されたオブジェクトクラスの整数インデックスを含むタプルと信頼値
class_idx, confidence = net.Classify(img, width, height)

# 結果の解釈
# クラスの説明を取得して、分類の結果を出力する
class_desc = net.GetClassDesc(class_idx)
# print out the result
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

