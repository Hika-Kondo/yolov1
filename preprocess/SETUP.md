# ファイルの説明
* `dataframe_preprocess.py` バウンディングボックスの答えが書いてあるcsvファイルをリサイズしてyolov1で扱いやすいファイル形式に変換する
* `imsize_resize.py` がぞの一番小さいサイズの画像に対してリサイズする
* `move.py` 画像データをimagesディレクトリに移動する
* `resize.py` 画像のサイズを1/8にクロップする

# 手順
1. `move.py` でファイルを移動する
1. `imsize_resize.py` サイズを均一にする
1. `resize.py` でファイルを1/8にクロップする
1. `dataframe_preprocess.py` でtxtファイルを作成する。

```
python move.py
python imsize_resize.py
python resize.py
python dataframe_preprocess.py
```