# ai-pptx-search

AIを使用したパワポの検索アプリのデモ

## 準備

このアプリを使うためには、Open AIのAPIが必要です。

https://platform.openai.com/ からAPIのkeyを取得して（有料です）、プロジェクト直下の`.env`に設定してください。

```.env
export OPENAI_API_KEY=/* 生成したAPI key */
```

## 使い方

#### ①pythonの仮想環境を有効化

```bash
source venv/bin/activate
```

#### ②パワポの読み込み

※このステップは一番最初、またはパワポを追加/削除したときに実行してください

- 使いたいパワポのファイルを`data`ディレクトリに入れる
- パワポを`json`ファイルにするため以下を実行する

```bash
python pptx_extract.py
```

#### ③読み込んだものをチャンクに分ける

※このステップは一番最初、またはパワポを追加/削除したときに実行してください

```bash
python chunks_slides.py
```

#### ④ベクトル化する

※このステップは一番最初、またはパワポを追加/削除したときに実行してください

```bash
python embed_chunks.py
```

#### ⑤アプリを立ち上げる

```bash
streamlit run app.py
```