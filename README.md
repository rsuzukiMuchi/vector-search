# ベクトル検索

## 手順
* cloud storage に画像をアップロード
* オブジェクトテーブル を作成する
```
CREATE OR REPLACE EXTERNAL TABLE
 `オブジェクトテーブル名`
WITH CONNECTION `your-connection`
OPTIONS
( object_metadata = 'SIMPLE',
   uris = ['gs://xxx/*']
);
```

* モデルの作成
```
CREATE OR REPLACE MODEL
 モデル名 REMOTE
WITH CONNECTION `your-connection`
OPTIONS (endpoint = 'multimodalembedding@001')
```

* エンベディング作成
```
CREATE OR REPLACE TABLE `エンベディングされたテーブル名`
AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `モデル名`,
  TABLE `オブジェクトテーブル名`)
WHERE content_type = 'image/jpeg'
```
