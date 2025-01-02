import sys
import boto3
import json
from botocore.exceptions import ClientError
import base64
from PIL import Image

# Bedrock runtime client の作成, Nova Lite モデルを使う。バージニア北部リージョンで利用可能 
bedrockRuntime = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.nova-canvas-v1:0"

def runjob():
    # プロンプト入力例
    if len(sys.argv) < 2:
        print("書式$ python3 bedrock_image.py 'ここに生成したい画像を説明する'")
        sys.exit(1)
    else:
        prompt = sys.argv[1]
        print( prompt )

    # モデルのネイティブ構造を使ってリクエストする
    body_json = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": 0
        }
    })

    # リクエスト実行
    try:
        print("invoke実行")
        response = bedrockRuntime.invoke_model(
            body = body_json,
            modelId = MODEL_ID,
            accept= "application/json",
            contentType = "application/json"
        )
        # 画像取得
        print("画像処理")
        response_body = json.loads( response.get("body").read() )
        base64_image = response_body.get("images")[0]
        base64_bytes = base64_image.encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)

        # generated_text の取得と表示 
        with open("generatedImage.png", 'wb') as file:
            file.write( image_bytes )
        print(f"画像を保存しました")

    except (ClientError, Exception) as e:
        print(f"ERROR: 実行できません '{MODEL_ID}', Reason: {e}")
        exit(1)

if __name__ == "__main__":
    runjob()

