import sys
import boto3
import json
from botocore.exceptions import ClientError

# Bedrock runtime client の作成, Nova Lite モデルを使う。バージニア北部リージョンで利用可能 
bedrockRuntime = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.nova-lite-v1:0"

def runjob():
    # プロンプト入力例
    if len(sys.argv) < 2:
        print("書式$ python3 bedrock.py 'ここに質問をテキストで記載する'")
        sys.exit(1)
    else:
        prompt = sys.argv[1]
        print( prompt )

    # モデルのネイティブ構造を使ってリクエストする
    requestPayload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 512,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9
        }
    }

    # リクエスト実行
    try:
        response = bedrockRuntime.invoke_model(
            modelId = MODEL_ID,
            body = json.dumps( requestPayload ),  # JSON形式のペイロードを渡す
            contentType = "application/json"  # リクエストのコンテンツタイプを指定
        )

        # generated_text の取得と表示 
        responseBody = json.loads( response['body'].read() )
        generatedText = responseBody['output']['message']['content'][0]['text']
        print( generatedText )

    except (ClientError, Exception) as e:
        print(f"ERROR: 実行できません '{MODEL_ID}', Reason: {e}")
        exit(1)

if __name__ == "__main__":
    runjob()

