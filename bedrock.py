import boto3
import json
from botocore.exceptions import ClientError

# Bedrock runtime client の作成, Nova Lite モデルを使う。バージニア北部リージョンで利用可能
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.nova-lite-v1:0"

# プロンプト入力例
prompt = "毎年お正月に日の出を楽しみにするのは何故？"

# モデルのネイティブ構造を使ってリクエストする
request_payload = {
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
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(request_payload),  # JSON形式のペイロードを渡す
        contentType="application/json"  # リクエストのコンテンツタイプを指定
    )

    # generated_text の取得と表示 
    response_body = json.loads(response['body'].read())
    generated_text = response_body['output']['message']['content'][0]['text']
    print(generated_text)

    # レスポンス全体を出力してデバッグ
    #response_body = json.loads(response['body'].read())
    #print("\nResponse Body:")
    #print(json.dumps(response_body, indent=4))  # 整形して表示

    # generated_text の取得と表示
    #if 'output' in response_body:
    #    generated_text = response_body['output']['message']['content'][0]['text']
    #    print(generated_text)
    #else:
    #    print("\nERROR: 'output' キーがレスポンスに含まれていません。")

except (ClientError, Exception) as e:
    print(f"ERROR: 実行できません '{model_id}', Reason: {e}")
    exit(1)
