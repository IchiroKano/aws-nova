import sys
import boto3
import json
from botocore.exceptions import ClientError

from ddtrace.llmobs import LLMObs # Datadog
from ddtrace.llmobs.decorators import embedding, llm, retrieval, workflow # Datadog

# Bedrock runtime client の作成, Nova Lite モデルを使う。バージニア北部リージョンで利用可能
bedrockRuntime = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.nova-lite-v1:0"

# モデルのネイティブ構造を使ってリクエストする
@llm(model_name="nova-lite-v1:0", model_provider="amazon")
def llm_call( prompt ):
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
            body = json.dumps( requestPayload ),
            contentType = "application/json"
        )

        # generated_text の取得と表示
        responseBody = json.loads( response['body'].read() )
        generatedText = responseBody['output']['message']['content'][0]['text']
        print( generatedText )

        LLMObs.annotate(
            span=None,
            input_data=[{"role": "user", "content": prompt}],
            output_data=[{"role": "assistant", "content": generatedText}],
            metadata={"temperature": 0.7, "max_tokens": 512},
            tags={"host": "game.funnygeekjp.com"},
        )

        return response

    except (ClientError, Exception) as e:
        print(f"ERROR: 実行できません '{MODEL_ID}', Reason: {e}")
        exit(1)

if __name__ == "__main__":
    # コマンドラインからプロンプトを取得
    if len(sys.argv) < 2:
        print("書式$ python3 bedrock.py 'ここに質問をテキストで記載する'")
        sys.exit(1)
    else:
        prompt = sys.argv[1]
        llm_call( prompt )




