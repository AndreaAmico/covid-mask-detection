import os
import io
import boto3
import json
import base64

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print(ENDPOINT_NAME)
    print("Received event: " + json.dumps(event, indent=2))
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json', Body=json.dumps(event))
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)
    
    return result
