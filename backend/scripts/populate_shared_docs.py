# import boto3

# s3 = boto3.resource('s3')
# s3_client = boto3.client('s3')

# source_bucket_name = 'shared_docs_serverless_pdf'
# destination_bucket_name = 'serverless-pdf-chat-us-east-1-257093326925'

# def list_subdirectories(bucket_name, prefix=''):
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
#     return [content['Prefix'] for content in response.get('CommonPrefixes', [])]

# def copy_directory(source_bucket, destination_bucket, source_prefix, destination_prefix):
#     for obj_summary in s3.Bucket(source_bucket).objects.filter(Prefix=source_prefix):
#         source = {'Bucket': source_bucket, 'Key': obj_summary.key}
#         destination_key = obj_summary.key.replace(source_prefix, destination_prefix, 1)
#         s3.meta.client.copy(source, destination_bucket, destination_key)

# def replicate_shared_directories():
#     shared_subdirectories = list_subdirectories(source_bucket_name)
#     target_subdirectories = list_subdirectories(destination_bucket_name)
    
#     for target_subdir in target_subdirectories:
#         for shared_subdir in shared_subdirectories:
#             destination_prefix = target_subdir + shared_subdir.split('/')[-2] + '/'
#             copy_directory(source_bucket_name, destination_bucket_name, shared_subdir, destination_prefix)
#             print(f"Copied {shared_subdir} to {destination_prefix}")

# if __name__ == "__main__":
#     replicate_shared_directories()

# import boto3

# s3 = boto3.resource('s3')
# s3_client = boto3.client('s3')

# source_bucket_name = 'shared_docs_serverless_pdf'
# destination_bucket_name = 'serverless-pdf-chat-us-east-1-257093326925'

# def list_subdirectories(bucket_name, prefix=''):
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
#     return [content['Prefix'] for content in response.get('CommonPrefixes', [])]

# def check_directory_exists(bucket_name, prefix):
#     # Check if any objects exist with the given prefix (folder path) in the specified bucket
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
#     return 'Contents' in response

# def copy_directory(source_bucket, destination_bucket, source_prefix, destination_prefix):
#     if not check_directory_exists(destination_bucket, destination_prefix):
#         for obj_summary in s3.Bucket(source_bucket).objects.filter(Prefix=source_prefix):
#             source = {'Bucket': source_bucket, 'Key': obj_summary.key}
#             destination_key = obj_summary.key.replace(source_prefix, destination_prefix, 1)
#             s3.meta.client.copy(source, destination_bucket, destination_key)
#         print(f"Copied {source_prefix} to {destination_prefix}")
#     else:
#         print(f"Directory {destination_prefix} already exists, skipping copy.")

# def replicate_shared_directories():
#     shared_subdirectories = list_subdirectories(source_bucket_name)
#     target_subdirectories = list_subdirectories(destination_bucket_name)
    
#     for target_subdir in target_subdirectories:
#         for shared_subdir in shared_subdirectories:
#             shared_folder_name = shared_subdir.rstrip('/').split('/')[-1]  # Get the folder name without trailing slash
#             destination_prefix = target_subdir + shared_folder_name + '/'
#             copy_directory(source_bucket_name, destination_bucket_name, shared_subdir, destination_prefix)

# if __name__ == "__main__":
#     replicate_shared_directories()

# import boto3

# s3_client = boto3.client('s3')
# bucket_name = 'serverless-pdf-chat-us-east-1-257093326925'
# source_prefix = 'shared_docs_serverless_pdf/'

# def list_subdirectories(bucket_name, prefix=''):
#     paginator = s3_client.get_paginator('list_objects_v2')
#     result = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
#     for page in result:
#         for prefix in page.get('CommonPrefixes', []):
#             yield prefix['Prefix']

# def destination_folder_exists(bucket_name, prefix):
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
#     return 'Contents' in response

# def copy_objects(source_bucket, destination_bucket, source_prefix, destination_prefix):
#     paginator = s3_client.get_paginator('list_objects_v2')
#     for page in paginator.paginate(Bucket=source_bucket, Prefix=source_prefix):
#         for obj in page.get('Contents', []):
#             copy_source = {'Bucket': source_bucket, 'Key': obj['Key']}
#             destination_key = obj['Key'].replace(source_prefix, destination_prefix, 1)
#             if not destination_folder_exists(destination_bucket, destination_key):
#                 s3_client.copy(copy_source, destination_bucket, destination_key)
#                 print(f"Copied {obj['Key']} to {destination_key}")

# def replicate_shared_directories():
#     shared_subdirectories = list(list_subdirectories(bucket_name, source_prefix))
#     all_subdirectories = list(list_subdirectories(bucket_name))

#     for target_subdir in all_subdirectories:
#         if target_subdir.startswith(source_prefix) or target_subdir == source_prefix:
#             continue  # Skip the source directory and its subdirectories
        
#         for shared_subdir in shared_subdirectories:
#             shared_folder_name = shared_subdir.rstrip('/').split('/')[-1]
#             destination_prefix = f"{target_subdir}{shared_folder_name}/"
#             copy_objects(bucket_name, bucket_name, shared_subdir, destination_prefix)
#             print(f"Completed copying to {destination_prefix}")

# if __name__ == "__main__":
#     replicate_shared_directories()


import boto3

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('serverless-pdf-chat-DocumentTable-1HQDF1EKYYW1W')
bucket_name = 'serverless-pdf-chat-us-east-1-257093326925'
source_prefix = 'shared_docs_serverless_pdf/'

def list_subdirectories(bucket_name, prefix=''):
    paginator = s3_client.get_paginator('list_objects_v2')
    result = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    for page in result:
        for prefix in page.get('CommonPrefixes', []):
            yield prefix['Prefix']

def destination_folder_exists(bucket_name, prefix):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
    return 'Contents' in response

def copy_objects(source_bucket, destination_bucket, source_prefix, destination_prefix):
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=source_bucket, Prefix=source_prefix):
        for obj in page.get('Contents', []):
            copy_source = {'Bucket': source_bucket, 'Key': obj['Key']}
            destination_key = obj['Key'].replace(source_prefix, destination_prefix, 1)
            if not destination_folder_exists(destination_bucket, destination_key):
                s3_client.copy(copy_source, destination_bucket, destination_key)
                print(f"Copied {obj['Key']} to {destination_key}")
                update_document_status(obj['Key'].split('/')[-1], "READY")

def update_document_status(filename, new_status):
    # Assuming 'filename' is a key or you have a Global Secondary Index to query on 'filename'
    response = table.scan(
        FilterExpression='filename = :filename AND docstatus = :status',
        ExpressionAttributeValues={
            ':filename': filename,
            ':status': "PROCESSING"
        }
    )
    for item in response.get('Items', []):
        response = table.update_item(
            Key={
                'filename': filename
            },
            UpdateExpression='SET docstatus = :new_status',
            ExpressionAttributeValues={
                ':new_status': new_status,
            },
            ReturnValues="UPDATED_NEW"
        )
        print(f"Updated docstatus for {filename} to {new_status}")

def replicate_shared_directories():
    shared_subdirectories = list(list_subdirectories(bucket_name, source_prefix))
    all_subdirectories = list(list_subdirectories(bucket_name))

    for target_subdir in all_subdirectories:
        if target_subdir.startswith(source_prefix) or target_subdir == source_prefix:
            continue  # Skip the source directory and its subdirectories
        
        for shared_subdir in shared_subdirectories:
            shared_folder_name = shared_subdir.rstrip('/').split('/')[-1]
            destination_prefix = f"{target_subdir}{shared_folder_name}/"
            copy_objects(bucket_name, bucket_name, shared_subdir, destination_prefix)
            print(f"Completed copying to {destination_prefix}")

if __name__ == "__main__":
    replicate_shared_directories()
